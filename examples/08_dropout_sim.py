"""08. 연결 끊김 시뮬레이션: 칼만 예측이 실제로 얼마나 쓸모 있나?

노트북 연결이 끊긴 뒤 파이가 할 수 있는 선택 세 가지를 비교합니다.
  A. 마지막 결과 그대로 유지 (hold-last)
  B. 칼만 예측 위치로 판정 (KF)
  C. 칼만 예측 + 불확실성만큼 통로를 넓혀 보수적으로 판정 (KF+2σ)  <- 우리 안
보행자 행동은 '계속 걷기 50% / 갑자기 멈춤 25% / 되돌아감 25%'로 섞어서, 모델에 없는 움직임도 넣습니다.

결과 지표 (끊긴 뒤 h초 시점):
  - 앞/옆 위치 오차 (중앙값, 90% 값)
  - 통로 안 사람을 '밖'이라고 놓친 비율 (가장 중요, 0에 가까워야 함)
  - 통로 밖 사람을 '안'이라고 본 비율 (헛경보)

실행:  python 08_dropout_sim.py   -> 표 출력 + dropout_result.csv 저장 (약 10초)
"""
import csv

import numpy as np

import config as C
from importlib import import_module

kt = import_module("07_kalman_tracker")

FPS = 10
HORIZONS = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
N_TRIALS = 3000
HALF = C.CORRIDOR_HALF_M


def make_trajectory(rng, t_drop, t_end):
    """사람 1명의 실제 궤적 (트램 기준 상대 좌표). 반환: 시간 배열, X, Z"""
    dt = 1 / FPS / 10                                   # 실제 움직임은 100 Hz로 촘촘히
    ts = np.arange(0, t_end + 1e-9, dt)
    v_tram = rng.uniform(20, 40) / 3.6
    side = rng.choice([-1, 1])
    x = side * rng.uniform(3.0, 7.0)
    vx = -side * rng.uniform(0.8, 1.8)                  # 선로 쪽으로 걷기
    z = rng.uniform(30, 60)
    kind = rng.choice(["walk", "stop", "turn"], p=[0.5, 0.25, 0.25])
    t_change = t_drop + rng.uniform(-0.5, 2.0)          # 끊김 전후 어딘가에서 행동 변화
    a_mag = rng.uniform(1.5, 3.0)
    X, Z = [], []
    for t in ts:
        if kind != "walk" and t >= t_change:
            target = 0.0 if kind == "stop" else side * 1.2  # 멈추거나 반대로 1.2 m/s
            if abs(vx - target) > a_mag * dt:
                vx += np.sign(target - vx) * a_mag * dt
            else:
                vx = target
        x += vx * dt
        z -= v_tram * dt
        X.append(x)
        Z.append(z)
    return ts, np.array(X), np.array(Z)


def run():
    rng = np.random.default_rng(1)
    rows = {h: {"ez_hold": [], "ez_kf": [], "ex_hold": [], "ex_kf": [],
                "miss": {"hold": 0, "kf": 0, "kf2s": 0}, "fa": {"hold": 0, "kf": 0, "kf2s": 0},
                "n_in": 0, "n_out": 0} for h in HORIZONS}
    for _ in range(N_TRIALS):
        t_drop = 2.0
        ts, X, Z = make_trajectory(rng, t_drop, t_drop + max(HORIZONS))
        frame_idx = np.arange(0, len(ts), 10)           # 10 FPS 측정
        kf, last = None, None
        for i in frame_idx:
            t = ts[i]
            if t > t_drop + 1e-9:
                break
            sx, sz = kt.meas_sigma(Z[i])
            mx, mz = X[i] + rng.normal(0, sx), Z[i] + rng.normal(0, sz)
            if kf is None:
                kf = kt.GroundKF(mx, mz, t)
            else:
                kf.predict(t)
                kf.update(mx, mz)
            last = (mx, mz)
        for h in HORIZONS:
            j = int(round((t_drop + h) / (ts[1] - ts[0])))
            tx, tz = X[j], Z[j]
            kf.predict(t_drop + h)                      # 측정 없이 예측만 (coast)
            px, pz = kf.x[0], kf.x[1]
            sxp = np.sqrt(kf.P[0, 0])
            r = rows[h]
            r["ez_hold"].append(abs(last[1] - tz))
            r["ez_kf"].append(abs(pz - tz))
            r["ex_hold"].append(abs(last[0] - tx))
            r["ex_kf"].append(abs(px - tx))
            truth_in = abs(tx) <= HALF
            dec = {"hold": abs(last[0]) <= HALF, "kf": abs(px) <= HALF,
                   "kf2s": abs(px) - 2 * sxp <= HALF}
            if truth_in:
                r["n_in"] += 1
                for k, v in dec.items():
                    r["miss"][k] += (not v)
            else:
                r["n_out"] += 1
                for k, v in dec.items():
                    r["fa"][k] += v
        # kf 객체를 계속 앞으로 예측했으므로 다음 h는 이어서 계산됨

    out = []
    print(f"시행 {N_TRIALS}회, 측정 {FPS} FPS, 끊김 직전 2초 추적, 통로 반폭 {HALF:.2f} m\n")
    print(" 끊긴 뒤 | 앞거리 오차 중앙값(90%) m   | 옆거리 오차 중앙값(90%) m  | 통로 안 사람 놓침 %        | 헛경보 %")
    print("   (초)  |   유지        칼만         |   유지        칼만         | 유지   칼만   칼만+2σ      | 유지  칼만  칼만+2σ")
    for h in HORIZONS:
        r = rows[h]
        q = lambda a: (np.median(a), np.percentile(a, 90))
        zh, zk, xh, xk = q(r["ez_hold"]), q(r["ez_kf"]), q(r["ex_hold"]), q(r["ex_kf"])
        ni, no = max(r["n_in"], 1), max(r["n_out"], 1)
        m = {k: 100 * v / ni for k, v in r["miss"].items()}
        f = {k: 100 * v / no for k, v in r["fa"].items()}
        print(f"  {h:4.1f}   | {zh[0]:5.2f}({zh[1]:5.2f})  {zk[0]:5.2f}({zk[1]:5.2f}) |"
              f" {xh[0]:5.2f}({xh[1]:5.2f})  {xk[0]:5.2f}({xk[1]:5.2f}) |"
              f" {m['hold']:5.1f}  {m['kf']:5.1f}  {m['kf2s']:5.1f}       |"
              f" {f['hold']:4.1f}  {f['kf']:4.1f}  {f['kf2s']:4.1f}")
        out.append({"horizon_s": h, "z_err_hold_med": zh[0], "z_err_hold_p90": zh[1],
                    "z_err_kf_med": zk[0], "z_err_kf_p90": zk[1],
                    "x_err_hold_med": xh[0], "x_err_hold_p90": xh[1],
                    "x_err_kf_med": xk[0], "x_err_kf_p90": xk[1],
                    "miss_hold_pct": m["hold"], "miss_kf_pct": m["kf"], "miss_kf2s_pct": m["kf2s"],
                    "fa_hold_pct": f["hold"], "fa_kf_pct": f["kf"], "fa_kf2s_pct": f["kf2s"],
                    "n_in": r["n_in"], "n_out": r["n_out"]})
    with open("dropout_result.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("\n저장: dropout_result.csv")


if __name__ == "__main__":
    run()
