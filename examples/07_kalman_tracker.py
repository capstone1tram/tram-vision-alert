"""07. 칼만 필터 추적기 (라즈베리 파이에서 돌릴 부분)

어떤 칼만 필터를 쓰나?
- 선형 칼만 필터(KF) + 등속도(CV) 모델.
- 상태는 '바닥 평면의 미터 좌표' [X, Z, Vx, Vz]  (X: 선로 중심에서 옆 거리, Z: 앞 거리, V: 상대 속도).
- 측정은 04번에서 이미 미터로 바꾼 (X, Z)라서 측정식이 선형(H = [I 0])입니다.
  -> EKF/UKF는 '픽셀을 직접 측정값으로 쓸 때'만 필요하므로 여기서는 쓰지 않습니다.
- 멀리 있을수록 궤간이 몇 픽셀밖에 안 돼서 Z 오차가 Z^2에 비례해 커집니다. 그래서 R(측정 잡음)을 Z에 따라 바꿉니다.

연결이 끊기면 update 없이 predict만 반복(coast)하면서 위치와 불확실성(공분산)을 같이 늘립니다.
numpy만 쓰므로 라즈베리 파이 4/5에서 물체 하나당 수십 마이크로초면 끝납니다.

실행:  python 07_kalman_tracker.py   -> 잡음 섞인 측정 vs 칼만 결과 오차, 1회 계산 시간 출력
"""
import time

import numpy as np

import config as C

FX_PX = (C.IMG_W / 2) / np.tan(np.radians(C.HFOV_DEG / 2))   # 초점거리(px). 보정값으로 바꾸세요.


def meas_sigma(z, px_err=1.5):
    """궤간 기반 거리의 1-시그마 오차.  Z = fx*G/w  ->  dZ = Z^2/(fx*G) * dw"""
    z = max(z, 1.0)
    sig_z = z * z / (FX_PX * C.GAUGE_M) * px_err
    sig_x = 0.05 + z / FX_PX * px_err * 2          # 옆 거리: 박스 중심 흔들림
    return sig_x, sig_z


class GroundKF:
    """상태 x = [X, Z, Vx, Vz],  등속도 모델."""

    def __init__(self, X, Z, t, acc_sigma=1.0):
        self.x = np.array([X, Z, 0.0, 0.0])
        sx, sz = meas_sigma(Z)
        self.P = np.diag([sx ** 2, sz ** 2, 2.0 ** 2, 8.0 ** 2])   # 속도는 처음엔 모름
        self.q = acc_sigma ** 2          # 모델에 없는 가속도(급정거·방향 전환)의 크기
        self.t = t
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])
        self.misses = 0                  # 측정 없이 예측만 한 횟수

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return self.x
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t
        return self.x

    def update(self, X, Z):
        sx, sz = meas_sigma(Z)
        R = np.diag([sx ** 2, sz ** 2])
        y = np.array([X, Z]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.misses = 0

    def gate(self, X, Z):
        """측정이 이 트랙 것인지 판단하는 마할라노비스 거리^2 (9.21 이하면 99% 안)."""
        sx, sz = meas_sigma(Z)
        S = self.H @ self.P @ self.H.T + np.diag([sx ** 2, sz ** 2])
        y = np.array([X, Z]) - self.H @ self.x
        return float(y @ np.linalg.inv(S) @ y)

    def coast(self, t):
        """연결 끊김: 측정 없이 예측만."""
        self.misses += 1
        return self.predict(t)

    def state(self):
        X, Z, vx, vz = self.x
        sx, sz = np.sqrt(self.P[0, 0]), np.sqrt(self.P[1, 1])
        return {"X": X, "Z": Z, "vx": vx, "v_close": -vz, "sigma_X": sx, "sigma_Z": sz}


class Tracker:
    """여러 물체: 가장 가까운(게이트 안) 측정에 붙이고, 못 붙으면 새 트랙."""

    def __init__(self, max_coast_s=1.5):
        self.tracks, self.next_id, self.max_coast_s = {}, 1, max_coast_s
        self.last_meas_t = {}

    def step(self, t, detections):
        """detections: [(X, Z), ...]  (없으면 빈 리스트 = 예측만)"""
        for tr in self.tracks.values():
            tr.predict(t)
        free = list(detections)
        for tid, tr in self.tracks.items():
            if not free:
                break
            d2 = [tr.gate(*m) for m in free]
            k = int(np.argmin(d2))
            if d2[k] < 9.21:
                tr.update(*free.pop(k))
                self.last_meas_t[tid] = t
        for m in free:
            self.tracks[self.next_id] = GroundKF(m[0], m[1], t)
            self.last_meas_t[self.next_id] = t
            self.next_id += 1
        for tid in [k for k, v in self.last_meas_t.items() if t - v > self.max_coast_s]:
            self.tracks.pop(tid)
            self.last_meas_t.pop(tid)
        return {tid: tr.state() for tid, tr in self.tracks.items()}


def demo():
    rng = np.random.default_rng(0)
    fps, v_tram = 10, 30 / 3.6                    # 10 FPS, 트램 30 km/h
    ts = np.arange(0, 5, 1 / fps)
    X = 4.0 - 1.4 * ts                            # 사람이 선로 쪽으로 1.4 m/s
    Z = 60.0 - v_tram * ts                        # 트램이 다가옴
    kf, raw_err, kf_err = None, [], []
    for t, x, z in zip(ts, X, Z):
        sx, sz = meas_sigma(z)
        mx, mz = x + rng.normal(0, sx), z + rng.normal(0, sz)
        if kf is None:
            kf = GroundKF(mx, mz, t)
            continue
        kf.predict(t)
        kf.update(mx, mz)
        if t > 1.0:                               # 처음 1초는 수렴 구간
            raw_err.append(abs(mz - z))
            kf_err.append(abs(kf.x[1] - z))
    s = kf.state()
    print(f"초점거리 fx = {FX_PX:.0f} px (HFOV {C.HFOV_DEG}°)")
    print(f"앞 거리 오차 평균  측정값 {np.mean(raw_err):.2f} m  ->  칼만 {np.mean(kf_err):.2f} m")
    print(f"마지막 추정: Z {s['Z']:.1f} m, 다가오는 속도 {s['v_close']:.1f} m/s (정답 {v_tram:.1f}), "
          f"옆 속도 {s['vx']:.2f} m/s (정답 -1.40)")

    n = 20000
    t0 = time.perf_counter()
    for i in range(n):
        kf.predict(kf.t + 0.1)
    dt_us = (time.perf_counter() - t0) / n * 1e6
    print(f"predict 1회 {dt_us:.0f} µs (이 PC 기준, 파이 5는 대략 2~4배 느림 = 그래도 0.1 ms 수준)")


if __name__ == "__main__":
    demo()
