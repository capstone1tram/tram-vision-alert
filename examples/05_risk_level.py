"""05. 움직임 계산(속도·TTC·필요 감속도) + 위험 단계 판정 + 깜빡임 방지

입력은 '물체 ID별로 시간에 따른 (Z, d_lat)'뿐이라서, 카메라 없이도 시험할 수 있습니다.

- 다가오는 속도 v   : 최근 N 프레임의 Z를 직선으로 맞춘 기울기의 반대 부호 (트램 속도 + 물체 속도)
- TTC               : Z / v
- 필요 감속도 a_req : v^2 / (2 * (Z - v * 반응시간))   -> 상용 제동(1.2 m/s^2)보다 크면 위험
- 진입 예상 시간    : 통로 경계까지 남은 옆 거리 / 선로 쪽 옆 속도

실행:  python 05_risk_level.py      -> 선로를 건너는 사람 시나리오에서 등급 변화를 출력
"""
from collections import deque

import numpy as np

import config as C


class TrackState:
    """물체 하나의 최근 기록과 움직임 계산."""

    def __init__(self, maxlen=10):
        self.hist = deque(maxlen=maxlen)          # (t, Z, d_lat)

    def add(self, t, z, d_lat):
        self.hist.append((t, z, d_lat))

    def _slope(self, idx):
        if len(self.hist) < 3:
            return 0.0
        a = np.array(self.hist)
        return float(np.polyfit(a[:, 0], a[:, idx], 1)[0])

    def metrics(self):
        t, _, _ = self.hist[-1]
        a = np.array(self.hist)
        z = float(np.median(a[-5:, 1]))                       # 중앙값으로 튀는 값 줄이기
        d = float(np.median(a[-5:, 2]))
        v_close = -self._slope(1)                              # Z가 줄면 +
        v_lat = self._slope(2)
        ttc = z / v_close if v_close > 0.1 else float("inf")
        brake_dist = z - v_close * C.T_REACT_S
        a_req = v_close ** 2 / (2 * brake_dist) if v_close > 0.1 and brake_dist > 0.1 else (
            float("inf") if v_close > 0.1 else 0.0)
        d_edge = abs(d) - C.CORRIDOR_HALF_M                    # 통로 경계까지 (안이면 음수)
        toward = -np.sign(d) * v_lat                           # 선로 쪽으로 움직이면 +
        tte = d_edge / toward if (d_edge > 0 and toward > 0.05) else float("inf")
        return {"Z": z, "d_lat": d, "v": v_close, "ttc": ttc, "a_req": a_req,
                "in_corridor": d_edge <= 0, "d_edge": d_edge,
                "approaching": toward > 0.05, "time_to_enter": tte}


def risk_level(m):
    """metrics 딕셔너리 -> 0 안전 / 1 주의 / 2 경고 / 3 위험."""
    if m["in_corridor"]:
        if m["ttc"] <= C.TTC_DANGER_S or m["a_req"] >= C.A_SERVICE:
            return 3
        if m["ttc"] <= C.TTC_WARN_S:
            return 2
        return 1
    if m["approaching"] and m["time_to_enter"] <= m["ttc"] <= C.TTC_WARN_S:
        return 2
    if m["d_edge"] <= C.BUFFER_M:
        return 1
    return 0


class LevelSmoother:
    """올릴 때는 빨리(RAISE_FRAMES 연속), 내릴 때는 천천히(LOWER_HOLD_S 유지)."""

    def __init__(self):
        self.level = 0
        self.up_count = 0
        self.low_since = None

    def update(self, raw, t):
        if raw > self.level:
            self.up_count += 1
            self.low_since = None
            if self.up_count >= C.RAISE_FRAMES:
                self.level, self.up_count = raw, 0
        elif raw < self.level:
            self.up_count = 0
            self.low_since = self.low_since if self.low_since is not None else t
            if t - self.low_since >= C.LOWER_HOLD_S:
                self.level, self.low_since = raw, None
        else:
            self.up_count, self.low_since = 0, None
        return self.level


if __name__ == "__main__":
    # 시나리오: 트램 30 km/h, 사람이 60 m 앞 오른쪽 4 m에서 1.4 m/s로 선로를 건넘 (00번 영상과 같음)
    fps, v_tram = 15, 30 / 3.6
    rng = np.random.default_rng(0)
    st, sm = TrackState(), LevelSmoother()
    print("  t(s)   Z(m)  옆(m)  TTC(s)  a_req   원래 -> 표시")
    for i in range(90):
        t = i / fps
        z = 60 - v_tram * t + rng.normal(0, 0.5)               # 측정 잡음 흉내
        d = 4.0 - 1.4 * t + rng.normal(0, 0.05)
        st.add(t, z, d)
        if len(st.hist) < 3:
            continue
        m = st.metrics()
        raw = risk_level(m)
        shown = sm.update(raw, t)
        if i % 5 == 0:
            print(f"{t:6.2f} {m['Z']:6.1f} {m['d_lat']:6.2f} {m['ttc']:7.2f} {m['a_req']:6.2f}"
                  f"   {C.LEVEL_NAMES_KO[raw]:>3} -> {C.LEVEL_NAMES_KO[shown]}")
