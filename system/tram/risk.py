"""움직임 계산(TTC·필요 감속도) + 4단계 위험 등급 + 깜빡임 방지

입력은 칼만 상태 하나(X, Z, 속도, 불확실성)뿐이라 카메라 없이도 시험할 수 있습니다.

- 다가오는 속도 v : 칼만 상태의 -Vz (트램 속도 + 물체 속도, 트램 기준 상대값)
- TTC            : Z / v
- 필요 감속도    : v² / (2 × (Z − v × 반응시간))   -> 상용 제동(1.2 m/s²)보다 크면 위험
- 진입 예상 시간 : 위험 구역 경계까지 남은 옆 거리 / 선로 쪽 옆 속도

등급 규칙 (발표 3·보고서 기준)
  위험(3): 위험 구역 안 + (TTC ≤ 3초 또는 필요 감속도 > 1.2)
  경고(2): 위험 구역 안 + TTC ≤ 5초, 또는 구역으로 들어오는 중이고 트램보다 먼저 들어옴(TTC ≤ 5초)
  주의(1): 위험 구역 안이지만 멂, 또는 완충 구역(바깥 1 m)
  안전(0): 그 외
"""
import math

from .config import cfg

INF = float("inf")


def metrics(s, widen=False):
    """칼만 상태 -> 판단 지표.  widen=True면 WIDEN_SIGMA·σ만큼 '더 가깝고 선로 쪽'이라고 가정(보수적)."""
    k = cfg.WIDEN_SIGMA if widen else 0.0
    X = max(abs(s["X"]) - k * s["sigma_X"], 0.0)
    Z = max(s["Z"] - k * s["sigma_Z"], 0.01)
    vc, vx = s["v_close"], s["vx"]
    approaching_tram = vc > cfg.V_MIN
    ttc = Z / vc if approaching_tram else INF
    brake = Z - vc * cfg.T_REACT_S
    if not approaching_tram:
        a_req = 0.0
    elif brake > 0.01:
        a_req = vc * vc / (2 * brake)
    else:
        a_req = INF                                   # 반응하는 사이에 이미 닿음
    d_edge = X - cfg.CORRIDOR_HALF_M                  # 위험 구역 경계까지 (안이면 음수)
    toward = -math.copysign(1.0, s["X"]) * vx         # 선로 쪽으로 움직이면 +
    entering = (toward > cfg.V_LAT_MIN + cfg.ENTER_SIGMA * s.get("sigma_vx", 0.0)   # 옆 속도가 오차보다 확실히 클 때만
                and s.get("age", 0.0) <= cfg.ENTER_MAX_AGE_S                      # 지금 보이는 물체만 (예측만 하는 트랙 제외)
                and s.get("hits", 99) >= cfg.ENTER_MIN_HITS)                       # 속도가 자리 잡은 트랙만
    tte = d_edge / toward if (d_edge > 0 and entering) else INF
    # 구역을 다 건너가는 데 걸리는 시간 (ENTER_EXIT_CHECK: 트램이 올 때 이미 지나가 버렸으면 경고 안 함)
    tx = tte + 2 * cfg.CORRIDOR_HALF_M / toward if (d_edge > 0 and entering) else INF
    return {"time_to_exit": tx, "Z": Z, "X": X if s["X"] >= 0 else -X, "v": vc, "ttc": ttc, "a_req": a_req,
            "in_corridor": d_edge <= 0, "d_edge": d_edge, "entering": entering, "time_to_enter": tte}


def level_of(m):
    """지표 -> 0 안전 / 1 주의 / 2 경고 / 3 위험"""
    if m["in_corridor"]:
        if m["ttc"] <= cfg.TTC_DANGER_S or m["a_req"] > cfg.A_SERVICE:
            return 3
        if m["ttc"] <= cfg.TTC_WARN_S:
            return 2
        return 1
    # 아직 밖이지만 트램이 오기 전에 구역 안으로 들어올 것 같음
    if m["entering"] and m["time_to_enter"] <= m["ttc"] <= cfg.TTC_WARN_S:
        if not cfg.ENTER_EXIT_CHECK or m["ttc"] <= m["time_to_exit"]:
            return 2
    if m["d_edge"] <= cfg.BUFFER_M:
        return 1
    return 0


def assess(s, widen=False):
    m = metrics(s, widen)
    return level_of(m), m


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
            if self.up_count >= cfg.RAISE_FRAMES:
                self.level, self.up_count = raw, 0
        elif raw < self.level:
            self.up_count = 0
            if self.low_since is None:
                self.low_since = t
            if t - self.low_since >= cfg.LOWER_HOLD_S:
                self.level, self.low_since = raw, None
        else:
            self.up_count, self.low_since = 0, None
        return self.level
