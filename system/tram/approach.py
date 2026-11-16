"""소실점 ROI 안에서 '다가오는 열차' 찾기: 배경 차분 -> 크기 검사 -> 확대율로 TTC.

왜 프레임 차분이 아니라 배경 차분인가
  정면으로 다가오는 열차는 화면에서 거의 움직이지 않고 '조금씩 커지기만' 합니다.
  (80 km/h, 500 m 앞이면 크기가 1초에 약 0.6 px 커짐) 앞뒤 프레임을 빼면 거의 0이라 못 잡습니다.
  그래서 '열차가 없을 때의 배경'과 비교합니다.

Background  : 픽셀마다 평균·분산(가우시안) 배경. 열차로 보이는 곳은 배경 갱신을 거의 멈춤(선택 갱신)
              -> 천천히 커지는 열차가 배경에 녹아 사라지지 않게. (갱신이 빠르면(alpha 0.01 = 약 3초) 멀리서
              거의 안 커지는 열차가 몇 초 만에 배경이 되어 사라지는 것을 합성 영상으로 확인함 -> 기본 0.003 ≈ 11초) 화면 전체 밝기 변화(구름)는 중앙값을 빼서 지움
              (mode="mog2"로 OpenCV MOG2와 비교 가능)
candidates  : 전경 덩어리 중 (1) 선로 중심선 근처이고 (2) 폭이 그 높이의 궤간 픽셀 폭 × (열차 폭/궤간) 근처인 것
              -> 옆 선로 열차, 사람·새(너무 작음), 구름 그림자(너무 큼)를 거름
ApproachTracker : 덩어리 하나를 따라가며 1~2초 창에서 크기 s=sqrt(폭×높이)의 역수 1/s를 직선 맞춤
              TTC = s / (ds/dt) = -(1/s) / (d(1/s)/dt).  커지지 않거나(멀어짐·정지) 맞춤이 불확실하면 TTC 없음
"""
from collections import deque

import cv2
import numpy as np


# ── 배경 차분 ─────────────────────────────────────────────────────────
class Background:
    def __init__(self, shape, mode="gauss", alpha=0.003, k_sigma=4.0, sigma_min=2.0,
                 warmup=30, fg_alpha_div=200.0, min_blob_px=4):
        self.mode = mode
        self.alpha, self.k, self.smin = alpha, k_sigma, sigma_min
        self.warmup, self.fg_div = warmup, fg_alpha_div
        self.mu = None
        self.var = None
        self.n = 0
        self.min_blob_px = min_blob_px
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k_hold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        if mode == "mog2":
            self.mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=k_sigma ** 2,
                                                          detectShadows=False)
        self.global_shift = 0.0

    @property
    def ready(self):
        return self.n >= self.warmup

    def noise_sigma(self):
        """배경 잡음 수준(밝기 단위): 픽셀별 시간 표준편차의 중앙값. 정지 장면 실측용."""
        if self.var is None:
            return None
        return float(np.sqrt(np.median(self.var)))

    def apply(self, roi_f32, hold_mask=None):
        """roi_f32: 흔들림 보정된 ROI (float32).  hold_mask: 배경 갱신을 멈출 곳(추적 중인 열차 상자) 255
        -> 전경 마스크(uint8 0/255). 워밍업 중에는 전부 0."""
        I = roi_f32
        self.n += 1
        if self.mode == "mog2":
            lr = -1 if self.n < self.warmup else self.alpha
            fg = self.mog.apply(np.clip(I, 0, 255).astype(np.uint8), learningRate=lr)
            if self.mu is None:
                self.mu, self.var = I.copy(), np.full_like(I, self.smin ** 2)
            d = I - self.mu
            self.mu += 0.05 * d
            self.var += 0.05 * (d * d - self.var)
            if not self.ready:
                return np.zeros(I.shape, np.uint8)
            return self._clean(fg)

        if self.mu is None:
            self.mu = I.copy()
            self.var = np.full_like(I, self.smin ** 2)
            return np.zeros(I.shape, np.uint8)
        d = I - self.mu
        # 화면 전체가 같이 밝아지거나 어두워진 만큼(구름, 남은 노출 변화)은 빼고 봄
        self.global_shift = float(np.median(d))
        d -= self.global_shift
        if not self.ready:                                # 워밍업: 빠르게 평균·분산 잡기
            a = 1.0 / self.n
            self.mu += a * (d + self.global_shift)
            self.var += a * (d * d - self.var)
            return np.zeros(I.shape, np.uint8)
        sigma = np.sqrt(np.maximum(self.var, self.smin ** 2))
        fg = (np.abs(d) > self.k * sigma).astype(np.uint8) * 255
        fg = self._clean(fg)
        # 선택 갱신: 전경·추적 상자는 아주 느리게 (fg_div 배 느리게), 나머지는 alpha
        a = np.full(I.shape, self.alpha, np.float32)
        hold = cv2.dilate(fg, self.k_hold)
        if hold_mask is not None:
            hold = cv2.bitwise_or(hold, hold_mask)
        a[hold > 0] = self.alpha / self.fg_div
        self.mu += a * (d + self.global_shift)
        self.var += a * (d * d - self.var)
        return fg

    def _clean(self, fg):
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)


# ── 후보 덩어리 ───────────────────────────────────────────────────────
def candidates(fg, box, calib, train_w_m=3.0, gauge_m=1.435, min_px=4,
               size_lo=0.4, size_hi=2.2, corridor_gauges=1.5, size_check=True):
    """전경 마스크 -> 후보 목록 [dict]. 좌표는 원본 프레임 기준.
    size_ratio = 실제 폭 / 기대 폭(그 덩어리 바닥 높이에서 궤간 px × 열차폭/궤간)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if max(w, h) < min_px or area < min_px:
            continue
        X, Y = x + box.x0, y + box.y0
        y_b = Y + h                                      # 덩어리 바닥 = 레일에 닿는 곳(으로 가정)
        cx = X + w / 2
        g = calib.gauge_px(y_b) if calib is not None else None
        # (1) 선로 위인가: 덩어리 가운데가 그 높이 선로 중심선에서 궤간 corridor_gauges배 안
        if calib is not None:
            half = max((g or 0) * corridor_gauges, w / 2 + 3, 6)
            if abs(cx - calib.center_x(y_b)) > half:
                continue
        # (2) 크기가 열차 폭 정도인가
        exp_w = g * train_w_m / gauge_m if g else None
        ratio = (w / exp_w) if exp_w and exp_w >= 2 else None
        size_ok = True
        if size_check and ratio is not None:
            size_ok = size_lo <= ratio <= size_hi
        out.append({"x": int(X), "y": int(Y), "w": int(w), "h": int(h), "area": int(area),
                    "y_bottom": int(y_b), "exp_w": exp_w, "ratio": ratio, "size_ok": size_ok})
    return out


# ── 추적 + TTC ────────────────────────────────────────────────────────
class ApproachTracker:
    """후보 하나를 따라가며 크기 변화로 TTC 계산.
    confirm_n / confirm_of : 최근 confirm_of 프레임 중 confirm_n번 이상 보이면 '확정'
    window_s : TTC 맞춤 창 (1~2초).  min_ttc_px : 크기가 이 px 이상일 때만 TTC.  min_growth_t : 기울기의 t값(기울기/표준오차) 이 이 이상이어야 TTC 인정"""

    def __init__(self, window_s=1.5, confirm_n=6, confirm_of=10, lost_s=1.0,
                 min_growth_t=3.0, ttc_smooth=5, min_ttc_px=20.0, jump=0.35, size_mode="area", max_window_s=3.0):
        self.window_s = window_s
        self.confirm_n, self.confirm_of = confirm_n, confirm_of
        self.lost_s = lost_s
        self.min_t = min_growth_t
        self.jump = jump
        self.size_mode = size_mode
        self.max_window_s = max_window_s
        self.min_ttc_px = min_ttc_px           # 이보다 작으면 TTC 계산 안 함 (1 px 오차가 크기의 10% 이상)
        self.hist = deque()                    # (t, s, w, h)
        self.seen = deque(maxlen=confirm_of)
        self.target = None                     # 마지막 후보 dict
        self.t_last = None
        self.ttc_hist = deque(maxlen=ttc_smooth)
        self.ttc = None
        self.growth = None                     # ds/dt (px/s)

    @property
    def confirmed(self):
        return self.target is not None and sum(self.seen) >= self.confirm_n

    def _pick(self, cands):
        ok = [c for c in cands if c["size_ok"]]
        if not ok:
            return None
        if self.target is None:
            return max(ok, key=lambda c: c["area"])
        tx = self.target["x"] + self.target["w"] / 2
        ty = self.target["y"] + self.target["h"] / 2
        lim = max(self.target["w"], self.target["h"]) * 1.5 + 8

        def dist(c):
            return np.hypot(c["x"] + c["w"] / 2 - tx, c["y"] + c["h"] / 2 - ty)
        near = [c for c in ok if dist(c) <= lim]
        return min(near, key=dist) if near else None

    def update(self, t, cands):
        c = self._pick(cands)
        self.seen.append(1 if c is not None else 0)
        if c is None:
            if self.t_last is not None and t - self.t_last > self.lost_s:
                self.reset()
            return self
        s = self.size_of(c)
        if self.hist and abs(s / self.hist[-1][1] - 1) > self.jump:
            # 한 프레임에 크기가 확 바뀜 = 다른 물체(사람·옆 선로 열차)와 붙었다 떨어짐 -> 크기 기록 새로 시작
            self.hist.clear()
            self.ttc_hist.clear()
        self.target, self.t_last = c, t
        self.hist.append((t, s, c["w"], c["h"]))
        while self.hist and t - self.hist[0][0] > self.window_now(s):
            self.hist.popleft()
        self._fit()
        return self

    def size_of(self, c):
        """크기 s (px).  area: sqrt(전경 픽셀 수) - 수백 픽셀의 합이라 상자 폭·높이(정수, 1 px씩 뜀)보다 매끈함.
        bbox: sqrt(폭×높이)."""
        if self.size_mode == "area":
            return float(np.sqrt(c["area"]))
        return float(np.sqrt(c["w"] * c["h"]))

    def window_now(self, s):
        """작을수록 긴 창: 20 px 미만이면 1초에 커지는 양이 1 px도 안 돼서 1.5초로는 잡음에 묻힘."""
        return float(np.clip(self.window_s * 20.0 / max(s, 1.0), self.window_s, self.max_window_s))

    def _fit(self):
        """크기 s(t)가 아니라 1/s(t)를 직선 맞춤.
        등속으로 다가오면 s = k/(T - t) 이므로 1/s = (T - t)/k 가 '정확히' 직선이고, TTC = -(1/s)/(d(1/s)/dt).
        s를 직선 맞춤하면 창 가운데의 기울기를 쓰게 되어 가까울수록 TTC를 길게(늦게) 잡음
        (합성 영상에서 실제 7초일 때 10초로 나옴)."""
        self.ttc, self.growth = None, None
        if len(self.hist) < 6:
            return
        a = np.array(self.hist)
        t, s = a[:, 0], a[:, 1]
        if t[-1] - t[0] < self.window_now(s[-1]) * 0.6:
            return
        u = 1.0 / s
        w = s * s                               # s의 1 px 오차가 1/s에서는 1/s² 만큼 -> 가중치 s²
        tm = (w * t).sum() / w.sum()
        um = (w * u).sum() / w.sum()
        tc = t - tm
        stt = (w * tc * tc).sum()
        slope = float((w * tc * (u - um)).sum() / stt)          # d(1/s)/dt  (다가오면 음수)
        resid = u - (um + slope * tc)
        se = float(np.sqrt((w * resid ** 2).sum() / max(len(s) - 2, 1) / stt))
        u_now = um + slope * tc[-1]
        s_now = 1.0 / u_now if u_now > 0 else s[-1]
        self.growth = -slope * s_now * s_now                    # ds/dt (px/s), 화면 표시·기록용
        if slope >= 0 or -slope < self.min_t * max(se, 1e-9):
            self.ttc_hist.clear()
            return
        if s_now < self.min_ttc_px:
            return
        self.ttc_hist.append(-u_now / slope)
        self.ttc = float(np.median(self.ttc_hist))

    def reset(self):
        self.hist.clear()
        self.ttc_hist.clear()
        self.target, self.t_last, self.ttc, self.growth = None, None, None, None


class LevelGate:
    """0 없음 / 1 열차로 보이는 물체 확정(주의) / 2 TTC < warn_s (경고) / 3 TTC < danger_s (위험)
    (alarm.Alarm.set 의 등급과 같은 번호)
    TTC 한두 번 튀는 값으로 경보가 울리지 않게, 올라가는 것은 hold_s 동안 계속 그 조건일 때만.
    내려가는 것도 hold_s 동안 조건이 풀려 있어야 (깜빡임 방지)."""

    def __init__(self, warn_s, danger_s, hold_s=0.5):
        self.warn_s, self.danger_s, self.hold_s = warn_s, danger_s, hold_s
        self.level = 0
        self.since = {}                         # 후보 등급 -> 그 조건이 처음 참이 된 시각

    def raw(self, tracker):
        if not tracker.confirmed:
            return 0
        if tracker.ttc is not None:
            if tracker.ttc < self.danger_s:
                return 3
            if tracker.ttc < self.warn_s:
                return 2
        return 1

    def update(self, t, tracker):
        r = self.raw(tracker)
        for lv in (1, 2, 3):                     # 등급 lv 이상 조건이 언제부터 계속 참이었나
            if r >= lv:
                self.since.setdefault(lv, t)
            else:
                self.since.pop(lv, None)
        up = max([lv for lv in (1, 2, 3) if lv in self.since and t - self.since[lv] >= self.hold_s] or [0])
        if up > self.level:
            self.level = up
        elif r < self.level:
            # 내려갈 때: 지금 등급 조건이 hold_s 동안 계속 거짓이면
            self.since.setdefault(("off", self.level), t)
            if t - self.since[("off", self.level)] >= self.hold_s:
                self.since.pop(("off", self.level), None)
                self.level = max(up, r)
        else:
            self.since.pop(("off", self.level), None)
        return self.level


def expected_px(dist_m, f_px, size_m):
    """거리 dist_m에서 크기 size_m 물체의 픽셀 수 (핀홀)."""
    return f_px * size_m / dist_m
