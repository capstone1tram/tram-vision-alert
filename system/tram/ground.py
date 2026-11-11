"""실제 선로 영상용 거리 계산 (바닥 평면 모델).  distance.py의 확장판입니다.

distance.py는 '레일을 찾은 줄'의 궤간 픽셀 폭만 씁니다. 실제 영상에서는 레일이 보통 20~30 m까지만
잡히는데, 차·사람은 그보다 멀리서도 보입니다. 그래서 바닥이 평평하다는 가정을 하나 더 씁니다.

  바닥이 평평하면, 화면 줄 y에서 궤간 픽셀 폭 g는 y에 '직선'으로 비례합니다.
      g(y) = a · (y − y_h)          y_h = 지평선(소실점) 줄
  (멀수록 좁아지다가 지평선에서 0. 선로가 휘어도 폭은 거리만의 함수라 그대로 성립)

  그러니 레일을 찾은 구간에서 g(y)를 직선으로 맞추면, 그 위쪽(더 먼 곳)까지 g를 연장할 수 있고
      앞 거리  Z = f · 궤간 / g(y)          (f = 초점거리 px)
      옆 거리  X = (x − 중심선 x) · 궤간 / g(y)
  중심선은 레일을 찾은 맨 위 줄에서 직선으로 연장합니다 (곡선이면 멀수록 오차 커짐 -> 방법 'ground'로 표시).

중요: 옆 거리 X와 위험 구역 판정은 궤간 '비율'이라 초점거리 f와 상관없고,
     TTC(= 거리 / 다가오는 속도)도 f가 위아래로 약분돼서 상관없습니다.
     f가 틀리면 '몇 m'와 '필요 감속도'만 같은 비율로 틀립니다. (pc_main.py의 c 키로 보정)
"""
import numpy as np

from .config import cfg


class GroundModel:
    """RailModel -> 궤간 폭 직선 g(y) = a·y + b,  지평선 y_h = −b/a."""

    def __init__(self, rails):
        self.rails = rails
        ys = np.linspace(rails.y_top, rails.y_bottom, 24)
        g = np.array([rails.gauge_px(y) for y in ys])
        ok = g > 2
        self.valid = ok.sum() >= 6
        if not self.valid:
            return
        self.a, self.b = np.polyfit(ys[ok], g[ok], 1)
        if self.a <= 1e-4:                          # 위로 갈수록 넓어지면 잘못 찾은 것
            self.valid = False
            return
        self.y_h = -self.b / self.a
        # 중심선: 맨 위쪽 구간 기울기로 직선 연장
        y0 = rails.y_top
        y1 = rails.y_top + max(10, 0.25 * (rails.y_bottom - rails.y_top))
        self.cx0 = rails.center_x(y0)
        self.cslope = (rails.center_x(y1) - self.cx0) / (y1 - y0)

    def gauge(self, y):
        return self.a * y + self.b

    def center_x(self, y):
        if y >= self.rails.y_top:
            return self.rails.center_x(y)
        return self.cx0 + self.cslope * (y - self.rails.y_top)

    def y_of_z(self, z):
        """앞 거리 z(m)가 화면 몇 번째 줄인지 (그림 그릴 때)."""
        g = cfg.focal_px * cfg.GAUGE_M / z
        return (g - self.b) / self.a

    def camera_height(self):
        """궤간 폭 기울기로 구한 카메라 높이 (m). g = G·(y − y_h)/h  ->  h = G / a  (f와 무관)."""
        return cfg.GAUGE_M / self.a


_cache = {"rails": None, "gm": None}


def ground_of(rails):
    if rails is None:
        return None
    if _cache["rails"] is not rails:
        _cache["rails"], _cache["gm"] = rails, GroundModel(rails)
    gm = _cache["gm"]
    return gm if gm.valid else None


def obj_distance(box, rails, name="person"):
    """박스(x1,y1,x2,y2) + RailModel -> (Z, X, 방법).
    방법: 'gauge'  = 레일을 찾은 구간 안 (가장 정확)
          'ground' = 그 위로 연장 (평평한 바닥 가정)
          'height' = 지평선 근처라 바닥 계산이 불안할 때 사람 키 1.7 m로 (사람만)
          'none'   = 계산 못 함"""
    gm = ground_of(rails)
    if gm is None:
        return None, None, "none"
    x1, y1, x2, y2 = box
    if x1 <= cfg.EDGE_PX or x2 >= cfg.IMG_W - 1 - cfg.EDGE_PX:
        return None, None, "edge"                   # 화면 옆 끝에 잘린 박스 (가운데·발 위치가 틀림)
    fx, fy = (x1 + x2) / 2, y2
    f = cfg.focal_px
    if fy >= rails.y_top:
        g = rails.gauge_px(min(fy, rails.y_bottom))
        if fy > rails.y_bottom:                     # 레일 찾은 범위보다 아래(가까움) -> 직선 연장
            g = gm.gauge(fy)
        if g > 2:
            return f * cfg.GAUGE_M / g, (fx - rails.center_x(fy)) * cfg.GAUGE_M / g, "gauge"
    g = gm.gauge(fy)
    if g > 2:
        z = f * cfg.GAUGE_M / g
        if z <= cfg.GROUND_MAX_Z:
            return z, (fx - gm.center_x(fy)) * cfg.GAUGE_M / g, "ground"
    if name == "person" and y2 - y1 > 8:
        z = f * cfg.PERSON_HEIGHT_M / (y2 - y1)
        if z <= cfg.GROUND_MAX_Z * 1.5:
            return z, (fx - gm.center_x(max(fy, gm.y_h + 1))) * z / f, "height"
    return None, None, "none"


def visible(s, rails):
    """칼만 예측 위치가 아직 카메라에 보이는 곳인지. 화면 맨 아래 줄보다 가깝거나 화면 옆으로 나가면 False."""
    Z, X = s["Z"], s["X"]
    gm = ground_of(rails)
    z_min = 0.5
    if gm is not None:
        g_bot = gm.gauge(cfg.IMG_H - 1)
        if g_bot > 2:
            z_min = cfg.focal_px * cfg.GAUGE_M / g_bot
    if Z < z_min:
        return False
    return abs(X) <= Z * (cfg.IMG_W / 2) / cfg.focal_px + 0.3
