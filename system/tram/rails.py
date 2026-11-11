"""선로 검출 (모형 시험장의 노란 테이프 선로) + 위험 구역(통로)

RailSafeNet과 같은 '줄 단위 스캔' 방식 (예제 03과 같은 알고리즘):
 1) 레일 색만 남긴 마스크 (HSV 색 범위 -> 모폴로지로 점 잡음 제거)
 2) 화면 아래쪽 여러 줄(y)에서 레일 픽셀의 x 위치를 찾아 왼쪽/오른쪽 레일 점을 모음
 3) 각 레일을 x = a*y + b 직선(곡선이면 2차식)으로 맞춤 (np.polyfit)
 4) 줄마다 궤간 픽셀 폭 g(y) = 오른쪽 x - 왼쪽 x  ->  1 px = 궤간 / g(y) m
 5) 중심선에서 양쪽으로 CORRIDOR_HALF_M 만큼이 위험 구역

실제 매립형 레일로 넓힐 때는 1단계 마스크만 YOLO-seg 결과로 바꾸면 2~5단계는 그대로 씁니다.
"""
import cv2
import numpy as np

from .config import cfg


class RailModel:
    """검출된 두 레일. y(화면 줄)를 넣으면 x 위치와 궤간 픽셀 폭을 알려 줌."""

    def __init__(self, left_coef, right_coef, y_top, y_bottom):
        self.l = np.poly1d(left_coef)
        self.r = np.poly1d(right_coef)
        self.y_top, self.y_bottom = int(y_top), int(y_bottom)

    def left_x(self, y):
        return float(self.l(y))

    def right_x(self, y):
        return float(self.r(y))

    def center_x(self, y):
        return (self.left_x(y) + self.right_x(y)) / 2

    def gauge_px(self, y):
        return self.right_x(y) - self.left_x(y)

    def m_per_px(self, y):
        g = self.gauge_px(y)
        return cfg.GAUGE_M / g if g > 1 else None

    def corridor_x(self, y, half_m=None):
        """줄 y에서 위험 구역 왼쪽/오른쪽 끝 x."""
        half_m = cfg.CORRIDOR_HALF_M if half_m is None else half_m
        mpp = self.m_per_px(y)
        if mpp is None:
            return None
        cx = self.center_x(y)
        return cx - half_m / mpp, cx + half_m / mpp

    # 노트북 -> 파이 전송용 ------------------------------------------------
    def to_dict(self):
        return {"left": self.l.coeffs.tolist(), "right": self.r.coeffs.tolist(),
                "y_top": self.y_top, "y_bottom": self.y_bottom}

    @classmethod
    def from_dict(cls, d, scale=1.0):
        """scale: 노트북이 받은 사진이 원본의 몇 배 크기였나 (원본 좌표로 되돌림).
        작은 사진 좌표의 x = Σ c_k y^k  ->  원본 X = s·Σ c_k (Y/s)^k = Σ c_k s^(1-k) Y^k"""
        if d is None:
            return None
        if scale == 1.0:
            return cls(d["left"], d["right"], d["y_top"], d["y_bottom"])
        s = 1.0 / scale                       # 작은 사진 -> 원본 배율

        def conv(coef):
            deg = len(coef) - 1               # poly1d는 높은 차수부터
            return [c * s ** (1 - (deg - i)) for i, c in enumerate(coef)]

        return cls(conv(d["left"]), conv(d["right"]), d["y_top"] * s, d["y_bottom"] * s)


def rail_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.RAIL_HSV_LO), np.array(cfg.RAIL_HSV_HI))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_rails(frame=None, mask=None):
    """사진(또는 마스크) -> RailModel. 못 찾으면 None."""
    if mask is None:
        mask = rail_mask(frame)
    h, w = mask.shape
    ys = np.linspace(int(h * cfg.RAIL_Y_START), h - 1, cfg.RAIL_ROWS).astype(int)
    left_pts, right_pts = [], []
    for y in ys:
        xs = np.flatnonzero(mask[y])
        if len(xs) < 2:
            continue
        # 픽셀 묶음 사이 가장 큰 빈칸을 기준으로 왼쪽/오른쪽 레일 나누기
        gaps = np.diff(xs)
        k = int(np.argmax(gaps))
        if gaps[k] < 3:                       # 레일이 하나만 보임
            continue
        left_pts.append((y, xs[: k + 1].mean()))
        right_pts.append((y, xs[k + 1:].mean()))
    if len(left_pts) < cfg.RAIL_MIN_POINTS:
        return None
    L, R = np.array(left_pts), np.array(right_pts)
    deg = cfg.RAIL_DEG
    model = RailModel(np.polyfit(L[:, 0], L[:, 1], deg), np.polyfit(R[:, 0], R[:, 1], deg),
                      L[:, 0].min(), L[:, 0].max())
    # 맨 위 줄에서 두 레일이 겹치거나 뒤집히면 잘못 맞춘 것
    if model.gauge_px(model.y_top) <= 2 or model.gauge_px(model.y_bottom) <= 2:
        return None
    return model
