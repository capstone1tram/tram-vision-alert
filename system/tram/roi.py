"""소실점 ROI: 고정 카메라에서 '선로 끝' 주변만 원본 해상도로 잘라 보기.

카메라는 선로 위/옆에 고정되어 선로 방향을 봅니다. 다가오는 열차는 처음에 소실점(두 레일이 만나는 점)
바로 아래·위의 몇 픽셀로 나타나므로, 화면 전체를 줄여서 보는 대신 그 주변(2304 모드에서 약 320x160)만
원본 그대로 잘라 봅니다.

1) RailCalib : 시작할 때 여러 프레임에서 선로를 찾고(rails_photo.find_rails_ego) 중앙값으로 고정
   - 평평한 바닥이면 궤간 픽셀 폭이 y에 대해 직선:  g(y) = k * (y - y_vp)
     -> g(y) = 0 이 되는 y가 소실점 높이(지평선), 그 y에서 선로 중심선 x가 소실점 x
   - 그래서 '레일 두 줄의 교점'을 직접 구하지 않고 g(y), center_x(y) 두 직선을 맞춰 구함 (곡선 선로에도 덜 흔들림)
2) Stabilizer : 레일 진동·삼각대 흔들림 보정. ROI 양옆(선로 밖) 두 조각에서 위상 상관으로 평행 이동량을 재서
   ROI를 되돌려 자름. 0.1° 흔들림이 2304 모드에서 약 3 px라, 보정 안 하면 배경 차분이 레일·전봇대 가장자리를 다 잡음.
"""
import json

import cv2
import numpy as np

from .rails_photo import find_rails_ego, tophat_mask

DETECT_W = 960          # 선로 검출은 이 폭으로 줄여서 (원본 2304는 tophat이 느림)


# ── 1) 소실점·궤간 보정 ────────────────────────────────────────────────
class RailCalib:
    """고정 카메라의 선로 기하 (원본 좌표).
    gauge_px(y) = k*(y - y_vp),  center_x(y) = x_vp + m*(y - y_vp)"""

    def __init__(self, x_vp, y_vp, k=None, m=0.0, n_frames=0, source="rails"):
        self.x_vp, self.y_vp = float(x_vp), float(y_vp)
        self.k = None if k is None else float(k)
        self.m = float(m)
        self.n_frames, self.source = n_frames, source

    @property
    def has_gauge(self):
        return self.k is not None and self.k > 0

    def gauge_px(self, y):
        if not self.has_gauge:
            return None
        return max(self.k * (y - self.y_vp), 0.0)

    def center_x(self, y):
        return self.x_vp + self.m * (y - self.y_vp)

    def to_dict(self):
        return {"x_vp": self.x_vp, "y_vp": self.y_vp, "k": self.k, "m": self.m,
                "n_frames": self.n_frames, "source": self.source}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x_vp"], d["y_vp"], d.get("k"), d.get("m", 0.0), d.get("n_frames", 0),
                   d.get("source", "file"))

    def save(self, path, frame_size):
        d = self.to_dict()
        d["frame_size"] = list(frame_size)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path, frame_size=None):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        c = cls.from_dict(d)
        fs = d.get("frame_size")
        if frame_size and fs and tuple(fs) != tuple(frame_size):   # 해상도가 다르면 비율대로
            s = frame_size[0] / fs[0]
            c.x_vp *= s
            c.y_vp *= s
        return c


def _fit_one(gray, y_start, far_part=0.6):
    """한 프레임 -> (k, y_vp, x_vp, m) 원본 좌표. 못 찾으면 None.
    far_part: 검출된 선로 중 먼 쪽(위쪽) 몇 %만 써서 직선을 맞춤 (가까운 곳의 곡선·렌즈 왜곡 영향 줄임)"""
    H, W = gray.shape
    s = DETECT_W / W
    small = cv2.resize(gray, (DETECT_W, int(round(H * s))), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    mask = tophat_mask(bgr, y_start)
    model = find_rails_ego(mask, y_start)
    if model is None:
        return None
    y0, y1 = model.y_top, model.y_bottom
    ys = np.linspace(y0, y0 + (y1 - y0) * far_part, 20)
    g = np.array([model.gauge_px(y) for y in ys])
    cx = np.array([model.center_x(y) for y in ys])
    if np.any(g <= 1):
        return None
    k, b = np.polyfit(ys, g, 1)                      # g = k*y + b
    if k <= 0:
        return None
    y_vp = -b / k
    m, c = np.polyfit(ys, cx, 1)
    x_vp = m * y_vp + c
    # 작은 그림 좌표 -> 원본 좌표 (k, m 은 기울기라 배율 그대로)
    return k, y_vp / s, x_vp / s, m


def calibrate(frames_gray, y_start=0.45, min_ok=5):
    """여러 프레임 -> RailCalib (중앙값). 성공 프레임이 min_ok보다 적으면 None."""
    res = [r for r in (_fit_one(g, y_start) for g in frames_gray) if r is not None]
    if len(res) < min_ok:
        return None, len(res)
    a = np.array(res)
    k, y_vp, x_vp, m = np.median(a, axis=0)
    return RailCalib(x_vp, y_vp, k, m, n_frames=len(res), source="rails"), len(res)


# ── ROI 위치 ─────────────────────────────────────────────────────────
class RoiBox:
    """소실점 주변 잘라 볼 사각형 (원본 좌표, 화면 밖으로 안 나가게 맞춤).
    dy_frac: 소실점을 ROI 세로의 어디에 둘지 (0.5=가운데, 0.6=조금 아래 -> 위쪽(열차 몸체)을 더 봄)"""

    def __init__(self, calib, frame_size, roi_w, roi_h, dy_frac=0.6):
        W, H = frame_size
        self.w, self.h = int(min(roi_w, W)), int(min(roi_h, H))
        x0 = int(round(calib.x_vp - self.w / 2))
        y0 = int(round(calib.y_vp - self.h * dy_frac))
        self.x0 = int(np.clip(x0, 0, W - self.w))
        self.y0 = int(np.clip(y0, 0, H - self.h))

    @property
    def rect(self):
        return self.x0, self.y0, self.w, self.h

    def crop(self, img):
        return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]


def _overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


# ── 2) 흔들림 보정 ───────────────────────────────────────────────────
class Stabilizer:
    """ROI 왼쪽·오른쪽(선로 밖) 조각에서 기준 프레임 대비 평행 이동(dx, dy)을 재고, 그만큼 되돌려 ROI를 자름.
    기준 프레임은 처음 한 번 잡고 고정 (매 프레임 앞 프레임과 비교하면 오차가 쌓여 천천히 흘러감).
    refresh(): 목표물이 없을 때 가끔 기준을 새로 잡아 조명 변화에 따라감."""

    def __init__(self, box, frame_size, max_shift=24, min_response=0.2, patch=(128, 96), min_texture=6.0):
        W, H = frame_size
        self.box = box
        self.max_shift = max_shift
        self.min_response = min_response
        self.min_texture = min_texture
        pw, ph = min(patch[0], W // 4), min(patch[1], H // 4)
        # ROI 왼쪽·오른쪽에 작은 조각 여러 개 (양쪽 2개씩 x 위/아래 2줄 = 8개).
        # 조각 하나에 사람·옆 선로 열차가 들어와도 '중앙값'을 쓰므로 나머지가 이김
        self.patches = []
        cy = box.y0 + box.h / 2
        bx = box.x0 + box.w / 2
        for side in (-1, 1):
            for k in (0, 1):
                xc = bx + side * (box.w / 2 + 8 + pw / 2 + k * (pw + 8))
                for oy in (-0.6, 0.6):
                    x = int(np.clip(xc - pw / 2, 0, W - pw))
                    y = int(np.clip(cy + oy * ph - ph / 2, 0, H - ph))
                    r = (x, y, pw, ph)
                    if r not in self.patches and not _overlap(r, box.rect):
                        self.patches.append(r)
        if not self.patches:                             # ROI가 화면만 하면 ROI 자체 사용
            pw, ph = box.w, box.h
            self.patches.append(box.rect)
        self.win = cv2.createHanningWindow((pw, ph), cv2.CV_32F)
        self.ref = None
        self.margin = max_shift + 2
        self.last = (0.0, 0.0, 0.0)

    def _cut(self, img, r):
        x, y, w, h = r
        return img[y:y + h, x:x + w].astype(np.float32)

    def set_ref(self, gray):
        """기준 프레임. 무늬가 없는 조각(맑은 하늘 등, 밝기 표준편차 < min_texture)은 위상 상관이 엉뚱한 값을
        내므로 빼고 씀. 다 빠지면 그나마 무늬가 가장 많은 조각 하나를 씀."""
        cuts = [self._cut(gray, r) for r in self.patches]
        tex = [float(c.std()) for c in cuts]
        keep = [i for i, t in enumerate(tex) if t >= self.min_texture] or [int(np.argmax(tex))]
        self.patches = [self.patches[i] for i in keep]
        self.ref = [cuts[i] for i in keep]

    def measure(self, gray):
        """-> (dx, dy, response).  현재 프레임이 기준보다 (dx, dy) 만큼 밀려 있음."""
        if self.ref is None:
            self.set_ref(gray)
            return 0.0, 0.0, 1.0
        sh, ws = [], []
        for ref, r in zip(self.ref, self.patches):
            # ref.copy(): 일부 OpenCV 버전(5.0 확인)은 phaseCorrelate가 입력 배열을 덮어써서 기준이 망가짐
            (dx, dy), resp = cv2.phaseCorrelate(ref.copy(), self._cut(gray, r), self.win)
            if resp >= self.min_response and abs(dx) <= self.max_shift and abs(dy) <= self.max_shift:
                sh.append((dx, dy))
                ws.append(resp)
        if not sh:
            return self.last[0], self.last[1], 0.0       # 측정 실패 -> 앞 값 유지
        sh, ws = np.array(sh), np.array(ws)
        dx, dy = np.median(sh, axis=0)                   # 움직이는 물체가 든 조각 몇 개는 무시
        self.last = (float(dx), float(dy), float(ws.mean()))
        return self.last

    def crop_aligned(self, gray, dx, dy):
        """흔들림을 되돌린 ROI (float32). 여유분을 붙여 자른 뒤 부분 픽셀 이동."""
        b, mg = self.box, self.margin
        H, W = gray.shape
        x0, y0 = max(b.x0 - mg, 0), max(b.y0 - mg, 0)
        x1, y1 = min(b.x0 + b.w + mg, W), min(b.y0 + b.h + mg, H)
        big = gray[y0:y1, x0:x1].astype(np.float32)
        # 원본 (b.x0 + dx, b.y0 + dy) 위치가 결과 (0, 0)이 되게
        M = np.float32([[1, 0, -(b.x0 - x0 + dx)], [0, 1, -(b.y0 - y0 + dy)]])
        return cv2.warpAffine(big, M, (b.w, b.h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
