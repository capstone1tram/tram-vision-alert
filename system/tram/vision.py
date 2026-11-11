"""인지 단계 (나중에 '노트북 쪽'이 될 부분): 사진 -> 선로 + 물체 박스

pc_main.py에서는 같은 PC 안에서 부르지만, 돌려주는 값(reply)은 laptop_server.py가 파이에 보내는
답과 모양이 같습니다.  {"t_capture", "rails"(dict), "dets", "scale", "proc_ms"}
그래서 나중에 나눌 때는 Perception.process()를 laptop_server의 handle 자리에 넣기만 하면 됩니다.

선로는 영상이라 앞 프레임 결과를 이용합니다 (RailTracker).
  1) 앞 프레임 선로 주변 띠만 남긴 마스크로 먼저 찾아 봄 (옆 선로·차선이 끼어들지 않게)
  2) 못 찾으면 화면 전체에서 다시 찾음
  3) 앞 프레임과 너무 다르면(궤간의 RAIL_JUMP 이상 튐) 의심 -> 앞 선로 유지.
     RAIL_REJECT_MAX번 연속이면 진짜 바뀐 것(분기·곡선)으로 보고 받아들임
  4) 받아들인 선로는 앞 선로와 섞어서(RAIL_SMOOTH) 화면 떨림을 줄임
"""
import time

import cv2
import numpy as np

from .config import cfg
from .rails import RailModel
from .rails_photo import find_rails_ego, tophat_mask


def _ys(model, n=24):
    return np.linspace(model.y_top, model.y_bottom, n)


class RailTracker:
    def __init__(self):
        self.model = None
        self.hits = 0
        self.rejects = 0
        self.misses = 0
        self.last_mask = None
        self.status = "-"

    def reset(self):
        self.__init__()

    def _mask(self, frame):
        h = frame.shape[0]
        m = tophat_mask(frame, cfg.PHOTO_Y_START)
        if cfg.PHOTO_Y_END < 1.0:
            m[int(h * cfg.PHOTO_Y_END):] = 0
        return m

    def _band(self, mask, model):
        """앞 선로 양옆으로 궤간 0.6배 띠만 남김."""
        h, w = mask.shape
        ys = np.arange(h)
        g = np.clip(model.r(ys) - model.l(ys), 4, w)
        lo = model.l(ys) - 0.6 * g
        hi = model.r(ys) + 0.6 * g
        xs = np.arange(w)[None, :]
        keep = (xs >= lo[:, None]) & (xs <= hi[:, None])
        keep[: max(0, model.y_top - int(0.08 * h))] = False      # 앞 선로 맨 위보다 조금 위까지만
        return np.where(keep, mask, 0).astype(np.uint8)

    def _find(self, mask):
        y_end = cfg.PHOTO_Y_END
        return find_rails_ego(mask, y_start=cfg.PHOTO_Y_START, center=cfg.RAIL_CENTER,
                              rows=int(80 * (y_end - cfg.PHOTO_Y_START) / 0.55))

    def _close(self, new, old):
        """화면 아래쪽 기준으로 위치·폭이 비슷한지."""
        y = 0.5 * (max(new.y_top, old.y_top) + min(new.y_bottom, old.y_bottom))
        go, gn = old.gauge_px(y), new.gauge_px(y)
        if go <= 2 or gn <= 2:
            return False
        dc = abs(new.center_x(y) - old.center_x(y))
        return dc < cfg.RAIL_JUMP * go and 1 / (1 + cfg.RAIL_JUMP) < gn / go < 1 + cfg.RAIL_JUMP

    def _blend(self, new, old):
        a = cfg.RAIL_SMOOTH
        ys = _ys(new)
        L = a * new.l(ys) + (1 - a) * old.l(ys)
        R = a * new.r(ys) + (1 - a) * old.r(ys)
        deg = len(new.l.coeffs) - 1
        top = int(a * new.y_top + (1 - a) * old.y_top)
        return RailModel(np.polyfit(ys, L, deg), np.polyfit(ys, R, deg), min(top, new.y_bottom - 5), new.y_bottom)

    def update(self, frame):
        """사진 -> (RailModel 또는 None, 이번에 새로 찾았는지)"""
        mask = self._mask(frame)
        self.last_mask = mask
        new = None
        if self.model is not None and self.hits >= 3:
            new = self._find(self._band(mask, self.model))
        if new is None:
            new = self._find(mask)
        if new is None:
            self.misses += 1
            self.status = "miss"
            return (self.model if self.misses <= 3 else None), False
        self.misses = 0
        if self.model is None:
            self.model, self.hits, self.status = new, 1, "new"
            return self.model, True
        if self._close(new, self.model):
            self.model = self._blend(new, self.model)
            self.hits += 1
            self.rejects = 0
            self.status = "track"
            return self.model, True
        self.rejects += 1
        if self.rejects >= cfg.RAIL_REJECT_MAX:
            self.model, self.hits, self.rejects, self.status = new, 1, 0, "jump"
            return self.model, True
        self.status = "hold"
        return self.model, False


class Perception:
    """사진 한 장 -> laptop_server와 같은 모양의 답."""

    def __init__(self, detector, rail_every=1, det_every=1):
        self.det = detector
        self.rails = RailTracker()
        self.rail_every, self.det_every = rail_every, det_every
        self.n = 0
        self.ms = {"rail": 0.0, "det": 0.0}
        self._last_rails = None

    def reset(self):
        self.rails.reset()
        self.n = 0

    def process(self, frame, t):
        self.n += 1
        t0 = time.perf_counter()
        rails_dict = None
        if self.n % self.rail_every == 0 or self.rails.model is None:
            model, _ = self.rails.update(frame)
            self._last_rails = model
        model = self._last_rails
        if model is not None:
            rails_dict = model.to_dict()
        t1 = time.perf_counter()
        dets = []
        ran_det = False
        if self.det is not None and self.n % self.det_every == 0:
            dets = [{"name": d["name"], "conf": round(d["conf"], 3), "box": d["box"]} for d in self.det(frame)]
            ran_det = True
        t2 = time.perf_counter()
        self.ms["rail"] = 0.8 * self.ms["rail"] + 0.2 * (t1 - t0) * 1000
        if ran_det:
            self.ms["det"] = 0.8 * self.ms["det"] + 0.2 * (t2 - t1) * 1000
        if not ran_det:
            return None, rails_dict
        return {"t_capture": t, "rails": rails_dict, "dets": dets, "scale": 1.0,
                "proc_ms": (t2 - t0) * 1000}, rails_dict

    def debug_mask(self):
        return self.rails.last_mask
