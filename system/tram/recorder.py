"""위험 장면 자동 저장 + 캡처.

- 등급이 min_level(기본 2 = 경고) 이상이 되면, 그 CLIP_PRE_S초 전부터 영상 저장을 시작하고
  등급이 내려간 뒤 CLIP_POST_S초까지 저장 -> clips/event_003_12m34s_DANGER.mp4
- 장면마다 가장 위험했던 순간을 사진으로 -> clips/event_003_..._peak.jpg
- events.csv : 장면 번호, 시작·끝 (영상 시각), 최고 등급, 그때 물체·거리·TTC, 최소 TTC
- capture()  : s 키로 지금 화면 저장 -> captures/
"""
import collections
import csv
import os

import cv2

from .config import LEVEL_NAMES, cfg


def mmss(t):
    return f"{int(t // 60):02d}m{int(t % 60):02d}s"


class EventRecorder:
    def __init__(self, out_dir, min_level=2, enable=True):
        self.dir = out_dir
        self.min_level = min_level
        self.enable = enable
        self.buf = collections.deque()
        self.writer = None
        self.ev = None
        self.n_ev = 0
        self.fps = 15.0
        os.makedirs(os.path.join(out_dir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "captures"), exist_ok=True)
        self.f = open(os.path.join(out_dir, "events.csv"), "w", newline="", encoding="utf-8-sig")
        self.w = csv.writer(self.f)
        self.w.writerow(["event", "start_video_t", "end_video_t", "start", "max_level", "peak_video_t",
                         "peak_obj", "peak_Z_m", "peak_X_m", "peak_ttc_s", "min_ttc_s", "clip"])

    def set_fps(self, fps):
        if fps and fps > 1:
            self.fps = float(fps)

    def _open(self, t, size):
        self.n_ev += 1
        name = f"event_{self.n_ev:03d}_{mmss(t)}"
        path = os.path.join(self.dir, "clips", name + ".mp4")
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        self.ev = {"name": name, "path": path, "start": t, "end": t, "max": 0, "peak": None,
                   "peak_img": None, "min_ttc": float("inf"), "below_since": None}
        for _, img in self.buf:                   # 미리 쌓아 둔 앞 몇 초
            self.writer.write(img)
        self.buf.clear()

    def _close(self):
        e = self.ev
        self.writer.release()
        lvname = LEVEL_NAMES.get(e["max"], str(e["max"]))
        final = e["path"].replace(".mp4", f"_{lvname}.mp4")
        try:
            os.replace(e["path"], final)
        except OSError:
            final = e["path"]
        if e["peak_img"] is not None:
            cv2.imwrite(final.replace(".mp4", "_peak.jpg"), e["peak_img"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        p = e["peak"] or {}
        self.w.writerow([self.n_ev, f"{e['start']:.2f}", f"{e['end']:.2f}", mmss(e["start"]), lvname,
                         f"{p.get('t', 0):.2f}", p.get("name", ""), f"{p.get('Z', 0):.1f}", f"{p.get('X', 0):+.2f}",
                         f"{p['ttc']:.2f}" if p.get("ttc", 99) < 99 else "",
                         f"{e['min_ttc']:.2f}" if e["min_ttc"] < 99 else "", os.path.basename(final)])
        self.f.flush()
        print(f"  [저장] 위험 장면 {self.n_ev}: {mmss(e['start'])}~{mmss(e['end'])} 최고 {lvname} -> {final}")
        self.writer, self.ev = None, None

    def push(self, img, t, res):
        """화면 그림 한 장 + 그 시각 + Guard 결과."""
        if not self.enable:
            return False
        lv = res["level"]
        if self.ev is None:
            self.buf.append((t, img))
            while self.buf and t - self.buf[0][0] > cfg.CLIP_PRE_S:
                self.buf.popleft()
            if lv >= self.min_level:
                self._open(self.buf[0][0] if self.buf else t, (img.shape[1], img.shape[0]))
            return self.ev is not None
        e = self.ev
        self.writer.write(img)
        e["end"] = t
        wo = res["worst"]
        if wo is not None:
            e["min_ttc"] = min(e["min_ttc"], wo["m"]["ttc"])
        if lv > e["max"] or (lv == e["max"] and wo is not None and e["peak"] is not None
                             and wo["m"]["ttc"] < e["peak"]["ttc"]):
            e["max"] = lv
            if wo is not None:
                e["peak"] = {"t": t, "name": wo["state"]["name"], "Z": wo["m"]["Z"], "X": wo["state"]["X"],
                             "ttc": wo["m"]["ttc"]}
            e["peak_img"] = img.copy()
        if lv >= self.min_level:
            e["below_since"] = None
        elif e["below_since"] is None:
            e["below_since"] = t
        elif t - e["below_since"] >= cfg.CLIP_POST_S:
            self._close()
        return True

    def capture(self, img, t):
        path = os.path.join(self.dir, "captures", f"cap_{mmss(t)}_{int((t % 1) * 1000):03d}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path

    def reset_buffer(self):
        """앞뒤로 건너뛸 때: 쌓아 둔 앞 장면 버리고, 저장 중이면 마무리."""
        if self.ev is not None:
            self._close()
        self.buf.clear()

    def close(self):
        if self.ev is not None:
            self._close()
        self.f.close()
