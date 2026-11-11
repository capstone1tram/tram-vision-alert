"""06. 전체 파이프라인: 영상 -> 선로 -> 검출·추적 -> 거리 -> 위험 단계 -> 화면

01~05번을 한 루프로 엮은 것입니다. 로드맵 4주차 'PC 데모'의 뼈대로 쓰면 됩니다.

검출기 두 가지:
  --detector yolo   : 실제 영상 (사람·자전거·차량, 02번)
  --detector color  : 빨간 물체를 사람 대신 찾음 (00번 합성 영상, 모형 시험장의 빨간 피규어용)

실행 예:
  python 00_make_sample_video.py
  python 06_pipeline.py --source sample.mp4 --detector color --save out_06.mp4 --no-show
  python 06_pipeline.py --source 트램영상.mp4 --detector yolo
"""
import argparse
import csv
import importlib

import cv2
import numpy as np

import config as C

rail = importlib.import_module("03_rail_detect")
dist = importlib.import_module("04_distance")
risk = importlib.import_module("05_risk_level")


class ColorDetector:
    """빨간 덩어리 -> 박스. 가까운 이전 박스와 같은 ID를 붙이는 아주 단순한 추적."""

    def __init__(self):
        self.prev = {}        # id -> 중심점
        self.next_id = 1

    def __call__(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 80), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (179, 255, 255))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        dets, now = [], {}
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < 20:
                continue
            c = cents[k]
            tid = min(self.prev, key=lambda i: np.linalg.norm(self.prev[i] - c), default=None)
            if tid is None or np.linalg.norm(self.prev[tid] - c) > 80 or tid in now:
                tid, self.next_id = self.next_id, self.next_id + 1
            now[tid] = c
            dets.append({"id": tid, "name": "person", "conf": 1.0, "box": [x, y, x + w, y + h]})
        self.prev = now
        return dets


class YoloDetector:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.det = importlib.import_module("02_yolo_detect_track").detect_frame

    def __call__(self, frame):
        return self.det(self.model, frame)


def draw_banner(img, level, text):
    color = C.LEVEL_COLORS[level]
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), color, -1)
    cv2.putText(img, f"{C.LEVEL_NAMES[level]}  {text}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    if level == 3:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample.mp4")
    ap.add_argument("--detector", choices=["yolo", "color"], default="yolo")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--save", default="")
    ap.add_argument("--log", default="events.csv")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    detect = ColorDetector() if args.detector == "color" else YoloDetector(args.model)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    tracks, smoother = {}, risk.LevelSmoother()
    rails, writer = None, None
    log = csv.writer(open(args.log, "w", newline=""))
    log.writerow(["frame", "t_s", "id", "Z_m", "d_lat_m", "ttc_s", "a_req", "method", "level"])

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if i % 2 == 0 or rails is None:                       # 선로는 2프레임마다
            found = rail.find_rails(rail.rail_mask(frame))
            rails = found if found is not None else rails
        out = frame.copy()
        if rails is not None:
            rail.draw_rails(out, rails)

        worst, worst_text = (0 if rails is not None else -1), ""
        for d in detect(frame):
            z, d_lat, method = dist.obj_distance(d["box"], rails)
            if z is None or d_lat is None:
                continue
            st = tracks.setdefault(d["id"], risk.TrackState())
            st.add(t, z, d_lat)
            if len(st.hist) < 3:
                continue
            m = st.metrics()
            lv = risk.risk_level(m)
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), C.LEVEL_COLORS[lv], 2)
            ttc_txt = f"{m['ttc']:.1f}s" if m["ttc"] < 99 else "-"
            cv2.putText(out, f"#{d['id']} {m['Z']:.1f}m {ttc_txt}", (x1, max(y1 - 8, 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C.LEVEL_COLORS[lv], 2)
            log.writerow([i, f"{t:.2f}", d["id"], f"{m['Z']:.2f}", f"{m['d_lat']:.2f}",
                          f"{m['ttc']:.2f}", f"{m['a_req']:.2f}", method, lv])
            if lv > worst:
                worst, worst_text = lv, f"{d['name']} {m['Z']:.1f}m  TTC {ttc_txt}"

        shown = smoother.update(worst, t) if worst >= 0 else -1
        draw_banner(out, shown, worst_text)
        i += 1

        if args.save:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
        if not args.no_show:
            cv2.imshow("06 pipeline", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"done: {i} frames, log -> {args.log}")


if __name__ == "__main__":
    main()
