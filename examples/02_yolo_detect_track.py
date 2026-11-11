"""02. YOLO로 사람·자전거·차량 검출 + 추적(ID 유지)

- YOLO("yolo11n.pt") : 처음 실행 때 가중치를 자동으로 내려받음 (n=가장 가벼운 모델)
- model.track(...)    : 검출 + 프레임 사이 같은 물체에 같은 ID
- persist=True        : 프레임을 하나씩 넣을 때 추적 상태를 이어감
- classes=[...]       : 원하는 종류만 (COCO 번호, config.TARGET_CLASSES)

결과 읽는 법: r.boxes.xyxy (x1,y1,x2,y2), r.boxes.cls (종류), r.boxes.conf (확신도), r.boxes.id (추적 ID)

실행 예:
  python 02_yolo_detect_track.py --source 트램영상.mp4
  python 02_yolo_detect_track.py --source 0 --model yolo11s.pt
  python 02_yolo_detect_track.py --source 트램영상.mp4 --save out_02.mp4 --no-show

참고: 00번 합성 영상의 빨간 사각형은 사람이 아니라서 YOLO가 못 찾는 게 정상입니다.
      실제 트램 영상이나 웹캠으로 시험하세요.
"""
import argparse
import time

import cv2
from ultralytics import YOLO

import config as C


def detect_frame(model, frame, conf=0.3, imgsz=640):
    """프레임 하나 -> [{'id', 'cls', 'name', 'conf', 'box'}] 리스트."""
    r = model.track(frame, persist=True, classes=list(C.TARGET_CLASSES), conf=conf,
                    imgsz=imgsz, tracker="bytetrack.yaml", verbose=False)[0]
    dets = []
    if r.boxes is None:
        return dets
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
    for box, c, p, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(),
                              r.boxes.conf.tolist(), ids):
        dets.append({"id": tid, "cls": c, "name": C.TARGET_CLASSES.get(c, str(c)),
                     "conf": p, "box": [int(v) for v in box]})
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--save", default="")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    writer = None
    frame_no = 0
    t_prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detect_frame(model, frame, args.conf)
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{d["name"]} #{d["id"]} {d["conf"]:.2f}', (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if frame_no % 15 == 0:
            print(frame_no, f"FPS {fps:.1f}", [(d["name"], d["id"]) for d in dets])
        frame_no += 1

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 15, (w, h))
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("02 YOLO track", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
