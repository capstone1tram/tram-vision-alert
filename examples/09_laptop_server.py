"""09. 노트북 쪽 서버: 파이가 보낸 사진에서 '검출 + 선로'만 해서 돌려줌

역할 분담 (로드맵 발표 2):
  노트북 = 무거운 일 (YOLO 검출, 선로 검출)  ->  결과는 숫자(JSON)만 돌려줌
  파이   = 나머지 전부 (촬영·전송, 칼만 추적, 거리, TTC, 등급, 경보)  ->  10_pi_client.py

통신: ZeroMQ REQ/REP.  파이가 [헤더 JSON, JPEG]를 보내면 [결과 JSON] 한 덩어리를 답합니다.

실행 (노트북):
  python 09_laptop_server.py --detector color            # 합성 영상·빨간 피규어
  python 09_laptop_server.py --detector yolo --model yolo11n_openvino_model
"""
import argparse
import importlib
import json
import time

import cv2
import numpy as np
import zmq

rail = importlib.import_module("03_rail_detect")
pipe = importlib.import_module("06_pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--detector", choices=["yolo", "color"], default="yolo")
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()

    detect = pipe.ColorDetector() if args.detector == "color" else pipe.YoloDetector(args.model)
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"노트북 서버 대기 중: 포트 {args.port}, 검출기 {args.detector}")

    while True:
        head, jpg = sock.recv_multipart()
        t0 = time.perf_counter()
        head = json.loads(head)
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        found = rail.find_rails(rail.rail_mask(frame))
        rails = None if found is None else {
            "left": found.l.coeffs.tolist(), "right": found.r.coeffs.tolist(),
            "y_top": found.y_top, "y_bottom": found.y_bottom}
        dets = [{"name": d["name"], "conf": round(float(d["conf"]), 3), "box": [int(v) for v in d["box"]]}
                for d in detect(frame)]
        reply = {"frame_id": head["frame_id"], "t_capture": head["t_capture"],
                 "dets": dets, "rails": rails,
                 "proc_ms": round((time.perf_counter() - t0) * 1000, 1)}
        sock.send(json.dumps(reply).encode())


if __name__ == "__main__":
    main()
