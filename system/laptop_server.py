"""노트북 서버: 파이가 보낸 사진에서 '사람·차 검출 + 선로 검출'만 해서 숫자로 돌려줌

역할 분담 (발표 3)
  노트북 = 무거운 일 (YOLO11n 검출, OpenCV 선로 검출)  -> 결과는 숫자(JSON)만
  파이   = 나머지 전부 (촬영·전송, 칼만 추적, 거리, TTC, 등급, 경보, 연결 감시)

실행 예
  python laptop_server.py --detector yolo --model yolo11n_openvino_model      # 실제 (OpenVINO, CPU)
  python laptop_server.py --detector yolo --model yolo11n_openvino_model --device intel:gpu   # 내장 GPU 시도
  python laptop_server.py --detector color                                     # 합성 영상·빨간 피규어
  --show 를 붙이면 받은 사진과 검출 결과를 창으로 보여 줌 (발표 시연용)
"""
import argparse
import time

import cv2

from tram import rails as rails_mod
from tram.config import add_common_args, apply_common_args
from tram.detectors import make_detector
from tram.link import serve


def make_handler(detector):
    def handle(frame):
        found = rails_mod.find_rails(frame)
        dets = [{"name": d["name"], "conf": round(d["conf"], 3), "box": d["box"]} for d in detector(frame)]
        return dets, (found.to_dict() if found is not None else None)
    return handle


def run(port, detector, stop=None, show=False, quiet=True):
    stat = {"n": 0, "t0": time.monotonic(), "ms": 0.0}

    def on_frame(frame, dets, rails, reply):
        stat["n"] += 1
        stat["ms"] += reply["proc_ms"]
        if not quiet and time.monotonic() - stat["t0"] >= 2.0:
            dt = time.monotonic() - stat["t0"]
            print(f"  {stat['n'] / dt:5.1f} 장/초  처리 평균 {stat['ms'] / stat['n']:.0f} ms  "
                  f"검출 {len(dets)}개  선로 {'O' if rails else 'X'}")
            stat.update(n=0, t0=time.monotonic(), ms=0.0)
        if show:
            img = frame.copy()
            if rails:
                from tram import display
                display.draw_rails(img, rails_mod.RailModel.from_dict(rails))
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{d['name']} {d['conf']:.2f}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, f"laptop: {reply['proc_ms']:.0f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imshow("tram laptop server", img)
            cv2.waitKey(1)

    serve(port, make_handler(detector), on_frame, stop)


def main():
    ap = argparse.ArgumentParser(description="트램 전방 위험 알림 - 노트북 쪽 서버")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--detector", choices=["yolo", "color"], default="yolo")
    ap.add_argument("--model", default="yolo11n_openvino_model",
                    help="yolo11n.pt / yolo11n_openvino_model (tools/export_models.py로 만듦)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="OpenVINO 장치. 예: intel:gpu (안 되면 빼고 실행 = CPU)")
    ap.add_argument("--show", action="store_true")
    add_common_args(ap)
    args = ap.parse_args()
    apply_common_args(args)

    print(f"[준비] 검출기 {args.detector} ({args.model if args.detector == 'yolo' else '빨간색'}) 불러오는 중...")
    det = make_detector(args.detector, args.model, args.imgsz, args.device)
    print(f"[대기] 포트 {args.port}. 파이에서 --server tcp://<이 노트북 IP>:{args.port} 로 연결하세요. (Ctrl+C 종료)")
    print("       IP 확인: 명령 프롬프트에서 ipconfig  (모바일 핫스팟이면 보통 192.168.137.1)")
    try:
        run(args.port, det, show=args.show, quiet=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
