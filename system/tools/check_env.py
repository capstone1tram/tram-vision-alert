"""설치 점검: 필요한 것들이 제대로 깔렸는지 하나씩 확인

  python tools/check_env.py                 # 기본 (opencv, numpy, zmq, ultralytics)
  python tools/check_env.py --picam         # 파이: 카메라 모듈 3에서 사진 1장
  python tools/check_env.py --gpio          # 파이: 부저·LED 1초씩 켜 보기
  python tools/check_env.py --ncnn yolo11n_ncnn_model
  python tools/check_env.py --hailo /usr/share/hailo-models/yolov8s_h8l.hef
  python tools/check_env.py --openvino yolo11n_openvino_model
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name, fn):
    t0 = time.perf_counter()
    try:
        msg = fn()
        print(f"[O] {name}  {msg or ''}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        return True
    except Exception as e:
        print(f"[X] {name}  -> {type(e).__name__}: {e}")
        return False


def bench(det, n=10):
    import numpy as np
    from tram.config import cfg
    f = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), np.uint8)
    t0 = time.perf_counter()
    for _ in range(n):
        det(f)
    return f"1장 {(time.perf_counter() - t0) / n * 1000:.0f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picam", action="store_true")
    ap.add_argument("--gpio", action="store_true")
    ap.add_argument("--ncnn", default=None)
    ap.add_argument("--openvino", default=None)
    ap.add_argument("--hailo", default=None)
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]}")
    check("numpy", lambda: __import__("numpy").__version__)
    check("opencv", lambda: __import__("cv2").__version__)
    check("pyzmq", lambda: __import__("zmq").__version__)
    check("ultralytics", lambda: __import__("ultralytics").__version__)
    if args.picam:
        def cam():
            from tram import camera
            _, _, _, f = next(camera.frames("picam"))
            import cv2
            cv2.imwrite("check_picam.jpg", f)
            return f"사진 {f.shape[1]}x{f.shape[0]} -> check_picam.jpg"
        check("카메라 모듈 3 (picamera2)", cam)
    if args.gpio:
        def gpio():
            from tram.alarm import Alarm
            a = Alarm()
            if not a.ok:
                raise RuntimeError("gpiozero 핀을 못 엶")
            for lv in (0, 1, 2, 3):
                a.set(lv)
                time.sleep(1)
            a.close()
            return "초록 -> 노랑 -> 빨강 깜빡+짧은 소리 -> 빨강+빠른 소리"
        check("부저·LED", gpio)
    from tram.detectors import make_detector
    if args.openvino:
        check(f"OpenVINO {args.openvino}", lambda: bench(make_detector("yolo", args.openvino, 640)))
    if args.ncnn:
        check(f"NCNN {args.ncnn}", lambda: bench(make_detector("yolo", args.ncnn, 320)))
    if args.hailo:
        check(f"Hailo {args.hailo}", lambda: bench(make_detector("hailo", args.hailo)))


if __name__ == "__main__":
    main()
