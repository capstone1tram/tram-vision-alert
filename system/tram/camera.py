"""영상 입력.  모두 (t, idx, video_t, frame) 를 차례로 내놓습니다.
  t       : time.monotonic() 기준 시각 (판단·통신 모두 이 시계를 씀)
  video_t : 영상 파일 안의 시각 (정답 CSV와 맞춰 볼 때 사용). 카메라면 t와 같은 의미
  frame   : OpenCV BGR 사진 (IMG_W x IMG_H)

source
  "picam"     : 라즈베리파이 카메라 모듈 3 (picamera2)
  "0", "1"    : USB 웹캠 / 노트북 카메라
  "파일.mp4"  : 영상 파일. 실제 카메라처럼 '실시간 속도'로 내보내고, 처리가 밀리면 프레임을 건너뜀
"""
import time

import cv2

from .config import cfg


def _picam(fps):
    from picamera2 import Picamera2
    cam = Picamera2()
    main = {"size": (cfg.IMG_W, cfg.IMG_H), "format": "RGB888"}   # picamera2의 RGB888 = 메모리상 BGR 순서
    ctrl = {"FrameRate": fps}
    try:
        # 2304x1296 센서 모드 = 화각을 자르지 않는 모드 (1536x864 모드는 가운데를 잘라 화각이 좁아짐)
        conf = cam.create_video_configuration(main=main, sensor={"output_size": (2304, 1296)}, controls=ctrl)
    except Exception:
        conf = cam.create_video_configuration(main=main, controls=ctrl)
    cam.configure(conf)
    try:
        from libcamera import controls
        cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})    # 카메라 모듈 3 자동 초점
    except Exception:
        pass
    cam.start()
    time.sleep(0.5)
    t0, i = time.monotonic(), 0
    try:
        while True:
            f = cam.capture_array("main")
            t = time.monotonic()
            yield t, i, t - t0, f
            i += 1
    finally:
        cam.stop()


def _cv(source, realtime=True, loop=False):
    is_cam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없음: {source}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.IMG_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.IMG_H)
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.FPS
    if fps <= 0 or fps > 240:
        fps = cfg.FPS
    t0 = time.monotonic()
    i = 0
    base = 0.0                                          # 반복 재생할 때 영상 시각 이어 붙이기
    while True:
        ok, f = cap.read()
        if not ok:
            if loop and not is_cam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                base += i / fps
                continue
            break
        if f.shape[1] != cfg.IMG_W or f.shape[0] != cfg.IMG_H:
            f = cv2.resize(f, (cfg.IMG_W, cfg.IMG_H))
        if is_cam:
            t = time.monotonic()
            yield t, i, t - t0, f
            i += 1
            continue
        vt = i / fps
        i += 1
        if realtime:
            target = t0 + base + vt
            now = time.monotonic()
            if now < target:
                time.sleep(target - now)
            elif now - target > 1.5 / fps:              # 처리가 밀림 -> 실제 카메라처럼 이 프레임은 건너뜀
                continue
        yield time.monotonic(), i - 1, base + vt, f
    cap.release()


def frames(source, realtime=True, loop=False):
    if source == "picam":
        return _picam(cfg.FPS)
    return _cv(source, realtime, loop)
