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


# ── 원본 해상도 흑백 입력 (소실점 접근 감지 approach_main.py 용) ─────────────────
# frames()는 1280x720으로 줄여서 내보내지만, 멀리 있는 열차는 몇 픽셀뿐이라 줄이면 사라집니다.
# 그래서 이쪽은 2304x1296 센서 모드를 '줄이지 않고' 흑백(Y 평면)으로 그대로 내보내고,
# 소실점 주변 ROI는 approach_main.py가 잘라 씁니다.

class FullFrames:
    """(t, idx, video_t, gray) 를 차례로 내놓음.  gray = 원본 해상도 흑백 (uint8, H x W).

    source
      "picam"   : 카메라 모듈 3. 초점 무한대 고정, 노출·게인·화이트밸런스 고정 (아래 _picam 참고)
      "0", "1"  : USB 웹캠
      "파일"    : 녹화 영상. realtime=False(기본)면 모든 프레임을 차례로 처리 (PC 시험용)
    info: 실제 해상도·fps·카메라 고정값 (요약 출력용)
    """

    def __init__(self, source, size=(2304, 1296), fps=30, realtime=False, lock="auto",
                 exposure_us=None, gain=None, colour_gains=None, lens=0.0, settle_s=1.5):
        self.source = source
        self.size, self.fps, self.realtime = size, fps, realtime
        self.lock, self.exposure_us, self.gain = lock, exposure_us, gain
        self.colour_gains, self.lens, self.settle_s = colour_gains, lens, settle_s
        self.info = {}

    def __iter__(self):
        if self.source == "picam":
            return self._picam()
        return self._cv()

    # 카메라 모듈 3 ------------------------------------------------------------
    def _picam(self):
        from picamera2 import Picamera2
        W, H = self.size
        cam = Picamera2()
        # YUV420의 Y 평면 = 흑백. RGB 변환을 안 하므로 2304x1296에서도 가벼움
        main = {"size": (W, H), "format": "YUV420"}
        frame_us = int(1e6 / self.fps)
        ctrl = {"FrameDurationLimits": (frame_us, frame_us)}      # fps 고정 (노출이 길어져 fps가 떨어지는 것 방지)
        try:
            conf = cam.create_video_configuration(main=main, sensor={"output_size": (W, H)},
                                                  controls=ctrl, buffer_count=4)
        except Exception:
            conf = cam.create_video_configuration(main=main, controls=ctrl, buffer_count=4)
        cam.configure(conf)
        # 초점: 무한대 고정 (LensPosition 단위는 디옵터 = 1/거리(m). 0 = 무한대)
        try:
            from libcamera import controls
            cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(self.lens)})
        except Exception as e:
            print(f"[카메라] 초점 고정 실패 ({type(e).__name__}) - 자동초점 그대로")
        cam.start()
        self._lock_exposure(cam)
        t0, i = time.monotonic(), 0
        try:
            while True:
                req = cam.capture_request()
                try:
                    yuv = req.make_array("main")
                    md = req.get_metadata()
                finally:
                    req.release()
                g = yuv[:H, :W]
                t = time.monotonic()
                if i == 0:
                    self.info["sensor_ts0"] = md.get("SensorTimestamp")
                yield t, i, t - t0, g
                i += 1
        finally:
            cam.stop()
            cam.close()

    def _lock_exposure(self, cam):
        """lock="auto": 처음 settle_s 초 동안 자동 노출·AWB로 맞춘 뒤 그 값을 읽어 고정.
        lock="manual": exposure_us / gain / colour_gains 로 바로 고정.  lock="off": 자동 그대로.
        자동 노출을 켜 두면 열차(밝은 전조등, 큰 어두운 물체)가 들어올 때 화면 전체 밝기가 바뀌어
        배경 차분이 화면 전체를 '변화'로 잡습니다. 그래서 고정합니다."""
        if self.lock == "off":
            self.info["lock"] = "off"
            return
        if self.lock == "auto":
            time.sleep(self.settle_s)
            md = cam.capture_metadata()
            exp = int(md.get("ExposureTime", 10000))
            gain = float(md.get("AnalogueGain", 1.0))
            cg = md.get("ColourGains", (1.5, 1.5))
        else:
            exp = int(self.exposure_us or 4000)
            gain = float(self.gain or 1.0)
            cg = self.colour_gains or (1.5, 1.5)
        exp = min(exp, int(1e6 / self.fps) - 500)        # 한 프레임보다 길면 fps가 떨어짐
        cam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": exp,
                          "AnalogueGain": gain, "ColourGains": tuple(float(c) for c in cg)})
        time.sleep(0.3)
        self.info.update(lock=self.lock, exposure_us=exp, gain=round(gain, 3),
                         colour_gains=tuple(round(float(c), 3) for c in cg), lens=self.lens)
        print(f"[카메라] 고정: 노출 {exp} us, 게인 {gain:.2f}, AWB {self.info['colour_gains']}, "
              f"초점 {self.lens} (0=무한대)")

    # 영상 파일 / 웹캠 ---------------------------------------------------------
    def _cv(self):
        is_cam = self.source.isdigit()
        cap = cv2.VideoCapture(int(self.source) if is_cam else self.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없음: {self.source}")
        if is_cam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            fps = self.fps
        self.info.update(fps_file=fps, lock="file" if not is_cam else "webcam")
        t0, i = time.monotonic(), 0
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                if is_cam:
                    t = time.monotonic()
                    yield t, i, t - t0, g
                    i += 1
                    continue
                vt = i / fps
                i += 1
                if self.realtime:
                    target = t0 + vt
                    now = time.monotonic()
                    if now < target:
                        time.sleep(target - now)
                    elif now - target > 1.5 / fps:      # 실시간 흉내: 밀리면 건너뜀
                        continue
                # 파일은 '영상 안의 시각'을 판단 시계로 씀 -> PC가 빠르든 느리든 TTC가 같게 나옴
                yield vt, i - 1, vt, g
        finally:
            cap.release()
