"""블랙박스 영상 + 노트북 성능 사전 점검 (한 번만 실행).

하는 일
  1) 영상 정보 출력: 해상도, FPS, 길이, 코덱
  2) 영상 전체에서 고르게 16장을 JPG로 저장          -> logs/probe/frames/
  3) 20초짜리 짧은 구간 4개를 960x540으로 저장         -> logs/probe/clips/   (프로그램 시험용)
  4) CPU 이름과 OpenVINO가 쓸 수 있는 장치(CPU/GPU/NPU) 출력
  5) YOLO 모델 준비: yolo11n·yolo11s를 받아서 OpenVINO 형식으로 변환  -> models/
  6) 이 노트북에서 YOLO가 1초에 몇 장 도는지 CPU/GPU 별로 측정
  결과 요약은 logs/probe/probe_report.txt 에도 저장됩니다.

사용 (G:\\tram_ 에서 가상환경 켠 뒤):
  python tools\\video_probe.py
  python tools\\video_probe.py --video G:\\rail_pic\\tram_blackbox.mp4
  python tools\\video_probe.py --skip-models        # 영상만 보고 싶을 때
"""
import argparse
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "?"


def probe_video(path, out_dir, n_frames, n_clips, clip_s):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        say(f"[X] 영상을 못 열었습니다: {path}")
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4))
    dur = n / fps if fps else 0
    say(f"[영상] {path}")
    say(f"  해상도 {w}x{h}, {fps:.2f} FPS, {n} 프레임, 길이 {dur / 60:.1f}분, 코덱 {fourcc}")

    # 디코딩 속도 (실시간 재생 가능한지)
    t0 = time.time()
    k = 0
    while k < 120:
        ok, _ = cap.read()
        if not ok:
            break
        k += 1
    dt = time.time() - t0
    if k:
        say(f"  영상 읽기 속도: {k / dt:.0f} 장/초 (원본 {fps:.0f} FPS보다 커야 실시간 가능)")

    fdir = os.path.join(out_dir, "frames")
    os.makedirs(fdir, exist_ok=True)
    idxs = np.linspace(n * 0.02, n * 0.98, n_frames).astype(int)
    for i, f in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            t = f / fps if fps else 0
            name = f"f{i:02d}_{int(t // 60):02d}m{int(t % 60):02d}s.jpg"
            cv2.imwrite(os.path.join(fdir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    say(f"  사진 {len(idxs)}장 저장: {fdir}")

    cdir = os.path.join(out_dir, "clips")
    os.makedirs(cdir, exist_ok=True)
    starts = np.linspace(n * 0.1, n * 0.85, n_clips).astype(int)
    cw, ch = 960, int(round(960 * h / w / 2) * 2)
    for i, s in enumerate(starts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
        t = s / fps if fps else 0
        name = os.path.join(cdir, f"clip{i}_{int(t // 60):02d}m{int(t % 60):02d}s.mp4")
        vw = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), fps or 30, (cw, ch))
        for _ in range(int(clip_s * (fps or 30))):
            ok, img = cap.read()
            if not ok:
                break
            vw.write(cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA))
        vw.release()
    say(f"  {clip_s}초 구간 {n_clips}개 저장 (960 폭): {cdir}")
    cap.release()
    return True


def prepare_models(model_dir, names):
    from ultralytics import YOLO
    os.makedirs(model_dir, exist_ok=True)
    out = {}
    for name in names:
        pt = os.path.join(model_dir, f"{name}.pt")
        ov = os.path.join(model_dir, f"{name}_openvino_model")
        if not os.path.exists(pt):
            src = os.path.join(ROOT, "tools", f"{name}.pt")
            if os.path.exists(src):
                import shutil
                shutil.copy(src, pt)
            else:
                cwd = os.getcwd()
                os.chdir(model_dir)                    # ultralytics는 현재 폴더에 받음
                try:
                    YOLO(f"{name}.pt")
                finally:
                    os.chdir(cwd)
        if not os.path.exists(ov):
            say(f"  {name} -> OpenVINO 변환 중 (1~2분)...")
            p = YOLO(pt).export(format="openvino", imgsz=640, half=True, verbose=False)
            if os.path.abspath(p) != os.path.abspath(ov) and os.path.exists(p):
                import shutil
                shutil.move(p, ov)
        out[name] = ov
        say(f"  [모델] {ov}")
    return out


def bench(models, frame, runs=40):
    from ultralytics import YOLO
    import openvino as ov
    core = ov.Core()
    devs = core.available_devices
    say(f"[OpenVINO] 쓸 수 있는 장치: {devs}")
    for d in devs:
        try:
            say(f"  {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
        except Exception:
            pass
    targets = [("CPU", "intel:cpu")]
    if any(d.startswith("GPU") for d in devs):
        targets.append(("GPU", "intel:gpu"))
    if "NPU" in devs:
        targets.append(("NPU", "intel:npu"))
    say("[YOLO 속도] (1080p 사진 한 장 넣고 사람·차 검출까지, 입력 640)")
    for name, path in models.items():
        for label, dev in targets:
            try:
                m = YOLO(path, task="detect")
                for _ in range(5):
                    m.predict(frame, imgsz=640, device=dev, verbose=False, classes=[0, 1, 2, 3, 5, 7])
                t0 = time.time()
                for _ in range(runs):
                    m.predict(frame, imgsz=640, device=dev, verbose=False, classes=[0, 1, 2, 3, 5, 7])
                dt = (time.time() - t0) / runs
                say(f"  {name:8s} {label}: {1 / dt:5.1f} 장/초 ({dt * 1000:.0f} ms)")
            except Exception as e:
                say(f"  {name:8s} {label}: 실패 ({type(e).__name__}: {str(e)[:120]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"G:\rail_pic\tram_blackbox.mp4")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "probe"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--clip-sec", type=int, default=20)
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    say(f"[PC] {cpu_name()} / {platform.platform()} / Python {platform.python_version()}")
    say(f"  OpenCV {cv2.__version__}, numpy {np.__version__}")
    ok = probe_video(args.video, args.out, args.frames, args.clips, args.clip_sec)

    if not args.skip_models:
        say("[모델 준비]")
        models = prepare_models(os.path.join(ROOT, "models"), ["yolo11n", "yolo11s"])
        frame = None
        if ok:
            cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
            ok2, frame = cap.read()
            cap.release()
        if frame is None:
            frame = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
        bench(models, frame)

    rep = os.path.join(args.out, "probe_report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    say(f"\n끝. 요약: {rep}")


if __name__ == "__main__":
    sys.exit(main())
