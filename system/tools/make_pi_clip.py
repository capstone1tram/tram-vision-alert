"""(노트북에서 실행) 큰 블랙박스 영상을 파이가 돌리기 쉬운 작은 영상으로 줄이기

파이 5는 1920x1080 영상을 풀면서 YOLO까지 돌리면 버겁습니다. 노트북에서 미리
폭 960(또는 640)으로 줄이고, 필요한 구간만 잘라서 USB로 옮기세요.

  python tools\\make_pi_clip.py                                       # 기본: 9분 40초부터 60초, 폭 960
  python tools\\make_pi_clip.py --start 3:30 --duration 120 --width 640
  python tools\\make_pi_clip.py --duration 0 --width 640              # 영상 전체 (몇 분 걸림)

결과: pi_clips\\tram_<시작>_<길이>s_<폭>.mp4   (OpenCV만 있으면 됨, ffmpeg 필요 없음)
"""
import argparse
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_time(s):
    if ":" in str(s):
        p = [float(x) for x in str(s).split(":")]
        return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"G:\rail_pic\tram_blackbox.mp4")
    ap.add_argument("--start", default="9:40", help="시작 위치 (초 또는 분:초)")
    ap.add_argument("--duration", type=float, default=60, help="길이(초). 0 = 끝까지")
    ap.add_argument("--width", type=int, default=960, help="줄일 폭 (960 또는 640 권장)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "pi_clips"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = args.width
    h = int(round(w * sh / sw / 2) * 2)
    start = parse_time(args.start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n_max = int(args.duration * fps) if args.duration > 0 else None

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{int(start // 60):02d}m{int(start % 60):02d}s"
    out = os.path.join(args.out_dir, f"tram_{tag}_{int(args.duration) or 'all'}s_{w}.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n_max is None or n < n_max:
        ok, f = cap.read()
        if not ok:
            break
        wr.write(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
        n += 1
        if n % int(fps * 10) == 0:
            print(f"  {n / fps:.0f}초 완료")
    wr.release()
    cap.release()
    mb = os.path.getsize(out) / 1e6
    print(f"완료: {out}  ({w}x{h}, {n / fps:.0f}초, {mb:.0f} MB)")


if __name__ == "__main__":
    main()
