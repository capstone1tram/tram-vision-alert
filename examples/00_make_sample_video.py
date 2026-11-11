"""00. 시험용 합성 영상 만들기

카메라나 트램 영상이 없어도 다른 예제를 돌려 볼 수 있게, 정답을 아는 가짜 영상을 만듭니다.
- 노란 선 2개 = 레일 (궤간 1.435 m)
- 빨간 사각형 = 선로를 가로지르는 사람 (키 1.7 m)
- 트램은 30 km/h로 전진, 사람은 오른쪽에서 1.4 m/s로 선로를 건넘

카메라 원리(핀홀 모델): 땅 위 점 (옆 X, 앞 Z)는 화면의
  u = cx + f * X / Z,   v = cy + f * h / Z   (h = 카메라 높이)
에 찍힙니다. 이 식을 거꾸로 쓰면 거리 추정이 됩니다(04번 예제).

실행:  python 00_make_sample_video.py          -> sample.mp4 와 sample_truth.csv 생성
"""
import argparse
import csv
import math

import cv2
import numpy as np

import config as C


def focal_px(width=C.IMG_W, hfov_deg=C.HFOV_DEG):
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def project(X, Z, Y=0.0, cam_h=2.5, f=None):
    """땅 기준 높이 Y인 점 (X, Y, Z)를 화면 좌표 (u, v)로."""
    f = f or focal_px()
    cx, cy = C.IMG_W / 2, C.IMG_H / 2
    return cx + f * X / Z, cy + f * (cam_h - Y) / Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tram-kmh", type=float, default=30.0)
    args = ap.parse_args()

    v_tram = args.tram_kmh / 3.6
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (C.IMG_W, C.IMG_H))
    truth = open(args.out.rsplit(".", 1)[0] + "_truth.csv", "w", newline="")
    tw = csv.writer(truth)
    tw.writerow(["frame", "t_s", "Z_m", "X_m"])

    n = int(args.fps * args.seconds)
    for i in range(n):
        t = i / args.fps
        img = np.full((C.IMG_H, C.IMG_W, 3), 90, np.uint8)               # 회색 도로
        img[: C.IMG_H // 2] = (200, 170, 140)                              # 하늘

        # 레일: 3 m ~ 200 m 구간을 짧은 선분으로 이어 그림
        for side in (-1, 1):
            X = side * C.GAUGE_M / 2
            pts = [project(X, z) for z in np.geomspace(3, 200, 60)]
            pts = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (0, 220, 255), 3, cv2.LINE_AA)

        # 사람: 트램이 다가오므로 앞 거리 Z가 줄고, 옆 위치 X도 선로 쪽으로 이동
        Z = 60.0 - v_tram * t
        X = 4.0 - 1.4 * t
        if Z > 4:
            u1, v1 = project(X - 0.25, Z, Y=1.7)
            u2, v2 = project(X + 0.25, Z, Y=0.0)
            cv2.rectangle(img, (int(u1), int(v1)), (int(u2), int(v2)), (40, 40, 220), -1)
        tw.writerow([i, f"{t:.3f}", f"{Z:.2f}", f"{X:.2f}"])
        writer.write(img)

    writer.release()
    truth.close()
    print(f"saved {args.out} ({n} frames) and truth csv")


if __name__ == "__main__":
    main()
