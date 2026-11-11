"""시험용 합성 영상 + 정답 CSV 만들기 (카메라·트램 영상 없이 전체 시험용)

- 노란 선 2개 = 레일 (궤간 1.435 m)
- 빨간 사각형 A = 오른쪽에서 선로를 천천히 건너는 사람 (0.8 m/s) -> 주의 -> 경고 -> 위험 -> 벗어남
- 빨간 사각형 B = 왼쪽 완충 구역에 서 있는 사람 -> 주의
- 트램은 30 km/h로 전진

카메라 원리(핀홀): 땅 위 점 (옆 X, 앞 Z)는 화면의  u = cx + f·X/Z,  v = cy + f·h/Z  (h = 카메라 높이)

실행 (프로젝트 폴더에서):
  python tools/make_sample_video.py                    -> sample.mp4, sample_truth.csv
  python tools/make_sample_video.py --scale 0.1 --out sample_model.mp4   -> 1:10 모형 크기
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tram.config import cfg  # noqa: E402


def project(X, Z, Y=0.0, cam_h=2.5):
    f = cfg.focal_px
    return cfg.IMG_W / 2 + f * X / Z, cfg.IMG_H / 2 + f * (cam_h - Y) / Z


def people(t, s, v_tram):
    """(이름, X, Z) 목록. s = 크기 배율."""
    return [("A", (4.0 - 0.8 * t) * s, 70.0 * s - v_tram * t),
            ("B", -2.5 * s, 45.0 * s - v_tram * t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--tram-kmh", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=1.0, help="1:10 모형이면 0.1")
    args = ap.parse_args()
    cfg.apply_scale(args.scale)
    s = args.scale
    v_tram = args.tram_kmh / 3.6 * s
    cam_h = 2.5 * s

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cfg.IMG_W, cfg.IMG_H))
    tw = csv.writer(open(os.path.splitext(args.out)[0] + "_truth.csv", "w", newline="", encoding="utf-8-sig"))
    tw.writerow(["frame", "t_s", "obj", "X_m", "Z_m"])
    rng = np.random.default_rng(0)
    n = int(args.fps * args.seconds)
    for i in range(n):
        t = i / args.fps
        img = np.full((cfg.IMG_H, cfg.IMG_W, 3), 90, np.uint8)            # 회색 바닥
        img[: cfg.IMG_H // 2] = (200, 170, 140)                            # 하늘
        for side in (-1, 1):
            X = side * cfg.GAUGE_M / 2
            pts = np.array([project(X, z, cam_h=cam_h) for z in np.geomspace(3 * s, 200 * s, 60)], np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 220, 255), 3, cv2.LINE_AA)
        # 먼 사람부터 그려야 가까운 사람이 앞에 보임
        for name, X, Z in sorted(people(t, s, v_tram), key=lambda p: -p[2]):
            if Z > 4 * s:
                u1, v1 = project(X - 0.25 * s, Z, Y=1.7 * s, cam_h=cam_h)
                u2, v2 = project(X + 0.25 * s, Z, Y=0.0, cam_h=cam_h)
                cv2.rectangle(img, (int(u1), int(v1)), (int(u2), int(v2)), (40, 40, 220), -1)
                tw.writerow([i, f"{t:.3f}", name, f"{X:.3f}", f"{Z:.3f}"])
        img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))  # 약한 잡음
        writer.write(img)
    writer.release()
    print(f"저장: {args.out} ({n}프레임, {args.seconds}초) + 정답 CSV")


if __name__ == "__main__":
    main()
