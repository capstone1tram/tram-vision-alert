"""(PC) 소실점 접근 감지 시험용 합성 영상 + 정답 CSV

고정 카메라(선로 위, 높이 --cam-h m)가 선로 방향을 보고, 열차가 멀리서 일정 속도로 다가오는 장면.
실제와 비슷하게 넣은 것: 흔들림(레일 진동, 열차가 가까울수록 커짐), 센서 잡음, 구름(화면 전체 밝기 천천히 변화),
선로를 가로지르는 사람(크기 검사로 걸러져야 함), 옆 선로에서 멀어지는 열차(선로 영역 검사로 걸러져야 함).

  python tools/make_approach_video.py                         # approach_sample.mp4 + approach_sample_truth.csv
  python tools/make_approach_video.py --speed 60 --start 1000 --cam-h 0.5 --out low.mp4
  python approach_main.py --source approach_sample.mp4 --truth approach_sample_truth.csv --show
"""
import argparse
import csv
import math
import os

import cv2
import numpy as np

SHIFT = 4                      # cv2 그리기 부분 픽셀 (1/16 px)
S = 1 << SHIFT


def P(x, y):
    return int(round(x * S)), int(round(y * S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="approach_sample.mp4")
    ap.add_argument("--width", type=int, default=2304)
    ap.add_argument("--height", type=int, default=1296)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--hfov", type=float, default=66.0)
    ap.add_argument("--cam-h", type=float, default=1.5, help="카메라 높이 (레일 윗면 기준, m)")
    ap.add_argument("--speed", type=float, default=80, help="열차 속도 km/h")
    ap.add_argument("--start", type=float, default=900, help="열차가 나타날 때 거리 (m)")
    ap.add_argument("--appear", type=float, default=3.0, help="열차가 나타나는 시각 (s, 그 전은 배경 학습)")
    ap.add_argument("--end-dist", type=float, default=60, help="이 거리까지 오면 끝")
    ap.add_argument("--noise", type=float, default=2.0, help="센서 잡음 표준편차")
    ap.add_argument("--jitter", type=float, default=1.0, help="기본 흔들림 (px)")
    ap.add_argument("--no-distractors", action="store_true")
    ap.add_argument("--no-train", action="store_true", help="열차 없이 (헛경보·잡음 실측용 정지 장면)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    W, H = a.width, a.height
    f = (W / 2) / math.tan(math.radians(a.hfov) / 2)
    cx, yh = W / 2, H * 0.46                     # 소실점 (지평선 높이)
    g2 = 1.435 / 2

    def proj(X, Y, Z):                            # 바닥 기준 Y(위로 +) -> 화면
        return cx + f * X / Z, yh + f * (a.cam_h - Y) / Z

    # ── 정적 배경 ──
    bg = np.zeros((H, W), np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]
    bg[:] = np.where(yy < yh, 190 - 60 * (yh - yy) / yh, 110)       # 하늘 / 땅
    # 지평선 근처 숲·건물 띠 (흔들림 측정에 필요한 무늬)
    band_h = int(f * 12 / 2500) + 6
    tex = cv2.resize(rng.uniform(60, 130, (band_h // 2 + 1, W // 6)).astype(np.float32), (W, band_h),
                     interpolation=cv2.INTER_LINEAR)
    for x0 in range(0, W, 37):                     # 들쭉날쭉한 윗선
        hh = int(rng.uniform(0.3, 1.0) * band_h)
        tex[: band_h - hh, x0:x0 + 37] = bg[int(yh) - band_h, 0]
    bg[int(yh) - band_h:int(yh)] = tex
    # 자갈 무늬 (멀수록 촘촘 -> 그냥 고운 잡음)
    ground = rng.normal(0, 12, (H - int(yh), W)).astype(np.float32)
    bg[int(yh):] += cv2.GaussianBlur(ground, (3, 3), 0)
    img = bg.copy()
    # 침목·레일·전봇대
    for Z in np.arange(3, 600, 0.6):
        xl, y = proj(-1.25, 0, Z)
        xr, _ = proj(1.25, 0, Z)
        th = max(1, int(f * 0.2 / Z * 0.3))
        cv2.line(img, P(xl, y), P(xr, y), 80, th, cv2.LINE_AA, SHIFT)
    for X in (-g2, g2, 4.5 - g2, 4.5 + g2):       # 내 선로 + 오른쪽 옆 선로
        pts = np.array([P(*proj(X, 0, Z)) for Z in np.geomspace(2.5, 3000, 400)], np.int32)
        cv2.polylines(img, [pts], False, 225, 2, cv2.LINE_AA, SHIFT)
    for Z in np.arange(20, 1500, 50):             # 왼쪽 전봇대
        x, yb = proj(-3.5, 0, Z)
        _, yt = proj(-3.5, 9, Z)
        cv2.line(img, P(x, yb), P(x, yt), 50, max(1, int(f * 0.25 / Z)), cv2.LINE_AA, SHIFT)
    static = img

    v = a.speed / 3.6
    t_end = a.appear + (a.start - a.end_dist) / v
    n = int(t_end * a.fps)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H), isColor=False)
    truth_path = os.path.splitext(a.out)[0] + "_truth.csv"
    ft = open(truth_path, "w", newline="")
    wr = csv.writer(ft)
    wr.writerow(["frame", "t", "dist_m", "ttc_true_s", "px_w", "px_h", "jitter_dx", "jitter_dy"])
    ph = rng.uniform(0, 6.28, 4)
    for i in range(n):
        t = i / a.fps
        fr = static.copy()
        Z = a.start - v * (t - a.appear) if t >= a.appear and not a.no_train else None
        # 옆 선로 열차: 2~12초에 멀어져 감 (선로 영역 검사로 걸러져야 함)
        if not a.no_distractors and 2.0 < t < 12.0:
            Zs = 60 + 15 * (t - 2.0)
            x0, y0 = proj(4.5 - 1.5, 3.6, Zs)
            x1, y1 = proj(4.5 + 1.5, 0, Zs)
            cv2.rectangle(fr, P(x0, y0), P(x1, y1), 70, -1, cv2.LINE_AA, SHIFT)
        # 선로를 가로지르는 사람 (Z=180 m, 5~14초)
        if not a.no_distractors and 5 < t < 14:
            Xp = -5 + (t - 5) * 1.2
            x0, y0 = proj(Xp - 0.25, 1.7, 180)
            x1, y1 = proj(Xp + 0.25, 0, 180)
            cv2.rectangle(fr, P(x0, y0), P(x1, y1), 40, -1, cv2.LINE_AA, SHIFT)
        if Z is not None:
            x0, y0 = proj(-1.5, 3.6, Z)
            x1, y1 = proj(1.5, 0.0, Z)
            cv2.rectangle(fr, P(x0, y0), P(x1, y1), 55, -1, cv2.LINE_AA, SHIFT)
            # 앞창(밝음)과 전조등
            wx0, wy0 = proj(-1.2, 3.2, Z)
            wx1, wy1 = proj(1.2, 2.3, Z)
            cv2.rectangle(fr, P(wx0, wy0), P(wx1, wy1), 120, -1, cv2.LINE_AA, SHIFT)
            hx, hy = proj(0, 0.8, Z)
            cv2.circle(fr, P(hx, hy), max(S, int(S * f * 0.25 / Z)), 255, -1, cv2.LINE_AA, SHIFT)
            pw, phh = x1 - x0, y1 - y0
        else:
            pw = phh = 0
        # 흔들림: 기본 진동 + 열차가 가까울수록 커지는 레일 진동
        amp = a.jitter * (1 + (3 * (200 / Z) if Z and Z < 600 else 0))
        dx = amp * (0.6 * math.sin(2 * math.pi * 7.3 * t + ph[0]) + 0.4 * math.sin(2 * math.pi * 13.1 * t + ph[1]))
        dy = amp * (0.6 * math.sin(2 * math.pi * 9.7 * t + ph[2]) + 0.4 * math.sin(2 * math.pi * 17.9 * t + ph[3]))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fr *= 1.0 + 0.04 * math.sin(2 * math.pi * t / 17)              # 구름
        fr += rng.normal(0, a.noise, fr.shape).astype(np.float32)
        vw.write(np.clip(fr, 0, 255).astype(np.uint8))
        wr.writerow([i, f"{t:.4f}", f"{Z:.1f}" if Z else "", f"{Z / v:.2f}" if Z else "",
                     f"{pw:.2f}", f"{phh:.2f}", f"{dx:.3f}", f"{dy:.3f}"])
    vw.release()
    ft.close()
    print(f"{a.out} ({W}x{H}, {n} 프레임, {t_end:.1f}s), {truth_path}")
    print(f"소실점 ≈ ({cx:.0f}, {yh:.0f}), f={f:.0f}px, 열차 {a.start:.0f} m -> {a.end_dist:.0f} m, {a.speed:.0f} km/h")


if __name__ == "__main__":
    main()
