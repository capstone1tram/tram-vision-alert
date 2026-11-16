"""소실점 접근 감지 (라즈베리 파이 + 카메라 모듈 3, 고정 카메라)

카메라를 선로 위/옆에 고정하고 선로 방향을 보게 둔 뒤 실행합니다.
  1) 시작 몇 초: 선로를 찾아 소실점·궤간 기울기를 고정 (calib.json 으로 저장 가능)
  2) 2304x1296 원본에서 소실점 주변 ROI(기본 320x160)만 잘라 흔들림 보정
  3) 배경 차분 -> 선로 위 + 열차 폭 크기 덩어리 -> 1.5초 창 확대율로 TTC -> 주의/경고/위험
  4) 프레임마다 CSV 기록, 끝나면 실측 요약(처리 fps, 흔들림, 배경 잡음, 헛검출 비율, 최대 픽셀 크기)

사용 예
  # 파이 (카메라)
  python approach_main.py --source picam --show
  python approach_main.py --source picam --no-show --save-calib calib.json      # ssh로 화면 없이
  python approach_main.py --source picam --calib calib.json --duration 120       # 저장한 소실점으로 바로
  # 파이에서 시험 영상 녹화 (나중에 PC에서 같은 알고리즘 돌리기)
  python approach_main.py --source picam --record raw.mp4 --duration 60 --no-detect
  # PC (녹화 영상 / 합성 영상)
  python tools/make_approach_video.py                       # approach_sample.mp4 + 정답 CSV
  python approach_main.py --source approach_sample.mp4 --show
  # 선로를 못 찾을 때: 소실점을 직접 (원본 좌표)
  python approach_main.py --source raw.mp4 --vp 1150,610
  # 거리별 예상 픽셀 수만 보기
  python approach_main.py --pixels
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tram.approach import ApproachTracker, Background, LevelGate, candidates, expected_px  # noqa: E402
from tram.camera import FullFrames                                                         # noqa: E402
from tram.roi import RailCalib, RoiBox, Stabilizer, calibrate                              # noqa: E402

LEVEL_NAME = ["-", "DETECT", "WARN", "DANGER"]
LEVEL_COLOR = [(160, 160, 160), (0, 220, 255), (0, 140, 255), (0, 0, 255)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="picam", help="picam / 웹캠 번호 / 영상 파일")
    ap.add_argument("--width", type=int, default=2304)
    ap.add_argument("--height", type=int, default=1296)
    ap.add_argument("--fps", type=int, default=30, help="카메라 fps (2304x1296은 30 또는 56)")
    ap.add_argument("--realtime", action="store_true", help="영상 파일을 실제 속도로 (밀리면 건너뜀)")
    ap.add_argument("--duration", type=float, default=0, help="이 초만큼만 실행 (0=끝까지/q)")
    # 카메라 고정
    ap.add_argument("--lock", choices=["auto", "manual", "off"], default="auto",
                    help="auto: 1.5초 자동 맞춤 후 고정 / manual: 아래 값으로 / off: 자동 그대로")
    ap.add_argument("--exposure-us", type=int)
    ap.add_argument("--gain", type=float)
    ap.add_argument("--lens", type=float, default=0.0, help="LensPosition (디옵터, 0=무한대)")
    # 소실점
    ap.add_argument("--calib", help="저장한 calib.json 사용 (선로 검출 생략)")
    ap.add_argument("--save-calib", help="찾은 소실점을 이 파일로 저장")
    ap.add_argument("--vp", help="소실점 직접 지정 'x,y' (원본 좌표). 카메라가 낮아 선로 검출이 안 될 때")
    ap.add_argument("--cam-h", type=float, help="--vp 와 함께: 레일 윗면에서 카메라 높이(m). 주면 궤간 px를 "
                    "계산해 크기 검사를 켬 (평평한 바닥: 궤간px(y) = 궤간 × (y - 소실점y) / 카메라높이)")
    ap.add_argument("--calib-frames", type=int, default=30, help="선로 검출에 쓸 프레임 수")
    ap.add_argument("--y-start", type=float, default=0.45, help="선로 검출 시작 높이 비율")
    # ROI
    ap.add_argument("--roi", default="320x160", help="ROI 크기 (2304 폭 기준, 해상도 다르면 비율대로)")
    ap.add_argument("--roi-dy", type=float, default=0.6, help="소실점을 ROI 세로 어디에 둘지 (0=위, 1=아래)")
    ap.add_argument("--no-stab", action="store_true", help="흔들림 보정 끄기 (비교용)")
    # 배경 차분·크기 검사
    ap.add_argument("--bg", choices=["gauss", "mog2"], default="gauss")
    ap.add_argument("--alpha", type=float, default=0.003, help="배경 갱신 속도 (프레임당, 0.003 ≈ 30fps에서 11초)")
    ap.add_argument("--k-sigma", type=float, default=4.0, help="전경 판정: 배경과 차이 > k * 픽셀 잡음")
    ap.add_argument("--sigma-min", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=30, help="배경 학습 프레임 수 (이 동안 열차가 없어야 함)")
    ap.add_argument("--min-px", type=int, default=4, help="이보다 작은 덩어리는 버림")
    ap.add_argument("--train-w", type=float, default=3.0, help="열차 폭 (m)")
    ap.add_argument("--train-h", type=float, default=3.6, help="열차 앞면 높이 (m, 거리 추정용)")
    ap.add_argument("--gauge", type=float, default=1.435)
    ap.add_argument("--no-size-check", action="store_true")
    # TTC·경보
    ap.add_argument("--window", type=float, default=1.5, help="TTC 맞춤 창 (초)")
    ap.add_argument("--max-window", type=float, default=3.0, help="작은(먼) 열차일 때 늘리는 최대 창 (초)")
    ap.add_argument("--size-mode", choices=["area", "bbox"], default="area",
                    help="크기 재는 법: area=sqrt(전경 픽셀 수, 매끈) / bbox=sqrt(폭×높이)")
    ap.add_argument("--min-ttc-px", type=float, default=20.0, help="열차 크기가 이 px 이상일 때만 TTC 계산")
    ap.add_argument("--warn", type=float, default=15.0, help="TTC < 이 초 -> 경고")
    ap.add_argument("--danger", type=float, default=8.0, help="TTC < 이 초 -> 위험")
    ap.add_argument("--hold", type=float, default=0.5, help="등급이 바뀌려면 이 초 동안 계속 그 조건이어야 함")
    ap.add_argument("--hfov", type=float, default=66.0, help="수평 화각 (카메라 모듈 3 표준 66°)")
    ap.add_argument("--gpio", action="store_true", help="부저·LED 사용 (tram/alarm.py 핀)")
    # 출력
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--out", default="approach_out", help="CSV·요약 저장 폴더")
    ap.add_argument("--save-video", help="화면(전체 축소 + ROI 확대)을 영상으로 저장")
    ap.add_argument("--record", help="원본 흑백 프레임을 영상으로 저장 (파이 -> PC 시험용)")
    ap.add_argument("--no-detect", action="store_true", help="감지 없이 녹화만")
    ap.add_argument("--truth", help="합성 영상 정답 CSV (make_approach_video.py) -> TTC 오차 채점")
    ap.add_argument("--pixels", action="store_true", help="거리별 예상 픽셀 수 표만 출력하고 끝")
    ap.set_defaults(show=False)
    return ap.parse_args()


def focal_px(width, hfov):
    return (width / 2) / math.tan(math.radians(hfov) / 2)


def print_pixel_table(args):
    print("거리별 예상 픽셀 수 (핀홀, 수평 화각 %.0f°)" % args.hfov)
    print("  해상도      f(px)   |  " + "  ".join(f"{d:>5d}m" for d in (100, 200, 300, 500, 800, 1000)))
    for w in (2304, 4608):           # 1536x864 모드는 가운데를 잘라 화각이 달라서 뺌
        f = focal_px(w, args.hfov)
        hs = "  ".join(f"{expected_px(d, f, args.train_h):6.1f}" for d in (100, 200, 300, 500, 800, 1000))
        ws = "  ".join(f"{expected_px(d, f, args.train_w):6.1f}" for d in (100, 200, 300, 500, 800, 1000))
        print(f"  {w:>4d} 높이 {f:7.0f}   | {hs}")
        print(f"  {w:>4d} 폭   {f:7.0f}   | {ws}")
    print("  (높이 %.1f m, 폭 %.1f m 기준. 있다/없다는 약 8 px, 커지는 속도(TTC)는 15~20 px부터 믿을 만함)"
          % (args.train_h, args.train_w))


def get_calib(args, it, frame_size, first_frames):
    """-> (calib, 이미 읽은 프레임 목록)"""
    if args.vp:
        x, y = (float(v) for v in args.vp.split(","))
        k = args.gauge / args.cam_h if args.cam_h else None
        return RailCalib(x, y, k, 0.0, source="manual"), first_frames
    if args.calib:
        return RailCalib.load(args.calib, frame_size), first_frames
    print(f"[소실점] 선로 검출 중 ({args.calib_frames} 프레임)...")
    grabbed = list(first_frames)
    while len(grabbed) < args.calib_frames:
        try:
            grabbed.append(next(it))
        except StopIteration:
            break
    calib, n_ok = calibrate([g for (_, _, _, g) in grabbed[::max(1, len(grabbed) // 15)]], args.y_start,
                            min_ok=3)
    if calib is None:
        raise SystemExit(f"[소실점] 선로를 찾지 못함 (성공 {n_ok}). --vp x,y 로 직접 지정하세요. "
                         f"(첫 프레임을 {args.out}/first_frame.png 로 저장함)")
    print(f"[소실점] ({calib.x_vp:.0f}, {calib.y_vp:.0f}),  궤간 기울기 k={calib.k:.4f} px/px,  "
          f"중심선 기울기 {calib.m:+.3f}  (성공 {n_ok} 프레임)")
    return calib, grabbed


def draw_view(gray, box, calib, roi_f, fg, cands, tr, lvl, info, scale_full=0.4, zoom=3):
    H, W = gray.shape
    small = cv2.resize(gray, (int(W * scale_full), int(H * scale_full)), interpolation=cv2.INTER_AREA)
    left = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    s = scale_full
    cv2.rectangle(left, (int(box.x0 * s), int(box.y0 * s)),
                  (int((box.x0 + box.w) * s), int((box.y0 + box.h) * s)), LEVEL_COLOR[lvl], 2)
    cv2.drawMarker(left, (int(calib.x_vp * s), int(calib.y_vp * s)), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    if calib.has_gauge:
        for yy in np.linspace(calib.y_vp + 5, H - 1, 12):
            g = calib.gauge_px(yy)
            cx = calib.center_x(yy)
            for x in (cx - g / 2, cx + g / 2):
                cv2.circle(left, (int(x * s), int(yy * s)), 2, (0, 255, 0), -1)
    # ROI 확대 + 전경 빨강
    r8 = np.clip(roi_f, 0, 255).astype(np.uint8)
    rv = cv2.cvtColor(r8, cv2.COLOR_GRAY2BGR)
    rv[fg > 0] = (0.5 * rv[fg > 0] + (0, 0, 127)).astype(np.uint8)
    rv = cv2.resize(rv, (box.w * zoom, box.h * zoom), interpolation=cv2.INTER_NEAREST)
    for c in cands:
        col = (0, 255, 0) if c["size_ok"] else (128, 128, 128)
        x, y = (c["x"] - box.x0) * zoom, (c["y"] - box.y0) * zoom
        cv2.rectangle(rv, (x, y), (x + c["w"] * zoom, y + c["h"] * zoom), col, 1)
    if tr.target is not None:
        c = tr.target
        x, y = (c["x"] - box.x0) * zoom, (c["y"] - box.y0) * zoom
        cv2.rectangle(rv, (x - 2, y - 2), (x + c["w"] * zoom + 2, y + c["h"] * zoom + 2), LEVEL_COLOR[lvl], 2)
    # 합치기
    hh = max(left.shape[0], rv.shape[0] + 110)
    canvas = np.zeros((hh, left.shape[1] + rv.shape[1] + 10, 3), np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    ox = left.shape[1] + 10
    canvas[:rv.shape[0], ox:ox + rv.shape[1]] = rv
    y = rv.shape[0] + 25
    for line in info:
        cv2.putText(canvas, line, (ox, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    cv2.putText(canvas, LEVEL_NAME[lvl], (10, left.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                LEVEL_COLOR[lvl], 3, cv2.LINE_AA)
    return canvas


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    args = parse_args()
    if args.pixels:
        print_pixel_table(args)
        return
    os.makedirs(args.out, exist_ok=True)
    src = FullFrames(args.source, (args.width, args.height), args.fps, args.realtime, args.lock,
                     args.exposure_us, args.gain, None, args.lens)
    it = iter(src)
    try:
        first = next(it)
    except StopIteration:
        raise SystemExit("프레임이 없음")
    H, W = first[3].shape
    cv2.imwrite(os.path.join(args.out, "first_frame.png"), first[3])
    print(f"[입력] {args.source}  {W}x{H}  {src.info}")

    rec = None
    if args.record:
        rec = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H), isColor=False)
    if args.no_detect:                                         # 녹화 전용
        t_start, n = time.monotonic(), 0
        if rec:
            rec.write(first[3])
        for t, i, vt, g in it:
            if rec:
                rec.write(g)
            n += 1
            if args.duration and vt >= args.duration:
                break
        dt = time.monotonic() - t_start
        print(f"[녹화] {n + 1} 프레임, {dt:.1f}초, {(n + 1) / max(dt, 1e-6):.1f} fps")
        if rec:
            rec.release()
        return

    calib, pending = get_calib(args, it, (W, H), [first])
    if args.save_calib:
        calib.save(args.save_calib, (W, H))
        print(f"[소실점] 저장: {args.save_calib}")
    rw, rh = (int(v) for v in args.roi.lower().split("x"))
    sc = W / 2304.0
    box = RoiBox(calib, (W, H), int(round(rw * sc)), int(round(rh * sc)), args.roi_dy)
    stab = Stabilizer(box, (W, H), max_shift=max(8, int(24 * sc)))
    bg = Background((box.h, box.w), args.bg, args.alpha, args.k_sigma, args.sigma_min, args.warmup,
                    min_blob_px=args.min_px)
    tr = ApproachTracker(window_s=args.window, min_ttc_px=args.min_ttc_px, size_mode=args.size_mode,
                         max_window_s=args.max_window)
    gate = LevelGate(args.warn, args.danger, args.hold)
    f_px = focal_px(W, args.hfov)
    print(f"[ROI] x={box.x0} y={box.y0} {box.w}x{box.h}  (원본 {W}x{H}, f≈{f_px:.0f}px)  "
          f"흔들림 측정 조각 {len(stab.patches)}개")

    alarm = None
    if args.gpio:
        from tram.alarm import Alarm
        alarm = Alarm(True)

    vw = None
    csv_path = os.path.join(args.out, "frames.csv")
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(fcsv)
    wr.writerow(["frame", "t", "proc_ms", "dx", "dy", "stab_resp", "bg_global", "fg_ratio", "n_cand",
                 "level", "tgt_x", "tgt_y", "tgt_w", "tgt_h", "tgt_area", "size_ratio", "growth_px_s", "ttc_s",
                 "dist_h_m"])

    proc_ms, shifts, fg_idle, n_frames = [], [], [], 0
    first_t = None
    max_tgt = (0, 0)
    first_confirm_t, first_alarm = None, {}
    wall0 = time.monotonic()
    ttc_log = []
    prev_cands = []

    def frames_all():
        yield from pending
        yield from it

    try:
        for t, i, vt, g in frames_all():
            if first_t is None:
                first_t = t
            if rec:
                rec.write(g)
            t0 = time.perf_counter()
            if args.no_stab:
                dx, dy, resp = 0.0, 0.0, 1.0
            else:
                dx, dy, resp = stab.measure(g)
            roi_f = stab.crop_aligned(g, dx, dy)
            # 배경 갱신을 멈출 곳: 추적 중인 열차 + 앞 프레임의 열차 크기 후보 (여유 있게 넓힘)
            hold = None
            held = ([tr.target] if tr.target is not None else []) + [c for c in prev_cands if c["size_ok"]]
            if held:
                hold = np.zeros((box.h, box.w), np.uint8)
                for c in held:
                    x, y = c["x"] - box.x0, c["y"] - box.y0
                    pad = max(6, max(c["w"], c["h"]))
                    cv2.rectangle(hold, (x - pad, y - pad), (x + c["w"] + pad, y + c["h"] + pad), 255, -1)
            fg = bg.apply(roi_f, hold)
            cands = candidates(fg, box, calib if calib.has_gauge else None, args.train_w,
                               args.gauge, args.min_px, size_check=not args.no_size_check and calib.has_gauge)
            tr.update(t, cands)
            prev_cands = cands
            lvl = gate.update(t, tr) if bg.ready else 0
            ms = (time.perf_counter() - t0) * 1000
            n_frames += 1
            if bg.ready:
                proc_ms.append(ms)
                shifts.append((dx, dy))
                if not tr.confirmed:
                    fg_idle.append(float((fg > 0).mean()))
            if alarm:
                alarm.set(lvl)
            c = tr.target if tr.confirmed else None
            dist = (args.train_h * f_px / c["h"]) if c else None
            if c:
                max_tgt = (max(max_tgt[0], c["w"]), max(max_tgt[1], c["h"]))
                if first_confirm_t is None:
                    first_confirm_t = t - first_t
            if lvl >= 2 and lvl not in first_alarm:
                first_alarm[lvl] = t - first_t
            if tr.ttc is not None and c:
                ttc_log.append((t - first_t, tr.ttc))
            wr.writerow([i, f"{t - first_t:.4f}", f"{ms:.2f}", f"{dx:.3f}", f"{dy:.3f}", f"{resp:.3f}",
                         f"{bg.global_shift:.2f}", f"{(fg > 0).mean():.5f}", len(cands), lvl,
                         c["x"] if c else "", c["y"] if c else "", c["w"] if c else "", c["h"] if c else "", c["area"] if c else "",
                         f"{c['ratio']:.2f}" if c and c["ratio"] else "",
                         f"{tr.growth:.3f}" if c and tr.growth is not None else "",
                         f"{tr.ttc:.2f}" if c and tr.ttc is not None else "",
                         f"{dist:.0f}" if dist else ""])
            if args.show or args.save_video:
                info = [f"t={t - first_t:6.2f}s  proc {ms:5.1f} ms",
                        f"shift dx={dx:+.2f} dy={dy:+.2f}  bg noise={bg.noise_sigma() or 0:.2f}",
                        ("target %dx%d px  ratio %s" % (c["w"], c["h"], f"{c['ratio']:.2f}" if c["ratio"] else "-"))
                        if c else ("warming up..." if not bg.ready else "no target"),
                        (f"growth {tr.growth:+.2f} px/s  TTC {tr.ttc:.1f}s  ~{dist:.0f} m"
                         if c and tr.ttc else (f"~{dist:.0f} m (TTC -)" if dist else ""))]
                view = draw_view(g, box, calib, roi_f, fg, cands, tr, lvl, info)
                if args.save_video:
                    if vw is None:
                        vw = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             min(args.fps, 30), (view.shape[1], view.shape[0]))
                    vw.write(view)
                if args.show:
                    cv2.imshow("approach", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if args.duration and t - first_t >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        fcsv.close()
        if vw:
            vw.release()
        if rec:
            rec.release()
        if alarm:
            alarm.close()
        if args.show:
            cv2.destroyAllWindows()

    wall = time.monotonic() - wall0
    sh = np.array(shifts) if shifts else np.zeros((0, 2))
    summary = {
        "source": args.source, "frame_size": [W, H], "roi": list(box.rect), "camera": src.info,
        "calib": calib.to_dict(),
        "frames": n_frames,
        "loop_fps": round(n_frames / max(wall, 1e-6), 1),
        "proc_ms_p50": round(pct(proc_ms, 50), 2), "proc_ms_p95": round(pct(proc_ms, 95), 2),
        "proc_fps_possible": round(1000 / pct(proc_ms, 50), 1) if proc_ms else None,
        "shift_std_px": [round(float(sh[:, 0].std()), 3), round(float(sh[:, 1].std()), 3)] if len(sh) else None,
        "shift_max_px": round(float(np.abs(sh).max()), 2) if len(sh) else None,
        "bg_noise_sigma": round(bg.noise_sigma() or 0, 2),
        "fg_ratio_idle_mean": round(float(np.mean(fg_idle)), 5) if fg_idle else None,
        "fg_ratio_idle_p99": round(pct(fg_idle, 99), 5) if fg_idle else None,
        "target_max_px_wh": list(max_tgt),
        "first_confirm_s": round(first_confirm_t, 2) if first_confirm_t is not None else None,
        "first_warn_s": round(first_alarm[2], 2) if 2 in first_alarm else None,
        "first_danger_s": round(first_alarm[3], 2) if 3 in first_alarm else None,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n── 실측 요약 ──")
    print(f"  프레임 {n_frames}개, 루프 {summary['loop_fps']} fps (입력 대기 포함)")
    print(f"  ROI 처리 시간 중앙값 {summary['proc_ms_p50']} ms, 95% {summary['proc_ms_p95']} ms "
          f"-> 처리만으로는 약 {summary['proc_fps_possible']} fps 가능")
    print(f"  흔들림 표준편차 dx/dy {summary['shift_std_px']} px, 최대 {summary['shift_max_px']} px")
    print(f"  배경 잡음(픽셀 시간 표준편차 중앙값) {summary['bg_noise_sigma']} (밝기 0~255)")
    print(f"  열차 없을 때 전경 비율 평균 {summary['fg_ratio_idle_mean']}, 99% {summary['fg_ratio_idle_p99']}")
    print(f"  잡힌 열차 최대 크기 {max_tgt[0]}x{max_tgt[1]} px,  처음 확정 {summary['first_confirm_s']} s, "
          f"처음 경고 {summary['first_warn_s']} s, 처음 위험 {summary['first_danger_s']} s")
    if args.truth:
        evaluate_truth(args.truth, ttc_log, first_confirm_t, summary, args)
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  기록: {csv_path},  {os.path.join(args.out, 'summary.json')}")


def evaluate_truth(path, ttc_log, first_confirm_t, summary, args):
    """정답 CSV와 비교: 처음 확정됐을 때 열차 거리, TTC 상대 오차, 경고가 실제 TTC 몇 초에 났는지."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    tt = np.array([float(r["t"]) for r in rows])
    dist = np.array([float(r["dist_m"]) if r["dist_m"] else np.nan for r in rows])
    ttc = np.array([float(r["ttc_true_s"]) if r["ttc_true_s"] else np.nan for r in rows])

    def at(arr, t):
        return float(arr[int(np.clip(np.searchsorted(tt, t), 0, len(tt) - 1))])
    out = {}
    if first_confirm_t is not None:
        out["confirm_dist_m"] = round(at(dist, first_confirm_t), 1)
    for k in ("first_warn_s", "first_danger_s"):
        if summary.get(k) is not None:
            out[k.replace("_s", "_true_ttc_s")] = round(at(ttc, summary[k]), 2)
    if ttc_log:
        e = np.array([(est - at(ttc, t)) / at(ttc, t) for t, est in ttc_log if not np.isnan(at(ttc, t))])
        if len(e):
            out["ttc_rel_err_median"] = round(float(np.median(e)), 3)
            out["ttc_rel_err_abs_p50"] = round(float(np.median(np.abs(e))), 3)
            out["ttc_rel_err_abs_p90"] = round(pct(np.abs(e), 90), 3)
            out["ttc_samples"] = int(len(e))
    summary["truth"] = out
    print("  [정답 비교] " + ", ".join(f"{k}={v}" for k, v in out.items()))


if __name__ == "__main__":
    main()
