"""PC 한 대로 전체 시스템 실행 (실제 트램 블랙박스 영상용)

  영상 -> [인지] 선로 검출 + 위험 구역 + YOLO 물체 검출       (tram/vision.py   : 나중에 '노트북 쪽')
       -> [판단] 거리 -> 칼만 추적 -> TTC·필요 감속도 -> 4단계 등급 (tram/guard.py : 나중에 '파이 쪽')
       -> 화면 + 위험 장면 자동 저장 + 기록(CSV)

인지가 돌려주는 값은 laptop_server.py가 파이에 보내는 답과 모양이 같아서, 나눌 때는
vision.Perception을 노트북 서버에, Guard를 파이에 두면 됩니다. (이 파일 맨 아래 '나눌 때' 참고)

실행 (G:\\tram_ 에서 가상환경 켠 뒤)
  python pc_main.py                                       # 기본 영상 G:\\rail_pic\\tram_blackbox.mp4, 실시간
  python pc_main.py --start 5:30                          # 5분 30초부터
  python pc_main.py --start 5:30 --duration 60            # 거기서 60초만
  python pc_main.py --no-realtime --no-show               # 창 없이 모든 프레임 처리 (가장 정확, 느림)
  python pc_main.py --device cpu                          # GPU가 안 되면
  python pc_main.py --model models\\yolo11s_openvino_model # 더 큰 모델 (멀리 있는 사람 더 잘 봄, 느림)

키 (영상 창에서)
  스페이스 = 멈춤/재생    d = 10초 앞으로    a = 10초 뒤로    s = 지금 화면 캡처
  m = 선로 마스크 보기(레일 후보 픽셀 초록)    c = 초점거리 보정 (멈춘 상태에서 두 점 클릭)    q = 끝

결과: logs\\pc\\<날짜_시각>\\
  clips\\event_001_05m31s_DANGER.mp4 (+ _peak.jpg)   경고 이상 장면 앞 3초 ~ 뒤 2초
  captures\\cap_*.jpg                                 s 키로 캡처한 화면
  events.csv / frames.csv / tracks.csv               장면 목록 / 프레임마다 / 물체마다 기록

거리에 대해
  옆 거리·위험 구역·TTC는 궤간(1.435 m) 비율로 계산해서 카메라 렌즈와 상관없습니다.
  앞 거리 '몇 m'와 필요 감속도는 초점거리(화각)가 필요해서, 처음엔 settings_video.json의 HFOV_DEG
  가정값으로 나옵니다.  c 키로 길이를 아는 바닥 표시(예: 차선 점선 한 칸)를 찍어 보정하세요.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2

from tram import ground, hud
from tram.config import LEVEL_NAMES_KO, add_common_args, apply_common_args, cfg
from tram.detectors import make_detector
from tram.guard import Guard
from tram.recorder import EventRecorder, mmss
from tram.vision import Perception

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = r"G:\rail_pic\tram_blackbox.mp4"
DEFAULT_SETTINGS = os.path.join(ROOT, "settings_video.json")
WIN = "tram PC (q: quit)"


def parse_time(s):
    if not s:
        return 0.0
    if ":" in str(s):
        p = [float(x) for x in str(s).split(":")]
        return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]
    return float(s)


def pick_model(arg):
    if arg:
        return arg
    for p in ("models/yolo11n_openvino_model", "yolo11n_openvino_model", "models/yolo11n.pt", "tools/yolo11n.pt"):
        q = os.path.join(ROOT, p)
        if os.path.exists(q):
            return q
    return "yolo11n.pt"


def pick_device(arg, model):
    if arg and arg != "auto":
        return None if arg == "cpu" else arg
    if "openvino" not in model:
        return None
    try:
        import openvino as ov
        if any(d.startswith("GPU") for d in ov.Core().available_devices):
            return "intel:gpu"
    except Exception:
        pass
    return None


class Video:
    """영상 파일 읽기 + 실시간 재생 시계 (처리가 느리면 프레임을 건너뜀) + 앞뒤 이동."""

    def __init__(self, path, width):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"영상을 열 수 없습니다: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1 < self.fps < 240):
            self.fps = 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dur = self.n / self.fps
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = width if width else self.src_w
        self.h = int(round(self.w * self.src_h / self.src_w / 2) * 2)
        self.idx = 0                                  # 다음에 읽을 프레임 번호

    def seek(self, t):
        t = min(max(t, 0.0), max(self.dur - 1, 0))
        self.idx = int(t * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx)

    def read(self, skip_to=None):
        """skip_to(영상 시각)가 있으면 그 시각까지 건너뛰고 읽음. -> (영상 시각, 작은 사진, 건너뛴 수)"""
        skipped = 0
        if skip_to is not None:
            target = int(skip_to * self.fps)
            while self.idx < target - 1:
                if not self.cap.grab():
                    return None, None, skipped
                self.idx += 1
                skipped += 1
        ok, f = self.cap.read()
        if not ok:
            return None, None, skipped
        t = self.idx / self.fps
        self.idx += 1
        if f.shape[1] != self.w:
            f = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return t, f, skipped


def calibrate(frame, res, length_m):
    """멈춘 화면에서 바닥 위 두 점(가까운 점, 먼 점, 둘 사이 실제 거리 length_m)을 클릭 -> 초점거리.
    두 점의 궤간 픽셀 폭 g1, g2로  Z = f·G/g  이므로  f = L / (G · (1/g2 − 1/g1))."""
    gm = ground.ground_of(res["rails"])
    if gm is None:
        print("  [보정] 선로를 찾은 화면에서만 할 수 있어요.")
        return None
    pts = []

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.setMouseCallback(WIN, on_mouse)
    print(f"  [보정] 바닥에서 길이 {length_m} m인 것의 '가까운 끝'을 클릭하고, 그다음 '먼 끝'을 클릭하세요. (Esc 취소)")
    print("         예: 차선 점선 한 칸(도시 도로 보통 3 m), 정류장 경계석 등. 선로와 나란한 방향이어야 해요.")
    while len(pts) < 2:
        v = frame.copy()
        for p in pts:
            cv2.circle(v, p, 6, (0, 255, 255), 2)
        cv2.putText(v, f"click NEAR end, then FAR end of a {length_m} m ground mark  (Esc: cancel)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow(WIN, v)
        if cv2.waitKey(30) & 0xFF == 27:
            cv2.setMouseCallback(WIN, lambda *a: None)
            return None
    cv2.setMouseCallback(WIN, lambda *a: None)
    (_, y1), (_, y2) = sorted(pts, key=lambda p: -p[1])       # 아래(가까운) 점이 먼저
    g1, g2 = gm.gauge(y1), gm.gauge(y2)
    if g2 <= 1 or g1 <= g2 + 0.5:
        print("  [보정] 두 점의 높이 차이가 너무 작아요. 다시 해 주세요.")
        return None
    f = length_m / (cfg.GAUGE_M * (1 / g2 - 1 / g1))
    z1, z2 = f * cfg.GAUGE_M / g1, f * cfg.GAUGE_M / g2
    print(f"  [보정] 초점거리 f = {f:.0f} px (처리 폭 {cfg.IMG_W} 기준, 수평 화각 약 "
          f"{2 * __import__('math').degrees(__import__('math').atan(cfg.IMG_W / 2 / f)):.0f}°). "
          f"두 점 거리 {z1:.1f} m, {z2:.1f} m.  카메라 높이(궤간 기울기) {gm.camera_height():.2f} m")
    return f


def save_settings(path, updates):
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    data.update(updates)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="트램 전방 위험 알림 - PC 통합 (블랙박스 영상)")
    ap.add_argument("--video", default="", help=f"영상 파일 (기본 {DEFAULT_VIDEO})")
    ap.add_argument("--start", default="0", help="시작 위치. 초 또는 분:초 (예: 5:30)")
    ap.add_argument("--duration", type=float, default=0, help="이만큼(초)만 처리 (0 = 끝까지)")
    ap.add_argument("--width", type=int, default=960, help="처리할 사진 폭 (작을수록 빠름)")
    ap.add_argument("--model", default="", help="YOLO 모델 (기본 models/yolo11n_openvino_model)")
    ap.add_argument("--device", default="auto", help="auto / cpu / intel:gpu")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO 입력 크기 (960이면 멀리 있는 것 더 잘 봄, 느림)")
    ap.add_argument("--conf", type=float, default=None, help="YOLO 확신도 문턱 (기본 설정값 0.35)")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 속도 (실시간 모드)")
    ap.add_argument("--no-realtime", action="store_true", help="모든 프레임 처리 (영상보다 느리게 재생될 수 있음)")
    ap.add_argument("--no-show", action="store_true", help="창 안 띄움")
    ap.add_argument("--clip-level", type=int, default=2, help="이 등급 이상이면 장면 저장 (1 주의, 2 경고, 3 위험)")
    ap.add_argument("--no-clips", action="store_true", help="위험 장면 자동 저장 끄기")
    ap.add_argument("--save-all", default="", help="화면 전체를 영상으로 저장 (예: out.mp4)")
    ap.add_argument("--calib-len", type=float, default=3.0, help="c 키 보정 때 두 점 사이 실제 거리 (m)")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "logs", "pc"))
    add_common_args(ap)
    args = ap.parse_args()

    video_path = args.video or DEFAULT_VIDEO
    if not os.path.exists(video_path):
        video_path = input(f"영상 파일 경로를 넣어 주세요 ({DEFAULT_VIDEO} 없음): ").strip().strip('"')
    settings_path = args.config or DEFAULT_SETTINGS
    if os.path.exists(settings_path):
        cfg.load_json(settings_path)
        args.config = ""
    ref_w = cfg.IMG_W
    apply_common_args(args)

    vid = Video(video_path, args.width)
    if cfg.FX_PX and ref_w and vid.w != ref_w:          # 보정값은 저장할 때 폭 기준 -> 폭이 바뀌면 비례
        cfg.FX_PX = cfg.FX_PX * vid.w / ref_w
    cfg.IMG_W, cfg.IMG_H = vid.w, vid.h
    if args.conf is not None:
        cfg.DET_CONF = args.conf

    model = pick_model(args.model)
    device = pick_device(args.device, model)
    print(f"[영상] {video_path}  {vid.src_w}x{vid.src_h} {vid.fps:.1f} FPS, {vid.dur / 60:.1f}분 -> 처리 {vid.w}x{vid.h}")
    print(f"[설정] {cfg.summary()}  ({'보정값' if cfg.FX_PX else f'화각 {cfg.HFOV_DEG}° 가정'})")
    print(f"[준비] YOLO {model} (장치 {device or 'CPU'}, 입력 {args.imgsz}) 불러오는 중...")
    try:
        det = make_detector("yolo", model, args.imgsz, device)
    except Exception as e:
        if device is None:
            raise
        print(f"  GPU로 못 올려서 CPU로 바꿉니다 ({type(e).__name__}: {str(e)[:100]})")
        device = None
        det = make_detector("yolo", model, args.imgsz, None)
    perc = Perception(det)
    guard = Guard(local_detector=None, use_link=True, dist_fn=ground.obj_distance,
                      visible_fn=ground.visible if cfg.DROP_OUT_OF_VIEW else None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.log_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    rec = EventRecorder(out_dir, args.clip_level, enable=not args.no_clips)
    f_frames = open(os.path.join(out_dir, "frames.csv"), "w", newline="", encoding="utf-8-sig")
    f_tracks = open(os.path.join(out_dir, "tracks.csv"), "w", newline="", encoding="utf-8-sig")
    lf, lt = csv.writer(f_frames), csv.writer(f_tracks)
    lf.writerow(["video_t_s", "frame", "level", "raw_level", "n_dets", "n_tracks", "rail", "rail_status",
                 "camera_h_m", "worst_id", "worst_name", "worst_Z_m", "worst_X_m", "worst_ttc_s", "worst_a_req",
                 "rail_ms", "det_ms", "skipped"])
    lt.writerow(["video_t_s", "frame", "id", "name", "X_m", "Z_m", "vx", "v_close", "sigma_X", "sigma_Z",
                 "ttc_s", "a_req", "in_corridor", "level"])

    start = parse_time(args.start)
    vid.seek(start)
    writer = None
    show = not args.no_show
    if show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, vid.w, vid.h)
    paused, mask_on = False, False
    wall0, vt0 = time.monotonic(), start
    fps_ema, t_prev_wall = None, None
    stats = {"frames": 0, "levels": {}, "rail": 0, "skipped": 0}
    last_out, last_res, last_frame, vt = None, None, None, start
    print(f"[시작] {mmss(start)}부터, {'실시간' if not args.no_realtime else '모든 프레임'}.  결과 폴더: {out_dir}")
    print("       키: 스페이스 멈춤 / d·a ±10초 / s 캡처 / m 마스크 / c 보정 / q 끝")

    def restart_from(t):
        nonlocal guard, wall0, vt0
        vid.seek(t)
        perc.reset()
        guard = Guard(local_detector=None, use_link=True, dist_fn=ground.obj_distance,
                      visible_fn=ground.visible if cfg.DROP_OUT_OF_VIEW else None)
        rec.reset_buffer()
        wall0, vt0 = time.monotonic(), t

    try:
        while True:
            # ── 키 ─────────────────────────────────────────────
            if show:
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                    if not paused:
                        wall0, vt0 = time.monotonic(), vt
                elif key == ord("d"):
                    restart_from(vt + 10)
                elif key == ord("a"):
                    restart_from(vt - 10)
                elif key == ord("m"):
                    mask_on = not mask_on
                elif key == ord("s") and last_out is not None:
                    print("  [캡처]", rec.capture(last_out, vt))
                elif key == ord("c") and last_res is not None:
                    paused = True
                    f = calibrate(last_out, last_res, args.calib_len)
                    if f:
                        cfg.FX_PX = f
                        save_settings(settings_path, {"FX_PX": round(f, 1), "IMG_W": cfg.IMG_W})
                        print(f"  [보정] {settings_path} 에 저장했어요. 다음 실행부터 자동으로 씁니다.")
                if paused:
                    if last_out is not None:
                        v = hud.mask_view(last_out, perc.debug_mask()) if mask_on else last_out.copy()
                        hud.paused_text(v)
                        cv2.imshow(WIN, v)
                    continue

            # ── 프레임 읽기 ────────────────────────────────────
            skip_to = None
            if not args.no_realtime:
                skip_to = vt0 + (time.monotonic() - wall0) * args.speed
            vt_new, frame, skipped = vid.read(skip_to)
            if frame is None:
                break
            vt = vt_new
            if args.duration and vt - start > args.duration:
                break
            if not args.no_realtime:                    # 너무 빠르면 영상 시각에 맞춰 기다림
                wait = (vt - vt0) / args.speed - (time.monotonic() - wall0)
                if wait > 0:
                    time.sleep(wait)

            # ── 인지 (나중에 노트북) ───────────────────────────
            rep, _ = perc.process(frame, vt)
            # ── 판단 (나중에 파이) ─────────────────────────────
            res = guard.step(vt, frame, [rep] if rep else [], age=0.0)

            # ── 화면·기록 ──────────────────────────────────────
            now = time.monotonic()
            if t_prev_wall is not None:
                inst = 1.0 / max(now - t_prev_wall, 1e-3)
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
            t_prev_wall = now
            rec.set_fps(fps_ema if not args.no_realtime else vid.fps)
            gm = ground.ground_of(res["rails"])
            bar = (f"{mmss(vt)} / {mmss(vid.dur)}   {fps_ema or 0:4.1f} FPS   rail {perc.ms['rail']:.0f}ms"
                   f"  yolo {perc.ms['det']:.0f}ms   rail:{perc.rails.status}"
                   f"   f={cfg.focal_px:.0f}px{'' if cfg.FX_PX else '(guess)'}"
                   + (f"  cam h~{gm.camera_height():.1f}m" if gm else ""))
            out = hud.draw(frame, res, {"bar": bar, "rec": rec.ev is not None})
            rec.push(out, vt, res)
            last_out, last_res, last_frame = out, res, frame
            if args.save_all:
                if writer is None:
                    writer = cv2.VideoWriter(args.save_all, cv2.VideoWriter_fourcc(*"mp4v"),
                                             vid.fps if args.no_realtime else 15, (out.shape[1], out.shape[0]))
                writer.write(out)
            if show:
                cv2.imshow(WIN, hud.mask_view(out, perc.debug_mask()) if mask_on else out)

            w = res["worst"]
            lf.writerow([f"{vt:.3f}", vid.idx - 1, res["level"], res["raw"], len(res["boxes"]), len(res["objs"]),
                         int(res["rails"] is not None and not res["no_rail"]), perc.rails.status,
                         f"{gm.camera_height():.2f}" if gm else "",
                         w["id"] if w else "", w["state"]["name"] if w else "",
                         f"{w['m']['Z']:.2f}" if w else "", f"{w['state']['X']:.2f}" if w else "",
                         f"{w['m']['ttc']:.2f}" if w and w["m"]["ttc"] < 1e6 else "",
                         f"{w['m']['a_req']:.2f}" if w and w["m"]["a_req"] < 1e6 else "",
                         f"{perc.ms['rail']:.1f}", f"{perc.ms['det']:.1f}", skipped])
            for o in res["objs"]:
                s, m = o["state"], o["m"]
                lt.writerow([f"{vt:.3f}", vid.idx - 1, o["id"], s["name"], f"{s['X']:.3f}", f"{s['Z']:.3f}",
                             f"{s['vx']:.2f}", f"{s['v_close']:.2f}", f"{s['sigma_X']:.3f}", f"{s['sigma_Z']:.3f}",
                             f"{m['ttc']:.2f}" if m["ttc"] < 1e6 else "",
                             f"{m['a_req']:.2f}" if m["a_req"] < 1e6 else "inf", int(m["in_corridor"]), o["level"]])
            stats["frames"] += 1
            stats["levels"][res["level"]] = stats["levels"].get(res["level"], 0) + 1
            stats["rail"] += int(res["rails"] is not None and not res["no_rail"])
            stats["skipped"] += skipped
            if args.no_show and stats["frames"] % 100 == 1:
                print(f"  {mmss(vt)}  {fps_ema or 0:4.1f} FPS  등급 {LEVEL_NAMES_KO[res['level']]}  "
                      f"물체 {len(res['objs'])}  선로 {perc.rails.status}")
    except KeyboardInterrupt:
        pass
    finally:
        rec.close()
        if writer:
            writer.release()
        f_frames.close()
        f_tracks.close()
        if show:
            cv2.destroyAllWindows()

    n = max(stats["frames"], 1)
    print("\n[요약]")
    print(f"  처리 {stats['frames']}장 (건너뜀 {stats['skipped']}장), 평균 {fps_ema or 0:.1f} FPS")
    print(f"  선로 찾은 비율 {stats['rail'] / n * 100:.0f}%")
    print("  등급 비율: " + ", ".join(f"{LEVEL_NAMES_KO[k]} {v / n * 100:.0f}%" for k, v in sorted(stats["levels"].items())))
    print(f"  위험 장면 {rec.n_ev}개 저장.  결과 폴더: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())

# ── 나눌 때 ─────────────────────────────────────────────────────────────
# 노트북: laptop_server.py의 make_handler 안에서
#           perc = Perception(detector);  rep, _ = perc.process(frame, t)
#         를 부르고 rep["rails"], rep["dets"]를 돌려주면 됩니다. (선로는 rails_photo + RailTracker)
# 파이  : pi_main.py에서 Guard(local, use_link, dist_fn=ground.obj_distance) 로 만들면
#         실제 선로 영상용 거리 계산(지평선 연장)을 씁니다. 나머지(칼만·등급·연결 끊김)는 그대로.
