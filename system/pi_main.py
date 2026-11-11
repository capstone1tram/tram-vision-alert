"""라즈베리 파이 메인: 촬영·전송 -> 칼만 추적 -> 거리·TTC·등급 -> 부저/LED·화면 -> 기록

실행 예
  # 1) 노트북 없이 PC 한 대로 전체 시험 (노트북 서버를 같은 프로그램 안에서 띄움)
  python pi_main.py --source sample.mp4 --server local --server-detector color --local color

  # 2) 연결 끊김 흉내: 3초부터 2.5초 동안 끊김 -> 정상 -> 예측 -> 자체 검출 -> 정상
  python pi_main.py --source sample.mp4 --server local --server-detector color --local color --drop-at 3 --drop-for 2.5

  # 3) 실제 파이 (노트북 IP가 192.168.137.1일 때, 자체 검출은 NCNN)
  python pi_main.py --source picam --server tcp://192.168.137.1:5555 --local yolo --local-model yolo11n_ncnn_model --local-imgsz 320

  # 4) AI HAT+를 자체 검출기로
  python pi_main.py --source picam --server tcp://192.168.137.1:5555 --local hailo

  # 5) 노트북 없이 파이 혼자 (AI HAT+를 주 검출기로)
  python pi_main.py --source picam --server none --local hailo

  1:10 모형 시험장이면 --scale 0.1 을 붙이세요.  창 없이: --no-show,  결과 영상 저장: --save out.mp4
"""
import argparse
import csv
import os
import threading
import time
from datetime import datetime

import cv2

from tram import camera, display
from tram.alarm import Alarm
from tram.config import LEVEL_NAMES_KO, add_common_args, apply_common_args, cfg
from tram.detectors import make_detector
from tram.guard import MODE_KO, Guard
from tram.link import LinkClient


def start_local_server(port, detector, model, imgsz):
    """--server local: 노트북 서버를 이 프로그램 안 스레드로 띄움 (PC 한 대 시험용)."""
    import laptop_server
    det = make_detector(detector, model, imgsz)
    stop = threading.Event()
    th = threading.Thread(target=laptop_server.run, kwargs=dict(port=port, detector=det, stop=stop.is_set),
                          daemon=True)
    th.start()
    time.sleep(0.3)
    return stop


def main():
    ap = argparse.ArgumentParser(description="트램 전방 위험 알림 - 라즈베리 파이 쪽")
    ap.add_argument("--source", default="picam", help="picam / 0(웹캠) / 영상 파일")
    ap.add_argument("--server", default="tcp://192.168.137.1:5555",
                    help="노트북 주소 tcp://IP:5555, local(같은 PC에서 서버 띄우기), none(파이 단독)")
    ap.add_argument("--port", type=int, default=5555, help="--server local일 때 포트")
    ap.add_argument("--server-detector", default="color", choices=["color", "yolo"], help="--server local일 때 검출기")
    ap.add_argument("--server-model", default="yolo11n.pt")
    ap.add_argument("--local", default="yolo", choices=["none", "color", "yolo", "hailo"],
                    help="연결이 끊겼을 때(또는 단독일 때) 파이가 직접 쓸 검출기")
    ap.add_argument("--local-model", default=None,
                    help="yolo: 기본 yolo11n_ncnn_model / hailo: .hef 경로 (기본 /usr/share/hailo-models/yolov8s_h8l.hef)")
    ap.add_argument("--local-imgsz", type=int, default=320)
    ap.add_argument("--drop-at", type=float, default=-1, help="시험용: 시작 후 이 시각(초)부터 연결을 끊은 척")
    ap.add_argument("--drop-for", type=float, default=0)
    ap.add_argument("--no-show", action="store_true", help="화면 창 안 띄움 (SSH 접속 등)")
    ap.add_argument("--save", default="", help="화면을 영상으로 저장 (예: out.mp4)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--no-gpio", action="store_true", help="부저·LED 안 씀")
    ap.add_argument("--loop", action="store_true", help="영상 파일 반복 재생")
    ap.add_argument("--duration", type=float, default=0, help="이 시간(초) 뒤 자동 종료 (0 = 끝까지)")
    add_common_args(ap)
    args = ap.parse_args()
    apply_common_args(args)
    print("[설정]", cfg.summary())

    # 준비 ------------------------------------------------------------------
    stop_server = None
    if args.server == "local":
        stop_server = start_local_server(args.port, args.server_detector, args.server_model, 640)
        args.server = f"tcp://127.0.0.1:{args.port}"
    use_link = args.server != "none"
    if args.local == "yolo" and not args.local_model:
        args.local_model = "yolo11n_ncnn_model"
    if args.local == "yolo" and not os.path.exists(args.local_model) and not args.local_model.endswith(".pt"):
        raise SystemExit(f"자체 검출 모델 '{args.local_model}'이 없습니다. tools/export_models.py --target ncnn 으로 만들거나 "
                         f"--local color / --local none 으로 실행하세요.")
    print(f"[준비] 자체 검출기 '{args.local}' 불러오는 중 (미리 올려 둬야 전환이 빠름)...")
    local = make_detector(args.local, args.local_model, args.local_imgsz)
    t_start = time.monotonic()

    def dropped():
        el = time.monotonic() - t_start
        return args.drop_at >= 0 and args.drop_at <= el < args.drop_at + args.drop_for

    link = LinkClient(args.server, drop=dropped) if use_link else None
    guard = Guard(local, use_link)
    alarm = Alarm(enable=not args.no_gpio)

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_frames = open(os.path.join(args.log_dir, f"{stamp}_frames.csv"), "w", newline="", encoding="utf-8-sig")
    f_tracks = open(os.path.join(args.log_dir, f"{stamp}_tracks.csv"), "w", newline="", encoding="utf-8-sig")
    lf, lt = csv.writer(f_frames), csv.writer(f_tracks)
    lf.writerow(["t_s", "video_t_s", "frame", "mode", "level", "raw_level", "n_tracks", "worst_id",
                 "worst_Z_m", "worst_X_m", "worst_ttc_s", "rtt_ms", "link_age_s", "no_rail"])
    lt.writerow(["t_s", "video_t_s", "frame", "mode", "id", "name", "X_m", "Z_m", "vx", "v_close",
                 "sigma_X", "sigma_Z", "ttc_s", "a_req", "in_corridor", "level"])

    writer = None
    fps_ema, t_prev = None, None
    stats = {"frames": 0, "modes": {}, "levels": {}, "rtt": []}
    print(f"[시작] 입력 {args.source}, 노트북 {args.server if use_link else '없음(단독)'}. 끝내려면 창에서 q 또는 Ctrl+C")

    try:
        for t, idx, vt, frame in camera.frames(args.source, realtime=True, loop=args.loop):
            el = t - t_start
            if args.duration and el > args.duration:
                break
            if link:
                link.submit(idx, t, frame)
                replies, age, rtt = link.poll(), link.age(time.monotonic()), link.rtt_ms
            else:
                replies, age, rtt = [], float("inf"), None
            res = guard.step(t, frame, replies, age)
            alarm.set(res["level"], res["degraded"])

            # 기록 --------------------------------------------------------
            w = res["worst"]
            lf.writerow([f"{el:.3f}", f"{vt:.3f}", idx, res["mode"], res["level"], res["raw"], len(res["objs"]),
                         w["id"] if w else "", f"{w['m']['Z']:.2f}" if w else "", f"{w['state']['X']:.2f}" if w else "",
                         f"{w['m']['ttc']:.2f}" if w and w["m"]["ttc"] < 1e6 else "",
                         f"{rtt:.0f}" if rtt is not None and res["mode"] == "ONLINE" else "",
                         f"{age:.2f}" if age < 1e6 else "", int(res["no_rail"])])
            for o in res["objs"]:
                s, m = o["state"], o["m"]
                lt.writerow([f"{el:.3f}", f"{vt:.3f}", idx, res["mode"], o["id"], s["name"], f"{s['X']:.3f}",
                             f"{s['Z']:.3f}", f"{s['vx']:.2f}", f"{s['v_close']:.2f}", f"{s['sigma_X']:.3f}",
                             f"{s['sigma_Z']:.3f}", f"{m['ttc']:.2f}" if m["ttc"] < 1e6 else "",
                             f"{m['a_req']:.2f}" if m["a_req"] < 1e6 else "inf", int(m["in_corridor"]), o["level"]])
            stats["frames"] += 1
            stats["modes"][res["mode"]] = stats["modes"].get(res["mode"], 0) + 1
            stats["levels"][res["level"]] = stats["levels"].get(res["level"], 0) + 1
            if rtt is not None and res["mode"] == "ONLINE":
                stats["rtt"].append(rtt)

            # 화면 --------------------------------------------------------
            if t_prev is not None:
                inst = 1.0 / max(t - t_prev, 1e-3)
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
            t_prev = t
            if not args.no_show or args.save:
                out = display.draw(frame, res, fps_ema, rtt, age)
                if args.save:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), cfg.FPS,
                                                 (out.shape[1], out.shape[0]))
                    writer.write(out)
                if not args.no_show:
                    cv2.imshow("tram guard (pi)", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if stats["frames"] % 15 == 1:
                wtxt = "-" if w is None else f"#{w['id']} {w['state']['name']} {w['m']['Z']:.1f} m"
                print(f"{el:6.1f}s  {MODE_KO[res['mode']]:<14} 등급 {LEVEL_NAMES_KO[res['level']]:<5} "
                      f"추적 {len(res['objs'])}개  가장 위험 {wtxt}"
                      + (f"  왕복 {rtt:.0f} ms" if rtt is not None and res["mode"] == "ONLINE" else "")
                      + (f"  {fps_ema:.1f} FPS" if fps_ema else ""))
    except KeyboardInterrupt:
        pass
    finally:
        alarm.close()
        if link:
            link.close()
        if stop_server:
            stop_server.set()
        if writer:
            writer.release()
        f_frames.close()
        f_tracks.close()
        cv2.destroyAllWindows()

    n = max(stats["frames"], 1)
    print("\n[요약]")
    print(f"  프레임 {stats['frames']}개, 평균 {stats['frames'] / max(time.monotonic() - t_start, 1e-3):.1f} FPS")
    print("  모드 비율: " + ", ".join(f"{MODE_KO[k]} {v / n * 100:.0f}%" for k, v in stats["modes"].items()))
    print("  등급 비율: " + ", ".join(f"{LEVEL_NAMES_KO[k]} {v / n * 100:.0f}%" for k, v in sorted(stats["levels"].items())))
    if stats["rtt"]:
        r = sorted(stats["rtt"])
        print(f"  왕복 시간: 중앙값 {r[len(r) // 2]:.0f} ms, 95% {r[int(len(r) * 0.95)]:.0f} ms (목표 300 ms 이하)")
    print(f"  기록: {f_frames.name}, {f_tracks.name}")


if __name__ == "__main__":
    main()
