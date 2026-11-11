"""10. 라즈베리 파이 쪽: 촬영·전송 + 칼만 추적 + 거리·TTC·등급 + 경보 + 연결 끊김 대응

  정상(ONLINE)   : 사진을 노트북(09번)에 보내고, 돌아온 박스로 칼만 추적·판정
  예측(PREDICT)  : 답이 안 오면 COAST_S초 동안 칼만 예측만으로 판정
                   (불확실성 2σ만큼 통로를 넓히고, 등급은 끊기기 직전보다 내리지 않음)
  자체(LOCAL)    : 그래도 안 오면 파이가 직접 검출 (YOLO NCNN 320 또는 색 검출) + '감시 저하' 표시
연결이 돌아오면 바로 ONLINE으로 복귀합니다. COAST_S = 1.0초는 08번 시뮬레이션 결과로 정한 제안값입니다.

PC 한 대에서 시험 (창 두 개):
  python 00_make_sample_video.py
  python 09_laptop_server.py --detector color
  python 10_pi_client.py --source sample.mp4 --local color --drop-at 3 --drop-for 2.5
파이에서:
  python 10_pi_client.py --server tcp://<노트북IP>:5555 --source picam --local yolo --local-model yolo11n_ncnn_model
"""
import argparse
import csv
import importlib
import json
import time

import cv2
import numpy as np
import zmq

import config as C

rail = importlib.import_module("03_rail_detect")
dist = importlib.import_module("04_distance")
risk = importlib.import_module("05_risk_level")
pipe = importlib.import_module("06_pipeline")
kt = importlib.import_module("07_kalman_tracker")

LINK_TIMEOUT_S = 0.3
COAST_S = 1.0
MODE_KO = {"ONLINE": "정상", "PREDICT": "예측", "LOCAL": "자체 검출(감시 저하)"}


class Link:
    """ZeroMQ REQ 소켓. 시간 안에 답이 없으면 소켓을 새로 만들어 다시 연결 (Lazy Pirate 패턴)."""

    def __init__(self, addr):
        self.addr, self.ctx, self.sock = addr, zmq.Context(), None
        self._open()

    def _open(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def ask(self, frame_id, t_capture, jpg, timeout_s):
        head = json.dumps({"frame_id": frame_id, "t_capture": t_capture}).encode()
        self.sock.send_multipart([head, jpg])
        if self.sock.poll(int(timeout_s * 1000)):
            return json.loads(self.sock.recv())
        self._open()
        return None


class Alarm:
    """부저·LED (gpiozero). 파이가 아니면 화면 출력만."""

    def __init__(self):
        try:
            from gpiozero import Buzzer, LED
            self.buz, self.leds = Buzzer(17), [LED(22), LED(23), LED(24)]   # 핀 번호는 배선에 맞게
        except Exception:
            self.buz = None

    def set(self, level):
        if self.buz is None:
            return
        for i, led in enumerate(self.leds):
            led.on() if level >= i + 1 else led.off()
        self.buz.on() if level >= 3 else self.buz.off()


def to_meas(dets, rails):
    """박스 -> 바닥 평면 (X 옆 거리, Z 앞 거리)."""
    out = []
    for d in dets:
        z, d_lat, _ = dist.obj_distance(d["box"], rails)
        if z is not None and d_lat is not None:
            out.append((d_lat, z))
    return out


def assess(s, widen):
    """칼만 상태 하나 -> (등급, 지표). widen이면 2σ만큼 가깝고 선로 쪽에 있다고 가정."""
    X = abs(s["X"]) - (2 * s["sigma_X"] if widen else 0)
    Z = max(s["Z"] - (2 * s["sigma_Z"] if widen else 0), 0.1)
    vc, vx = s["v_close"], s["vx"]
    ttc = Z / vc if vc > 0.1 else float("inf")
    brake = Z - vc * C.T_REACT_S
    a_req = vc ** 2 / (2 * brake) if vc > 0.1 and brake > 0.1 else (float("inf") if vc > 0.1 else 0.0)
    d_edge = X - C.CORRIDOR_HALF_M
    toward = -np.sign(s["X"]) * vx
    tte = d_edge / toward if (d_edge > 0 and toward > 0.05) else float("inf")
    m = {"Z": Z, "ttc": ttc, "a_req": a_req, "in_corridor": d_edge <= 0, "d_edge": d_edge,
         "approaching": toward > 0.05, "time_to_enter": tte}
    return risk.risk_level(m), m


def frames(source):
    if source == "picam":
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (C.IMG_W, C.IMG_H), "format": "RGB888"}))
        cam.start()
        t0 = time.monotonic()
        while True:
            yield time.monotonic() - t0, cam.capture_array()
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield i / fps, f
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="tcp://127.0.0.1:5555")
    ap.add_argument("--source", default="sample.mp4")
    ap.add_argument("--local", choices=["yolo", "color"], default="color")
    ap.add_argument("--local-model", default="yolo11n_ncnn_model")
    ap.add_argument("--drop-at", type=float, default=-1, help="시험용: 이 시각부터 전송을 끊음 (초)")
    ap.add_argument("--drop-for", type=float, default=0)
    ap.add_argument("--log", default="pi_log.csv")
    args = ap.parse_args()

    link, alarm = Link(args.server), Alarm()
    tracker = kt.Tracker(max_coast_s=1.5)
    # 자체 검출기는 처음부터 메모리에 올려 둠 -> 전환할 때 로딩 시간 0
    local = pipe.ColorDetector() if args.local == "color" else pipe.YoloDetector(args.local_model)
    rails, t_lost, level_at_loss, shown = None, None, 0, 0
    smoother = risk.LevelSmoother()
    log = csv.writer(open(args.log, "w", newline="", encoding="utf-8"))
    log.writerow(["t_s", "mode", "n_tracks", "level", "worst_Z_m", "rtt_ms"])

    for fid, (t, frame) in enumerate(frames(args.source)):
        dropped = args.drop_at >= 0 and args.drop_at <= t < args.drop_at + args.drop_for
        reply, rtt = None, None
        if not dropped:
            jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            t0 = time.perf_counter()
            reply = link.ask(fid, t, jpg, LINK_TIMEOUT_S)
            rtt = (time.perf_counter() - t0) * 1000

        if reply is not None:
            mode, t_lost = "ONLINE", None
            r = reply["rails"]
            rails = rail.RailModel(r["left"], r["right"], r["y_top"], r["y_bottom"]) if r else rails
            states = tracker.step(t, to_meas(reply["dets"], rails))
        else:
            if t_lost is None:
                t_lost, level_at_loss = t, shown
            if t - t_lost < COAST_S:
                mode = "PREDICT"
                states = tracker.step(t, [])
            else:
                mode = "LOCAL"
                found = rail.find_rails(rail.rail_mask(frame))     # 선로 검출은 가벼워서 파이도 가능
                rails = found if found is not None else rails
                states = tracker.step(t, to_meas(local(frame), rails))

        worst, worst_z = 0, None
        for s in states.values():
            lv, m = assess(s, widen=(mode != "ONLINE"))
            if worst_z is None or lv > worst or (lv == worst and m["Z"] < worst_z):
                worst, worst_z = lv, m["Z"]
        if mode == "PREDICT":
            worst = max(worst, level_at_loss)          # 예측 중에는 등급을 내리지 않음
        if mode == "LOCAL":
            worst = max(worst, 1)                      # 감시 저하 = 최소 '주의'
        shown = smoother.update(worst, t)
        alarm.set(shown)

        log.writerow([f"{t:.2f}", mode, len(states), shown,
                      "" if worst_z is None else f"{worst_z:.1f}", "" if rtt is None else f"{rtt:.0f}"])
        if fid % 5 == 0:
            ztxt = "-" if worst_z is None else f"{worst_z:5.1f} m"
            print(f"t={t:5.1f}s  {MODE_KO[mode]:<14} 추적 {len(states)}개  등급 {C.LEVEL_NAMES_KO[shown]}  "
                  f"가장 위험한 물체 {ztxt}" + ("" if rtt is None else f"  왕복 {rtt:.0f} ms"))
    print(f"끝. 기록 -> {args.log}")


if __name__ == "__main__":
    main()
