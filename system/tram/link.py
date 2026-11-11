"""파이 <-> 노트북 통신 (ZeroMQ REQ/REP)

  파이 -> 노트북 : [헤더 JSON, JPEG 사진]       헤더 = {frame_id, t_capture, scale}
  노트북 -> 파이 : [결과 JSON]                  결과 = {frame_id, t_capture, dets, rails, proc_ms}

LinkClient는 '배경 스레드'에서 돕니다. 파이의 메인 루프는 절대 기다리지 않고:
  - submit(): 최신 사진을 맡김 (아직 보내는 중이면 덮어써서 항상 최신 사진만 감)
  - poll()  : 새로 도착한 답이 있으면 꺼내 감
  - age()   : 마지막 답이 온 뒤 몇 초 지났나 -> 0.3초 넘으면 예측, 1초 넘으면 자체 검출
답이 LINK_REQ_TIMEOUT_S 안에 안 오면 소켓을 새로 만들어 다시 연결합니다 (Lazy Pirate 패턴).
"""
import json
import threading
import time

import cv2

from .config import cfg


class LinkClient:
    def __init__(self, addr, drop=None):
        import zmq
        self.zmq = zmq
        self.addr, self.ctx, self.sock = addr, zmq.Context.instance(), None
        self.drop = drop or (lambda: False)            # 시험용: True를 돌려주는 동안 '끊긴 척'
        self._lock = threading.Condition()
        self._pending = None                           # (frame_id, t_capture, jpg, scale)
        self._replies = []
        self.t_last_reply = None                       # 마지막 답이 '도착한' 시각 (monotonic)
        self.rtt_ms = None
        self._stop = False
        self._open()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _open(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(self.zmq.REQ)
        self.sock.setsockopt(self.zmq.LINGER, 0)
        self.sock.connect(self.addr)

    # 메인 루프에서 부르는 것 ------------------------------------------------
    def submit(self, frame_id, t_capture, frame):
        h, w = frame.shape[:2]
        scale = 1.0
        if cfg.SEND_WIDTH and w > cfg.SEND_WIDTH:
            scale = cfg.SEND_WIDTH / w
            frame = cv2.resize(frame, (cfg.SEND_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])[1].tobytes()
        with self._lock:
            self._pending = (frame_id, t_capture, jpg, scale)
            self._lock.notify()

    def poll(self):
        with self._lock:
            out, self._replies = self._replies, []
        return out

    def age(self, now):
        """마지막 답 이후 경과 시간 (한 번도 못 받았으면 무한대)."""
        return float("inf") if self.t_last_reply is None else now - self.t_last_reply

    def close(self):
        self._stop = True
        with self._lock:
            self._lock.notify()
        self.th.join(timeout=2)
        self.sock.close(linger=0)

    # 배경 스레드 ------------------------------------------------------------
    def _run(self):
        while not self._stop:
            with self._lock:
                while self._pending is None and not self._stop:
                    self._lock.wait(0.1)
                if self._stop:
                    return
                fid, t_cap, jpg, scale = self._pending
                self._pending = None
            if self.drop():
                time.sleep(0.02)
                continue
            head = json.dumps({"frame_id": fid, "t_capture": t_cap, "scale": scale}).encode()
            t0 = time.monotonic()
            self.sock.send_multipart([head, jpg])
            if self.sock.poll(int(cfg.LINK_REQ_TIMEOUT_S * 1000)):
                reply = json.loads(self.sock.recv())
                now = time.monotonic()
                if self.drop():                        # 기다리는 사이 '끊김'이 시작됐으면 버림
                    continue
                reply["scale"] = scale
                reply["rtt_ms"] = (now - t0) * 1000
                with self._lock:
                    self._replies.append(reply)
                    self.t_last_reply = now
                    self.rtt_ms = reply["rtt_ms"]
            else:
                self._open()                           # 답 없음 -> 다시 연결


def serve(port, handle, on_frame=None, stop=None):
    """노트북 쪽 서버 루프. handle(frame) -> (dets, rails_dict).  stop(): True면 종료."""
    import numpy as np
    import zmq
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{port}")
    try:
        while not (stop and stop()):
            if not sock.poll(200):
                continue
            head, jpg = sock.recv_multipart()
            t0 = time.perf_counter()
            head = json.loads(head)
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            dets, rails = handle(frame)
            reply = {"frame_id": head["frame_id"], "t_capture": head["t_capture"],
                     "dets": dets, "rails": rails,
                     "proc_ms": round((time.perf_counter() - t0) * 1000, 1)}
            sock.send(json.dumps(reply).encode())
            if on_frame:
                on_frame(frame, dets, rails, reply)
    finally:
        sock.close(linger=0)
