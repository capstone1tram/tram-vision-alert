"""2주차 관문: 파이 <-> 노트북 왕복 시간 측정 (목표 0.3초 = 300 ms 이하)

노트북에서 laptop_server.py 를 켜 둔 상태로 파이에서:
  python tools/latency_test.py --server tcp://192.168.137.1:5555 --n 200
카메라 없이 영상 크기의 가짜 사진을 보내므로 '통신 + 노트북 처리' 시간만 잽니다.
노트북 모바일 핫스팟 / 휴대폰 핫스팟 둘 다 재서 비교하세요.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tram.config import cfg  # noqa: E402
from tram.link import LinkClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="tcp://127.0.0.1:5555")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--send-width", type=int, default=None, help="보낼 사진 폭 (기본 config의 SEND_WIDTH)")
    args = ap.parse_args()
    if args.send_width is not None:
        cfg.SEND_WIDTH = args.send_width

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)
    frame[:] = cv2_blur(frame)
    link = LinkClient(args.server)
    rtts, lost = [], 0
    for i in range(args.n):
        link.submit(i, time.monotonic(), frame)
        t0 = time.monotonic()
        while True:
            reps = link.poll()
            if reps:
                rtts.append(reps[-1]["rtt_ms"])
                break
            if time.monotonic() - t0 > cfg.LINK_REQ_TIMEOUT_S + 0.2:
                lost += 1
                break
            time.sleep(0.002)
    link.close()
    if not rtts:
        print("답이 하나도 없음: 노트북 서버가 켜져 있는지, IP·방화벽(포트 5555)을 확인하세요.")
        return
    r = np.array(rtts)
    print(f"보낸 사진 {args.n}장, 답 {len(r)}장, 시간 초과 {lost}장")
    print(f"왕복 시간(ms): 중앙값 {np.median(r):.0f}, 평균 {r.mean():.0f}, 95% {np.percentile(r, 95):.0f}, 최대 {r.max():.0f}")
    print("판정:", "통과 (0.3초 이하)" if np.percentile(r, 95) <= 300 else
          "불합격 -> SEND_WIDTH·JPEG_QUALITY 낮추기, 5 GHz 핫스팟, 또는 AI HAT+ 단독(--server none) 고려")


def cv2_blur(img):
    import cv2
    return cv2.GaussianBlur(img, (0, 0), 3)        # 실제 사진처럼 JPEG 크기가 나오도록 잡음을 뭉갬


if __name__ == "__main__":
    main()
