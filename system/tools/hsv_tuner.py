"""노란 테이프 색 범위(HSV) 맞추기. 시험장 조명이 바뀌면 다시 하세요.

  python tools/hsv_tuner.py --source 0              (웹캠)
  python tools/hsv_tuner.py --source track.jpg      (사진)
  python tools/hsv_tuner.py --source picam          (파이 카메라)
슬라이더를 움직여 레일만 하얗게 남기고, 'q'를 누르면 설정 JSON에 넣을 값을 출력합니다.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tram import camera, display, rails as rails_mod  # noqa: E402
from tram.config import cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    args = ap.parse_args()

    still = None
    if os.path.splitext(args.source)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        still = cv2.resize(cv2.imread(args.source), (cfg.IMG_W, cfg.IMG_H))
    else:
        gen = camera.frames(args.source, realtime=True, loop=True)

    win = "hsv tuner"
    cv2.namedWindow(win)
    names = ["H lo", "S lo", "V lo", "H hi", "S hi", "V hi"]
    init = cfg.RAIL_HSV_LO + cfg.RAIL_HSV_HI
    for n, v in zip(names, init):
        cv2.createTrackbar(n, win, int(v), 179 if n.startswith("H") else 255, lambda _: None)
    while True:
        frame = still if still is not None else next(gen)[3]
        vals = [cv2.getTrackbarPos(n, win) for n in names]
        cfg.RAIL_HSV_LO, cfg.RAIL_HSV_HI = vals[:3], vals[3:]
        mask = rails_mod.rail_mask(frame)
        rails = rails_mod.find_rails(mask=mask)
        vis = frame.copy()
        if rails is not None:
            display.draw_rails(vis, rails)
        cv2.putText(vis, "RAIL OK" if rails else "NO RAIL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if rails else (0, 0, 255), 2)
        both = np.hstack([cv2.resize(vis, (640, 360)), cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (640, 360))])
        cv2.imshow(win, both)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    print(f'설정 JSON에:  "RAIL_HSV_LO": {cfg.RAIL_HSV_LO}, "RAIL_HSV_HI": {cfg.RAIL_HSV_HI}')


if __name__ == "__main__":
    main()
