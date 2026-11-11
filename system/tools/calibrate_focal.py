"""초점거리(px) 보정: 거리 계산의 정확도를 정하는 숫자 하나를 실제로 재서 맞춤

방법: 선로 위, 카메라에서 거리를 줄자로 잰 지점(예: 1:10 모형이면 2.00 m)에 표시를 하고 사진을 찍습니다.
      이 도구에서 그 표시가 있는 줄을 마우스로 클릭하면  f = Z × (그 줄의 궤간 픽셀 폭) / 궤간  을 계산합니다.
      여러 지점(가까이·멀리)을 찍으면 평균을 냅니다.  결과를 설정 JSON의 "FX_PX"에 적으세요.

  python tools/calibrate_focal.py --image calib.jpg --dist 2.0 --scale 0.1
  python tools/calibrate_focal.py --image calib.jpg --dist 2.0 --row 512     (창 없이, 줄 번호 직접)
  사진 찍기(파이):  rpicam-still -o calib.jpg --width 1280 --height 720   (또는 pi_main.py 화면 저장)
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tram import rails as rails_mod  # noqa: E402
from tram.config import add_common_args, apply_common_args, cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dist", type=float, required=True, help="표시한 지점까지 실제 거리 (m)")
    ap.add_argument("--row", type=int, default=None, help="표시가 있는 화면 줄(y). 없으면 마우스로 클릭")
    add_common_args(ap)
    args = ap.parse_args()
    apply_common_args(args)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"사진을 못 읽음: {args.image}")
    img = cv2.resize(img, (cfg.IMG_W, cfg.IMG_H))
    rails = rails_mod.find_rails(img)
    if rails is None:
        sys.exit("선로를 못 찾음 -> tools/hsv_tuner.py로 노란색 범위부터 맞추세요")

    def report(y):
        g = rails.gauge_px(y)
        f = args.dist * g / cfg.GAUGE_M
        print(f"줄 y={y}: 궤간 {g:.1f} px  ->  초점거리 f = {f:.0f} px  "
              f"(지금 설정 {cfg.focal_px:.0f} px = HFOV {cfg.HFOV_DEG}°)")
        return f

    if args.row is not None:
        f = report(args.row)
        print(f'설정 JSON에:  "FX_PX": {f:.0f}')
        return

    vals = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            vals.append(report(y))
            cv2.line(img, (0, y), (cfg.IMG_W, y), (0, 0, 255), 1)
            cv2.imshow("calibrate", img)

    from tram import display
    display.draw_rails(img, rails)
    cv2.imshow("calibrate", img)
    cv2.setMouseCallback("calibrate", on_mouse)
    print("표시가 있는 줄을 클릭하세요. 끝나면 아무 키나 누르기.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if vals:
        f = sum(vals) / len(vals)
        print(f'평균 f = {f:.0f} px  ->  설정 JSON에:  "FX_PX": {f:.0f}')


if __name__ == "__main__":
    main()
