"""03. 선로 검출 (모형 시험장의 노란 테이프 선로용) + 위험 통로 그리기

RailSafeNet과 같은 '줄 단위 스캔' 방식입니다.
 1) 레일 색만 남긴 마스크를 만든다 (HSV 색 범위)
 2) 화면 아래쪽 여러 줄(y)에서 레일 픽셀의 x 위치를 찾아 왼쪽/오른쪽 레일 점을 모은다
 3) 각 레일을 x = a*y + b 직선(곡선이면 2차식)으로 맞춘다
 4) 줄마다 궤간 픽셀 폭 g(y) = 오른쪽 x - 왼쪽 x  ->  1 px = 1.435 / g(y) m
 5) 기준선(두 레일 가운데)에서 양쪽으로 CORRIDOR_HALF_M 만큼을 통로로 그린다

실제 도로의 매립형 레일은 색으로 안 잡히므로, 1단계 마스크만 YOLO-seg 결과로 바꾸면
나머지 2~5단계는 그대로 씁니다.

실행 예:
  python 03_rail_detect.py --source sample.mp4 --save out_03.mp4 --no-show
"""
import argparse

import cv2
import numpy as np

import config as C

# 노란 테이프 HSV 범위. 조명에 따라 조정하세요 (H: 0~179)
YELLOW_LO = np.array([20, 80, 120])
YELLOW_HI = np.array([35, 255, 255])


class RailModel:
    """검출된 두 레일. y(화면 줄)를 넣으면 x 위치와 궤간 픽셀 폭을 알려 줌."""

    def __init__(self, left_coef, right_coef, y_top, y_bottom):
        self.l = np.poly1d(left_coef)
        self.r = np.poly1d(right_coef)
        self.y_top, self.y_bottom = y_top, y_bottom

    def left_x(self, y):
        return float(self.l(y))

    def right_x(self, y):
        return float(self.r(y))

    def center_x(self, y):
        return (self.left_x(y) + self.right_x(y)) / 2

    def gauge_px(self, y):
        return self.right_x(y) - self.left_x(y)

    def m_per_px(self, y):
        g = self.gauge_px(y)
        return C.GAUGE_M / g if g > 1 else None

    def corridor_x(self, y, half_m=C.CORRIDOR_HALF_M):
        """줄 y에서 통로 왼쪽/오른쪽 끝 x."""
        half_px = half_m / self.m_per_px(y)
        cx = self.center_x(y)
        return cx - half_px, cx + half_px


def rail_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_rails(mask, n_rows=25, y_start_ratio=0.55, deg=1, min_points=5):
    """마스크 -> RailModel (못 찾으면 None)."""
    h, w = mask.shape
    ys = np.linspace(int(h * y_start_ratio), h - 1, n_rows).astype(int)
    left_pts, right_pts = [], []
    for y in ys:
        xs = np.flatnonzero(mask[y])
        if len(xs) < 2:
            continue
        # 픽셀 묶음 사이 가장 큰 빈칸을 기준으로 왼쪽/오른쪽 레일 나누기
        gaps = np.diff(xs)
        k = int(np.argmax(gaps))
        if gaps[k] < 3:            # 레일이 하나만 보임
            continue
        left_pts.append((y, xs[: k + 1].mean()))
        right_pts.append((y, xs[k + 1:].mean()))
    if len(left_pts) < min_points:
        return None
    L, R = np.array(left_pts), np.array(right_pts)
    return RailModel(np.polyfit(L[:, 0], L[:, 1], deg), np.polyfit(R[:, 0], R[:, 1], deg),
                     int(L[:, 0].min()), int(L[:, 0].max()))


def draw_rails(img, rails, color=(0, 0, 255)):
    """레일, 기준선, 위험 통로(반투명)를 그림."""
    ys = np.arange(rails.y_top, rails.y_bottom + 1, 4)
    poly_l, poly_r = [], []
    for y in ys:
        a, b = rails.corridor_x(y)
        poly_l.append((a, y))
        poly_r.append((b, y))
    poly = np.array(poly_l + poly_r[::-1], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    center = np.array([(rails.center_x(y), y) for y in ys], np.int32)
    cv2.polylines(img, [center], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample.mp4")
    ap.add_argument("--save", default="")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    writer = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rails = find_rails(rail_mask(frame))
        if rails is None:
            cv2.putText(frame, "NO RAIL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        else:
            draw_rails(frame, rails)
            y = rails.y_bottom
            if i == 0:
                print(f"bottom row y={y}: gauge={rails.gauge_px(y):.1f}px, 1px={rails.m_per_px(y)*100:.2f}cm")
        i += 1
        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("03 rails", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
