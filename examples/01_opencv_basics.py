"""01. OpenCV 기초: 영상 읽기 -> 처리 -> 그리기 -> 보여주기/저장

이것만 익히면 나머지 예제의 '틀'은 다 같습니다.
- cv2.VideoCapture : 카메라(0, 1, ...) 또는 영상 파일 열기
- cv2.cvtColor     : 색 공간 바꾸기 (BGR -> 회색, HSV)
- cv2.Canny        : 엣지(경계선) 찾기
- cv2.HoughLinesP  : 엣지에서 직선 찾기 (레일 같은 긴 선)
- cv2.line / rectangle / putText : 그리기
- cv2.imshow / VideoWriter : 화면 표시 / 저장

실행 예:
  python 01_opencv_basics.py --source sample.mp4             # 창으로 보기 (q로 종료)
  python 01_opencv_basics.py --source 0                      # 웹캠
  python 01_opencv_basics.py --source sample.mp4 --save out_01.mp4 --no-show
"""
import argparse
import time

import cv2
import numpy as np


def open_source(src):
    """'0' 같은 숫자면 카메라, 아니면 파일 경로."""
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {src}")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample.mp4")
    ap.add_argument("--save", default="")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cap = open_source(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 15
    writer = None
    t_prev = time.time()

    while True:
        ok, frame = cap.read()                 # frame: (높이, 너비, 3) BGR 배열
        if not ok:
            break
        h, w = frame.shape[:2]

        # 1) 회색 -> 흐리게 -> 엣지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2) 화면 아래 절반(선로가 보이는 곳)에서만 직선 찾기
        roi = np.zeros_like(edges)
        roi[h // 2:, :] = 255
        lines = cv2.HoughLinesP(edges & roi, rho=1, theta=np.pi / 180, threshold=40,
                                minLineLength=40, maxLineGap=20)

        # 3) 그리기
        out = frame.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        n_lines = 0 if lines is None else len(lines)
        cv2.putText(out, f"lines: {n_lines}  FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 엣지 화면을 오른쪽 위에 작게 붙이기
        small = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (w // 4, h // 4))
        out[0:h // 4, w - w // 4:w] = small

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(out)
        if not args.no_show:
            cv2.imshow("01 OpenCV basics", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
