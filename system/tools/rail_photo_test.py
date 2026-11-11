"""실제 선로 사진(RailSem19)으로 선로 인식 실험.

방법 4가지를 같은 사진에 돌리고, 정답 라벨과 비교해 '내 선로를 맞게 찾았는지' 채점합니다.
  hsv       : 지금 rails.py 그대로 (노란 테이프 색). 실제 사진에선 안 되는 게 정상 -> 기준선
  tophat    : 학습 없는 OpenCV (밝고 가는 세로 줄 = 레일) + 내 선로 고르기
  label_old : 정답 레일 마스크 + rails.py의 '가장 큰 빈칸' 나누기 (모델이 완벽해도 고르는 단계가 틀리는지)
  label_ego : 정답 레일 마스크 + 새 '내 선로 고르기' (seg 모델이 완벽할 때의 상한)

사용 예 (G:\\tram_ 에서, 가상환경 켠 뒤):
  python tools\\rail_photo_test.py --data G:\\rail_pic --n 200
  python tools\\rail_photo_test.py --data G:\\rail_pic --names rs00000 rs04000 --save-all
결과: rail_test_out\\ 에 결과.csv, 요약.txt, 사진(성공/실패 예시)
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tram.rails import find_rails                        # noqa: E402
from tram.rails_photo import (find_rails_ego, gt_ego_edges, label_rail_mask,  # noqa: E402
                              score, tophat_mask)

METHODS = ["hsv", "tophat", "label_old", "label_ego"]
OK_ERR = 0.2          # 궤간 폭의 20% 이내면 성공
WORK_W = 960          # 처리 폭 (1920 -> 960)


def draw(frame, model, gt, title, err):
    v = frame.copy()
    h, w = v.shape[:2]
    for y, (l, r) in gt.items():
        if y % 3 == 0:
            cv2.circle(v, (int(l), y), 1, (0, 255, 0), -1)
            cv2.circle(v, (int(r), y), 1, (0, 255, 0), -1)
    if model is not None:
        ys = np.arange(model.y_top, model.y_bottom + 1)
        over = v.copy()
        poly = []
        for y in ys:
            c = model.corridor_x(y)
            if c:
                poly.append((c[0], y))
        for y in ys[::-1]:
            c = model.corridor_x(y)
            if c:
                poly.append((c[1], y))
        if len(poly) > 3:
            cv2.fillPoly(over, [np.array(poly, np.int32)], (0, 0, 255))
            v = cv2.addWeighted(over, 0.25, v, 0.75, 0)
        for f in (model.left_x, model.right_x):
            pts = np.array([(f(y), y) for y in ys], np.int32)
            cv2.polylines(v, [pts], False, (0, 0, 255), 2)
    ok = err is not None and err <= OK_ERR
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.rectangle(v, (0, 0), (w, 28), (0, 0, 0), -1)
    state = "not found" if model is None else ("OK" if ok else "WRONG")
    cv2.putText(v, f"{title}  {state}" + ("" if err is None else f"  err={err:.2f}"),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="RailSem19 폴더 (jpgs, uint8 가 들어 있는 곳)")
    ap.add_argument("--n", type=int, default=200, help="고르게 뽑을 사진 수 (0 = 전부)")
    ap.add_argument("--names", nargs="*", help="특정 사진만 (예: rs00000)")
    ap.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    ap.add_argument("--out", default="rail_test_out")
    ap.add_argument("--save-all", action="store_true", help="모든 사진의 결과 그림 저장 (기본은 앞 30장)")
    a = ap.parse_args()

    jdir = os.path.join(a.data, "jpgs", "rs19_val")
    ldir = os.path.join(a.data, "uint8", "rs19_val")
    if a.names:
        names = a.names
    else:
        names = sorted(f[:-4] for f in os.listdir(jdir) if f.endswith(".jpg"))
        if a.n and a.n < len(names):
            idx = np.linspace(0, len(names) - 1, a.n).astype(int)
            names = [names[i] for i in idx]
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)

    rows, ms = [], {m: [] for m in a.methods}
    t0 = time.time()
    for k, n in enumerate(names):
        img = cv2.imread(os.path.join(jdir, n + ".jpg"))
        lab = cv2.imread(os.path.join(ldir, n + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or lab is None:
            print("건너뜀:", n)
            continue
        s = WORK_W / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        h = img.shape[0]
        gt = gt_ego_edges(lab)
        row = {"name": n, "gt_rows": len(gt)}
        pics = []
        for m in a.methods:
            t = time.perf_counter()
            if m == "hsv":
                model = find_rails(img)
            elif m == "tophat":
                model = find_rails_ego(tophat_mask(img))
            elif m == "label_old":
                model = find_rails(mask=label_rail_mask(lab))
            else:
                model = find_rails_ego(label_rail_mask(lab))
            ms[m].append((time.perf_counter() - t) * 1000)
            err = score(model, gt, h)
            row[m + "_found"] = int(model is not None)
            row[m + "_err"] = "" if err is None else round(err, 3)
            row[m + "_ok"] = int(err is not None and err <= OK_ERR)
            pics.append(draw(img, model, gt, m, err))
        rows.append(row)
        if a.save_all or k < 30:
            pair = [np.hstack(pics[i:i + 2]) if i + 1 < len(pics) else
                    np.hstack([pics[i], np.zeros_like(pics[i])]) for i in range(0, len(pics), 2)]
            cv2.imwrite(os.path.join(a.out, "img", n + ".jpg"),
                        cv2.resize(np.vstack(pair), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(names)}  ({time.time() - t0:.0f}초)")

    with open(os.path.join(a.out, "결과.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    valid = [r for r in rows if r["gt_rows"] > 20]
    lines = [f"사진 {len(rows)}장 중 정답 선로가 있는 사진 {len(valid)}장 기준 (성공 = 궤간 폭의 {OK_ERR:.0%} 이내)", ""]
    for m in a.methods:
        found = sum(r[m + "_found"] for r in valid)
        ok = sum(r[m + "_ok"] for r in valid)
        errs = [r[m + "_err"] for r in valid if r[m + "_ok"]]
        lines.append(f"{m:10s} 찾음 {found:4d}  맞음 {ok:4d} ({ok / max(len(valid), 1):5.1%})  "
                     f"맞은 것 평균 오차 {np.mean(errs) if errs else float('nan'):.3f}  "
                     f"처리 {np.mean(ms[m]):.1f} ms/장")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, "요약.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
