"""기록(tracks.csv)을 정답 CSV와 비교해서 거리 오차와 경보 시점을 계산

  python tools/evaluate.py logs/20261028_170000_tracks.csv sample_truth.csv

- 같은 영상 시각(video_t)에서 각 정답 물체와 가장 가까운 트랙을 짝지어 앞 거리(Z)·옆 거리(X) 오차를 냄
- 정답 물체가 위험 구역 안에 있었는데 등급이 0(안전)이었던 프레임 수 = 놓침
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tram.config import add_common_args, apply_common_args, cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("truth")
    add_common_args(ap)
    args = ap.parse_args()
    apply_common_args(args)

    truth = defaultdict(list)                       # 프레임 -> [(obj, X, Z)]
    times = {}
    for r in csv.DictReader(open(args.truth, encoding="utf-8-sig")):
        truth[int(r["frame"])].append((r["obj"], float(r["X_m"]), float(r["Z_m"])))
        times[int(r["frame"])] = float(r["t_s"])
    tracks = defaultdict(list)
    for r in csv.DictReader(open(args.tracks, encoding="utf-8-sig")):
        tracks[int(r["frame"])].append(r)

    err = defaultdict(lambda: {"z": [], "x": [], "rel": []})
    miss, inside = 0, 0
    for fr, objs in truth.items():
        trs = tracks.get(fr, [])
        for name, X, Z in objs:
            if not trs:
                continue
            d = [np.hypot(float(t["X_m"]) - X, (float(t["Z_m"]) - Z) * 0.3) for t in trs]
            t = trs[int(np.argmin(d))]
            if min(d) > max(2.0 * cfg.GAUGE_M, 0.15 * Z):
                continue
            err[name]["z"].append(float(t["Z_m"]) - Z)
            err[name]["x"].append(float(t["X_m"]) - X)
            err[name]["rel"].append(abs(float(t["Z_m"]) - Z) / Z)
            if abs(X) <= cfg.CORRIDOR_HALF_M:
                inside += 1
                if int(t["level"]) == 0:
                    miss += 1

    print("물체 | 짝지은 프레임 | 앞 거리 오차 평균(절대) | 상대 오차 | 옆 거리 오차 평균(절대)")
    for name, e in sorted(err.items()):
        z, x, rel = map(np.array, (e["z"], e["x"], e["rel"]))
        print(f"  {name}  | {len(z):5d} | {np.mean(np.abs(z)):.2f} m (치우침 {np.mean(z):+.2f}) "
              f"| {np.mean(rel) * 100:.1f}% | {np.mean(np.abs(x)):.2f} m")
    print(f"위험 구역 안 프레임 {inside}개 중 '안전'으로 본 프레임: {miss}개 (목표 0)")


if __name__ == "__main__":
    main()
