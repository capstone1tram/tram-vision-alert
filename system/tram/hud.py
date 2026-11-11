"""PC 통합 프로그램(pc_main.py) 화면.  display.py의 선로·위험 구역 그리기를 그대로 쓰고,
실제 영상에 맞게 박스마다 등급 색·거리·TTC를 붙입니다.  (OpenCV 글자라 영어로 표시)

  빨간 면   : 위험 구역 (트램이 지나갈 자리 = 중심선 ± 차체 폭/2 + 여유)
  노란 선   : 완충 구역 바깥 끝
  흰 선     : 선로 중심선 (점선 = 레일을 찾은 범위 위로 연장한 부분)
  박스 색   : 그 물체의 등급 (초록 안전 / 노랑 주의 / 주황 경고 / 빨강 위험 / 회색 판단 전)
  가로 눈금 : 10 m, 20 m, 30 m ... (초점거리 가정값 기준)
"""
import cv2
import numpy as np

from . import ground
from .config import LEVEL_COLORS, LEVEL_NAMES, cfg
from .display import draw_rails

GRAY = (170, 170, 170)


def _t(x):
    return f"{x:.1f}s" if x < 99 else "-"


def _txt(img, s, org, scale=0.5, col=(255, 255, 255), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def draw_ground(img, rails):
    gm = ground.ground_of(rails)
    if gm is None:
        return
    h, w = img.shape[:2]
    # 연장 중심선 (점선)
    y_lo = int(max(gm.y_h + 0.03 * h, 0))
    for y in range(rails.y_top, y_lo, -12):
        p1 = (int(gm.center_x(y)), y)
        p2 = (int(gm.center_x(y - 6)), y - 6)
        cv2.line(img, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
    # 거리 눈금
    for z in (10, 20, 30, 50):
        y = gm.y_of_z(z)
        if not (gm.y_h < y < h):
            continue
        y = int(y)
        g = gm.gauge(y)
        cx = gm.center_x(y)
        half = (cfg.CORRIDOR_HALF_M + cfg.BUFFER_M) * g / cfg.GAUGE_M
        cv2.line(img, (int(cx - half), y), (int(cx + half), y), (255, 255, 255), 1, cv2.LINE_AA)
        _txt(img, f"{z}m", (int(cx + half) + 4, y + 4), 0.4)


def draw(img, res, info):
    """res: Guard.step 결과, info: 화면 아래 띠에 쓸 값들(dict)."""
    out = img.copy()
    h, w = out.shape[:2]
    rails = res["rails"]
    if rails is not None and not res["no_rail"]:
        draw_rails(out, rails)
        draw_ground(out, rails)

    # 박스 <-> 트랙 짝 (이번 프레임 측정 번호로)
    by_k = {}
    for o in res["objs"]:
        k = o["state"].get("meas_k")
        if k is not None:
            by_k[k] = o
    for d in res["boxes"]:
        x1, y1, x2, y2 = d["box"]
        o = by_k.get(d.get("meas_k"))
        if o is None:
            col = GRAY
            label = d["name"] if d.get("Z") is None else f"{d['name']} {d['Z']:.0f}m"
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
            _txt(out, label, (x1, max(y1 - 4, 12)), 0.4, col)
            continue
        col = LEVEL_COLORS[o["level"]]
        m, s = o["m"], o["state"]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if o["level"] >= 2 else 1)
        label = f"#{o['id']} {s['name']} {m['Z']:.1f}m"
        if m["ttc"] < 99:
            label += f" TTC {_t(m['ttc'])}"
        if d.get("method") in ("ground", "height"):
            label += "*"                       # 레일 찾은 범위 밖 (연장 계산)
        _txt(out, label, (x1, max(y1 - 4, 12)), 0.45, col)

    # 위 띠: 등급
    lv = res["level"]
    col = LEVEL_COLORS[lv]
    cv2.rectangle(out, (0, 0), (w, 40), col, -1)
    title = LEVEL_NAMES[lv]
    wo = res["worst"]
    if wo is not None and lv >= 1:
        wm = wo["m"]
        title += f"  #{wo['id']} {wo['state']['name']} {wm['Z']:.1f}m"
        if wm["ttc"] < 99:
            title += f"  TTC {_t(wm['ttc'])}"
        if wm["a_req"] > 0.05:
            title += f"  a_req {min(wm['a_req'], 9.9):.1f}"
    if res["no_rail"]:
        title += "   NO RAIL"
    cv2.putText(out, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    if lv == 3:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, 10)

    # 아래 띠
    cv2.rectangle(out, (0, h - 26), (w, h), (35, 35, 35), -1)
    _txt(out, info.get("bar", ""), (8, h - 8), 0.5)
    if info.get("rec"):
        cv2.circle(out, (w - 18, 58), 8, (0, 0, 255), -1)
        _txt(out, "REC", (w - 58, 63), 0.45, (0, 0, 255))
    return out


def paused_text(img):
    _txt(img, "PAUSED  (space: play, d/a: +-10s, s: capture, c: calibrate, m: mask, q: quit)",
         (8, 62), 0.5, (0, 255, 255))


def mask_view(img, mask):
    """선로 마스크 확인용: 레일 후보 픽셀을 초록으로 겹쳐 그림 (m 키)."""
    if mask is None:
        return img
    out = img.copy()
    out[mask > 0] = (0, 255, 0)
    return out
