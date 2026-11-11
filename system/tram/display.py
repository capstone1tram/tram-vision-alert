"""화면 그리기 (파이 모니터 / 노트북 창 / 저장 영상 공통).

OpenCV 글자는 한글이 안 나와서 화면 글자는 영어로 씁니다.
"""
import cv2
import numpy as np

from .config import LEVEL_COLORS, LEVEL_NAMES, cfg

MODE_TXT = {"ONLINE": "LINK OK", "PREDICT": "LINK LOST - PREDICT", "LOCAL": "PI DETECT (DEGRADED)",
            "ALONE": "PI ONLY"}


def _poly(rails, half_m, ys):
    pl, pr = [], []
    for y in ys:
        c = rails.corridor_x(y, half_m)
        if c is None:
            continue
        pl.append((c[0], y))
        pr.append((c[1], y))
    return np.array(pl + pr[::-1], np.int32)


def draw_rails(img, rails, widen_m=0.0):
    """레일 중심선, 위험 구역(빨강 반투명), 완충 구역(노랑 선)."""
    if rails is None:
        return img
    ys = np.arange(rails.y_top, rails.y_bottom + 1, 4)
    overlay = img.copy()
    poly = _poly(rails, cfg.CORRIDOR_HALF_M + widen_m, ys)
    if len(poly) >= 4:
        cv2.fillPoly(overlay, [poly], (0, 0, 255))
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
    buf = _poly(rails, cfg.CORRIDOR_HALF_M + cfg.BUFFER_M, ys)
    if len(buf) >= 4:
        cv2.polylines(img, [buf], True, (0, 220, 255), 1, cv2.LINE_AA)
    center = np.array([(rails.center_x(y), y) for y in ys], np.int32)
    cv2.polylines(img, [center], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ground_to_image(rails, X, Z, h):
    """바닥 좌표 (X, Z) -> 화면 (u, v).  궤간 픽셀 폭이 f·G/Z 인 줄을 찾음."""
    if rails is None or Z <= 0:
        return None
    ys = np.arange(0, h)
    g = rails.r(ys) - rails.l(ys)
    target = cfg.focal_px * cfg.GAUGE_M / Z
    k = int(np.argmin(np.abs(g - target)))
    if g[k] <= 1:
        return None
    v = ys[k]
    return int(rails.center_x(v) + X * g[k] / cfg.GAUGE_M), int(v)


def _fmt_t(x):
    return f"{x:.1f}s" if x < 99 else "-"


def draw(img, res, fps=None, rtt_ms=None, age=None):
    out = img.copy()
    h, w = out.shape[:2]
    mode = res["mode"]
    widen = 0.0
    if mode == "PREDICT" and res["worst"] is not None:
        widen = cfg.WIDEN_SIGMA * res["worst"]["state"]["sigma_X"]
    draw_rails(out, res["rails"], widen)

    # 검출 박스 (노트북/파이가 마지막으로 본 것)
    for d in res["boxes"]:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # 칼만 트랙 (지금 시각까지 예측한 위치)
    for o in res["objs"]:
        s, m, lv = o["state"], o["m"], o["level"]
        p = ground_to_image(res["rails"], s["X"], s["Z"], h)
        if p is None:
            continue
        col = LEVEL_COLORS[lv]
        r = 7 if mode != "PREDICT" else 11
        cv2.circle(out, p, r, col, 2 if mode != "PREDICT" else 1, cv2.LINE_AA)
        txt = f"#{o['id']} {s['name']} Z{s['Z']:.1f} X{s['X']:+.1f} TTC {_fmt_t(m['ttc'])}"
        cv2.putText(out, txt, (p[0] + 10, max(p[1] - 10, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # 위 띠: 등급 + 모드
    lv = res["level"]
    col = LEVEL_COLORS[lv]
    cv2.rectangle(out, (0, 0), (w, 54), col, -1)
    title = LEVEL_NAMES[lv]
    if res["worst"] is not None and lv >= 0:
        wm = res["worst"]["m"]
        title += f"   {res['worst']['state']['name']} {wm['Z']:.1f}m  TTC {_fmt_t(wm['ttc'])}"
    if res["no_rail"]:
        title += "   NO RAIL"
    cv2.putText(out, title, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    if lv == 3:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, 14)

    # 아래 띠: 모드, 통신, FPS
    info = MODE_TXT[mode]
    if rtt_ms is not None and mode == "ONLINE":
        info += f"  RTT {rtt_ms:.0f}ms"
    if age is not None and mode in ("PREDICT", "LOCAL") and age < 1e6:
        info += f"  lost {age:.1f}s"
    if fps:
        info += f"  {fps:.1f} FPS"
    bg = (40, 40, 40) if mode in ("ONLINE", "ALONE") else (0, 90, 170)
    cv2.rectangle(out, (0, h - 34), (w, h), bg, -1)
    cv2.putText(out, info, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
