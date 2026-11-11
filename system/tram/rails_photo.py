"""실제 선로 사진(RailSem19 등)용 선로 검출 실험 모듈.

rails.py(노란 테이프)와 같은 RailModel을 돌려주므로, 거리·위험 구역 계산은 그대로 씁니다.
바뀌는 건 두 군데뿐입니다.

1) 레일 마스크 만들기 (사진 -> 레일 픽셀이 흰색인 흑백 그림)
   - label  : RailSem19 정답 라벨(17 rail-raised, 18 rail-embedded). 'seg 모델이 완벽하면' 의 상한선
   - tophat : 학습 없음. 레일 윗면은 주변 자갈·침목보다 밝고 가늘다 -> 흰색 탑햇 + 세로 방향 경계
   - (나중) seg : YOLO11n-seg를 RailSem19로 학습한 결과 마스크

2) 내 선로 고르기 (find_rails_ego)
   rails.py의 find_rails는 '한 줄에서 가장 큰 빈칸'으로 왼쪽/오른쪽을 나눕니다.
   모형처럼 선로가 하나면 맞지만, 실제 사진은 옆 선로·분기기가 같이 보여서 틀립니다.
   그래서 화면 아래에서 출발해, 화면 가운데에 가장 가까운 레일 한 쌍을 잡고
   위로 한 줄씩 올라가며 '바로 아래 줄에서 예측한 위치'에 가장 가까운 레일을 이어 붙입니다.
"""
import cv2
import numpy as np

from .rails import RailModel

RAIL_IDS = (17, 18)         # rail-raised, rail-embedded
TRACK_IDS = (3, 12)         # tram-track, rail-track (두 레일 사이 영역)


# ── 1) 마스크 ─────────────────────────────────────────────────────────
def label_rail_mask(label):
    return np.isin(label, RAIL_IDS).astype(np.uint8) * 255


def tophat_mask(frame, y_start=0.45):
    """학습 없는 레일 마스크: 밝고 가는 세로 줄무늬만 남김."""
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    k = max(5, int(w * 0.02)) | 1                       # 레일 폭보다 넓은 가로 막대
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    vertical = gx > 1.2 * gy                             # 침목(가로 줄)은 버림
    roi = np.zeros_like(g, bool)
    roi[int(h * y_start):] = True
    t = np.percentile(th[roi], 93)
    m = ((th > max(t, 12)) & roi & vertical).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
    m[~roi] = 0
    # 세로로 이어진 조각만 남김 (짧은 점·자갈 잡음 제거)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_HEIGHT] > h * 0.06
    m = (keep[lab]).astype(np.uint8) * 255
    return m


# ── 2) 내 선로 고르기 ─────────────────────────────────────────────────
def _runs(row):
    """한 줄의 흰 픽셀 묶음 -> 각 묶음의 가운데 x 목록."""
    xs = np.flatnonzero(row)
    if len(xs) == 0:
        return np.empty(0)
    cut = np.flatnonzero(np.diff(xs) > 2) + 1
    return np.array([s.mean() for s in np.split(xs, cut)])


def _chains(rows_pts, ys, tol):
    """줄마다의 x 목록 -> 아래에서 위로 이어 붙인 줄기(chain) 목록. 각 줄기 = [(y, x), ...]"""
    chains = [[(ys[0], x)] for x in rows_pts[0]]
    for y, xs in zip(ys[1:], rows_pts[1:]):
        used = set()
        for ch in sorted(chains, key=len, reverse=True):
            if len(xs) == 0:
                break
            if len(ch) >= 2:
                (y1, x1), (y2, x2) = ch[-2], ch[-1]
                p = x2 + (x2 - x1) * (y - y2) / (y2 - y1)
            else:
                p = ch[-1][1]
            i = int(np.argmin(np.abs(xs - p)))
            if i not in used and abs(xs[i] - p) <= tol:
                ch.append((y, xs[i]))
                used.add(i)
    return [c for c in chains if len(c) >= 4]


def _ego_cost(sl, sr, mid, cx, w):
    """내 선로일수록 작은 값.
    평행한 선로들은 소실점이 같고, 카메라가 올라탄 선로는 '가운데 선'이 화면에서 거의 수직이다
    (옆 선로일수록 가운데 선이 소실점 쪽으로 많이 기움). 그래서
    가운데 선 기울기 |(왼쪽+오른쪽)/2| 가 작고, 화면 가운데에 가까운 쌍을 고른다.
    위로 갈수록 폭이 좁아지지 않으면(sl >= sr) 레일 한 쌍이 아님."""
    if sl >= sr:
        return 9.0
    return abs((sl + sr) / 2) + 3.0 * abs(mid - cx) / w


def _slope(ch):
    a = np.array(ch)
    return np.polyfit(a[:, 0], a[:, 1], 1)[0]           # dx/dy


def pick_ego_pair(rows_pts, ys, w, center=0.5, gauge_min=0.08, gauge_max=0.9):
    """화면 아래 띠에서 '내 선로' 두 레일 줄기를 고름.
    기준은 _ego_cost (가운데 선이 수직에 가깝고 화면 가운데에 가까운 쌍)."""
    tol = max(4, w * 0.012)
    chains = _chains(rows_pts, ys, tol)
    if len(chains) < 2:
        return None
    chains.sort(key=lambda c: c[0][1])
    cx = center * w
    # 같은 줄에서 평행한 선로들의 궤간 픽셀 폭은 거의 같고, 선로 '사이' 빈칸은 그보다 넓다.
    # 그래서 이웃한 레일 간격 중 가장 좁은 값(= 궤간 추정)보다 많이 넓은 쌍은 선로 사이로 보고 벌점.
    gaps = [chains[i + 1][0][1] - chains[i][0][1] for i in range(len(chains) - 1)]
    ok_gaps = [g for g in gaps if g >= gauge_min * w]
    g_min = min(ok_gaps) if ok_gaps else None
    best = None
    for i in range(len(chains) - 1):
        a, b = chains[i], chains[i + 1]
        g = b[0][1] - a[0][1]
        if not (gauge_min * w <= g <= gauge_max * w):
            continue
        sa, sb = _slope(a), _slope(b)
        mid = (a[0][1] + b[0][1]) / 2
        cost = _ego_cost(sa, sb, mid, cx, w)
        if g_min and g > 1.4 * g_min and abs(a[0][0] - b[0][0]) < 3:
            cost += 0.5
        if best is None or cost < best[0]:
            best = (cost, a, b)
    return None if best is None or best[0] >= 9.0 else (best[1], best[2])


def find_rails_ego(mask, y_start=0.45, rows=80, deg=2, min_points=8,
                   gauge_min=0.08, gauge_max=0.9, center=0.5, band=0.25):
    """레일 마스크 -> 내 선로(화면 가운데 선로)의 RailModel. 못 찾으면 None.
    1) 화면 아래 band 비율 띠에서 레일 줄기들을 만들고 기울기로 내 선로 한 쌍을 고름
    2) 그 두 줄기를 위로 계속 이어 감 (앞 두 점으로 예측, 가장 가까운 레일 묶음)"""
    h, w = mask.shape
    ys = np.linspace(h - 1, int(h * y_start), rows).astype(int)
    pts = [_runs(mask[y]) for y in ys]
    # 맨 아래에서 레일이 2개 이상 보이는 첫 줄부터 띠 시작
    k0 = next((k for k, p in enumerate(pts) if len(p) >= 2), None)
    if k0 is None:
        return None
    nb = max(6, int(rows * band / (1 - y_start)))
    pair = pick_ego_pair(pts[k0:k0 + nb], ys[k0:k0 + nb], w, center, gauge_min, gauge_max)
    if pair is None:
        return None
    L, R = list(pair[0]), list(pair[1])
    # 두 줄기에서 이미 쓴 마지막 줄 다음부터 이어 가기
    miss = 0
    last_y = min(L[-1][0], R[-1][0])
    for y, c in zip(ys, pts):
        if y >= last_y:
            continue
        if len(c) < 2:
            miss += 1
            if miss > rows // 5:
                break
            continue

        def pred(P):
            (y1, x1), (y2, x2) = P[-2], P[-1]
            return x2 + (x2 - x1) * (y - y2) / (y2 - y1)
        pl, pr = pred(L), pred(R)
        g = max(pr - pl, 4)
        tol = 0.25 * g + 3
        il = int(np.argmin(np.abs(c - pl)))
        ir = int(np.argmin(np.abs(c - pr)))
        if il == ir or abs(c[il] - pl) > tol or abs(c[ir] - pr) > tol or c[ir] - c[il] < 3:
            miss += 1
            if miss > rows // 5:
                break
            continue
        miss = 0
        L.append((y, c[il]))
        R.append((y, c[ir]))
    if len(L) < min_points or len(R) < min_points:
        return None
    L, R = np.array(L), np.array(R)
    span = min(np.ptp(L[:, 0]), np.ptp(R[:, 0]))
    d = deg if span > h * 0.15 else 1
    model = RailModel(np.polyfit(L[:, 0], L[:, 1], d), np.polyfit(R[:, 0], R[:, 1], d),
                      max(L[:, 0].min(), R[:, 0].min()), min(L[:, 0].max(), R[:, 0].max()))
    if model.gauge_px(model.y_top) <= 2 or model.gauge_px(model.y_bottom) <= 2:
        return None
    return model


# ── 정답 (RailSem19 라벨에서 내 선로 뽑기) ─────────────────────────────
def _segs(row, min_w=3):
    xs = np.flatnonzero(row)
    if len(xs) == 0:
        return []
    cut = np.flatnonzero(np.diff(xs) > 3) + 1
    return [(int(s[0]), int(s[-1])) for s in np.split(xs, cut) if s[-1] - s[0] >= min_w]


def _trace_between(between, y0, seg, y_end):
    """두 레일 사이 영역 한 묶음을 아래(y0)에서 위로 겹침으로 따라감 -> {y: (l, r)}"""
    out = {y0: (float(seg[0]), float(seg[1]))}
    prev = seg
    for y in range(y0 - 1, y_end, -1):
        segs = _segs(between[y])
        if not segs:
            break

        def ov(s):
            return min(s[1], prev[1]) - max(s[0], prev[0])
        s = max(segs, key=ov)
        if ov(s) <= 0:
            break
        wprev = prev[1] - prev[0]
        if s[1] - s[0] > 1.3 * wprev + 4:      # 분기기에서 옆 선로와 붙음 -> 아래 줄 폭 기준으로 자름
            mid = (prev[0] + prev[1]) / 2
            s = (max(s[0], int(mid - wprev * 0.6)), min(s[1], int(mid + wprev * 0.6)))
        out[y] = (float(s[0]), float(s[1]))
        prev = s
    return out


def gt_ego_edges(label, y_start=0.45, center=0.5):
    """RailSem19 라벨 -> 내 선로의 {y: (왼쪽 x, 오른쪽 x)} (두 레일 사이 영역의 양 끝 = 레일 안쪽 면).
    화면 맨 아래 쪽에서 사이 영역 묶음을 모두 따라가 본 뒤,
    find_rails_ego와 같은 기준(_ego_cost)으로 고름."""
    h, w = label.shape
    between = np.isin(label, TRACK_IDS)
    y_end = int(h * y_start)
    y0 = next((y for y in range(h - 1, y_end, -1) if _segs(between[y])), None)
    if y0 is None:
        return {}
    best = None
    for seg in _segs(between[y0]):
        tr = _trace_between(between, y0, seg, y_end)
        if len(tr) < 20:
            continue
        a = np.array([(y, l, r) for y, (l, r) in tr.items()][: max(20, len(tr) // 3)])
        sl = np.polyfit(a[:, 0], a[:, 1], 1)[0]
        sr = np.polyfit(a[:, 0], a[:, 2], 1)[0]
        mid = (seg[0] + seg[1]) / 2
        cost = _ego_cost(sl, sr, mid, center * w, w)
        if best is None or cost < best[0]:
            best = (cost, tr)
    return {} if best is None else best[1]


def score(model, gt, h, y_lo=0.6, y_hi=0.97):
    """화면 아래쪽(y_lo~y_hi) 줄에서 예측 레일과 정답의 차이 / 정답 궤간 폭 (평균).
    None = 비교할 줄이 없음. 0.2 이하면 '내 선로를 맞게 찾음'으로 봄."""
    if model is None or not gt:
        return None
    errs = []
    for y in range(int(h * y_lo), int(h * y_hi), 4):
        if y not in gt:
            continue
        gl, gr = gt[y]
        gw = gr - gl
        if gw < 8:
            continue
        el = abs(model.left_x(y) - gl)
        er = abs(model.right_x(y) - gr)
        errs.append((el + er) / 2 / gw)
    return float(np.mean(errs)) if len(errs) >= 5 else None
