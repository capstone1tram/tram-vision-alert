"""선로 사진을 한 장씩 (기본 3초) 보여 주며 선로 인식 결과를 눈으로 확인하는 뷰어.

왼쪽 = 원본 사진, 오른쪽 = 인식 결과
  빨간 선  : 찾은 '내 선로' 두 레일
  빨간 면  : 위험 구역 (레일 가운데에서 양옆 CORRIDOR_HALF_M)
  주황 면  : 그 바깥 완충 구역 (BUFFER_M)
  초록 점  : 정답 (--gt 를 켰을 때, RailSem19 라벨에서 뽑은 내 선로)
  위 글자  : OK(정답과 궤간 폭 20% 안) / WRONG / NOT FOUND

키: 스페이스 = 멈춤/다시, d 또는 → = 다음, a 또는 ← = 이전, q 또는 Esc = 끝

사용 예 (G:\\tram_ 에서, 가상환경 켠 뒤):
  python tools\\rail_viewer.py --data G:\\rail_pic
  python tools\\rail_viewer.py --data G:\\rail_pic --sec 3 --method tophat --gt --shuffle
  python tools\\rail_viewer.py --data G:\\rail_pic --method label         # seg 모델이 완벽하면 이렇게 보임
  python tools\\rail_viewer.py --images 내사진폴더                          # 라벨 없는 내 사진(정답 비교 없음)
  python tools\\rail_viewer.py --data G:\\rail_pic --yolo --shuffle        # 물체(사람·차 등)도 찾아서 표시
  python tools\\rail_viewer.py --data G:\\rail_pic --yolo --n 50           # 50장만 (고르게 뽑음, --shuffle이면 무작위 50장)
  python tools\\rail_viewer.py                                            # 아무것도 안 넣으면(VS Code ▷ 버튼) 터미널에서 물어봄

--yolo 를 켜면 오른쪽 그림에 YOLO11n이 찾은 물체 박스가 같이 나옵니다 (사람·자전거·차·오토바이·버스·트럭).
  박스 색 : 빨강 = 위험 구역 안, 주황 = 완충 구역, 초록 = 구역 밖, 흰색 = 선로를 못 찾아 판단 불가
  글자    : 종류 확신도  Z(앞 거리, 대략)  X(선로 가운데에서 옆 거리, +는 오른쪽)
  사진 한 장이라 속도를 몰라서 TTC·4단계 등급 대신 '어느 구역에 있나'까지만 봅니다.
  옆 거리 X와 구역 판정은 궤간 비율로 계산해서 카메라와 상관없지만, 앞 거리 Z는 초점거리가 필요해
  카메라 모듈 3(화각 66도)을 가정한 값이라 RailSem19 사진(카메라 제각각)에서는 대략값입니다.
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tram.config import cfg                                                    # noqa: E402
from tram.distance import obj_distance                                         # noqa: E402
from tram.rails_photo import (find_rails_ego, gt_ego_edges, label_rail_mask,  # noqa: E402
                              score, tophat_mask)

WORK_W = 960
OK_ERR = 0.2


def zone_polygon(model, half_m):
    ys = np.arange(model.y_top, model.y_bottom + 1)
    left, right = [], []
    for y in ys:
        c = model.corridor_x(y, half_m)
        if c:
            left.append((c[0], y))
            right.append((c[1], y))
    return np.array(left + right[::-1], np.int32) if len(left) > 2 else None


def draw_result(img, model, gt, show_gt, err, method, has_label):
    v = img.copy()
    if model is not None:
        over = v.copy()
        p = zone_polygon(model, cfg.CORRIDOR_HALF_M + cfg.BUFFER_M)
        if p is not None:
            cv2.fillPoly(over, [p], (0, 165, 255))
        p = zone_polygon(model, cfg.CORRIDOR_HALF_M)
        if p is not None:
            cv2.fillPoly(over, [p], (0, 0, 255))
        v = cv2.addWeighted(over, 0.3, v, 0.7, 0)
        ys = np.arange(model.y_top, model.y_bottom + 1)
        for f in (model.left_x, model.right_x):
            cv2.polylines(v, [np.array([(f(y), y) for y in ys], np.int32)], False, (0, 0, 255), 3)
        cx = [(model.center_x(y), y) for y in ys]
        cv2.polylines(v, [np.array(cx, np.int32)], False, (255, 255, 255), 1)
    if show_gt:
        for y, (l, r) in gt.items():
            if y % 4 == 0:
                cv2.circle(v, (int(l), y), 2, (0, 255, 0), -1)
                cv2.circle(v, (int(r), y), 2, (0, 255, 0), -1)
    if model is None:
        state, color = "NOT FOUND", (0, 0, 255)
    elif not has_label or err is None:
        state, color = "FOUND", (255, 255, 255)
    elif err <= OK_ERR:
        state, color = f"OK  err={err:.2f}", (0, 220, 0)
    else:
        state, color = f"WRONG  err={err:.2f}", (0, 0, 255)
    cv2.rectangle(v, (0, 0), (v.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(v, f"[{method}] {state}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return v


ZONE_COLOR = {"IN": (0, 0, 255), "BUFFER": (0, 165, 255), "OUT": (0, 200, 0), "?": (255, 255, 255)}


def classify(dets, model):
    """검출 박스마다 앞 거리 Z, 옆 거리 X, 구역(IN / BUFFER / OUT / ?)."""
    out = []
    for d in dets:
        z, x, how = obj_distance(d["box"], model, d["name"])
        if x is None:
            zone = "?"
        elif abs(x) <= cfg.CORRIDOR_HALF_M:
            zone = "IN"
        elif abs(x) <= cfg.CORRIDOR_HALF_M + cfg.BUFFER_M:
            zone = "BUFFER"
        else:
            zone = "OUT"
        out.append(dict(d, z=z, x=x, how=how, zone=zone))
    return out


def draw_objects(v, objs):
    for o in objs:
        x1, y1, x2, y2 = o["box"]
        c = ZONE_COLOR[o["zone"]]
        cv2.rectangle(v, (x1, y1), (x2, y2), c, 2)
        txt = f"{o['name']} {o['conf']:.2f}"
        if o["z"] is not None:
            txt += f"  Z~{o['z']:.0f}m X{o['x']:+.1f}m"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, 40 + th)
        cv2.rectangle(v, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(v, txt, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    n_in = sum(o["zone"] == "IN" for o in objs)
    n_buf = sum(o["zone"] == "BUFFER" for o in objs)
    txt = f"objects {len(objs)}  in zone {n_in}  buffer {n_buf}"
    cv2.putText(v, txt, (v.shape[1] - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if n_in else (255, 255, 255), 2)
    return v


def find_folder(d):
    """폴더 찾기. 그대로 -> 코드가 있는 드라이브의 맨 위(G:\\이름) -> 코드 폴더 옆 순서로 찾아봄."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 코드 폴더 (예: G:\\tram_)
    drive = os.path.splitdrive(here)[0]
    for c in (d, os.path.join(drive + os.sep, d), os.path.join(os.path.dirname(here), d)):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return None


def ask(a):
    """인자 없이 실행했을 때(VS Code ▷ 버튼 등) 터미널에서 물어보기. Enter = 괄호 안 기본값."""
    print("실행 설정을 물어볼게요. 그냥 Enter를 누르면 [ ] 안의 값을 씁니다.")
    while True:
        d = input("사진 폴더 [G:\\rail_pic]: ").strip().strip('"').strip("'") or r"G:\rail_pic"
        d = find_folder(d)
        if d:
            break
        print("  그 폴더를 못 찾았어요. 예: G:\\rail_pic 처럼 드라이브부터 넣거나, rail_pic 처럼 이름만 넣어 보세요.")
    print(f"  -> {d}")
    if os.path.isdir(os.path.join(d, "jpgs")) or os.path.basename(d.rstrip("\\/")) == "rs19_val":
        a.data = d
    else:
        a.images = d                                   # RailSem19 구조가 아니면 그냥 사진 폴더로
    n = input("몇 장 볼까요? 0 = 전부 [50]: ").strip()
    a.n = int(n) if n.isdigit() else 50
    a.shuffle = input("무작위 순서로 볼까요? y/n [y]: ").strip().lower() != "n"
    a.yolo = input("물체(사람·차)도 찾을까요? y/n [y]: ").strip().lower() != "n"
    if a.data:
        a.gt = input("정답 선로(초록 점)도 보일까요? y/n [y]: ").strip().lower() != "n"
    s = input("한 장에 몇 초? [3]: ").strip()
    a.sec = float(s) if s.replace(".", "", 1).isdigit() else 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="RailSem19 폴더 (jpgs, uint8 이 들어 있는 곳)")
    ap.add_argument("--images", help="라벨 없는 사진 폴더 (jpg/png)")
    ap.add_argument("--sec", type=float, default=3.0, help="한 장을 보여 주는 시간(초)")
    ap.add_argument("--method", default="tophat", choices=["tophat", "label"],
                    help="tophat = 학습 없는 OpenCV, label = 정답 레일 마스크 사용")
    ap.add_argument("--gt", action="store_true", help="정답 레일(초록 점)도 표시")
    ap.add_argument("--shuffle", action="store_true", help="무작위 순서")
    ap.add_argument("--start", default=None, help="이 이름부터 시작 (예: rs04000)")
    ap.add_argument("--width", type=int, default=1600, help="창 전체 폭(px)")
    ap.add_argument("--yolo", action="store_true", help="YOLO로 물체(사람·차 등)도 찾아서 표시")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="YOLO 모델 (yolo11n.pt / yolo11n_openvino_model 등, 처음엔 자동 내려받음)")
    ap.add_argument("--conf", type=float, default=0.3, help="YOLO 확신도 기준 (낮추면 더 많이 찾음)")
    ap.add_argument("--n", type=int, default=0, help="볼 사진 수 (0 = 전부). 고르게 뽑고, --shuffle이면 무작위")
    a = ap.parse_args()
    if not a.data and not a.images:
        ask(a)

    det = None
    if a.yolo:
        cfg.IMG_W, cfg.IMG_H = WORK_W, WORK_W * 9 // 16      # 거리 계산 초점거리를 처리 크기에 맞춤
        from tram.detectors import YoloDetector
        print("YOLO 불러오는 중 ... (처음이면 yolo11n.pt 내려받기)")
        det = YoloDetector(a.model, conf=a.conf)
    obj_stats = {"photos": 0, "objects": 0, "in": 0}

    for k in ("data", "images"):                    # 이름만 넣어도(rail_pic) 찾아 줌
        v = getattr(a, k)
        if v:
            f = find_folder(v.strip('"'))
            if not f:
                sys.exit(f"폴더를 찾을 수 없습니다: {v}\n예: G:\\rail_pic 처럼 드라이브부터 전체 경로를 넣어 주세요.")
            setattr(a, k, f)
    if a.images:
        folder, ldir = a.images, None
        names = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if a.method == "label":
            sys.exit("--images 에는 정답 라벨이 없어서 --method tophat 만 됩니다.")
    elif a.data:
        root = a.data.rstrip("\\/")
        # G:\rail_pic 대신 G:\rail_pic\jpgs\rs19_val 을 넣어도 되게
        if os.path.basename(root) == "rs19_val" and os.path.basename(os.path.dirname(root)) == "jpgs":
            root = os.path.dirname(os.path.dirname(root))
        folder = os.path.join(root, "jpgs", "rs19_val")
        ldir = os.path.join(root, "uint8", "rs19_val")
        if not os.path.isdir(folder):
            sys.exit(f"RailSem19 사진 폴더가 없습니다: {folder}\n"
                     "--data 에는 jpgs·uint8 폴더가 들어 있는 곳(예: G:\\rail_pic)을 넣어 주세요.")
        names = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    else:
        sys.exit("--data G:\\rail_pic 또는 --images 폴더 를 넣어 주세요.")
    if not names:
        sys.exit("사진이 없습니다: " + folder)
    if a.shuffle:
        random.shuffle(names)
    if a.n and a.n < len(names):
        step = len(names) / a.n
        names = [names[int(k * step)] for k in range(a.n)]     # 전체에서 고르게 n장
    print(f"사진 {len(names)}장을 봅니다. (스페이스 = 멈춤, a/d = 이전/다음, q = 끝)")
    i = 0
    if a.start:
        i = next((k for k, n in enumerate(names) if n.startswith(a.start)), 0)

    win = "rail viewer  (space=pause  a/d=prev/next  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused = False
    stats = {"ok": 0, "wrong": 0, "none": 0}
    seen = set()
    seen_obj = set()
    while 0 <= i < len(names):
        n = names[i]
        img = cv2.imread(os.path.join(folder, n))
        if img is None:
            i += 1
            continue
        s = WORK_W / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        lab = None
        if ldir:
            lab = cv2.imread(os.path.join(ldir, os.path.splitext(n)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
            if lab is not None:
                lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if a.method == "label":
            if lab is None:
                i += 1
                continue
            model = find_rails_ego(label_rail_mask(lab))
        else:
            model = find_rails_ego(tophat_mask(img))
        gt = gt_ego_edges(lab) if lab is not None else {}
        err = score(model, gt, img.shape[0]) if gt else None

        if n not in seen:
            seen.add(n)
            if model is None:
                stats["none"] += 1
            elif err is not None:
                stats["ok" if err <= OK_ERR else "wrong"] += 1

        left = img.copy()
        cv2.rectangle(left, (0, 0), (left.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(left, f"{n}   ({i + 1}/{len(names)})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)
        right = draw_result(img, model, gt, a.gt, err, a.method, lab is not None)
        if det is not None:
            objs = classify(det(img), model)
            right = draw_objects(right, objs)
            if n not in seen_obj:
                seen_obj.add(n)
                obj_stats["photos"] += 1
                obj_stats["objects"] += len(objs)
                obj_stats["in"] += sum(o["zone"] == "IN" for o in objs)
        both = np.hstack([left, right])
        tot = stats["ok"] + stats["wrong"] + stats["none"]
        if tot:
            txt = f"so far: OK {stats['ok']}  WRONG {stats['wrong']}  NOT FOUND {stats['none']}"
            cv2.putText(both, txt, (10, both.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
        sc = a.width / both.shape[1]
        cv2.imshow(win, cv2.resize(both, None, fx=sc, fy=sc))

        k = cv2.waitKeyEx(0 if paused else int(a.sec * 1000))
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
            continue
        if k in (ord("a"), 2424832, 65361):        # a / ← (Windows, Linux)
            i = max(0, i - 1)
            continue
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        i += 1                                     # 시간이 지나거나 d / → / 다른 키
    cv2.destroyAllWindows()
    tot = stats["ok"] + stats["wrong"] + stats["none"]
    if tot:
        print(f"본 사진 {tot}장: 맞음 {stats['ok']}, 틀림 {stats['wrong']}, 못 찾음 {stats['none']}")
    if obj_stats["photos"]:
        print(f"물체: 사진 {obj_stats['photos']}장에서 {obj_stats['objects']}개, 그중 위험 구역 안 {obj_stats['in']}개")


if __name__ == "__main__":
    main()
