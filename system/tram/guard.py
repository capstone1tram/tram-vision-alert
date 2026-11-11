"""파이 쪽 판단 엔진.  사진 한 장이 들어올 때마다 step()을 부르면 등급과 모드를 돌려줍니다.

모드 (노트북 마지막 답 이후 시간 = age 기준)
  ONLINE  정상     : age ≤ 0.3초.  노트북이 보낸 박스·선로로 칼만 갱신
  PREDICT 예측     : 0.3초 < age < 1초.  측정 없이 칼만 예측만. 불확실성 2σ만큼 '더 가깝고 선로 쪽'으로
                     보고 판정하며, 등급은 끊기기 직전보다 내리지 않음
  LOCAL   자체 검출 : age ≥ 1초.  파이가 직접 검출(NCNN 또는 Hailo) + 선로 검출, '감시 저하' 표시, 최소 '주의'
  ALONE   단독     : 노트북 없이 파이만으로 (--server none, 예: AI HAT+를 주 검출기로 쓸 때)
노트북 답이 다시 오면 바로 ONLINE으로 돌아갑니다.

시간은 모두 '사진을 찍은 시각' 기준입니다. 노트북 답이 0.1초 늦게 와도 칼만은 찍은 시각으로
갱신하고, 판정은 지금 시각까지 예측한 위치로 하므로 지연만큼 보정됩니다.
"""
from . import distance, rails as rails_mod, risk
from .config import cfg
from .kalman import Tracker

MODE_KO = {"ONLINE": "정상", "PREDICT": "예측", "LOCAL": "자체 검출(감시 저하)", "ALONE": "단독"}


def scale_box(box, scale):
    if scale == 1.0:
        return [int(v) for v in box]
    return [int(round(v / scale)) for v in box]


def to_meas(dets, rails, dist_fn=None):
    """박스 목록 -> 칼만 측정 [(X, Z, 이름)] + 박스마다 계산 결과(화면 표시용).
    dist_fn: 거리 계산 함수 (기본 distance.obj_distance, 실제 영상은 ground.obj_distance)"""
    dist_fn = dist_fn or distance.obj_distance
    meas = []
    for d in dets:
        z, x, method = dist_fn(d["box"], rails, d.get("name", "person"))
        d["Z"], d["X"], d["method"] = z, x, method
        d["meas_k"] = None
        if z is not None and z > 0:
            d["meas_k"] = len(meas)
            meas.append((x, z, d.get("name", "person")))
    return meas


class Guard:
    def __init__(self, local_detector=None, use_link=True, dist_fn=None, visible_fn=None):
        self.local = local_detector
        self.dist_fn = dist_fn
        self.visible_fn = visible_fn          # (상태, 선로) -> 화면 안이면 True. 없으면 안 거름
        self.use_link = use_link
        self.tracker = Tracker()
        self.smoother = risk.LevelSmoother()
        self.rails, self.rails_t = None, None
        self.boxes, self.boxes_src = [], ""
        self.mode = "LOCAL" if use_link else "ALONE"
        self.level_at_loss = 0
        self.shown = 0
        self.n = 0

    # ------------------------------------------------------------------
    def _apply_reply(self, rep):
        scale = rep.get("scale", 1.0)
        t_cap = rep["t_capture"]
        if rep.get("rails"):
            self.rails, self.rails_t = rails_mod.RailModel.from_dict(rep["rails"], scale), t_cap
        dets = [dict(d, box=scale_box(d["box"], scale)) for d in rep["dets"]]
        if self.tracker.update(t_cap, to_meas(dets, self.rails, self.dist_fn), source="laptop"):
            self.boxes, self.boxes_src = dets, "laptop"

    def _local_step(self, t, frame):
        if self.n % 2 == 0 or self.rails is None:      # 선로는 2프레임마다 (가벼움)
            found = rails_mod.find_rails(frame)
            if found is not None:
                self.rails, self.rails_t = found, t
        if self.local is None:
            return
        dets = self.local(frame)
        self.tracker.update(t, to_meas(dets, self.rails, self.dist_fn), source="pi")
        self.boxes, self.boxes_src = dets, "pi"

    # ------------------------------------------------------------------
    def step(self, t, frame, replies=(), age=float("inf")):
        """t: 지금 시각, frame: 지금 사진, replies: 새로 도착한 노트북 답들, age: 마지막 답 이후 시간"""
        self.n += 1
        prev_mode = self.mode
        for rep in sorted(replies, key=lambda r: r["t_capture"]):
            self._apply_reply(rep)

        if not self.use_link:
            mode = "ALONE"
        elif age <= cfg.LINK_PREDICT_S:
            mode = "ONLINE"
        elif age < cfg.LINK_LOCAL_S:
            mode = "PREDICT"
        else:
            mode = "LOCAL"
        if mode == "PREDICT" and prev_mode == "ONLINE":
            self.level_at_loss = self.shown
        if mode in ("LOCAL", "ALONE"):
            self._local_step(t, frame)
        else:
            self.tracker.prune(t)
        self.mode = mode

        # 판정 ------------------------------------------------------------
        widen = mode == "PREDICT"
        objs, raw, worst = [], 0, None
        for tid, s in self.tracker.states_at(t).items():
            if s["hits"] < cfg.CONFIRM_HITS:           # 아직 확정 안 된 트랙은 판단에 안 씀 (헛검출 방지)
                continue
            if self.visible_fn and s["age"] > 0 and not self.visible_fn(s, self.rails):
                continue                               # 예측 위치가 화면 옆·뒤로 나감 = 이미 지나간 물체
            lv, m = risk.assess(s, widen)
            objs.append({"id": tid, "level": lv, "state": s, "m": m})
            if worst is None or lv > worst["level"] or (lv == worst["level"] and m["Z"] < worst["m"]["Z"]):
                worst = objs[-1]
            raw = max(raw, lv)
        if mode == "PREDICT":
            raw = max(raw, self.level_at_loss)         # 예측 중에는 등급을 내리지 않음
        if mode == "LOCAL":
            raw = max(raw, 1)                          # 감시 저하 = 최소 '주의'
        if mode == "LOCAL" and self.local is None:
            raw = -1                                   # 자체 검출기가 없으면 감시 불가

        no_rail = self.rails_t is None or t - self.rails_t > cfg.NO_RAIL_S
        if no_rail and mode != "PREDICT":
            raw = -1
        if raw < 0:
            shown = -1
        else:
            shown = self.smoother.update(raw, t)
        self.shown = shown if shown >= 0 else self.shown
        return {"t": t, "mode": mode, "raw": raw, "level": shown, "objs": objs, "worst": worst,
                "rails": self.rails, "boxes": self.boxes, "boxes_src": self.boxes_src,
                "no_rail": no_rail, "degraded": mode in ("LOCAL",)}
