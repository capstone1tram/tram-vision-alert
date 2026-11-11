"""모든 모듈이 함께 쓰는 설정값.

숫자는 발표 3·상세 보고서의 '제안 초기값'입니다. 실험하면서 바꾸세요.
- 바꾸는 방법 1: 이 파일을 직접 고치기
- 바꾸는 방법 2: JSON 파일을 만들어 실행할 때 --config my.json  (예: settings_model.json)
- 1:10 모형 시험장: 실행할 때 --scale 0.1  (길이·속도·감속도만 1/10, 시간 기준은 그대로)

다른 모듈은 반드시 `cfg.이름`으로 '쓸 때' 읽습니다. (함수 기본값으로 굳혀 두지 않음)
"""
import json
import math


class Config:
    def __init__(self):
        # ── 카메라 ──────────────────────────────────────────
        self.IMG_W, self.IMG_H = 1280, 720   # 처리할 영상 크기 (px)
        self.HFOV_DEG = 66.0                 # 수평 화각. 카메라 모듈 3 표준 렌즈 66°(Wide는 102°)
        self.FX_PX = None                    # 보정한 초점거리(px). 있으면 HFOV보다 우선 (tools/calibrate_focal.py)
        self.FPS = 15                        # 카메라 촬영 FPS 목표

        # ── 선로·차량 (실제 크기, m) ────────────────────────
        self.GAUGE_M = 1.435                 # 궤간
        self.TRAM_WIDTH_M = 2.65             # 차체 폭 (가정값)
        self.MARGIN_M = 0.5                  # 차체 바깥 여유
        self.BUFFER_M = 1.0                  # 위험 구역 바깥 '주의' 완충 구역 폭

        # ── 위험 판단 ───────────────────────────────────────
        self.TTC_WARN_S = 5.0                # 경고 TTC
        self.TTC_DANGER_S = 3.0              # 위험 TTC
        self.A_SERVICE = 1.2                 # 상용 제동 감속도 (m/s²)
        self.T_REACT_S = 1.0                 # 반응 시간
        self.V_MIN = 0.1                     # 이보다 느리게 다가오면 '다가오지 않음' (m/s)
        self.V_LAT_MIN = 0.05                # 옆으로 이보다 느리면 '안 움직임' (m/s)
        self.PERSON_HEIGHT_M = 1.7           # 선로가 가려졌을 때 쓰는 백업 거리용

        # ── 등급 안정화 ─────────────────────────────────────
        self.RAISE_FRAMES = 2                # 이만큼 연속으로 높으면 올림
        self.LOWER_HOLD_S = 1.0              # 이만큼 계속 낮아야 내림

        # ── 칼만 필터 ───────────────────────────────────────
        self.KF_ACC_SIGMA = 1.0              # 모델에 없는 가속도 크기 (m/s²)
        self.KF_PX_ERR = 1.5                 # 사람 발 위치의 궤간 픽셀 측정 오차 (px)
        self.KF_VEHICLE_PX_MULT = 3.0        # 차량은 박스 밑변이 덜 또렷해서 오차를 이만큼 크게 봄
        self.KF_GATE = 9.21                  # 연결 판정 문턱 (카이제곱 2자유도 99%)
        self.TRACK_MAX_COAST_S = 1.5         # 측정 없이 이만큼 지나면 트랙 삭제
        self.CONFIRM_HITS = 3                # 이만큼 잡혀야 '확정' 트랙 -> 판정에 씀 (한두 번 헛검출 거르기)
        self.TRACK_TENTATIVE_COAST_S = 0.5   # 확정 전 트랙은 이만큼만 기다리고 삭제
        self.WIDEN_SIGMA = 2.0               # 예측 모드에서 통로를 넓히는 정도 (σ 배수)

        # ── 연결 감시 ───────────────────────────────────────
        self.LINK_PREDICT_S = 0.3            # 답이 이만큼 없으면 '예측' 모드
        self.LINK_LOCAL_S = 1.0              # 끊긴 지 이만큼이면 '자체 검출' 모드
        self.LINK_REQ_TIMEOUT_S = 1.0        # 요청 하나를 기다리는 최대 시간 (넘으면 소켓 재연결)
        self.JPEG_QUALITY = 80
        self.SEND_WIDTH = 960                # 노트북으로 보낼 때 줄이는 폭 (0이면 원본)
        self.NO_RAIL_S = 2.0                 # 선로를 이만큼 못 찾으면 '감시 불가'

        # ── 선로 검출 (노란 테이프, HSV) ────────────────────
        self.RAIL_HSV_LO = [20, 80, 120]
        self.RAIL_HSV_HI = [35, 255, 255]
        self.RAIL_ROWS = 25                  # 스캔할 줄 수
        self.RAIL_Y_START = 0.55             # 화면 높이의 이 비율부터 아래만 스캔
        self.RAIL_DEG = 1                    # 1 = 직선, 2 = 곡선 선로
        self.RAIL_MIN_POINTS = 5

        # ── 실제 선로 영상 (블랙박스, pc_main.py) ───────────
        self.PHOTO_Y_START = 0.45            # 화면 높이의 이 비율부터 아래에서 레일을 찾음 (지평선 조금 위)
        self.PHOTO_Y_END = 1.0               # 이 비율 아래는 안 봄 (대시보드·차체가 찍히면 0.9 등으로)
        self.RAIL_CENTER = 0.5               # 내 선로가 화면 가로 어디쯤에서 시작하나 (0.5 = 가운데)
        self.RAIL_SMOOTH = 0.5               # 프레임 사이 선로 부드럽게 (새 값 비율, 1이면 안 섞음)
        self.RAIL_JUMP = 0.5                 # 앞 프레임보다 궤간의 이 비율 넘게 튀면 의심
        self.RAIL_REJECT_MAX = 6             # 이만큼 연속으로 의심되면 새 선로를 그냥 받아들임
        self.GROUND_MAX_Z = 80.0             # 선로 위쪽 연장으로 계산하는 최대 거리 (m)
        self.CLIP_PRE_S = 3.0                # 위험 장면 저장: 등급이 오르기 전 몇 초부터
        self.CLIP_POST_S = 2.0               # 등급이 내려간 뒤 몇 초까지
        self.DROP_OUT_OF_VIEW = False        # True: 측정 없이 예측만 하는 트랙이 화면 밖(옆·뒤)으로 나가면 판단에서 뺌
        self.ENTER_EXIT_CHECK = False        # True: '들어오는 중' 경고는 트램이 올 때 아직 구역 안에 있을 때만 (다 건너가면 경고 안 함)
        self.ENTER_SIGMA = 0.0               # '들어오는 중' 판정: 옆 속도가 V_LAT_MIN + 이 배수 × 옆 속도 오차(σ)보다 커야 함
        self.ENTER_MAX_AGE_S = 1e9           # '들어오는 중' 판정은 마지막 측정이 이 시간 안인 트랙만 (영상용 0.2)
        self.ENTER_MIN_HITS = 0              # '들어오는 중' 판정은 이만큼 측정된 트랙만 (처음 몇 장은 속도가 출렁임, 영상용 10)
        self.EDGE_PX = 3                     # 화면 좌우 끝에 걸린 박스는 잘려서 위치가 틀림 -> 거리 계산 안 함

        # ── 감시 대상 (COCO 번호: 이름) ─────────────────────
        self.TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.DET_CONF = 0.35

        # ── 경보 핀 (BCM 번호, 배선에 맞게) ─────────────────
        self.PIN_BUZZER = 17
        self.PIN_LED_GREEN = 22
        self.PIN_LED_YELLOW = 23
        self.PIN_LED_RED = 24

        self.SCALE = 1.0

    # 계산되는 값 -----------------------------------------------------
    @property
    def CORRIDOR_HALF_M(self):
        """선로 중심선에서 위험 구역 끝까지 (기본 약 1.8 m)."""
        return self.TRAM_WIDTH_M / 2 + self.MARGIN_M

    @property
    def focal_px(self):
        if self.FX_PX:
            return float(self.FX_PX)
        return (self.IMG_W / 2) / math.tan(math.radians(self.HFOV_DEG) / 2)

    # 바꾸기 ----------------------------------------------------------
    _LENGTHS = ["GAUGE_M", "TRAM_WIDTH_M", "MARGIN_M", "BUFFER_M", "A_SERVICE", "V_MIN",
                "V_LAT_MIN", "PERSON_HEIGHT_M", "KF_ACC_SIGMA"]

    def apply_scale(self, s):
        """1:10 모형이면 s = 0.1. 길이(m)·속도(m/s)·감속도(m/s²)를 s배. 시간(초) 기준은 그대로."""
        if s == self.SCALE:
            return
        k = s / self.SCALE
        for name in self._LENGTHS:
            setattr(self, name, getattr(self, name) * k)
        self.SCALE = s

    def load_json(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k.startswith("_"):
                continue                       # "_설명" 같은 메모용 키
            if not hasattr(self, k):
                raise KeyError(f"설정 파일 {path}: 모르는 이름 {k}")
            if k == "SCALE":
                self.apply_scale(float(v))
                continue
            if k == "TARGET_CLASSES":
                v = {int(a): b for a, b in v.items()}
            setattr(self, k, v)

    def summary(self):
        return (f"scale={self.SCALE}  궤간={self.GAUGE_M:.4g}m  위험구역 반폭={self.CORRIDOR_HALF_M:.3g}m  "
                f"완충={self.BUFFER_M:.3g}m  TTC {self.TTC_WARN_S}/{self.TTC_DANGER_S}s  "
                f"감속도 기준={self.A_SERVICE:.3g}m/s²  f={self.focal_px:.0f}px")


cfg = Config()

LEVEL_NAMES = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "DANGER", -1: "NO VIEW"}
LEVEL_NAMES_KO = {0: "안전", 1: "주의", 2: "경고", 3: "위험", -1: "감시 불가"}
LEVEL_COLORS = {0: (0, 200, 0), 1: (0, 220, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (150, 150, 150)}  # BGR


def add_common_args(ap):
    """모든 실행 파일이 같이 쓰는 --config, --scale 옵션."""
    ap.add_argument("--config", default="", help="설정 덮어쓰기 JSON 파일")
    ap.add_argument("--scale", type=float, default=None, help="1:10 모형이면 0.1")


def apply_common_args(args):
    """--scale 먼저, 그다음 JSON (JSON에 적은 값이 최종값)."""
    if args.scale is not None:
        cfg.apply_scale(args.scale)
    if args.config:
        cfg.load_json(args.config)
