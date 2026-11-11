"""검출기 모음.  모두 같은 모양으로 씁니다:

    det = make_detector("yolo", model="yolo11n_openvino_model")
    boxes = det(frame)    # frame: OpenCV BGR 사진
    # -> [{"name": "person", "cls": 0, "conf": 0.87, "box": [x1, y1, x2, y2]}, ...]

종류
  color : 빨간 물체를 사람으로 봄 (합성 영상, 모형 시험장의 빨간 피규어). 아무 설치 없이 동작
  yolo  : Ultralytics YOLO. 모델 경로에 따라 자동으로
          yolo11n.pt (PyTorch) / yolo11n_openvino_model (노트북) / yolo11n_ncnn_model (파이) 사용
  hailo : AI HAT+ (Hailo-8L 13 TOPS). Hailo용 .hef 모델 (picamera2의 Hailo 도우미 사용)

추적(ID 붙이기)은 파이의 칼만 추적기가 하므로 검출기는 박스만 돌려줍니다.
"""
import cv2
import numpy as np

from .config import cfg

# COCO 80개 이름 (Hailo 결과를 이름으로 바꿀 때 사용. 감시 대상은 cfg.TARGET_CLASSES)
COCO_N = 80


class ColorDetector:
    """빨간 덩어리 -> 박스."""

    def __init__(self, min_area=20):
        self.min_area = min_area

    def __call__(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 80), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (179, 255, 255))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        dets = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area >= self.min_area:
                dets.append({"name": "person", "cls": 0, "conf": 1.0,
                             "box": [int(x), int(y), int(x + w), int(y + h)]})
        return dets


class YoloDetector:
    """Ultralytics YOLO. .pt / OpenVINO 폴더 / NCNN 폴더 모두 같은 코드로 동작."""

    def __init__(self, model="yolo11n.pt", imgsz=640, device=None, conf=None):
        from ultralytics import YOLO
        self.model = YOLO(model, task="detect")
        self.imgsz, self.device = imgsz, device
        self.conf = conf if conf is not None else cfg.DET_CONF
        # 처음 한 번은 느리므로 미리 한 번 돌려 둠 (자체 검출로 바뀔 때 지연 없게)
        self(np.zeros((cfg.IMG_H, cfg.IMG_W, 3), np.uint8))

    def __call__(self, frame):
        kw = {"device": self.device} if self.device else {}
        r = self.model.predict(frame, classes=list(cfg.TARGET_CLASSES), conf=self.conf,
                               imgsz=self.imgsz, verbose=False, **kw)[0]
        dets = []
        if r.boxes is None:
            return dets
        for box, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            dets.append({"name": cfg.TARGET_CLASSES.get(c, str(c)), "cls": c, "conf": float(p),
                         "box": [int(v) for v in box]})
        return dets


class HailoDetector:
    """AI HAT+ (Hailo-8L). 설치: sudo apt install hailo-all  (picamera2 Hailo 도우미 포함)

    model: NMS가 들어간 COCO YOLO .hef (예: /usr/share/hailo-models/yolov8s_h8l.hef).
    출력은 클래스별로 [y0, x0, y1, x1, score] (0~1 비율) 목록입니다.
    *실제 HAT에서는 아직 시험하지 못했습니다. 처음 붙였을 때 tools/check_env.py --hailo로 확인하세요.*
    """

    def __init__(self, model="/usr/share/hailo-models/yolov8s_h8l.hef", conf=None):
        from picamera2.devices import Hailo
        self.hailo = Hailo(model)
        self.in_h, self.in_w = self.hailo.get_input_shape()[:2]
        self.conf = conf if conf is not None else cfg.DET_CONF
        self(np.zeros((cfg.IMG_H, cfg.IMG_W, 3), np.uint8))

    def __call__(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame, (self.in_w, self.in_h)), cv2.COLOR_BGR2RGB)
        out = self.hailo.run(rgb)
        if len(out) == 1 and len(out[0]) == COCO_N:  # 배치 차원이 붙어 오는 경우
            out = out[0]
        dets = []
        for c, arr in enumerate(out):
            if c not in cfg.TARGET_CLASSES:
                continue
            for d in np.asarray(arr).reshape(-1, 5):
                y0, x0, y1, x1, score = d
                if score >= self.conf:
                    dets.append({"name": cfg.TARGET_CLASSES[c], "cls": c, "conf": float(score),
                                 "box": [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)]})
        return dets

    def close(self):
        self.hailo.close()


def make_detector(kind, model=None, imgsz=640, device=None):
    if kind == "color":
        return ColorDetector()
    if kind == "yolo":
        return YoloDetector(model or "yolo11n.pt", imgsz=imgsz, device=device)
    if kind == "hailo":
        return HailoDetector(model) if model else HailoDetector()
    if kind == "none":
        return None
    raise ValueError(f"모르는 검출기: {kind}")
