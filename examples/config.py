"""모든 예제가 함께 쓰는 설정값.

숫자는 로드맵의 '제안 초기값'입니다. 모형(1:10) 시험에서는 길이 값만 1/10로 바꾸면 됩니다.
"""

# 카메라
IMG_W, IMG_H = 1280, 720      # 처리할 영상 크기 (px)
HFOV_DEG = 60.0               # 수평 화각 (도). 카메라 사양서 값으로 바꾸세요.

# 선로·차량 (실제 크기, m)
GAUGE_M = 1.435               # 궤간: 두 레일 사이 거리
TRAM_WIDTH_M = 2.65           # 차체 폭 (가정값, 실제 차량 사양으로 교체)
MARGIN_M = 0.5                # 차체 바깥 여유
BUFFER_M = 1.0                # 통로 바깥 '주의' 완충 구역 폭
CORRIDOR_HALF_M = TRAM_WIDTH_M / 2 + MARGIN_M   # 기준선에서 통로 끝까지 (약 1.8 m)

# 위험 판단
TTC_WARN_S = 5.0              # 경고 기준 TTC
TTC_DANGER_S = 3.0            # 위험 기준 TTC
A_SERVICE = 1.2               # 상용 제동 감속도 (m/s², 예시)
T_REACT_S = 1.0               # 운전자 반응 시간
PERSON_HEIGHT_M = 1.7         # 레일이 안 보일 때 쓰는 백업 거리용

# 등급 안정화 (깜빡임 방지)
RAISE_FRAMES = 2              # 이만큼 연속으로 높으면 등급을 올림
LOWER_HOLD_S = 1.0            # 이만큼 계속 낮아야 등급을 내림

# 감시 대상 (COCO 클래스 번호: 이름)
TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

LEVEL_NAMES = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "DANGER", -1: "NO VIEW"}
LEVEL_NAMES_KO = {0: "안전", 1: "주의", 2: "경고", 3: "위험", -1: "감시 불가"}
# OpenCV 색은 (B, G, R) 순서
LEVEL_COLORS = {0: (0, 200, 0), 1: (0, 220, 255), 2: (0, 140, 255), 3: (0, 0, 255), -1: (150, 150, 150)}
