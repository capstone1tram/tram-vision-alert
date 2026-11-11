"""경보 출력: 부저 + LED 3개 (gpiozero).  파이가 아니면 아무것도 안 함 (화면 경보만).

  안전     : 초록 켜짐
  주의     : 노랑 켜짐
  경고     : 빨강 깜빡임 + 부저 짧게 (1초에 한 번)
  위험     : 빨강 켜짐 + 부저 빠르게 (1초에 5번)
  감시 불가 : 노랑 깜빡임 + 부저 길게 (2초에 한 번)
  감시 저하(자체 검출 모드)는 위 등급 표시 + 초록 깜빡임

핀 번호는 config.py의 PIN_* (BCM 번호). 부저는 능동 부저(전원만 주면 소리 나는 것) 기준.
"""
from .config import cfg


class Alarm:
    def __init__(self, enable=True):
        self.ok = False
        self.state = None
        if not enable:
            return
        try:
            from gpiozero import LED, Buzzer
            self.buz = Buzzer(cfg.PIN_BUZZER)
            self.g, self.y, self.r = LED(cfg.PIN_LED_GREEN), LED(cfg.PIN_LED_YELLOW), LED(cfg.PIN_LED_RED)
            self.ok = True
        except Exception as e:                          # 노트북, 핀 없음 등
            print(f"[경보] GPIO 없음 -> 화면 경보만 사용 ({type(e).__name__})")

    def set(self, level, degraded=False):
        st = (level, degraded)
        if not self.ok or st == self.state:             # 바뀔 때만 (깜빡임 스레드 재시작 방지)
            return
        self.state = st
        for d in (self.g, self.y, self.r, self.buz):
            d.off()
        if level == 0:
            self.g.on()
        elif level == 1:
            self.y.on()
        elif level == 2:
            self.r.blink(0.25, 0.25)
            self.buz.beep(0.15, 0.85)
        elif level == 3:
            self.r.on()
            self.buz.beep(0.1, 0.1)
        else:                                          # -1 감시 불가
            self.y.blink(0.5, 0.5)
            self.buz.beep(0.5, 1.5)
        if degraded and level >= 0:
            self.g.blink(0.2, 0.8)

    def close(self):
        if self.ok:
            for d in (self.g, self.y, self.r, self.buz):
                d.off()
