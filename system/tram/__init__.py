"""트램 전방 위험 알림 시스템 (동아리 캡스톤).

모듈 구성
  config    : 모든 설정값 (궤간, 통로 폭, TTC 기준, 핀 번호 ...)
  rails     : 노란 테이프 선로 검출 + 위험 구역(통로)
  distance  : 궤간 기하로 앞/옆 거리 계산
  kalman    : 바닥 평면 칼만 필터 추적기 (파이)
  risk      : TTC, 필요 감속도, 4단계 등급, 깜빡임 방지
  detectors : 검출기 (color / yolo(.pt·OpenVINO·NCNN) / hailo)
  link      : 파이 <-> 노트북 통신 (ZeroMQ)
  alarm     : 부저·LED (gpiozero)
  camera    : 영상 입력 (카메라 모듈 3, 웹캠, 영상 파일)
  display   : 화면 그리기
  guard     : 파이 쪽 판단 엔진 (모드 전환: 정상 / 예측 / 자체 검출)
  vision    : 실제 영상 인지 (선로 추적 + YOLO) - pc_main.py, 나중에 노트북 쪽
  ground    : 실제 영상 거리 (궤간 폭 지평선까지 연장)
  hud       : PC 화면 그리기
  recorder  : 위험 장면 자동 저장·캡처
"""
