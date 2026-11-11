# 트램 전방 위험 알림: 코드 예제

내일 모임 문서(OpenCV·YOLO 기초와 RailSafeNet 검토)의 처리 순서를 그대로 따라가는 예제 7개입니다.
설명과 전체 코드는 `코드예제_설명서.pdf`에 있습니다.

## 설치 (Python 3.10 이상, 가상환경 권장)

이 폴더에서 프로젝트 전용 가상환경(.venv)을 만들어 설치하면 다른 프로그램과 섞이지 않습니다. PC 환경에 따라 약 1~5 GB를 씁니다.

    # Windows (PowerShell) - Python 3.12 권장 (python.org 설치 때 'Add python.exe to PATH' 체크)
    py -3.12 -m venv .venv
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   # 처음 한 번만, Y 입력
    .\.venv\Scripts\Activate.ps1                                            # 줄 앞에 (.venv)가 보이면 성공
    python -m pip install --upgrade pip
    pip install -r requirements.txt

    # PowerShell 대신 명령 프롬프트(cmd)를 쓰면 활성화는:  .venv\Scripts\activate.bat

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

YOLO 가중치(yolo11n.pt)는 02·06번을 처음 실행할 때 자동으로 내려받습니다.

## 바로 해 보기 (카메라 없이)

    python 00_make_sample_video.py
    python 06_pipeline.py --source sample.mp4 --detector color

## 파일

| 파일 | 하는 일 |
| --- | --- |
| config.py | 궤간·차체 폭·TTC 기준 등 공통 설정 |
| 00_make_sample_video.py | 정답을 아는 합성 영상 + 정답 CSV 만들기 |
| 01_opencv_basics.py | 영상 읽기, 엣지, 직선 찾기, 그리기, 저장 |
| 02_yolo_detect_track.py | YOLO로 사람·자전거·차량 검출과 ID 추적 |
| 03_rail_detect.py | 테이프 선로 검출, 궤간 픽셀 폭, 위험 통로 |
| 04_distance.py | 궤간·사람 키로 앞/옆 거리 계산 |
| 05_risk_level.py | 속도·TTC·필요 감속도, 4단계 판정, 깜빡임 방지 |
| 06_pipeline.py | 위 전부를 묶은 PC 데모 + 이벤트 로그 |
| 07_kalman_tracker.py | 바닥 평면 칼만 필터(등속도) 추적기, 거리 따라 바뀌는 측정 잡음, 예측만 하기(coast) |
| 08_dropout_sim.py | 연결 끊김 3,000회 시뮬레이션: 마지막 값 유지 vs 칼만 예측 vs 칼만+2σ 오차·놓침 비교 |
| 09_laptop_server.py | 노트북: 파이가 보낸 사진에서 검출·선로만 하고 숫자로 답함 (ZeroMQ) |
| 10_pi_client.py | 파이: 촬영·전송, 칼만 추적, 거리·TTC·등급, 부저/LED, 끊김 시 예측 → 자체 검출 |

## 파이·노트북 분담 시험 (PC 한 대로도 가능)

    python 09_laptop_server.py --detector color          # 창 1 (노트북 역할)
    python 10_pi_client.py --source sample.mp4 --local color --drop-at 3 --drop-for 2.5   # 창 2 (파이 역할)

`--drop-at 3 --drop-for 2.5`는 3초부터 2.5초 동안 연결이 끊긴 상황을 흉내 냅니다. 화면에 정상 → 예측 → 자체 검출(감시 저하) → 정상 순서로 바뀌는 것이 보이면 성공입니다.
파이에서는 `--server tcp://<노트북IP>:5555 --source picam --local yolo --local-model yolo11n_ncnn_model`로 실행합니다.

창 없이 결과 영상만 저장하려면 `--save out.mp4 --no-show`를 붙이세요.

## 출처

선로를 줄 단위로 스캔해 궤간으로 거리 구역을 만드는 방식은 RailSafeNet(Ondřej Valach, MIT License,
https://github.com/oValach/RailSafeNet)을 참고해 새로 작성했습니다. YOLO는 Ultralytics(AGPL-3.0)를 씁니다.
