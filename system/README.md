# 트램 전방 위험 알림 시스템 (실행용 전체 코드)

발표 3·상세 보고서의 구상을 그대로 돌아가는 프로그램으로 묶은 것입니다.
`code_examples/`의 예제 03~10을 정리해서 재사용했고, 하드웨어 없이 노트북 한 대로도 전체를 시험할 수 있습니다.

```
 [라즈베리 파이 5 + 카메라 모듈 3]                       [노트북 (Windows)]
  촬영 ──JPEG──────────────────────────────────────────▶ YOLO11n 검출 (OpenVINO)
                                                         + 선로 검출 (OpenCV)
  칼만 추적 ◀──────────── 박스·선로 숫자(JSON) ───────────┘
  거리(궤간) → TTC·필요 감속도 → 4단계 등급 → 부저·LED·화면, 기록(CSV)
  연결 감시: 0.3초 무응답 → 예측(2σ 넓힘, 등급은 올리기만)
            1초 무응답   → 파이 자체 검출(NCNN 또는 AI HAT+) + "감시 저하"
```

## 1. 폴더 구조

| 파일 | 하는 일 |
| --- | --- |
| `pi_main.py` | **파이에서 실행.** 촬영·전송, 칼만, 거리, TTC, 등급, 경보, 연결 감시, 기록 |
| `laptop_server.py` | **노트북에서 실행.** 받은 사진에서 검출 + 선로만 하고 숫자로 답함 |
| `tram/config.py` | 모든 설정값 (궤간, 위험 구역 폭, TTC 5/3초, 감속도 1.2, 핀 번호 …) |
| `tram/rails.py` | 노란 테이프 선로 검출(줄 단위 스캔 + 직선 맞춤), 위험 구역 |
| `tram/rails_photo.py` | 실제 선로 사진용: 레일 마스크(학습 없는 tophat / 정답 라벨) + 내 선로 고르기 (12절) |
| `tram/distance.py` | 궤간 기하로 앞 거리 Z·옆 거리 X (멀면 사람 키 백업) |
| `tram/kalman.py` | 바닥 평면 선형 칼만 필터(등속도) + 여러 물체 추적 |
| `tram/risk.py` | TTC, 필요 감속도, 4단계 등급, 깜빡임 방지 |
| `tram/guard.py` | 모드 전환(정상 / 예측 / 자체 검출 / 단독)과 최종 등급 |
| `tram/detectors.py` | 검출기: `color`(빨간 물체), `yolo`(.pt·OpenVINO·NCNN), `hailo`(AI HAT+) |
| `tram/link.py` | 파이 ↔ 노트북 통신 (ZeroMQ, 배경 스레드, 자동 재연결) |
| `tram/alarm.py` | 부저·LED (gpiozero) |
| `tram/camera.py` | 입력: 카메라 모듈 3 / 웹캠 / 영상 파일 |
| `tram/display.py` | 화면 그리기 (등급 띠, 위험 구역, 추적 물체, 통신 상태) |
| `tools/selftest.py` | 설치 후 한 번에 합격/불합격 확인 |
| `tools/make_sample_video.py` | 정답을 아는 합성 영상 + 정답 CSV |
| `tools/evaluate.py` | 기록과 정답을 비교해 거리 오차·놓침 계산 |
| `tools/export_models.py` | yolo11n → OpenVINO(노트북) / NCNN 입력 320(파이) 변환 |
| `tools/latency_test.py` | 2주차 관문: 파이 ↔ 노트북 왕복 시간 측정 |
| `tools/hsv_tuner.py` | 노란 테이프 색 범위 맞추기 (조명 바뀌면 다시) |
| `tools/calibrate_focal.py` | 초점거리 보정 (거리 정확도를 정하는 숫자) |
| `tools/rail_photo_test.py` | RailSem19 사진으로 선로 인식 실험·채점 (12절) |
| `tools/rail_viewer.py` | 사진을 3초씩 넘기며 원본과 인식 결과를 나란히 보여 주는 뷰어 (12절) |
| `tools/check_env.py` | 설치 점검 (카메라, 부저·LED, NCNN, OpenVINO, Hailo) |
| `settings_model_example.json` | 1:10 모형 시험장 설정 예시 |
| `pc_main.py` | **PC 한 대로 전체 실행 (실제 블랙박스 영상).** 선로·위험 구역·검출·거리·TTC·등급 + 위험 장면 자동 저장 (13절) |
| `tram/vision.py` | 인지 단계(나중에 노트북 쪽): 프레임 사이 선로 추적(RailTracker) + YOLO, laptop_server와 같은 모양의 답 |
| `tram/ground.py` | 실제 영상용 거리: 궤간 폭을 지평선까지 직선 연장(평평한 바닥 가정), 화면 밖으로 나간 트랙 거르기 |
| `tram/hud.py` | PC 화면: 박스마다 등급 색·거리·TTC, 10/20/30/50 m 눈금 |
| `tram/recorder.py` | 경고 이상 장면 자동 저장(앞 3초~뒤 2초) + 캡처 + events.csv |
| `settings_video.json` | 블랙박스 영상용 설정 (화각 추정 73°, 헛경보 줄이는 값들) |
| `tools/video_probe.py` | 영상 정보·사진·짧은 구간 뽑기 + YOLO 모델 받기·OpenVINO 변환·CPU/GPU 속도 측정 |

## 2. 빠른 시작: 노트북 한 대, 하드웨어 없이 (약 10분)

PowerShell을 이 폴더(`tram_system`)에서 엽니다.

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   # 처음 한 번만, Y
.\.venv\Scripts\Activate.ps1                                            # 줄 앞에 (.venv) 보이면 성공
python -m pip install --upgrade pip
pip install -r requirements_laptop.txt

python tools\selftest.py          # 약 20초. 마지막 줄에 '합격'이 나오면 설치 끝
```

이제 창으로 직접 봅니다. 노트북 서버를 같은 프로그램 안에서 띄우는 `--server local` 모드입니다.

```powershell
python tools\make_sample_video.py                               # sample.mp4 + sample_truth.csv
python pi_main.py --source sample.mp4 --server local --local color --no-gpio
python pi_main.py --source sample.mp4 --server local --local color --no-gpio --drop-at 3 --drop-for 2.5
```

두 번째 줄은 3초부터 2.5초 동안 연결이 끊긴 상황입니다. 화면 아래 띠가
`LINK OK` → `LINK LOST - PREDICT` → `PI DETECT (DEGRADED)` → `LINK OK` 로 바뀌면 정상입니다.
창에서 `q`를 누르면 끝나고, 터미널에 요약(모드 비율, 등급 비율, 왕복 시간)이 나옵니다.

창을 두 개 띄워서 실제처럼 나눠 돌릴 수도 있습니다.

```powershell
python laptop_server.py --detector color --show                                   # 창 1 (노트북 역할)
python pi_main.py --source sample.mp4 --server tcp://127.0.0.1:5555 --local color --no-gpio   # 창 2 (파이 역할)
```

## 3. 실제 YOLO로 노트북에서 시험 (영상 파일·웹캠)

```powershell
python tools\export_models.py --target all     # yolo11n.pt 내려받고 OpenVINO·NCNN 폴더 만듦 (몇 분)
python laptop_server.py --detector yolo --model yolo11n_openvino_model --show      # 창 1
python pi_main.py --source 내영상.mp4 --server tcp://127.0.0.1:5555 --local yolo --local-model yolo11n_ncnn_model --no-gpio   # 창 2
```

- 웹캠이면 `--source 0`. 휴대폰으로 찍은 모형 시험장 영상도 그대로 쓸 수 있습니다.
- 합성 영상의 빨간 사각형은 사람이 아니라서 YOLO가 못 찾는 게 정상입니다(합성 영상은 `color`).
- 노트북 내장 GPU(Arc 140T)를 쓰려면 서버에 `--device intel:gpu`를 붙여 보세요. 오류가 나면 빼고 CPU로 돌리면 됩니다(CPU도 1장 수십 ms).
- 선로가 노란 테이프가 아니면 등급이 `NO VIEW`(감시 불가)로 나옵니다. 이건 정상 동작입니다.

## 4. 노트북 + 파이 실제 연결

### 4-1. 노트북
1. 설정 → 네트워크 및 인터넷 → **모바일 핫스팟** 켜기. 노트북 주소는 보통 `192.168.137.1`입니다(`ipconfig`로 확인).
2. `python laptop_server.py --detector yolo --model yolo11n_openvino_model --show`
3. 처음 실행할 때 Windows 방화벽 창이 뜨면 **개인 네트워크 허용**을 누릅니다.

### 4-2. 파이 (Raspberry Pi OS 64-bit)
이 폴더를 통째로 파이에 복사합니다(USB, `scp -r tram_system pi@<파이IP>:~` 등). NCNN 모델 폴더(`yolo11n_ncnn_model`)도 같이 복사하세요.

```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero
cd ~/tram_system
python3 -m venv --system-site-packages .venv     # 반드시 --system-site-packages (picamera2·gpiozero가 보이게)
source .venv/bin/activate
pip install -r requirements_pi.txt

python tools/check_env.py --picam --gpio --ncnn yolo11n_ncnn_model
```

`picamera2`를 불러올 때 `numpy.dtype size changed` 같은 오류가 나면 `pip install "numpy<2"` 후 다시 해 보세요(OS의 numpy와 pip numpy가 섞인 경우).

### 4-3. 2주차 관문: 왕복 시간 측정
```bash
python tools/latency_test.py --server tcp://192.168.137.1:5555 --n 200
```
95% 값이 300 ms 이하면 통과입니다. 노트북 모바일 핫스팟과 휴대폰 핫스팟을 둘 다 재서 비교하세요.
넘으면 `config.py`의 `SEND_WIDTH`(기본 960)·`JPEG_QUALITY`(기본 80)를 낮추거나 AI HAT+ 단독 모드(5장)를 고려합니다.

### 4-4. 실행
```bash
python pi_main.py --source picam --server tcp://192.168.137.1:5555 --local yolo --local-model yolo11n_ncnn_model --local-imgsz 320
```
- SSH로 접속해서 모니터가 없으면 `--no-show`, 화면을 영상으로 남기려면 `--save run1.mp4`.
- 노트북이 켜지기 전에는 파이가 자체 검출 모드로 시작하고, 노트북 답이 오면 바로 정상 모드로 바뀝니다.
- 시연 중 노트북 서버를 Ctrl+C로 끄면 실제 끊김 대응(예측 → 자체 검출)을 보여 줄 수 있습니다.

## 5. AI HAT+ (13 TOPS, Hailo-8L)를 쓸 때 (선택)

```bash
sudo apt install -y hailo-all && sudo reboot
ls /usr/share/hailo-models/                      # 들어 있는 .hef 모델 이름 확인 (예: yolov8s_h8l.hef)
python tools/check_env.py --hailo /usr/share/hailo-models/yolov8s_h8l.hef
```
- 끊겼을 때만 HAT 사용 (권장, 발표 구조 그대로):
  `python pi_main.py --source picam --server tcp://192.168.137.1:5555 --local hailo`
- 노트북 없이 파이 단독:
  `python pi_main.py --source picam --server none --local hailo`
- 다른 .hef를 쓰려면 `--local-model 경로.hef`. 미리 컴파일된 COCO 모델을 쓰므로 Hailo 컴파일러는 필요 없습니다.
- 검출기 형식이 같아서(이름·확신도·박스) 거리·칼만·등급·경보 코드는 하나도 안 바뀝니다.

## 6. 1:10 모형 시험장

1. `python tools/hsv_tuner.py --source picam` (또는 `--source 0`, 사진 파일): 레일만 하얗게 남게 슬라이더 조정 → 출력된 `RAIL_HSV_LO/HI`를 설정 JSON에 넣기
2. 선로 위 줄자로 잰 지점(예: 2.00 m)에 표시를 하고 사진을 찍은 뒤
   `python tools/calibrate_focal.py --image calib.jpg --dist 2.0 --scale 0.1` → 그 줄을 클릭 → `FX_PX` 값을 설정 JSON에 넣기
3. 실행할 때 `--config settings_model_example.json`(안에 `"SCALE": 0.1`) 또는 `--scale 0.1`

`--scale 0.1`은 길이·속도·감속도만 1/10로 바꾸고 시간 기준(TTC 5초/3초, 0.3초/1초)은 그대로 둡니다.
그래서 모형 트램이 실제 30 km/h의 1/10인 3 km/h(0.83 m/s)로 달리면 실제와 같은 시점에 같은 등급이 나옵니다.
1:10 합성 영상: `python tools/make_sample_video.py --scale 0.1 --out sample_model.mp4`

## 7. 부저·LED 배선 (BCM 번호, `config.py`에서 변경)

| 부품 | 핀 | 동작 |
| --- | --- | --- |
| 능동 부저 | GPIO17 | 경고: 1초에 한 번 짧게, 위험: 1초에 5번, 감시 불가: 2초에 한 번 길게 |
| 초록 LED | GPIO22 | 안전에서 켜짐, 감시 저하(자체 검출)일 때 깜빡임 |
| 노랑 LED | GPIO23 | 주의에서 켜짐, 감시 불가일 때 깜빡임 |
| 빨강 LED | GPIO24 | 경고에서 깜빡임, 위험에서 켜짐 |

LED는 220 Ω 저항을 거쳐 GND로 연결합니다. `python tools/check_env.py --gpio`로 순서대로 켜 볼 수 있습니다.

## 8. 판단 규칙 (발표 3·보고서 기준, 값은 `tram/config.py`)

- 위험 구역: 선로 중심선에서 양옆으로 차체 폭 절반(2.65 m ÷ 2) + 여유 0.5 m ≈ 1.8 m. 그 바깥 1 m는 완충 구역
- 거리: Z = 초점거리 × 궤간 ÷ (발 위치 줄의 궤간 픽셀 폭), X = (발 x − 중심선 x) × 궤간 ÷ 궤간 픽셀 폭
- 칼만: 상태 [X, Z, Vx, Vz], 등속도 모델, 측정 잡음은 Z²에 비례. 노트북 답은 '찍은 시각'으로 갱신하고 판정은 '지금'까지 예측해서 함(지연 보정)
- TTC = Z ÷ 다가오는 속도, 필요 감속도 = v² ÷ (2 × (Z − v × 반응 1초))

| 등급 | 조건 |
| --- | --- |
| 위험 | 위험 구역 안 + (TTC ≤ 3초 또는 필요 감속도 > 1.2 m/s²) |
| 경고 | 위험 구역 안 + TTC ≤ 5초, 또는 구역으로 들어오는 중이고 트램보다 먼저 들어올 때 |
| 주의 | 위험 구역 안이지만 멂, 또는 완충 구역 |
| 안전 | 그 외 |
| 감시 불가 | 선로를 2초 넘게 못 찾음, 또는 끊겼는데 자체 검출기가 없음 |

등급은 2프레임 연속이면 올리고, 1초 동안 계속 낮아야 내립니다. 새 물체는 3번 잡혀야 판정에 씁니다(헛검출 거르기).

## 9. 기록 읽기

실행할 때마다 `logs/`에 두 파일이 생깁니다(엑셀로 열림).
- `날짜_frames.csv`: 프레임마다 모드, 등급, 가장 위험한 물체의 Z·X·TTC, 왕복 시간, 끊긴 시간
- `날짜_tracks.csv`: 프레임마다 물체별 X, Z, 속도, 불확실성(σ), TTC, 필요 감속도, 등급

합성 영상 결과 채점: `python tools/evaluate.py logs\날짜_tracks.csv sample_truth.csv`

## 10. 이 코드로 확인한 것과 아직 못 한 것

클라우드 PC(Python 3.13, OpenCV 5.0, Ultralytics 8.4)에서 확인한 것:
- 합성 영상: 앞 거리 오차 평균 약 1.2%(0.3~0.5 m), 위험 구역 안 프레임 69개 중 '안전'으로 본 것 0개, '위험' 경보 나옴
- 끊김 시험: 정상 → 0.3초에 예측 → 1초에 자체 검출 → 복구 시 정상, 순서대로 동작
- 실제 YOLO11n: 노트북 OpenVINO(1장 약 22 ms) + 파이용 NCNN 입력 320 조합으로 전체 흐름 동작, 정지 장면에서 헛 '위험' 없음
- 1:10 크기(`--scale 0.1`) 합성 영상도 같은 등급 흐름

하드웨어가 없어 아직 못 한 것 (처음 붙일 때 `tools/check_env.py`로 확인):
- 카메라 모듈 3(picamera2), 부저·LED(gpiozero), 파이 5에서의 실제 속도
- AI HAT+(Hailo) 검출기: picamera2의 Hailo 도우미 사용법대로 짰지만 실제 HAT에서는 미확인
- Windows에서 `--device intel:gpu` 동작 여부

## 11. 예제와의 관계

| 예제 | 여기서 |
| --- | --- |
| config.py | `tram/config.py` (+ `--config` JSON, `--scale`) |
| 03_rail_detect | `tram/rails.py` |
| 04_distance | `tram/distance.py` |
| 05_risk_level | `tram/risk.py` |
| 06_pipeline (검출기) | `tram/detectors.py` (+ Hailo) |
| 07_kalman_tracker | `tram/kalman.py` (지연 보정, 종류별 묶음, 검출기 전환 보정 추가) |
| 09_laptop_server | `laptop_server.py` |
| 10_pi_client | `pi_main.py` + `tram/guard.py`, `tram/link.py`(배경 스레드로 바꿔서 끊겨도 화면이 안 멈춤) |

## 12. 실제 선로 사진(RailSem19)으로 선로 인식 실험 (2026-11-03 추가)

`tram/rails_photo.py` + `tools/rail_photo_test.py`. 노란 테이프용 `rails.py`는 그대로 두고, 실제 사진용을 따로 만들었습니다. 결과는 같은 `RailModel`이라 거리·위험 구역 계산은 그대로 씁니다.

```powershell
python tools\rail_photo_test.py --data G:\rail_pic --n 200            # 고르게 200장 (몇 분)
python tools\rail_photo_test.py --data G:\rail_pic --n 0              # 8,500장 전부 (오래 걸림)
python tools\rail_photo_test.py --data G:\rail_pic --names rs00000 rs04400 --save-all
```
눈으로 보기 (사진 3초씩, 왼쪽 원본 / 오른쪽 인식 결과, 스페이스 = 멈춤, a·d = 이전·다음, q = 끝):
```powershell
python tools\rail_viewer.py --data G:\rail_pic --gt --shuffle
python tools\rail_viewer.py --data G:\rail_pic --method label    # 정답 마스크로 (seg 모델이 완벽할 때 모습)
python tools\rail_viewer.py --data G:\rail_pic --yolo --shuffle  # 물체(사람·차)도 찾아 구역별 색으로 표시 (빨강=위험 구역 안, 주황=완충, 초록=밖)
```

채점 결과는 `rail_test_out\` 에 생깁니다: `요약.txt`(방법별 성공률), `결과.csv`(사진별), `img\`(4칸 그림: 초록 점 = 정답 레일 안쪽, 빨간 선 = 찾은 레일, 빨간 면 = 위험 구역).

| 방법 | 내용 | 클라우드 60장 결과 |
|---|---|---|
| hsv | 지금 `rails.py` 그대로(노란 테이프 색) | 0% (예상대로) |
| tophat | 학습 없음. 밝고 가는 세로 줄 = 레일 → 내 선로 고르기 | 55% |
| label_old | 정답 레일 마스크 + `rails.py`의 '가장 큰 빈칸' 나누기 | 7% |
| label_ego | 정답 레일 마스크 + 새 '내 선로 고르기' | 75% |

성공 = 화면 아래쪽에서 찾은 레일이 정답과 궤간 폭의 20% 안. 정답 '내 선로'도 라벨에서 규칙으로 뽑은 것이라 분기기·트램 교차로에서는 정답 자체가 애매합니다.

읽는 법: label_old가 7%라는 건 **마스크가 완벽해도 옆 선로가 보이면 기존 나누기가 틀린다**는 뜻이라, 실제 영상에서는 '내 선로 고르기'가 꼭 필요합니다. label_ego 75%는 YOLO-seg를 학습했을 때 기대할 수 있는 대략의 상한이고, 남은 25%는 분기기·트램 교차로·급곡선입니다. 학습 없는 tophat은 직선·맑은 날엔 잘 되지만 어둡거나 레일이 녹슨 사진에서 못 찾습니다.

'내 선로 고르기' 원리: 평행한 선로는 소실점이 같고, 카메라가 올라탄 선로는 가운데 선이 화면에서 거의 수직입니다. 그래서 화면 아래 띠에서 레일 줄기들을 만들고, 이웃한 두 줄기 중 가운데 선이 가장 수직이고 화면 가운데에 가까운 쌍을 고른 뒤, 위로 한 줄씩 이어 갑니다.

라이선스: RailSem19 사진·라벨과 이 스크립트가 만든 그림은 등록한 본인만 비상업 연구용으로 쓸 수 있습니다. 팀원에게 복사해 주거나 저장소에 올리지 말고, 발표에 쓸 땐 얼굴·번호판을 가려 주세요.

## 13. PC 한 대로 전체 실행: 실제 트램 블랙박스 영상 (2026-11-04 추가)

`pc_main.py`. 영상을 재생하면서 선로 검출 → 위험 구역 → 사람·차 검출 → 거리 → 칼만 추적 → TTC·필요 감속도 → 4단계 등급까지 한 프로그램에서 합니다.

```
 영상 ─▶ [인지] tram/vision.py  : 선로(rails_photo + 프레임 사이 추적) + YOLO11n      ← 나중에 노트북
        ─▶ 답 {t_capture, rails, dets}  (laptop_server.py가 파이에 보내는 것과 같은 모양)
        ─▶ [판단] tram/guard.py : 거리(ground.py) → 칼만 → TTC·감속도 → 등급     ← 나중에 파이
        ─▶ 화면(hud.py) + 위험 장면 저장(recorder.py) + CSV
```

```bat
cd /d G:\tram_
.venv\Scripts\activate.bat
python pc_main.py
```

| 명령 | 뜻 |
|---|---|
| `python pc_main.py` | 기본 영상 `G:\rail_pic\tram_blackbox.mp4`, 실시간 재생 |
| `python pc_main.py --start 9:40` | 9분 40초부터 |
| `python pc_main.py --start 9:40 --duration 60` | 거기서 60초만 |
| `python pc_main.py --model models\yolo11s_openvino_model` | 더 큰 모델 (멀리 있는 사람 더 잘 봄, 조금 느림) |
| `python pc_main.py --no-realtime --no-show` | 창 없이 모든 프레임 처리 (정확한 기록용, 느림) |
| `python pc_main.py --device cpu` | GPU에서 오류가 나면 |

키: 스페이스 = 멈춤/재생, d / a = 10초 앞/뒤, s = 지금 화면 캡처, m = 선로 마스크(레일 후보 픽셀 초록) 보기, c = 초점거리 보정, q = 끝.

결과 `logs\pc\<날짜_시각>\`: `clips\event_001_05m31s_WARNING.mp4`(+ `_peak.jpg`, 경고 이상 장면 앞 3초~뒤 2초), `captures\`(s 키), `events.csv`(장면 목록), `frames.csv`, `tracks.csv`.

화면 읽는 법: 빨간 면 = 위험 구역(트램이 지나갈 자리), 노란 선 = 완충 구역 끝, 흰 점선 = 레일 찾은 범위 위로 연장한 중심선, 가로 눈금 = 10/20/30/50 m, 박스 색 = 그 물체의 등급(회색 = 아직 확정 전·거리 계산 못 함), 거리 뒤 `*` = 레일 찾은 범위 밖(연장 계산).

**기존 모형용 코드와 달라진 점** (파이·노트북 쪽 동작은 기본값이면 그대로)
1. 선로: `rails_photo`(tophat + 내 선로 고르기)에 **프레임 사이 추적**을 더함. 앞 선로 주변 띠에서 먼저 찾고, 궤간의 50% 넘게 튀면 6프레임까지는 의심하고 앞 선로 유지, 앞 선로와 섞어 떨림 줄임.
2. 거리: 레일은 보통 20~30 m까지만 잡히므로, 바닥이 평평하면 궤간 픽셀 폭이 화면 줄에 직선으로 비례한다는 점을 이용해 지평선까지 연장 (`ground.py`, 최대 80 m). 이 기울기로 카메라 높이도 나옴(이 영상은 약 1.8~2.2 m).
3. 헛경보 줄이기(`settings_video.json`에서만 켬): 측정 없이 예측만 하는 트랙이 화면 옆·뒤로 나가면 판단에서 뺌 / 화면 좌우 끝에 잘린 박스는 거리 계산 안 함 / '구역으로 들어오는 중' 경고는 지금 보이고(0.2초 안) 10번 이상 측정된 트랙이, 옆 속도 오차의 2배보다 확실히 빠르게, 트램이 올 때 아직 구역 안에 있을 때만.
4. 칼만: 박스 밑변 흔들림을 3 px로, 가속도 잡음을 2 m/s²로 (도로 주행·정지·출발 반복).

**거리 정확도에 대해 (교수님 질문 대비)**: 옆 거리 X, 위험 구역 판정, TTC는 모두 궤간 '비율'이라 초점거리 f와 상관없습니다(TTC = 거리 ÷ 다가오는 속도에서 f가 약분됨). f가 틀리면 '몇 m'와 '필요 감속도'만 같은 비율로 틀립니다. 이 영상의 카메라 정보가 없어서 화각 73°(처리 폭 960에서 f ≈ 650 px)는 추정값입니다(차선 점선이 1 m 선 + 1 m 빈칸이라고 가정 + 시내 속도가 시속 25 km 안팎이 되도록). 정확히 하려면 멈춘 화면에서 c를 누르고, 길이를 아는 바닥 표시의 가까운 끝·먼 끝을 차례로 클릭하세요(`--calib-len`, 기본 3 m). 예: 선로와 나란히 주차한 승용차의 앞·뒤 바퀴 닿는 점(축간거리 약 2.6 m → `--calib-len 2.6`). 결과는 `settings_video.json`의 `FX_PX`에 저장돼 다음부터 자동으로 씁니다.

**클라우드에서 20초 구간 4개(3분 31초, 12분 19초, 21분 7초, 29분 55초)로 확인한 것**: 내 선로 찾음 100%(4개 구간 모두), 처리 속도 CPU만으로 약 18장/초(노트북 GPU면 YOLO가 14 ms라 더 빠름). 헛경보 조정 전에는 '위험' 32~40%, '경고' 최대 50%였고, 조정 후 '위험' 0%, '경고' 0~5%. 남은 경고 1건은 화면 오른쪽 끝에 세워 둔 자전거가 처음 잡힐 때 속도가 출렁인 0.1초짜리 헛경보입니다. 실제 위험 장면이 섞인 정답이 없어서 **놓침은 아직 확인 못 했습니다** (위험 장면을 찾아 눈으로 확인하는 게 다음 단계).

한계: 렌즈 왜곡(화면 가장자리), 분기기·교차로에서 옆 선로로 잘못 붙는 경우, 정류장 이름 자막·워터마크, 차 뒤에 가려진 사람, 영상의 트램 속도는 알 수 없음(상대 속도만 씀). 영상 저작권은 원 채널에 있으니 발표에 쓸 땐 출처를 밝히고 짧게만 쓰세요.

**나중에 나눌 때**: 노트북은 `laptop_server.py`의 `make_handler` 안에서 `Perception(detector).process(frame, t)`를 부르면 되고, 파이는 `Guard(..., dist_fn=ground.obj_distance, visible_fn=ground.visible)`로 만들면 실제 영상용 거리 계산을 씁니다. 칼만·등급·연결 끊김 처리는 그대로입니다.

## 출처·라이선스

선로를 줄 단위로 스캔해 궤간으로 거리 구역을 만드는 방식은 RailSafeNet(Ondřej Valach, MIT License)을 참고해 새로 작성했습니다.
YOLO는 Ultralytics(AGPL-3.0)를 씁니다. 대회·발표용 비상업 사용은 문제없지만, 코드를 공개 배포하면 AGPL 조건(소스 공개)을 따라야 합니다.
