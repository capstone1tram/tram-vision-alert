"""설치 후 한 번에 확인: 합성 영상으로 전체 흐름(정상 -> 예측 -> 자체 검출 -> 정상)을 돌려 보고 합격/불합격

  python tools/selftest.py
(창 없이 약 20초. 하드웨어·YOLO 필요 없음)
"""
import csv
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(args):
    print("$ python " + " ".join(args))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")       # Windows 한글 출력 깨짐 방지
    r = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("불합격: 실행 오류")
    return r.stdout


def main():
    log_dir = os.path.join(ROOT, "logs", "selftest")
    os.makedirs(log_dir, exist_ok=True)
    for f in glob.glob(os.path.join(log_dir, "*.csv")):
        os.remove(f)
    run(["tools/make_sample_video.py", "--out", "logs/selftest/sample.mp4"])
    run(["pi_main.py", "--source", "logs/selftest/sample.mp4", "--server", "local", "--port", "5599",
         "--local", "color", "--drop-at", "3", "--drop-for", "2.5", "--no-show", "--no-gpio",
         "--log-dir", "logs/selftest"])
    frames = glob.glob(os.path.join(log_dir, "*_frames.csv"))[0]
    tracks = glob.glob(os.path.join(log_dir, "*_tracks.csv"))[0]
    out = run(["tools/evaluate.py", tracks, "logs/selftest/sample_truth.csv"])
    print(out)

    modes, levels = [], []
    for r in csv.DictReader(open(frames, encoding="utf-8-sig")):
        if not modes or modes[-1] != r["mode"]:
            modes.append(r["mode"])
        levels.append(int(r["level"]))
    ok_modes = "ONLINE" in modes and "PREDICT" in modes and "LOCAL" in modes and modes[-1] == "ONLINE"
    ok_levels = max(levels) == 3
    miss = int(out.strip().splitlines()[-1].split(":")[-1].split("개")[0])
    print(f"모드 순서: {' -> '.join(modes)}  {'O' if ok_modes else 'X'}")
    print(f"'위험' 경보가 나왔나: {'O' if ok_levels else 'X'}")
    print(f"위험 구역 안 놓침 0: {'O' if miss == 0 else 'X'}")
    print("\n합격" if ok_modes and ok_levels and miss == 0 else "\n불합격")


if __name__ == "__main__":
    main()
