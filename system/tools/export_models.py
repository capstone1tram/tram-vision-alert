"""YOLO11n 가중치를 노트북용(OpenVINO)·파이용(NCNN)으로 변환

  python tools/export_models.py --target openvino      # 노트북 -> yolo11n_openvino_model/
  python tools/export_models.py --target ncnn          # 파이   -> yolo11n_ncnn_model/ (입력 320)
  python tools/export_models.py --target all

- yolo11n.pt 가 없으면 처음 실행할 때 자동으로 내려받습니다 (인터넷 필요).
- NCNN 변환은 노트북에서 해서 폴더째 파이로 복사해도 되고, 파이에서 직접 해도 됩니다.
- 입력 크기(imgsz)는 실행할 때 쓰는 값과 같아야 합니다 (파이: --local-imgsz 320).
- AI HAT+(Hailo)용 .hef는 여기서 만들지 않습니다. 'sudo apt install hailo-all' 하면 들어오는
  /usr/share/hailo-models/ 의 미리 컴파일된 COCO 모델을 그대로 씁니다.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["openvino", "ncnn", "all"], default="all")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--laptop-imgsz", type=int, default=640)
    ap.add_argument("--pi-imgsz", type=int, default=320)
    ap.add_argument("--int8", action="store_true", help="OpenVINO INT8 양자화 (더 빠르지만 정확도 약간 손해, 시간 오래 걸림)")
    args = ap.parse_args()

    from ultralytics import YOLO
    if args.target in ("openvino", "all"):
        p = YOLO(args.weights).export(format="openvino", imgsz=args.laptop_imgsz, int8=args.int8)
        print(f"노트북용 완료: {p}")
    if args.target in ("ncnn", "all"):
        p = YOLO(args.weights).export(format="ncnn", imgsz=args.pi_imgsz)
        print(f"파이용 완료: {p}  (파이로 폴더째 복사)")


if __name__ == "__main__":
    main()
