"""
[2단계] YOLOv8 모델 학습 스크립트
Transfer Learning으로 폐기물 분류 모델 학습

실행 환경: Google Colab (GPU T4 무료 사용)
실행 방법:
  1. dataset 폴더 전체를 Google Drive에 업로드
  2. 이 파일을 Colab에서 열어서 실행
"""

# ── Colab에서 첫 셀에 아래 명령어 실행 ──────────────────
# !pip install ultralytics
# from google.colab import drive
# drive.mount('/content/drive')
# ──────────────────────────────────────────────────────────

from ultralytics import YOLO
import yaml
from pathlib import Path

# ────────────────────────────────────────────────
# 설정값
# ────────────────────────────────────────────────
# Google Drive에 업로드한 dataset 폴더 경로
DATA_YAML   = "/content/drive/MyDrive/waste_project/dataset/data.yaml"
MODEL_SIZE  = "yolov8n.pt"   # n=nano(빠름) / s=small / m=medium
EPOCHS      = 50             # 학습 반복 횟수 (50~100 권장)
BATCH_SIZE  = 16             # GPU 메모리에 따라 조정 (T4: 16 권장)
IMG_SIZE    = 640            # 입력 이미지 크기
PROJECT_DIR = "/content/drive/MyDrive/waste_project/runs"  # 결과 저장 경로


def train():
    print("=" * 50)
    print("[2단계] YOLOv8 모델 학습 시작")
    print("=" * 50)
    print(f"  모델  : {MODEL_SIZE}")
    print(f"  Epoch : {EPOCHS}")
    print(f"  Batch : {BATCH_SIZE}")
    print(f"  Image : {IMG_SIZE}x{IMG_SIZE}")

    # 사전 학습된 YOLOv8 모델 로드 (처음 실행 시 자동 다운로드)
    model = YOLO(MODEL_SIZE)

    # Transfer Learning 학습
    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        batch     = BATCH_SIZE,
        imgsz     = IMG_SIZE,
        project   = PROJECT_DIR,
        name      = "waste_classifier",
        patience  = 15,       # 15 epoch 동안 개선 없으면 조기 종료
        optimizer = "AdamW",
        lr0       = 0.001,
        augment   = True,     # 데이터 증강 (flip, crop, brightness 등)
        cache     = True,     # 이미지 캐싱으로 학습 속도 향상
        device    = 0,        # GPU 사용 (CPU면 "cpu"로 변경)
        verbose   = True,
    )

    print("\n✅ 학습 완료!")
    print(f"결과 저장 위치: {PROJECT_DIR}/waste_classifier/")

    return results


def evaluate(weights_path=None):
    """
    학습 완료 후 Test 세트로 성능 평가
    weights_path 예시: PROJECT_DIR + "/waste_classifier/weights/best.pt"
    """
    if weights_path is None:
        weights_path = f"{PROJECT_DIR}/waste_classifier/weights/best.pt"

    print("\n" + "=" * 50)
    print("[성능 평가] Test 세트 mAP 측정")
    print("=" * 50)

    model = YOLO(weights_path)
    metrics = model.val(
        data  = DATA_YAML,
        split = "test",
        imgsz = IMG_SIZE,
    )

    print(f"\n📊 성능 지표")
    print(f"  mAP@50    : {metrics.box.map50:.4f}")    # 포트폴리오 핵심 수치
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    train_results = train()
    evaluate()
