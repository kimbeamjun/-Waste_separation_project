"""
[2단계] YOLOv8 Classification 모델 학습
사전 학습된 모델로 폐기물 분류 Fine-tuning

실행 환경: Google Colab (무료 GPU T4)
실행 방법:
  1. dataset/ 폴더를 Google Drive에 업로드
  2. 이 파일을 Colab에 올려서 셀 단위로 실행

-- Colab 첫 셀에 실행할 명령어 --
!pip install ultralytics
from google.colab import drive
drive.mount('/content/drive')
"""

from ultralytics import YOLO

# ════════════════════════════════════════════════
# ★ 경로 설정 (Google Drive 업로드 경로에 맞게 수정)
# ════════════════════════════════════════════════
DATA_DIR    = "/content/drive/MyDrive/waste_project/dataset"  # dataset 폴더 경로
PROJECT_DIR = "/content/drive/MyDrive/waste_project/runs"     # 결과 저장 경로
MODEL_SIZE  = "yolov8n-cls.pt"   # cls = Classification 모델
                                  # n(nano/빠름) → s → m → l → x(느리고 정확)
EPOCHS      = 50
BATCH_SIZE  = 32                  # Classification은 batch 크게 줘도 됨
IMG_SIZE    = 224                 # Classification 표준 입력 크기
WORKERS     = 2


def train():
    print("=" * 55)
    print("  [2단계] YOLOv8 Classification 학습 시작")
    print("=" * 55)
    print(f"  데이터  : {DATA_DIR}")
    print(f"  모델    : {MODEL_SIZE}")
    print(f"  Epoch   : {EPOCHS}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  ImgSize : {IMG_SIZE}")
    print()

    # 사전 학습 모델 로드 (자동 다운로드)
    model = YOLO(MODEL_SIZE)

    # Fine-tuning 학습
    results = model.train(
        data      = DATA_DIR,     # Classification은 data.yaml 없이 폴더 경로만
        epochs    = EPOCHS,
        batch     = BATCH_SIZE,
        imgsz     = IMG_SIZE,
        project   = PROJECT_DIR,
        name      = "waste_cls",
        patience  = 15,           # 15 epoch 개선 없으면 조기 종료
        optimizer = "AdamW",
        lr0       = 0.001,
        augment   = True,         # 자동 데이터 증강
        workers   = WORKERS,
        device    = 0,            # GPU 사용 (CPU면 "cpu")
        verbose   = True,
    )

    print("\n✅ 학습 완료!")
    print(f"   결과 저장: {PROJECT_DIR}/waste_cls/")
    print(f"   best.pt 위치: {PROJECT_DIR}/waste_cls/weights/best.pt")
    return results


def evaluate(weights_path=None):
    """Test 세트로 최종 성능 평가"""
    if weights_path is None:
        weights_path = f"{PROJECT_DIR}/waste_cls/weights/best.pt"

    print("\n" + "=" * 55)
    print("  [성능 평가] Test 세트 정확도 측정")
    print("=" * 55)

    model = YOLO(weights_path)
    metrics = model.val(
        data  = DATA_DIR,
        split = "test",
        imgsz = IMG_SIZE,
    )

    print(f"\n📊 성능 지표 (포트폴리오 기재용)")
    print(f"  Top-1 Accuracy : {metrics.top1:.4f} ({metrics.top1*100:.1f}%)")
    print(f"  Top-5 Accuracy : {metrics.top5:.4f} ({metrics.top5*100:.1f}%)")


if __name__ == "__main__":
    train()
    evaluate()
