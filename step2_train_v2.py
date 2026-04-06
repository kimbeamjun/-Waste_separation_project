"""
[2단계 - 개선판] YOLOv8s Classification 모델 학습
nano → small 모델 업그레이드 + epoch 100 + 학습률 최적화

실행 환경: Google Colab (무료 GPU T4)
변경사항:
  - yolov8n-cls → yolov8s-cls (더 정확한 모델)
  - epoch 50 → 100
  - lr0 0.001 → 0.0005 (과적합 방지)
  - patience 15 → 20

-- Colab 첫 셀 --
!pip install ultralytics
from google.colab import drive
drive.mount('/content/drive')
"""

from ultralytics import YOLO

# ════════════════════════════════════════════════
# ★ 설정값
# ════════════════════════════════════════════════
DATA_DIR    = "/content/drive/MyDrive/waste_project/dataset"
PROJECT_DIR = "/content/drive/MyDrive/waste_project/runs"
MODEL_SIZE  = "yolov8s-cls.pt"   # ★ nano → small 업그레이드
EPOCHS      = 100                 # ★ 50 → 100
BATCH_SIZE  = 32
IMG_SIZE    = 224
WORKERS     = 2


def train():
    print("=" * 55)
    print("  [2단계 개선] YOLOv8s Classification 학습")
    print("=" * 55)
    print(f"  모델    : {MODEL_SIZE}  (nano → small 업그레이드)")
    print(f"  Epoch   : {EPOCHS}     (50 → 100)")
    print(f"  lr0     : 0.0005       (과적합 방지)")
    print()

    model = YOLO(MODEL_SIZE)

    results = model.train(
        data      = DATA_DIR,
        epochs    = EPOCHS,
        batch     = BATCH_SIZE,
        imgsz     = IMG_SIZE,
        project   = PROJECT_DIR,
        name      = "waste_cls_v2",       # v2로 구분
        patience  = 20,                   # ★ 15 → 20
        optimizer = "AdamW",
        lr0       = 0.0005,               # ★ 0.001 → 0.0005
        lrf       = 0.01,                 # 최종 lr 비율
        warmup_epochs = 5,                # 워밍업 5 epoch
        augment   = True,
        workers   = WORKERS,
        device    = 0,
        verbose   = True,
    )

    print("\n✅ 학습 완료!")
    print(f"   결과: {PROJECT_DIR}/waste_cls_v2/weights/best.pt")
    return results


def evaluate(weights_path=None):
    if weights_path is None:
        weights_path = f"{PROJECT_DIR}/waste_cls_v2/weights/best.pt"

    print("\n" + "=" * 55)
    print("  [성능 평가] v1 vs v2 비교")
    print("=" * 55)

    model   = YOLO(weights_path)
    metrics = model.val(data=DATA_DIR, split="test", imgsz=IMG_SIZE)

    print(f"\n📊 v2 성능 지표")
    print(f"  Top-1 Accuracy : {metrics.top1:.4f} ({metrics.top1*100:.1f}%)")
    print(f"  Top-5 Accuracy : {metrics.top5:.4f} ({metrics.top5*100:.1f}%)")
    print(f"\n  [참고] v1 (yolov8n): Top-1 80.0%")
    print(f"  [결과] v2 (yolov8s): Top-1 {metrics.top1*100:.1f}%")


if __name__ == "__main__":
    train()
    evaluate()
