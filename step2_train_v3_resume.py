"""
[2단계 v3 - 이어서 학습] epoch 24에서 재개
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
현재 상태:
  - 완료: 24 epoch
  - 현재 best Top-1: 80.6%
  - train loss: 0.072 (잘 수렴 중)
  - 남은 epoch: 76 (목표 100)

이어서 학습하는 방법:
  model.train(..., resume=True) 사용
  → last.pt에서 optimizer 상태, lr 스케줄러까지 모두 복원
  → epoch 25부터 자동으로 이어서 시작

Colab 첫 셀:
  !pip install ultralytics
  from google.colab import drive
  drive.mount('/content/drive')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from pathlib import Path
from ultralytics import YOLO

# ════════════════════════════════════════════════
# ★ 경로 설정
# ════════════════════════════════════════════════
# last.pt 를 Drive에 업로드한 경로로 수정
LAST_PT_PATH = "/content/drive/MyDrive/waste_project/last.pt"

# 원본 학습 데이터 경로 (기존과 동일)
DATA_DIR     = "/content/drive/MyDrive/waste_project/dataset"

# 결과 저장 경로 — 기존 waste_cls_v3 폴더에 이어서 저장
PROJECT_DIR  = "/content/drive/MyDrive/waste_project/runs"

IMG_SIZE     = 224


def resume_train():
    print("=" * 55)
    print("  [재개] epoch 24 → 100 이어서 학습")
    print("=" * 55)
    print(f"  체크포인트 : {LAST_PT_PATH}")
    print(f"  현재 best  : Top-1 80.6% (epoch 24)")
    print(f"  목표       : epoch 100 완주 또는 early stopping")
    print()

    # last.pt 존재 확인
    if not Path(LAST_PT_PATH).exists():
        print(f"❌ 파일 없음: {LAST_PT_PATH}")
        print("   last.pt 를 Google Drive에 업로드했는지 확인하세요.")
        return

    # last.pt 로드
    model = YOLO(LAST_PT_PATH)

    # ★ resume=True 가 핵심
    # optimizer 상태, lr 스케줄러, epoch 카운터 모두 복원됨
    results = model.train(
        resume = True,   # ← 이것만 있으면 나머지는 자동 복원
    )

    print("\n✅ 학습 완료!")
    weights_dir = f"{PROJECT_DIR}/waste_cls_v3/weights"
    print(f"   best.pt : {weights_dir}/best.pt")
    print(f"   last.pt : {weights_dir}/last.pt")
    return results


def evaluate():
    """학습 완료 후 test 세트 최종 평가"""
    weights_path = f"{PROJECT_DIR}/waste_cls_v3/weights/best.pt"

    if not Path(weights_path).exists():
        print(f"❌ best.pt 없음: {weights_path}")
        return

    print("\n" + "=" * 55)
    print("  [최종 평가] Test 세트 Top-1 Accuracy")
    print("=" * 55)

    model   = YOLO(weights_path)
    metrics = model.val(
        data  = DATA_DIR,
        split = "test",
        imgsz = IMG_SIZE,
        plots = True,
    )

    top1 = metrics.top1 * 100
    top5 = metrics.top5 * 100

    print(f"\n📊 최종 성능")
    print(f"  Top-1 Accuracy : {top1:.1f}%")
    print(f"  Top-5 Accuracy : {top5:.1f}%")
    print()
    print(f"  이전 (epoch 24) : 80.6%")
    print(f"  최종 (epoch 완주): {top1:.1f}%")
    diff = top1 - 80.6
    if diff > 0:
        print(f"  향상폭 : +{diff:.1f}%p ✅")
    else:
        print(f"  변화   : {diff:.1f}%p (early stopping으로 이미 best 달성)")

    if top1 >= 85:
        print(f"\n  ✅ 85% 이상 달성 — 포트폴리오 제출 가능!")
    elif top1 >= 70:
        print(f"\n  ✅ 요구사항 70% 달성")
    else:
        print(f"\n  ⚠️  70% 미달 — 데이터 추가 또는 모델 업그레이드 필요")

    return metrics


def download_best():
    """Colab에서 로컬 PC로 best.pt 다운로드"""
    try:
        from google.colab import files
        best_path = f"{PROJECT_DIR}/waste_cls_v3/weights/best.pt"
        if Path(best_path).exists():
            files.download(best_path)
            print("✅ best.pt 다운로드 시작")
        else:
            print(f"Drive에서 직접 다운로드: {best_path}")
    except ImportError:
        print(f"Drive에서 직접 다운로드: {PROJECT_DIR}/waste_cls_v3/weights/best.pt")


# ════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. 이어서 학습
    resume_train()

    # 2. 최종 성능 평가
    evaluate()

    # 3. best.pt 다운로드
    download_best()
