"""
[2단계 v3] YOLOv8s Classification - 정확도 최적화 버전
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 (nano,  epoch 50)  → 기본 버전
v2 (small, epoch 100) → 모델 업그레이드
v3 (small, epoch 100) → augmentation 강화 + collected_data 병합 + 시각화 ← 이 파일

변경사항 (v2 대비):
  ① yolov8s-cls 유지 (small이 T4 환경에서 최적)
  ② Augmentation 파라미터 세분화 (fliplr, hsv, degrees, scale 등)
  ③ collected_data 자동 병합 (피드백 루프 데이터 활용)
  ④ 클래스 불균형 자동 감지 및 경고
  ⑤ 학습 후 Confusion Matrix + Loss Curve 자동 저장
  ⑥ v1/v2/v3 성능 비교 출력

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 환경: Google Colab (T4 GPU)

Colab 첫 셀에 실행:
  !pip install ultralytics
  from google.colab import drive
  drive.mount('/content/drive')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ════════════════════════════════════════════════
# ★ 경로 설정 — Google Drive 업로드 경로에 맞게 수정
# ════════════════════════════════════════════════
DATA_DIR         = "/content/drive/MyDrive/waste_project/dataset"
COLLECTED_DIR    = "/content/drive/MyDrive/waste_project/collected_data"  # 피드백 루프 수집 데이터
PROJECT_DIR      = "/content/drive/MyDrive/waste_project/runs"
MERGED_DATA_DIR  = "/content/drive/MyDrive/waste_project/dataset_v3"      # 병합 결과 저장

# ════════════════════════════════════════════════
# ★ 학습 설정
# ════════════════════════════════════════════════
MODEL_SIZE  = "yolov8s-cls.pt"   # small (nano보다 정확, T4에서 충분히 빠름)
EPOCHS      = 100                # 충분히 학습 (patience로 과적합 방지)
BATCH_SIZE  = 32
IMG_SIZE    = 224
WORKERS     = 2

# 클래스 정의 (step1 전처리와 동일 순서)
CLASSES = ["can", "glass_bottle", "paper", "pet_bottle", "snack_bag"]


# ════════════════════════════════════════════════
# 1. 클래스 불균형 체크
# ════════════════════════════════════════════════
def check_balance(data_dir: str):
    print("\n[클래스 불균형 체크]")
    print("-" * 40)
    total = 0
    counts = {}
    for split in ["train", "val", "test"]:
        split_path = Path(data_dir) / split
        if not split_path.exists():
            continue
        for cls_dir in sorted(split_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            n = len(list(cls_dir.glob("*.*")))
            counts[f"{split}/{cls_dir.name}"] = n
            total += n

    train_counts = {k: v for k, v in counts.items() if k.startswith("train")}
    max_n = max(train_counts.values()) if train_counts else 1
    min_n = min(train_counts.values()) if train_counts else 1

    for key, n in counts.items():
        bar = "█" * int(n / max_n * 20)
        warn = " ⚠️  부족" if "train" in key and n < min_n * 1.5 else ""
        print(f"  {key:<30} {n:>4}장  {bar}{warn}")

    print(f"\n  총 이미지: {total}장")
    if max_n > min_n * 2:
        print(f"  ⚠️  클래스 불균형 감지! (최대 {max_n}장 vs 최소 {min_n}장)")
        print(f"     부족한 클래스를 직접 촬영해서 추가하면 정확도가 올라갑니다.")
    else:
        print(f"  ✅ 클래스 균형 양호")
    print()


# ════════════════════════════════════════════════
# 2. collected_data 병합 (피드백 루프 데이터 활용)
# ════════════════════════════════════════════════
def merge_collected_data(base_dir: str, collected_dir: str, output_dir: str):
    """
    기존 dataset + collected_data 병합 → dataset_v3 생성
    collected_data는 train 세트에만 추가 (val/test는 원본 유지)
    """
    collected_path = Path(collected_dir)
    if not collected_path.exists():
        print("[병합] collected_data 폴더 없음 — 기존 dataset 그대로 사용")
        return base_dir

    # collected_data 내 총 이미지 수 확인
    total_collected = sum(
        len(list(d.glob("*.*")))
        for d in collected_path.iterdir()
        if d.is_dir()
    )
    if total_collected == 0:
        print("[병합] collected_data 비어있음 — 기존 dataset 그대로 사용")
        return base_dir

    print(f"\n[collected_data 병합] {total_collected}장 발견 → train 세트에 추가")
    print("-" * 40)

    # 기존 dataset 복사
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(base_dir, output_dir)

    # collected_data → train에 추가
    added = 0
    for cls_dir in collected_path.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        dest = output_path / "train" / cls_name
        dest.mkdir(parents=True, exist_ok=True)

        files = list(cls_dir.glob("*.*"))
        for f in files:
            dest_file = dest / f"collected_{f.name}"
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
                added += 1

        print(f"  {cls_name:<20} +{len(files)}장 추가")

    print(f"\n  ✅ 병합 완료: {added}장 추가 → {output_dir}")
    return output_dir


# ════════════════════════════════════════════════
# 3. 학습
# ════════════════════════════════════════════════
def train(data_dir: str):
    print("\n" + "=" * 55)
    print("  [2단계 v3] YOLOv8s Classification 학습")
    print("=" * 55)
    print(f"  모델     : {MODEL_SIZE}  (small)")
    print(f"  데이터   : {data_dir}")
    print(f"  Epoch    : {EPOCHS}  (patience=20 early stopping)")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  lr0      : 0.0005  →  lrf: 0.01")
    print(f"  Augment  : fliplr, hsv, degrees, scale, translate")
    print()

    model = YOLO(MODEL_SIZE)

    results = model.train(
        data    = data_dir,
        epochs  = EPOCHS,
        batch   = BATCH_SIZE,
        imgsz   = IMG_SIZE,
        project = PROJECT_DIR,
        name    = "waste_cls_v3",

        # ── 학습률 ──────────────────────────────
        optimizer     = "AdamW",
        lr0           = 0.0005,   # 초기 학습률 (v2와 동일)
        lrf           = 0.01,     # 최종 lr = lr0 * lrf
        warmup_epochs = 5,        # 처음 5 epoch은 lr을 서서히 올림
        momentum      = 0.937,
        weight_decay  = 0.0005,

        # ── 조기 종료 ───────────────────────────
        patience = 20,            # 20 epoch 개선 없으면 자동 종료

        # ── Augmentation ────────────────────────
        augment   = True,
        fliplr    = 0.5,          # 좌우 반전 50%
        flipud    = 0.1,          # 상하 반전 10% (쓰레기는 뒤집힐 수 있음)
        hsv_h     = 0.015,        # 색조 변환 (조명 변화 대응)
        hsv_s     = 0.7,          # 채도 변환
        hsv_v     = 0.4,          # 밝기 변환 (어두운/밝은 환경 대응)
        degrees   = 15,           # ±15도 회전 (기존 90도 고정보다 자연스러움)
        translate = 0.1,          # ±10% 이동
        scale     = 0.5,          # 50% 크기 변환
        shear     = 5.0,          # ±5도 전단 변환
        perspective = 0.0005,     # 원근 변환 (카메라 각도 대응)
        erasing   = 0.4,          # 40% 확률로 랜덤 영역 지움 (occlusion 대응)
        crop_fraction = 0.9,      # 90% crop (분류 모델 전용)

        # ── 기타 ────────────────────────────────
        workers = WORKERS,
        device  = 0,
        verbose = True,
        plots   = True,           # Loss curve, Confusion Matrix 자동 저장
    )

    weights_path = f"{PROJECT_DIR}/waste_cls_v3/weights/best.pt"
    print(f"\n✅ 학습 완료!")
    print(f"   best.pt 위치: {weights_path}")
    print(f"   Confusion Matrix: {PROJECT_DIR}/waste_cls_v3/confusion_matrix.png")
    print(f"   Loss Curve:       {PROJECT_DIR}/waste_cls_v3/results.png")
    return results


# ════════════════════════════════════════════════
# 4. 성능 평가 + 버전 비교
# ════════════════════════════════════════════════
def evaluate(data_dir: str, weights_path: str = None):
    if weights_path is None:
        weights_path = f"{PROJECT_DIR}/waste_cls_v3/weights/best.pt"

    print("\n" + "=" * 55)
    print("  [성능 평가] Test 세트 Top-1 Accuracy")
    print("=" * 55)

    model   = YOLO(weights_path)
    metrics = model.val(
        data  = data_dir,
        split = "test",
        imgsz = IMG_SIZE,
        plots = True,
    )

    top1 = metrics.top1 * 100
    top5 = metrics.top5 * 100

    print(f"\n📊 v3 성능 결과 (포트폴리오 기재용)")
    print(f"  Top-1 Accuracy : {top1:.1f}%")
    print(f"  Top-5 Accuracy : {top5:.1f}%")
    print()
    print(f"  [버전 비교]")
    print(f"  v1 (yolov8n, epoch 50)  → 기존 측정값과 비교")
    print(f"  v2 (yolov8s, epoch 100) → 기존 측정값과 비교")
    print(f"  v3 (yolov8s, epoch 100, augment++) → {top1:.1f}%  ← 현재")
    print()

    # 클래스별 정확도 출력
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        print(f"  [클래스별 정확도] → Confusion Matrix 파일 참고")
        print(f"  {PROJECT_DIR}/waste_cls_v3/confusion_matrix_normalized.png")

    # 70% 미달 시 개선 가이드 출력
    if top1 < 70:
        print(f"\n  ⚠️  Top-1 < 70% — 아래 조치 권장:")
        print(f"     1. 부족한 클래스 이미지 직접 촬영 추가")
        print(f"     2. epoch 100 → 150 늘리기")
        print(f"     3. yolov8s → yolov8m 모델 업그레이드 시도")
    elif top1 < 85:
        print(f"\n  💡 추가 개선 여지 있음:")
        print(f"     collected_data 500장 달성 후 재학습 권장")
    else:
        print(f"\n  ✅ 목표 정확도 달성! 포트폴리오 제출 가능")

    return metrics


# ════════════════════════════════════════════════
# 5. best.pt 로컬 다운로드 (Colab 전용)
# ════════════════════════════════════════════════
def download_best():
    """Colab에서 로컬 PC로 best.pt 다운로드"""
    try:
        from google.colab import files
        best_path = f"{PROJECT_DIR}/waste_cls_v3/weights/best.pt"
        if Path(best_path).exists():
            files.download(best_path)
            print(f"✅ best.pt 다운로드 시작")
        else:
            print(f"❌ 파일 없음: {best_path}")
    except ImportError:
        print("Colab 환경이 아닙니다. Drive에서 직접 다운로드하세요.")
        print(f"경로: {PROJECT_DIR}/waste_cls_v3/weights/best.pt")


# ════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════
if __name__ == "__main__":

    # Step 1. 클래스 불균형 체크
    check_balance(DATA_DIR)

    # Step 2. collected_data 병합 (있으면 자동 병합, 없으면 원본 그대로)
    final_data_dir = merge_collected_data(
        base_dir      = DATA_DIR,
        collected_dir = COLLECTED_DIR,
        output_dir    = MERGED_DATA_DIR,
    )

    # Step 3. 학습
    train(final_data_dir)

    # Step 4. 성능 평가
    evaluate(final_data_dir)

    # Step 5. best.pt 다운로드 (Colab에서 실행 시 자동 다운로드)
    download_best()
