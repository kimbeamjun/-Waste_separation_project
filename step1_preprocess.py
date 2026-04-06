"""
[1단계] 데이터 전처리 스크립트 - Classification 버전
이미지를 클래스별 폴더로 정리 (YOLOv8 Classification 포맷)

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step1_preprocess.py

결과 폴더 구조:
dataset/
├── train/
│   ├── snack_bag/
│   ├── glass_bottle/
│   ├── paper/
│   ├── can/
│   └── pet_bottle/
├── val/
│   └── (동일 구조)
└── test/
    └── (동일 구조)
"""

import shutil
import random
from pathlib import Path

# ════════════════════════════════════════════════
# ★ 경로 설정
# ════════════════════════════════════════════════
TRAINING_DIR   = r"C:\Users\lms\Downloads\생활 폐기물 이미지\Training"
VALIDATION_DIR = r"C:\Users\lms\Downloads\생활 폐기물 이미지\Validation"
OUTPUT_ROOT    = "./dataset"

SAMPLE_PER_CLASS     = 300   # Train 클래스당 최대 이미지 수
VAL_SAMPLE_PER_CLASS = 80    # Validation 클래스당 최대 이미지 수
TEST_RATIO           = 0.2   # Validation 중 Test로 분리할 비율
RANDOM_SEED          = 42

# ════════════════════════════════════════════════
# 클래스 정의 (폴더명 키워드 → 클래스명)
# ════════════════════════════════════════════════
FOLDER_CLASS_MAP = {
    "비닐":    "snack_bag",
    "과자봉지": "snack_bag",
    "유리병":  "glass_bottle",
    "음료수병": "glass_bottle",
    "종이류":  "paper",
    "포장상자": "paper",
    "캔":      "can",
    "음료수캔": "can",
    "페트병":  "pet_bottle",
}

CLASS_NAMES = ["snack_bag", "glass_bottle", "paper", "can", "pet_bottle"]


def get_class_name(folder_name: str):
    """폴더명 키워드로 클래스명 반환"""
    for keyword, cls_name in FOLDER_CLASS_MAP.items():
        if keyword in folder_name:
            return cls_name
    return None


def collect_images(base_dir: Path, sample_per_class: int):
    """
    base_dir 하위 폴더를 탐색해서 클래스별로 이미지 경로 수집 후 샘플링
    반환: [(img_path, class_name), ...]
    """
    class_buckets = {name: [] for name in CLASS_NAMES}

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        cls_name = get_class_name(folder.name)
        if cls_name is None:
            print(f"  스킵 (매핑 없음): {folder.name}")
            continue

        print(f"  탐색: {folder.name} → [{cls_name}]")

        # 하위 폴더까지 재귀 탐색
        imgs = (list(folder.rglob("*.jpg")) +
                list(folder.rglob("*.jpeg")) +
                list(folder.rglob("*.png")) +
                list(folder.rglob("*.JPG")) +
                list(folder.rglob("*.JPEG")) +
                list(folder.rglob("*.PNG")))

        class_buckets[cls_name].extend(imgs)

    # 클래스별 샘플링
    random.seed(RANDOM_SEED)
    result = []
    print()
    for cls_name in CLASS_NAMES:
        imgs = class_buckets[cls_name]
        n = min(sample_per_class, len(imgs))
        sampled = random.sample(imgs, n) if n > 0 else []
        print(f"  [{cls_name:15s}] 전체 {len(imgs):5d}개 → {n}개 샘플링")
        for img_path in sampled:
            result.append((img_path, cls_name))

    return result


def build_dataset():
    print("=" * 55)
    print("  [1단계] 데이터 전처리 시작 (Classification)")
    print("=" * 55)
    print(f"  Training   : {TRAINING_DIR}")
    print(f"  Validation : {VALIDATION_DIR}")
    print(f"  출력        : {OUTPUT_ROOT}")
    print()

    # 출력 폴더 생성 (클래스별 하위 폴더 포함)
    for split in ["train", "val", "test"]:
        for cls_name in CLASS_NAMES:
            Path(f"{OUTPUT_ROOT}/{split}/{cls_name}").mkdir(parents=True, exist_ok=True)

    # ── Train 수집 ────────────────────────────────
    print("[Train 이미지 수집]")
    train_data = collect_images(Path(TRAINING_DIR), SAMPLE_PER_CLASS)
    random.shuffle(train_data)

    # ── Validation → Val + Test 분리 ──────────────
    print("\n[Validation 이미지 수집]")
    val_all = collect_images(Path(VALIDATION_DIR), VAL_SAMPLE_PER_CLASS)
    random.shuffle(val_all)

    n_test    = int(len(val_all) * TEST_RATIO)
    test_data = val_all[:n_test]
    val_data  = val_all[n_test:]

    print(f"\n최종 데이터셋 구성:")
    print(f"  Train : {len(train_data)}개")
    print(f"  Val   : {len(val_data)}개")
    print(f"  Test  : {len(test_data)}개")
    print()

    # ── 이미지 복사 ───────────────────────────────
    for split_name, data in [("train", train_data),
                              ("val",   val_data),
                              ("test",  test_data)]:
        print(f"[{split_name}] 복사 중... ({len(data)}개)")
        for idx, (img_path, cls_name) in enumerate(data):
            # 파일명 충돌 방지: 클래스명_인덱스.확장자
            dst = Path(f"{OUTPUT_ROOT}/{split_name}/{cls_name}/"
                       f"{cls_name}_{idx:05d}{img_path.suffix.lower()}")
            shutil.copy2(img_path, dst)
        print(f"  → 완료\n")

    print("=" * 55)
    print("✅ 전처리 완료!")
    print(f"   결과 폴더: {Path(OUTPUT_ROOT).resolve()}")
    print()
    print("다음 단계:")
    print("  1. dataset/ 폴더를 Google Drive에 업로드")
    print("  2. step2_train.py 를 Colab에서 실행")
    print("=" * 55)


if __name__ == "__main__":
    errors = []
    if not Path(TRAINING_DIR).exists():
        errors.append(f"❌ Training 폴더 없음:\n   {TRAINING_DIR}")
    if not Path(VALIDATION_DIR).exists():
        errors.append(f"❌ Validation 폴더 없음:\n   {VALIDATION_DIR}")

    if errors:
        print("\n".join(errors))
        print("\n▶ 경로를 확인하고 ZIP 압축 해제 후 다시 실행하세요.")
    else:
        build_dataset()
