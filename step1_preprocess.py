"""
[1단계] 데이터 전처리 스크립트 - 회전 증강 버전
이미지를 클래스별 폴더로 정리 + 0°/90°/180°/270° 회전 증강

변경사항:
  - Train 데이터에 회전 증강 적용 (4배 증가)
  - Val/Test는 원본 그대로 유지 (평가 공정성 보장)
  - 기존 dataset 폴더 자동 삭제 후 재생성

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step1_preprocess.py

결과:
  Train: 원본 + 90°/180°/270° 회전본 → 약 6,080장
  Val  : 원본만                       → 약 320장
  Test : 원본만                       → 약 80장
"""

import cv2
import shutil
import random
import numpy as np
from pathlib import Path

# ════════════════════════════════════════════════
# ★ 경로 설정
# ════════════════════════════════════════════════
TRAINING_DIR   = r"C:\Users\lms\Downloads\생활 폐기물 이미지\Training"
VALIDATION_DIR = r"C:\Users\lms\Downloads\생활 폐기물 이미지\Validation"
OUTPUT_ROOT    = "./dataset"

SAMPLE_PER_CLASS     = 300    # Train 클래스당 원본 샘플 수
                               # 회전 증강 후 → 300 × 4 = 1,200장/클래스
VAL_SAMPLE_PER_CLASS = 80     # Validation 클래스당 샘플 수
TEST_RATIO           = 0.2    # Validation 중 Test로 분리
RANDOM_SEED          = 42

# 회전 증강 설정
# Train만 적용, Val/Test는 원본 유지
AUGMENT_ANGLES = [0, 90, 180, 270]   # 적용할 회전 각도

# ════════════════════════════════════════════════
# 클래스 정의
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


# ════════════════════════════════════════════════
# 유틸 함수
# ════════════════════════════════════════════════

def get_class_name(folder_name: str):
    for keyword, cls_name in FOLDER_CLASS_MAP.items():
        if keyword in folder_name:
            return cls_name
    return None


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    이미지를 angle도 회전
    0°   → 원본
    90°  → 시계 반대 방향 90°
    180° → 상하 반전
    270° → 시계 방향 90°
    """
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def collect_images(base_dir: Path, sample_per_class: int):
    """
    base_dir 하위 폴더 탐색 → 클래스별 샘플링
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
        imgs = (list(folder.rglob("*.jpg"))  +
                list(folder.rglob("*.jpeg")) +
                list(folder.rglob("*.png"))  +
                list(folder.rglob("*.JPG"))  +
                list(folder.rglob("*.JPEG")) +
                list(folder.rglob("*.PNG")))
        class_buckets[cls_name].extend(imgs)

    random.seed(RANDOM_SEED)
    result = []
    print()
    for cls_name in CLASS_NAMES:
        imgs = class_buckets[cls_name]
        n    = min(sample_per_class, len(imgs))
        sampled = random.sample(imgs, n) if n > 0 else []
        print(f"  [{cls_name:15s}] 전체 {len(imgs):5d}개 → {n}개 샘플링")
        for img_path in sampled:
            result.append((img_path, cls_name))

    return result


def save_with_augment(data, split: str):
    """
    Train: 원본 + 90°/180°/270° 회전본 저장 (4배 증강)
    Val/Test: 원본만 저장
    """
    is_train = (split == "train")
    angles   = AUGMENT_ANGLES if is_train else [0]
    total    = 0
    errors   = 0

    print(f"[{split}] {'회전 증강 포함 ' if is_train else ''}저장 중...")
    print(f"  원본 {len(data)}개 × {len(angles)}각도 = 최대 {len(data)*len(angles)}개")

    idx = 0
    for (img_path, cls_name) in data:
        # 한글 경로 대응 이미지 읽기 (numpy 바이트 읽기)
        try:
            with open(str(img_path), "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            img = None

        if img is None:
            errors += 1
            continue

        for angle in angles:
            rotated   = rotate_image(img, angle)
            angle_tag = f"_r{angle:03d}" if angle != 0 else ""
            dst_name  = f"{cls_name}_{idx:05d}{angle_tag}.jpg"
            dst_path  = Path(f"{OUTPUT_ROOT}/{split}/{cls_name}/{dst_name}")

            # 한글 경로 대응 이미지 저장
            ret, encoded = cv2.imencode(".jpg", rotated,
                                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ret:
                with open(str(dst_path), "wb") as f:
                    f.write(encoded.tobytes())
                total += 1

        idx += 1

    print(f"  → 저장 완료: {total}개  (오류 스킵: {errors}개)\n")
    return total


# ════════════════════════════════════════════════
# 메인 파이프라인
# ════════════════════════════════════════════════

def build_dataset():
    print("=" * 55)
    print("  [1단계] 데이터 전처리 + 회전 증강")
    print("=" * 55)
    print(f"  Training   : {TRAINING_DIR}")
    print(f"  Validation : {VALIDATION_DIR}")
    print(f"  출력        : {OUTPUT_ROOT}")
    print(f"  회전 각도   : {AUGMENT_ANGLES} (Train만 적용)")
    print()

    # 기존 dataset 폴더 삭제 후 재생성
    out = Path(OUTPUT_ROOT)
    if out.exists():
        print("  기존 dataset 폴더 삭제 중...")
        shutil.rmtree(out)

    for split in ["train", "val", "test"]:
        for cls_name in CLASS_NAMES:
            Path(f"{OUTPUT_ROOT}/{split}/{cls_name}").mkdir(
                parents=True, exist_ok=True)

    # ── Train 수집 ────────────────────────────────
    print("[Train 이미지 수집]")
    train_data = collect_images(Path(TRAINING_DIR), SAMPLE_PER_CLASS)
    random.shuffle(train_data)

    # ── Validation → Val + Test 분리 ──────────────
    print("\n[Validation 이미지 수집]")
    val_all   = collect_images(Path(VALIDATION_DIR), VAL_SAMPLE_PER_CLASS)
    random.shuffle(val_all)

    n_test    = int(len(val_all) * TEST_RATIO)
    test_data = val_all[:n_test]
    val_data  = val_all[n_test:]

    print(f"원본 데이터셋 구성:")
    print(f"  Train 원본 : {len(train_data)}개")
    print(f"  Val        : {len(val_data)}개")
    print(f"  Test       : {len(test_data)}개")
    print(f"\n증강 후 예상 Train: {len(train_data) * len(AUGMENT_ANGLES)}개")
    print()

    # ── 저장 (Train은 증강, Val/Test는 원본) ──────
    t_total = save_with_augment(train_data, "train")
    v_total = save_with_augment(val_data,   "val")
    e_total = save_with_augment(test_data,  "test")

    print("=" * 55)
    print("✅ 전처리 완료!")
    print(f"   Train : {t_total}장 (원본 {len(train_data)}장 × 4각도)")
    print(f"   Val   : {v_total}장")
    print(f"   Test  : {e_total}장")
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
        print("\n▶ 경로 확인 후 다시 실행하세요.")
    else:
        build_dataset()
