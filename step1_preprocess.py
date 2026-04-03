"""
[1단계] 데이터 전처리 스크립트
AI 허브 데이터셋 → YOLO 포맷 변환 + Train/Val/Test 분할

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step1_preprocess.py
"""

import os
import json
import shutil
import random
from pathlib import Path

# ────────────────────────────────────────────────
# 설정값 (본인 경로에 맞게 수정)
# ────────────────────────────────────────────────
TRAINING_DIR   = r"C:\Users\lms\Downloads\생활 폐기물 이미지\Training"
VALIDATION_DIR = r"C:\Users\lms\Downloads\생활 폐기물 이미지\Validation"
OUTPUT_ROOT   = "./dataset"           # YOLO 포맷으로 변환된 결과 폴더
SAMPLE_RATIO  = 0.15                  # 전체 데이터 중 사용할 비율 (15%)
SPLIT_RATIO   = (0.8, 0.1, 0.1)      # Train / Val / Test 비율

# 5개 클래스 정의 (AI 허브 카테고리명 → 우리 클래스명)
CLASS_MAP = {
    "비닐_과자봉지": 0,   # snack_bag (비닐/과자봉지/과자봉지)
    "과자봉지":      0,
    "유리병":        1,   # glass_bottle (유리병/음료수병)
    "음료수병":      1,
    "종이류":        2,   # paper (종이류/포장상자)
    "포장상자":      2,
    "캔":            3,   # can (캔/음료수캔)
    "음료수캔":      3,
    "페트병":        4,   # pet_bottle (페트병)
}

CLASS_NAMES = ["snack_bag", "glass_bottle", "paper", "can", "pet_bottle"]


def convert_bbox_to_yolo(img_w, img_h, x1, y1, x2, y2):
    """
    AI 허브 바운딩박스(절대좌표) → YOLO 포맷(상대좌표) 변환
    YOLO 포맷: cx cy w h (모두 0~1 사이 비율값)
    """
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def parse_aihub_json(json_path):
    """
    AI 허브 JSON 라벨 파일 파싱
    반환값: [(class_id, cx, cy, w, h), ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = []
    img_w = data["images"]["width"]
    img_h = data["images"]["height"]

    for ann in data.get("annotations", []):
        category = ann.get("category", "")
        if category not in CLASS_MAP:
            continue  # 우리 클래스가 아닌 항목은 스킵

        class_id = CLASS_MAP[category]
        bbox = ann["bbox"]           # [x1, y1, x2, y2] 형식
        cx, cy, w, h = convert_bbox_to_yolo(img_w, img_h, *bbox)
        annotations.append((class_id, cx, cy, w, h))

    return annotations


def build_dataset():
    """
    전체 데이터 전처리 파이프라인
    1. JSON 라벨 파싱
    2. 샘플링
    3. Train/Val/Test 분할
    4. YOLO 포맷 저장
    """
    print("=" * 50)
    print("[1단계] 데이터 전처리 시작")
    print("=" * 50)

    # 출력 폴더 생성
    for split in ["train", "val", "test"]:
        Path(f"{OUTPUT_ROOT}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_ROOT}/labels/{split}").mkdir(parents=True, exist_ok=True)

    # 이미지-라벨 쌍 수집
    all_pairs = []  # [(img_path, json_path), ...]
    raw_path = Path(RAW_DATA_ROOT)

    for img_path in raw_path.rglob("*.jpg"):
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            all_pairs.append((img_path, json_path))

    print(f"총 발견된 이미지-라벨 쌍: {len(all_pairs)}개")

    # 샘플링
    random.seed(42)
    sample_size = int(len(all_pairs) * SAMPLE_RATIO)
    sampled = random.sample(all_pairs, sample_size)
    print(f"샘플링 후: {len(sampled)}개 ({SAMPLE_RATIO*100:.0f}%)")

    # Train/Val/Test 분할
    random.shuffle(sampled)
    n = len(sampled)
    n_train = int(n * SPLIT_RATIO[0])
    n_val   = int(n * SPLIT_RATIO[1])

    splits = {
        "train": sampled[:n_train],
        "val":   sampled[n_train:n_train + n_val],
        "test":  sampled[n_train + n_val:],
    }

    # YOLO 포맷으로 저장
    for split_name, pairs in splits.items():
        print(f"\n[{split_name}] {len(pairs)}개 처리 중...")
        skip_count = 0

        for img_path, json_path in pairs:
            try:
                annotations = parse_aihub_json(json_path)
                if not annotations:
                    skip_count += 1
                    continue

                # 이미지 복사
                dst_img = f"{OUTPUT_ROOT}/images/{split_name}/{img_path.name}"
                shutil.copy(img_path, dst_img)

                # YOLO 라벨 저장 (.txt)
                dst_label = f"{OUTPUT_ROOT}/labels/{split_name}/{img_path.stem}.txt"
                with open(dst_label, "w") as f:
                    for cls_id, cx, cy, w, h in annotations:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            except Exception as e:
                print(f"  오류 스킵: {img_path.name} → {e}")
                skip_count += 1

        print(f"  완료 ({skip_count}개 스킵)")

    # data.yaml 생성 (YOLOv8 학습에 필요)
    yaml_path = f"{OUTPUT_ROOT}/data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        abs_path = str(Path(OUTPUT_ROOT).resolve()).replace("\\", "/")
        f.write(f"path: {abs_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n")
        f.write(f"test:  images/test\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\n✅ data.yaml 생성 완료: {yaml_path}")
    print("\n[1단계] 전처리 완료!")


if __name__ == "__main__":
    build_dataset()
