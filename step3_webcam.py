"""
[3단계] 웹캠 실시간 추론 스크립트
학습된 모델로 웹캠에서 실시간 폐기물 감지

실행 환경: PyCharm 로컬 (Windows 10) ← 반드시 로컬에서 실행
실행 방법: python step3_webcam.py
준비물: step2 학습 후 best.pt 파일을 로컬에 복사해올 것
"""

import cv2
from ultralytics import YOLO
from pathlib import Path

# ────────────────────────────────────────────────
# 설정값
# ────────────────────────────────────────────────
WEIGHTS_PATH = "./best.pt"   # Colab에서 다운받은 best.pt 경로
CAMERA_INDEX = 0             # 웹캠 번호 (보통 0, 안되면 1로 변경)
CONF_THRESHOLD = 0.5         # 신뢰도 임계값 (0~1, 높을수록 엄격)
IMG_SIZE = 640

# 클래스별 색상 (BGR 포맷)
CLASS_COLORS = {
    0: (0,   165, 255),  # snack_bag   - 주황
    1: (255, 0,   0  ),  # glass_bottle - 파랑
    2: (0,   200, 0  ),  # paper       - 초록
    3: (0,   0,   255),  # can         - 빨강
    4: (255, 0,   255),  # pet_bottle  - 보라
}

CLASS_KOR = {
    0: "과자봉지/비닐",
    1: "유리병/음료수병",
    2: "종이류/포장상자",
    3: "캔/음료수캔",
    4: "페트병",
}


def draw_boxes(frame, results):
    """
    프레임에 바운딩박스 + 클래스명 + 신뢰도 표시
    """
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_KOR.get(cls_id, cls_id)} {conf:.0%}"

        # 바운딩박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 라벨 배경
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)

        # 라벨 텍스트
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def run_webcam():
    print("=" * 50)
    print("[3단계] 웹캠 실시간 추론 시작")
    print("=" * 50)

    if not Path(WEIGHTS_PATH).exists():
        print(f"❌ 오류: {WEIGHTS_PATH} 파일이 없습니다.")
        print("   Colab에서 best.pt를 다운받아 이 폴더에 넣어주세요.")
        return

    model = YOLO(WEIGHTS_PATH)
    cap   = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"❌ 웹캠을 열 수 없습니다. CAMERA_INDEX={CAMERA_INDEX} 확인하세요.")
        return

    print("✅ 웹캠 연결 성공!")
    print("   [Q] 종료  |  [S] 현재 프레임 저장")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        # 추론
        results = model.predict(
            source    = frame,
            imgsz     = IMG_SIZE,
            conf      = CONF_THRESHOLD,
            verbose   = False,
        )

        # 바운딩박스 그리기
        frame = draw_boxes(frame, results)

        # FPS 표시
        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 화면 출력
        cv2.imshow("폐기물 분류 AI - Q:종료 / S:저장", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("종료합니다.")
            break
        elif key == ord("s"):
            save_path = f"./capture_{frame_count}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"📸 저장됨: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
