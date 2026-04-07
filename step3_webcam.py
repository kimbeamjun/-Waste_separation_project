"""
[3단계 - Detection 버전] 웹캠 클라이언트
YOLOv8 Detection 모델로 여러 폐기물을 동시에 감지 + 바운딩박스 표시

방식: YOLOv8n (COCO 사전학습) Detection
  - 별도 학습 없이 바로 사용 가능
  - 여러 물체 동시 감지 + 정확한 바운딩박스
  - COCO 클래스 → 폐기물 클래스 매핑

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step3_webcam.py
사전 조건: step5_api_server.py 실행 중 (통계/수집용)
           yolov8n.pt 자동 다운로드 (첫 실행 시)
"""

import cv2
import numpy as np
import time
import requests
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pathlib import Path

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
DETECTION_MODEL  = "yolov8n.pt"    # COCO 사전학습 Detection 모델 (자동 다운로드)
CAMERA_INDEX     = 0
CONF_THRESHOLD   = 0.45            # 감지 최소 신뢰도
IOU_THRESHOLD    = 0.45            # NMS IoU 임계값
IMG_SIZE         = 640
FONT_PATH        = "C:/Windows/Fonts/malgun.ttf"

# AI 서버 (통계/수집용 — 없어도 Detection 자체는 동작)
SERVER_URL       = "http://localhost:8000"
HEALTH_URL       = f"{SERVER_URL}/health"

# ════════════════════════════════════════════════
# COCO 클래스 → 폐기물 클래스 매핑
# COCO 80개 클래스 중 폐기물 관련 항목만 선별
# ════════════════════════════════════════════════
COCO_TO_WASTE = {
    # 병류 → glass_bottle / pet_bottle
    "bottle":       ("pet_bottle",   "페트병"),

    # 캔류
    "cup":          ("can",          "캔/컵"),

    # 종이/박스류
    "book":         ("paper",        "종이류"),
    "suitcase":     ("paper",        "포장상자"),

    # 음식 포장 관련
    "banana":       ("snack_bag",    "과자봉지류"),
    "apple":        ("snack_bag",    "과자봉지류"),
    "orange":       ("snack_bag",    "과자봉지류"),
    "sandwich":     ("snack_bag",    "과자봉지류"),
    "pizza":        ("snack_bag",    "과자봉지류"),
    "cake":         ("snack_bag",    "과자봉지류"),

    # 기타 용기
    "bowl":         ("can",          "캔/그릇"),
    "vase":         ("glass_bottle", "유리병"),
    "wine glass":   ("glass_bottle", "유리병/음료수병"),
    "sports ball":  ("pet_bottle",   "페트병"),
    "cell phone":   ("snack_bag",    "비닐류"),
    "remote":       ("paper",        "종이류"),
    "scissors":     ("can",          "금속류"),
    "toothbrush":   ("snack_bag",    "비닐류"),
    "fork":         ("can",          "금속류"),
    "knife":        ("can",          "금속류"),
    "spoon":        ("can",          "금속류"),
}

# 폐기물 클래스별 색상 (BGR)
WASTE_COLORS = {
    "pet_bottle":   (200, 80,  200),   # 보라
    "glass_bottle": (255, 80,  80 ),   # 파랑
    "can":          (50,  50,  255),   # 빨강
    "paper":        (50,  200, 50 ),   # 초록
    "snack_bag":    (0,   165, 255),   # 주황
}

# 매핑 안된 COCO 클래스도 표시할지 여부
SHOW_UNMAPPED = False   # True: 모든 COCO 객체 표시 / False: 폐기물만 표시


# ════════════════════════════════════════════════
# 유틸 함수
# ════════════════════════════════════════════════

def put_kor(frame, text, pos, size=28, color=(255,255,255), bg=None):
    """PIL로 한글 텍스트 렌더링"""
    try:
        font    = ImageFont.truetype(FONT_PATH, size)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)
        if bg is not None:
            bbox = draw.textbbox(pos, text, font=font)
            pad  = 5
            draw.rectangle(
                [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                fill=(bg[2], bg[1], bg[0])
            )
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame


def draw_box(frame, x1, y1, x2, y2, color, label, conf):
    """바운딩박스 + 모서리 강조 + 라벨 그리기"""

    # 반투명 내부 채우기
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # 테두리
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    # 모서리 강조선
    L, T = 22, 3
    corners = [
        ((x1,y1),(x1+L,y1)), ((x1,y1),(x1,y1+L)),
        ((x2,y1),(x2-L,y1)), ((x2,y1),(x2,y1+L)),
        ((x1,y2),(x1+L,y2)), ((x1,y2),(x1,y2-L)),
        ((x2,y2),(x2-L,y2)), ((x2,y2),(x2,y2-L)),
    ]
    for p1, p2 in corners:
        cv2.line(frame, p1, p2, color, T)

    # 라벨 텍스트
    label_text = f"{label}  {conf:.0%}"
    lbl_y = max(y1 - 6, 36)
    frame = put_kor(frame, label_text, (x1+4, lbl_y-30),
                    size=24, color=(255,255,255), bg=color)

    return frame


def draw_summary(frame, detected_classes):
    """우측 상단 감지 요약 패널"""
    h, w = frame.shape[:2]
    if not detected_classes:
        return frame

    items     = list(detected_classes.items())  # {waste_cls: (kor, count)}
    panel_w   = 260
    panel_h   = 38 + len(items) * 34
    x0, y0   = w - panel_w - 10, 10

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+panel_w, y0+panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x0,y0), (x0+panel_w, y0+panel_h), (70,70,70), 1)

    total = sum(v[1] for v in items)
    frame = put_kor(frame, f"감지된 폐기물  {total}개",
                    (x0+10, y0+6), size=20, color=(160,160,160))

    for i, (cls, (kor, count)) in enumerate(items):
        color = WASTE_COLORS.get(cls, (200,200,200))
        y     = y0 + 38 + i * 34
        cv2.circle(frame, (x0+14, y+12), 8, color, -1)
        frame = put_kor(frame, f"{kor}  ×{count}",
                        (x0+30, y), size=22, color=(220,220,220))

    return frame


def check_server():
    """AI 서버 연결 확인 (없어도 Detection 자체는 동작)"""
    try:
        res = requests.get(HEALTH_URL, timeout=1).json()
        if res.get("model_ready"):
            print(f"  ✅ AI 서버 연결: {SERVER_URL} (통계 수집 활성화)")
            return True
    except Exception:
        pass
    print(f"  ⚠️  AI 서버 없음 — Detection만 단독 실행")
    return False


# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════

def run():
    print("=" * 55)
    print("  폐기물 복합 감지 AI - Detection 버전")
    print("  (YOLOv8 COCO 사전학습 모델)")
    print("=" * 55)

    # Detection 모델 로드
    print(f"\n  Detection 모델 로딩: {DETECTION_MODEL}")
    print("  (첫 실행 시 자동 다운로드 — 약 6MB)")
    model = YOLO(DETECTION_MODEL)
    print("  ✅ 모델 로드 완료\n")

    # AI 서버 연결 확인 (선택사항)
    check_server()

    # 웹캠 연결
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ 웹캠 연결 실패 (CAMERA_INDEX={CAMERA_INDEX})")
        return

    print("\n✅ 웹캠 연결 성공")
    print("  [Q] 종료  |  [S] 저장  |  [A] 전체 클래스 토글\n")

    frame_idx    = 0
    show_all     = SHOW_UNMAPPED

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ── Detection 추론 ──────────────────────
        results = model.predict(
            source    = frame,
            imgsz     = IMG_SIZE,
            conf      = CONF_THRESHOLD,
            iou       = IOU_THRESHOLD,
            verbose   = False,
        )

        # ── 결과 파싱 ──────────────────────────
        display        = frame.copy()
        detected_cls   = {}   # {waste_cls: (kor_name, count)}

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf            = float(box.conf[0])
            cls_id          = int(box.cls[0])
            coco_name       = model.names[cls_id]

            if coco_name in COCO_TO_WASTE:
                # 폐기물 클래스로 매핑
                waste_cls, kor = COCO_TO_WASTE[coco_name]
                color          = WASTE_COLORS.get(waste_cls, (200,200,200))
                display        = draw_box(display, x1,y1,x2,y2, color, kor, conf)

                # 요약 패널용 집계
                if waste_cls not in detected_cls:
                    detected_cls[waste_cls] = (kor, 0)
                detected_cls[waste_cls] = (kor, detected_cls[waste_cls][1] + 1)

            elif show_all:
                # 매핑 안된 COCO 클래스도 회색으로 표시
                display = draw_box(display, x1,y1,x2,y2,
                                   (120,120,120), coco_name, conf)

        # ── UI 렌더링 ──────────────────────────
        # 우측 요약 패널
        display = draw_summary(display, detected_cls)

        # 감지 없을 때 안내
        if not detected_cls:
            display = put_kor(display, "폐기물을 카메라 앞에 놓아주세요",
                              (10, 14), size=26, color=(130,130,130))

        # 하단 상태 바
        mode_txt = "전체표시 ON" if show_all else "폐기물만"
        cv2.rectangle(display, (0,h-36), (w,h), (20,20,20), -1)
        display = put_kor(display,
                          f"모드: {mode_txt}  |  Q:종료  S:저장  A:전체클래스토글",
                          (10, h-30), size=19, color=(150,150,150))

        cv2.imshow("Waste Classifier - Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("종료합니다.")
            break
        elif key == ord("s"):
            path = f"./capture_{frame_idx:05d}.jpg"
            cv2.imwrite(path, display)
            print(f"저장: {path}")
        elif key == ord("a"):
            show_all = not show_all
            print(f"전체 클래스 표시: {'ON' if show_all else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
