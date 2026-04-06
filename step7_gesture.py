"""
[7단계] 손 제스처 인식 모델 (두 번째 AI 모델 - 요구사항 #9)
MediaPipe로 손 랜드마크 추출 → 제스처 분류 → 폐기물 분류 AI 제어

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step7_gesture.py
설치 필요: pip install mediapipe

제스처 종류:
  ✋ 손 펼치기 (5손가락)  → 분류 시작
  ✊ 주먹 (0손가락)        → 분류 일시정지
  ☝️ 검지만 세우기 (1손가락) → 현재 프레임 캡처 저장
  ✌️ 브이 (2손가락)        → 최근 결과 화면 고정
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
import time
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
SERVER_URL      = "http://localhost:8000"
PREDICT_URL     = f"{SERVER_URL}/predict"
HEALTH_URL      = f"{SERVER_URL}/health"
CAMERA_INDEX    = 0
SEND_INTERVAL   = 0.8     # 제스처 인식 중엔 전송 간격 늘림
REQUEST_TIMEOUT = 3
FONT_PATH       = "C:/Windows/Fonts/malgun.ttf"

# 클래스별 색상 (BGR)
CLASS_COLORS = {
    "snack_bag":    (0,   165, 255),
    "glass_bottle": (255, 0,   0  ),
    "paper":        (0,   200, 0  ),
    "can":          (0,   0,   255),
    "pet_bottle":   (200, 0,   200),
}
CLASS_LABEL = {
    "snack_bag":    "과자봉지 / 비닐",
    "glass_bottle": "유리병 / 음료수병",
    "paper":        "종이류 / 포장상자",
    "can":          "캔 / 음료수캔",
    "pet_bottle":   "페트병",
}

# 제스처 정의
GESTURE = {
    0: ("✊ 일시정지",  "PAUSE",   (100, 100, 100)),
    1: ("☝️ 캡처 저장", "CAPTURE", (0,   200, 200)),
    2: ("✌️ 화면 고정", "FREEZE",  (200, 150, 0  )),
    5: ("✋ 분류 시작", "START",   (0,   200, 0  )),
}


# ════════════════════════════════════════════════
# MediaPipe 초기화
# ════════════════════════════════════════════════
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode       = False,
    max_num_hands           = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence  = 0.6,
)


def count_fingers(hand_landmarks) -> int:
    """
    손 랜드마크로 펼쳐진 손가락 수 계산
    엄지: x축 비교 / 나머지: y축 비교 (손끝 vs 두 번째 마디)
    """
    lm = hand_landmarks.landmark
    fingers = 0

    # 엄지 (4번 끝, 3번 마디)
    if lm[4].x < lm[3].x:
        fingers += 1

    # 검지~소지 (끝 마디가 중간 마디보다 위에 있으면 펼침)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            fingers += 1

    return fingers


def put_kor(frame, text, pos, size=32, color=(255,255,255)):
    """PIL로 한글 렌더링"""
    try:
        font    = ImageFont.truetype(FONT_PATH, size)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame


def send_frame(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        res = requests.post(PREDICT_URL,
                            files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")},
                            timeout=REQUEST_TIMEOUT)
        return res.json()
    except Exception as e:
        return {"error": str(e)}


def run():
    print("=" * 50)
    print("  손 제스처 + 폐기물 분류 AI")
    print("=" * 50)

    # 서버 연결 확인
    try:
        res = requests.get(HEALTH_URL, timeout=2).json()
        if not res.get("model_ready"):
            print("⚠️  AI 서버 모델 미로드 — step5_api_server.py 먼저 실행하세요.")
            return
        print("✅ AI 서버 연결 성공")
    except Exception:
        print("⚠️  AI 서버 없음 — 제스처 인식만 단독 실행합니다.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ 웹캠 연결 실패")
        return

    print("✅ 웹캠 연결 성공")
    print("  ✋ 5손가락: 분류 시작  ✊ 주먹: 일시정지")
    print("  ☝️ 검지: 캡처 저장   ✌️ 브이: 화면 고정\n")

    # 상태 변수
    mode        = "START"      # START / PAUSE / FREEZE / CAPTURE
    last_send   = 0.0
    last_result = {}
    frozen_result = {}
    frame_idx   = 0
    last_gesture_name = "✋ 분류 시작"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── 손 감지 ──────────────────────────────────
        result_hands = hands.process(rgb)
        finger_count = -1
        gesture_name = last_gesture_name
        gesture_cmd  = mode

        if result_hands.multi_hand_landmarks:
            for hlm in result_hands.multi_hand_landmarks:
                # 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,200,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
                )
                finger_count = count_fingers(hlm)

            if finger_count in GESTURE:
                gesture_name, gesture_cmd, _ = GESTURE[finger_count]
                last_gesture_name = gesture_name
                mode = gesture_cmd

                # 캡처 저장 1회성 처리
                if gesture_cmd == "CAPTURE":
                    path = f"./gesture_capture_{frame_idx:05d}.jpg"
                    cv2.imwrite(path, frame)
                    print(f"📸 캡처 저장: {path}")
                    mode = "START"   # 저장 후 다시 START로

        # ── AI 분류 (START 모드일 때만) ──────────────
        now = time.time()
        if mode == "START" and (now - last_send) >= SEND_INTERVAL:
            last_result = send_frame(frame)
            last_send   = now

        if mode == "FREEZE":
            frozen_result = last_result   # 마지막 결과 고정

        display_result = frozen_result if mode == "FREEZE" else last_result

        # ── 화면 렌더링 ──────────────────────────────
        # 상단 제스처 상태 바
        g_color = (0,200,0) if mode=="START" else (100,100,100) if mode=="PAUSE" else (0,200,200)
        cv2.rectangle(frame, (0,0), (w, 48), g_color, -1)
        frame = put_kor(frame, f"{gesture_name}  |  손가락: {finger_count if finger_count>=0 else '-'}개",
                        (10, 8), size=26, color=(255,255,255))

        # 분류 결과 바
        if display_result and "class_name" in display_result:
            cls_name = display_result.get("class_name","")
            conf     = display_result.get("confidence", 0.0)
            label    = CLASS_LABEL.get(cls_name, cls_name)
            c_color  = CLASS_COLORS.get(cls_name, (200,200,200))

            frozen_tag = "  [고정]" if mode=="FREEZE" else ""
            cv2.rectangle(frame, (0, 48), (w, 100), c_color, -1)
            frame = put_kor(frame, f"{label}  {conf:.0%}{frozen_tag}",
                            (12, 55), size=34, color=(255,255,255))

        elif mode == "PAUSE":
            cv2.rectangle(frame, (0, 48), (w, 100), (80,80,80), -1)
            frame = put_kor(frame, "⏸ 분류 일시정지",
                            (12, 55), size=34, color=(200,200,200))

        # 하단 안내 바
        cv2.rectangle(frame, (0, h-36), (w, h), (30,30,30), -1)
        frame = put_kor(frame, "✋분류시작  ✊일시정지  ☝️캡처  ✌️화면고정  Q:종료",
                        (8, h-30), size=19, color=(180,180,180))

        cv2.imshow("Waste Classifier + Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    run()
