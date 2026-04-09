"""
[7단계] 손 제스처 인식 (mediapipe 0.10.30+ 호환 버전)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mediapipe 0.10.30+ 에서 mp.solutions 삭제됨
→ mp.tasks.vision.HandLandmarker API로 전면 교체

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step7_gesture.py
설치:      pip install mediapipe  (0.10.30 이상)

제스처:
  ✋ 5손가락 → 분류 시작
  ✊ 주먹    → 일시정지
  ☝ 검지    → 캡처 저장
  ✌ 브이    → 화면 고정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
import time
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
SERVER_URL       = "http://localhost:8000"
PREDICT_URL      = f"{SERVER_URL}/predict"
HEALTH_URL       = f"{SERVER_URL}/health"
CAMERA_INDEX     = 0
CAMERA_BACKEND = cv2.CAP_DSHOW  # Windows DirectShow (MSMF 오류 방지)
SEND_INTERVAL    = 0.8
REQUEST_TIMEOUT  = 3
FONT_PATH        = "C:/Windows/Fonts/malgun.ttf"

# hand_landmarker.task 모델 파일 경로 (자동 다운로드)
LANDMARKER_PATH  = "./hand_landmarker.task"
LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# 클래스별 색상 (BGR)
CLASS_COLORS = {
    "snack_bag":    (0,   165, 255),
    "glass_bottle": (255, 0,   0  ),
    "paper":        (0,   200, 0  ),
    "can":          (0,   0,   255),
    "pet_bottle":   (200, 0,   200),
}
CLASS_LABEL = {
    "snack_bag":    "Snack Bag",
    "glass_bottle": "Glass Bottle",
    "paper":        "Paper / Box",
    "can":          "Can",
    "pet_bottle":   "PET Bottle",
}

# 제스처 정의 (손가락 수 → 동작)
GESTURE = {
    0: ("Fist  PAUSE",   "PAUSE",   (80,  80,  80 )),
    1: ("Index CAPTURE", "CAPTURE", (0,   200, 200)),
    2: ("V     FREEZE",  "FREEZE",  (200, 150, 0  )),
    5: ("Open  START",   "START",   (0,   200, 0  )),
}


# ════════════════════════════════════════════════
# hand_landmarker.task 모델 자동 다운로드
# ════════════════════════════════════════════════
def ensure_landmarker():
    if Path(LANDMARKER_PATH).exists():
        return True
    print(f"  hand_landmarker.task 다운로드 중...")
    try:
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print(f"  ✅ 다운로드 완료: {LANDMARKER_PATH}")
        return True
    except Exception as e:
        print(f"  ❌ 다운로드 실패: {e}")
        print(f"  수동 다운로드: {LANDMARKER_URL}")
        print(f"  → {LANDMARKER_PATH} 에 저장 후 재실행")
        return False


# ════════════════════════════════════════════════
# HandLandmarker 초기화 (최신 API)
# ════════════════════════════════════════════════
def create_hand_landmarker():
    BaseOptions        = mp.tasks.BaseOptions
    HandLandmarker     = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options             = BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode             = VisionRunningMode.IMAGE,
        num_hands                = 1,
        min_hand_detection_confidence = 0.7,
        min_hand_presence_confidence  = 0.7,
        min_tracking_confidence       = 0.6,
    )
    return HandLandmarker.create_from_options(options)


# ════════════════════════════════════════════════
# 손가락 수 계산
# ════════════════════════════════════════════════
def count_fingers(landmarks) -> int:
    """
    NormalizedLandmark 리스트로 펼쳐진 손가락 수 계산
    엄지: x축 비교 / 나머지 4개: y축 비교
    """
    lm = landmarks
    fingers = 0

    # 엄지 (4번 끝 vs 3번 마디)
    if lm[4].x < lm[3].x:
        fingers += 1

    # 검지~소지 (끝이 중간 마디보다 위 = 펼침)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            fingers += 1

    return fingers


# ════════════════════════════════════════════════
# 랜드마크 그리기 (최신 API — draw_landmarks 없음)
# ════════════════════════════════════════════════
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # 엄지
    (0,5),(5,6),(6,7),(7,8),         # 검지
    (0,9),(9,10),(10,11),(11,12),    # 중지
    (0,13),(13,14),(14,15),(15,16),  # 약지
    (0,17),(17,18),(18,19),(19,20),  # 소지
    (5,9),(9,13),(13,17),            # 손바닥
]

def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for i, pt in enumerate(pts):
        color = (0, 200, 0) if i in [4,8,12,16,20] else (0,150,255)
        cv2.circle(frame, pt, 4, color, -1)


# ════════════════════════════════════════════════
# 한글 텍스트 렌더링 (PIL)
# ════════════════════════════════════════════════
def put_text(frame, text, pos, size=28, color=(255, 255, 255)):
    try:
        font    = ImageFont.truetype(FONT_PATH, size)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame


# ════════════════════════════════════════════════
# AI 서버에 프레임 전송
# ════════════════════════════════════════════════
def send_frame(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        res = requests.post(
            PREDICT_URL,
            files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")},
            timeout=REQUEST_TIMEOUT,
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════
def run():
    print("=" * 52)
    print("  손 제스처 + 폐기물 분류 AI  (mediapipe 0.10+)")
    print("=" * 52)

    # 모델 파일 확인
    if not ensure_landmarker():
        return

    # AI 서버 연결 확인
    server_ok = False
    try:
        res = requests.get(HEALTH_URL, timeout=2).json()
        if res.get("model_ready"):
            print("✅ AI 서버 연결 성공")
            server_ok = True
        else:
            print("⚠️  AI 서버 모델 미로드 — step5 먼저 실행하세요")
    except Exception:
        print("⚠️  AI 서버 없음 — 제스처 인식만 단독 실행")

    # HandLandmarker 생성
    print("  HandLandmarker 초기화 중...")
    try:
        landmarker = create_hand_landmarker()
        print("✅ HandLandmarker 준비 완료")
    except Exception as e:
        print(f"❌ HandLandmarker 초기화 실패: {e}")
        return

    # 웹캠
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 웹캠 연결 실패")
        landmarker.close()
        return

    print("✅ 웹캠 연결 성공")
    print("  ✋ 5손가락:분류시작  ✊ 주먹:일시정지")
    print("  ☝ 검지:캡처저장    ✌ 브이:화면고정  Q:종료\n")

    # 상태 변수
    mode              = "START"
    last_send         = 0.0
    last_result       = {}
    frozen_result     = {}
    frame_idx         = 0
    last_gesture_label = "Open  START"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ── 손 감지 (최신 API) ─────────────────────
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        detection = landmarker.detect(mp_image)

        finger_count  = -1
        gesture_label = last_gesture_label

        if detection.hand_landmarks:
            landmarks = detection.hand_landmarks[0]
            draw_landmarks(frame, landmarks)
            finger_count = count_fingers(landmarks)

            if finger_count in GESTURE:
                gesture_label, gesture_cmd, _ = GESTURE[finger_count]
                last_gesture_label = gesture_label
                mode = gesture_cmd

                if gesture_cmd == "CAPTURE":
                    path = f"./gesture_capture_{frame_idx:05d}.jpg"
                    cv2.imwrite(path, frame)
                    print(f"  📸 캡처 저장: {path}")
                    mode = "START"

        # ── AI 분류 ────────────────────────────────
        now = time.time()
        if server_ok and mode == "START" and (now - last_send) >= SEND_INTERVAL:
            last_result = send_frame(frame)
            last_send   = now

        if mode == "FREEZE":
            frozen_result = last_result

        display = frozen_result if mode == "FREEZE" else last_result

        # ── 화면 렌더링 ────────────────────────────
        # 상단 제스처 바
        g_color = (
            (0, 180, 0)   if mode == "START"   else
            (80, 80, 80)  if mode == "PAUSE"   else
            (0, 180, 180) if mode == "CAPTURE" else
            (180, 120, 0)
        )
        cv2.rectangle(frame, (0, 0), (w, 44), g_color, -1)
        frame = put_text(
            frame,
            f"{gesture_label}   fingers: {finger_count if finger_count >= 0 else '-'}",
            (10, 8), size=24, color=(255, 255, 255),
        )

        # 분류 결과 바
        if display and "class_name" in display:
            cls   = display.get("class_name", "")
            conf  = display.get("confidence", 0.0)
            label = CLASS_LABEL.get(cls, cls)
            c_col = CLASS_COLORS.get(cls, (180, 180, 180))
            tag   = "  [FROZEN]" if mode == "FREEZE" else ""
            cv2.rectangle(frame, (0, 44), (w, 96), c_col, -1)
            frame = put_text(
                frame,
                f"{label}  {conf:.0%}{tag}",
                (12, 52), size=32, color=(255, 255, 255),
            )
        elif mode == "PAUSE":
            cv2.rectangle(frame, (0, 44), (w, 96), (60, 60, 60), -1)
            frame = put_text(frame, "PAUSED", (12, 52),
                             size=32, color=(180, 180, 180))

        # 하단 안내 바
        cv2.rectangle(frame, (0, h - 32), (w, h), (20, 20, 20), -1)
        frame = put_text(
            frame,
            "Open:Start  Fist:Pause  Index:Capture  V:Freeze  Q:Quit",
            (8, h - 28), size=17, color=(160, 160, 160),
        )

        cv2.imshow("Waste Classifier + Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    run()
