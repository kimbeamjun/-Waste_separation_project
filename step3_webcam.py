"""
[3+7 통합] 웹캠 Detection + 손 제스처 제어
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step3 기능 (Detection + LED + 캡처) 그대로 유지 +
step7 기능 (손 제스처 제어) 추가

제스처:
  ✋ 5손가락 → 분류 시작 (기본 상태)
  ✊ 주먹    → 분류 일시정지 (LED OFF)
  ☝  검지   → 현재 프레임 캡처 저장
  ✌  브이   → 화면 고정 (마지막 결과 유지)

기존 키보드:
  Q → 종료
  S → 저장
  A → 전체 클래스 토글
  G → 제스처 인식 ON/OFF 토글

실행 방법: python step3_webcam.py
설치 필요: pip install mediapipe ultralytics
사전 조건: step5_api_server.py 실행 중
           hand_landmarker.task (자동 다운로드)
"""

import cv2
import threading
import numpy as np
import requests
import time
import urllib.request
import mediapipe as mp
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
DETECTION_MODEL  = "yolov8n.pt"
CAMERA_INDEX     = 0
CONF_THRESHOLD   = 0.45
IOU_THRESHOLD    = 0.45
IMG_SIZE         = 640
FONT_PATH        = "C:/Windows/Fonts/malgun.ttf"
SAVE_DIR         = "./collected_data/captures"

SERVER_URL  = "http://localhost:8000"
HEALTH_URL  = f"{SERVER_URL}/health"
MULTI_URL   = f"{SERVER_URL}/detect_multi"
STATE_URL   = f"{SERVER_URL}/webcam/state"
DONE_URL    = f"{SERVER_URL}/webcam/capture_done"

LANDMARKER_PATH = "./hand_landmarker.task"
LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ════════════════════════════════════════════════
# COCO → 폐기물 클래스 매핑  (step3 그대로)
# ════════════════════════════════════════════════
COCO_TO_WASTE = {
    "bottle":      ("pet_bottle",   "페트병"),
    "cup":         ("can",          "캔/컵"),
    "book":        ("paper",        "종이류"),
    "suitcase":    ("paper",        "포장상자"),
    "banana":      ("snack_bag",    "과자봉지류"),
    "apple":       ("snack_bag",    "과자봉지류"),
    "orange":      ("snack_bag",    "과자봉지류"),
    "sandwich":    ("snack_bag",    "과자봉지류"),
    "pizza":       ("snack_bag",    "과자봉지류"),
    "cake":        ("snack_bag",    "과자봉지류"),
    "bowl":        ("can",          "캔/그릇"),
    "vase":        ("glass_bottle", "유리병"),
    "wine glass":  ("glass_bottle", "유리병/음료수병"),
    "sports ball": ("pet_bottle",   "페트병"),
    "cell phone":  ("snack_bag",    "비닐류"),
    "remote":      ("paper",        "종이류"),
    "scissors":    ("can",          "금속류"),
    "toothbrush":  ("snack_bag",    "비닐류"),
    "fork":        ("can",          "금속류"),
    "knife":       ("can",          "금속류"),
    "spoon":       ("can",          "금속류"),
}

WASTE_COLORS = {
    "pet_bottle":   (200, 80,  200),
    "glass_bottle": (255, 80,  80 ),
    "can":          (50,  50,  255),
    "paper":        (50,  200, 50 ),
    "snack_bag":    (0,   165, 255),
}

# 제스처 정의 (손가락 수 → (표시명, 커맨드))
GESTURE = {
    0: ("Fist   PAUSE",   "PAUSE"),
    1: ("Index  CAPTURE", "CAPTURE"),
    2: ("V      FREEZE",  "FREEZE"),
    5: ("Open   START",   "START"),
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ════════════════════════════════════════════════
# MediaPipe 모델 자동 다운로드
# ════════════════════════════════════════════════
def ensure_landmarker() -> bool:
    if Path(LANDMARKER_PATH).exists():
        return True
    print("  hand_landmarker.task 다운로드 중... (약 22MB)")
    try:
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print("  ✅ 다운로드 완료")
        return True
    except Exception as e:
        print(f"  ⚠️  다운로드 실패: {e}")
        print("  → 제스처 없이 Detection 모드로 실행합니다.")
        return False


# ════════════════════════════════════════════════
# HandLandmarker 초기화
# ════════════════════════════════════════════════
def create_landmarker():
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    return HandLandmarker.create_from_options(options)


# ════════════════════════════════════════════════
# 손가락 수 계산
# ════════════════════════════════════════════════
def count_fingers(landmarks) -> int:
    lm = landmarks
    fingers = 0
    if lm[4].x < lm[3].x:
        fingers += 1
    for tip, pip in zip([8,12,16,20], [6,10,14,18]):
        if lm[tip].y < lm[pip].y:
            fingers += 1
    return fingers


# ════════════════════════════════════════════════
# 랜드마크 그리기
# ════════════════════════════════════════════════
def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)
    for i, pt in enumerate(pts):
        color = (0, 220, 0) if i in [4,8,12,16,20] else (0,140,255)
        cv2.circle(frame, pt, 4, color, -1)


# ════════════════════════════════════════════════
# 유틸 — step3 그대로
# ════════════════════════════════════════════════
def put_kor(frame, text, pos, size=28, color=(255,255,255), bg=None):
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


def save_image(img, folder, prefix="capture"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    path = str(Path(folder) / f"{prefix}_{ts}.jpg")
    ret, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ret:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
        return path
    return None


def draw_box(frame, x1, y1, x2, y2, color, label, conf):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    L, T = 22, 3
    for p1, p2 in [
        ((x1,y1),(x1+L,y1)), ((x1,y1),(x1,y1+L)),
        ((x2,y1),(x2-L,y1)), ((x2,y1),(x2,y1+L)),
        ((x1,y2),(x1+L,y2)), ((x1,y2),(x1,y2-L)),
        ((x2,y2),(x2-L,y2)), ((x2,y2),(x2,y2-L)),
    ]:
        cv2.line(frame, p1, p2, color, T)
    lbl_y = max(y1-6, 36)
    frame = put_kor(frame, f"{label}  {conf:.0%}",
                    (x1+4, lbl_y-30), size=24,
                    color=(255,255,255), bg=color)
    return frame


def draw_summary(frame, detected_cls):
    h, w = frame.shape[:2]
    if not detected_cls:
        return frame
    items   = list(detected_cls.items())
    panel_w = 260
    panel_h = 38 + len(items) * 34
    x0, y0  = w - panel_w - 10, 60   # 제스처 바 아래로 내림
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+panel_w, y0+panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x0,y0), (x0+panel_w, y0+panel_h), (70,70,70), 1)
    total = sum(v[1] for _,v in items)
    frame = put_kor(frame, f"감지된 폐기물  {total}개",
                    (x0+10, y0+6), size=20, color=(160,160,160))
    for i, (cls, (kor, count)) in enumerate(items):
        color = WASTE_COLORS.get(cls, (200,200,200))
        y     = y0 + 38 + i * 34
        cv2.circle(frame, (x0+14, y+12), 8, color, -1)
        frame = put_kor(frame, f"{kor}  ×{count}",
                        (x0+30, y), size=22, color=(220,220,220))
    return frame


def draw_gesture_bar(frame, gesture_label, finger_count, mode, gesture_on):
    """상단 제스처 상태 바"""
    h, w = frame.shape[:2]
    if not gesture_on:
        cv2.rectangle(frame, (0,0), (w, 36), (30,30,30), -1)
        frame = put_kor(frame, "Gesture OFF  (G키로 켜기)",
                        (10,4), size=22, color=(100,100,100))
        return frame

    g_color = (
        (0, 170, 0)   if mode == "START"   else
        (70, 70, 70)  if mode == "PAUSE"   else
        (0, 170, 170) if mode == "FREEZE"  else
        (170, 120, 0)
    )
    cv2.rectangle(frame, (0,0), (w, 44), g_color, -1)
    fc_str = str(finger_count) if finger_count >= 0 else "-"
    frame = put_kor(
        frame,
        f"Gesture: {gesture_label}   fingers: {fc_str}",
        (10, 8), size=24, color=(255,255,255)
    )
    return frame


def post_detect(cls_list):
    try:
        requests.post(MULTI_URL, json={"classes": cls_list}, timeout=0.8)
    except Exception:
        pass


def check_server():
    try:
        res = requests.get(HEALTH_URL, timeout=1).json()
        if res.get("model_ready"):
            print(f"  ✅ AI 서버 연결: {SERVER_URL}")
            return True
    except Exception:
        pass
    print("  ⚠️  AI 서버 없음 — Detection만 단독 실행")
    return False


# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════
def run():
    print("=" * 55)
    print("  폐기물 분류 AI — Detection + 제스처 통합")
    print("=" * 55)

    # Detection 모델 로드
    print(f"\n  Detection 모델 로딩: {DETECTION_MODEL}")
    det_model = YOLO(DETECTION_MODEL)
    print("  ✅ 모델 로드 완료")

    # AI 서버 확인
    server_ok = check_server()

    # MediaPipe HandLandmarker 초기화
    gesture_ok = ensure_landmarker()
    landmarker = None
    if gesture_ok:
        try:
            landmarker = create_landmarker()
            print("  ✅ HandLandmarker 준비 완료")
        except Exception as e:
            print(f"  ⚠️  HandLandmarker 초기화 실패: {e}")
            gesture_ok = False

    # 웹캠
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ 웹캠 연결 실패")
        if landmarker:
            landmarker.close()
        return

    print("\n✅ 웹캠 연결 성공")
    print("  [Q]종료  [S]저장  [A]전체클래스토글  [G]제스처ON/OFF")
    print("  ✋START  ✊PAUSE  ☝CAPTURE  ✌FREEZE\n")

    # 상태 변수
    frame_idx      = 0
    show_all       = False
    gesture_on     = True       # G키로 토글
    mode           = "START"    # START / PAUSE / FREEZE
    gesture_label  = "Open  START"
    finger_count   = -1
    frozen_frame   = None
    last_detected  = {}

    # 캡처 중복 방지
    last_capture_time = 0.0
    CAPTURE_COOLDOWN  = 2.0   # 캡처 후 최소 2초 대기
    prev_finger_count = -1    # 이전 프레임 손가락 수 (검지 뗐다 다시 올려야 재캡처)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ── 손 제스처 감지 ─────────────────────
        if gesture_on and landmarker is not None:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            detection_result = landmarker.detect(mp_image)

            if detection_result.hand_landmarks:
                lm = detection_result.hand_landmarks[0]
                draw_hand(frame, lm)
                finger_count = count_fingers(lm)

                if finger_count in GESTURE:
                    gesture_label, gesture_cmd = GESTURE[finger_count]

                    # CAPTURE: 쿨다운 + 검지 새로 올렸을 때만 1회 실행
                    if gesture_cmd == "CAPTURE":
                        now = time.time()
                        finger_just_raised = (prev_finger_count != 1)  # 이전 프레임이 검지가 아닐 때
                        cooldown_ok = (now - last_capture_time) >= CAPTURE_COOLDOWN
                        if finger_just_raised and cooldown_ok:
                            path = save_image(frame, SAVE_DIR, "gesture_capture")
                            if path:
                                print(f"  📸 제스처 캡처: {path}")
                            last_capture_time = now
                        # mode는 START 유지 (CAPTURE로 바꾸지 않음)
                    else:
                        mode = gesture_cmd

                prev_finger_count = finger_count
            else:
                finger_count      = -1
                prev_finger_count = -1

        # ── Detection 추론 ──────────────────────
        # FREEZE: 웹캠 프레임은 live지만 detection 결과는 고정
        # PAUSE:  웹캠 프레임 live, detection 스킵, 결과 유지
        # START:  정상 추론

        display      = frame.copy()
        detected_cls = last_detected  # PAUSE/FREEZE 중 이전 결과 유지

        if mode == "START":
            results = det_model.predict(
                source=frame, imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                verbose=False,
            )
            detected_cls = {}
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf            = float(box.conf[0])
                cls_id          = int(box.cls[0])
                coco_name       = det_model.names[cls_id]
                if coco_name in COCO_TO_WASTE:
                    waste_cls, kor = COCO_TO_WASTE[coco_name]
                    color          = WASTE_COLORS.get(waste_cls, (200,200,200))
                    display        = draw_box(display, x1,y1,x2,y2, color, kor, conf)
                    if waste_cls not in detected_cls:
                        detected_cls[waste_cls] = (kor, 0)
                    detected_cls[waste_cls] = (kor, detected_cls[waste_cls][1]+1)
                elif show_all:
                    display = draw_box(display, x1,y1,x2,y2,
                                       (120,120,120), coco_name, conf)
            last_detected = detected_cls

            if server_ok:
                threading.Thread(
                    target=post_detect,
                    args=(list(detected_cls.keys()),),
                    daemon=True
                ).start()

        elif mode == "FREEZE":
            # FREEZE: 바운딩박스는 frozen_frame 기준으로 display에 오버레이
            if frozen_frame is not None:
                # frozen 결과를 현재 live 프레임 위에 반투명 오버레이
                cv2.addWeighted(frozen_frame, 0.55, display, 0.45, 0, display)

        elif mode == "PAUSE":
            # PAUSE: LED OFF 신호 전송
            if server_ok:
                threading.Thread(
                    target=post_detect, args=([],), daemon=True
                ).start()

        # ── 아두이노 버튼1 캡처 요청 확인 ──────
        if server_ok and frame_idx % 5 == 0:
            try:
                state   = requests.get(STATE_URL, timeout=0.3).json()
                pending = state.get("pending", 0)
                for _ in range(pending):
                    path = save_image(display, SAVE_DIR, "btn_capture")
                    if path:
                        print(f"  📸 버튼 캡처: {path}")
                    requests.post(DONE_URL, timeout=0.3)
            except Exception:
                pass

        # ── UI 렌더링 ──────────────────────────
        # FREEZE면 이미 위에서 display 처리됨
        if mode != "FREEZE":
            display = draw_summary(display, detected_cls)

            if not detected_cls:
                display = put_kor(display, "폐기물을 카메라 앞에 놓아주세요",
                                  (10, 52), size=26, color=(120,120,120))

        # PAUSE 오버레이 — 화면은 살아있고 반투명 배너만 표시
        if mode == "PAUSE":
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 44), (w, 100), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
            display = put_kor(display, "PAUSED  —  Open hand to resume",
                              (12, 54), size=28, color=(180, 180, 180))

        # 제스처 바 — FREEZE는 위에서 이미 그렸으므로 스킵
        if mode != "FREEZE":
            display = draw_gesture_bar(display, gesture_label,
                                       finger_count, mode, gesture_on)

        # 하단 안내
        mode_txt = "전체ON" if show_all else "폐기물만"
        g_txt    = "제스처ON" if gesture_on else "제스처OFF"
        cv2.rectangle(display, (0,h-36), (w,h), (20,20,20), -1)
        display = put_kor(display,
            f"{mode_txt} | {g_txt} | Q:종료 S:저장 A:클래스 G:제스처",
            (10,h-30), size=19, color=(140,140,140))

        # FREEZE 스냅샷 — START→FREEZE 전환 순간 1회만 저장
        if mode == "FREEZE" and frozen_frame is None:
            frozen_frame = frame.copy()  # 바운딩박스 포함 현재 display 저장
        elif mode != "FREEZE":
            frozen_frame = None   # FREEZE 해제 시 초기화

        cv2.imshow("Waste Classifier + Gesture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("종료합니다.")
            break
        elif key == ord("s"):
            path = save_image(display, SAVE_DIR, "key_capture")
            if path:
                print(f"  📸 S키 저장: {path}")
        elif key == ord("a"):
            show_all = not show_all
            print(f"  전체 클래스: {'ON' if show_all else 'OFF'}")
        elif key == ord("g"):
            gesture_on = not gesture_on
            print(f"  제스처 인식: {'ON' if gesture_on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    if landmarker:
        landmarker.close()


if __name__ == "__main__":
    run()
