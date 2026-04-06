"""
[3단계] 웹캠 클라이언트 (API 서버 연동 버전)
웹캠 영상을 캡처하여 AI 서버로 전송, 결과를 화면에 표시

실행 환경: PyCharm 로컬 (Windows 10) - 클라이언트 역할
실행 방법: python step3_webcam.py
사전 조건: step5_api_server.py 가 먼저 실행되어 있어야 함

[구조]
이 파일 (클라이언트)
    ↓ HTTP POST /predict (이미지 전송)
step5_api_server.py (AI 서버)
    ↓ JSON 응답
이 파일에서 결과를 화면에 표시
"""

import cv2
import requests
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
SERVER_URL      = "http://localhost:8000"   # AI 서버 주소
                                            # 다른 PC면 IP로 변경: "http://192.168.0.10:8000"
PREDICT_URL     = f"{SERVER_URL}/predict"
HEALTH_URL      = f"{SERVER_URL}/health"

CAMERA_INDEX    = 0      # 웹캠 번호
SEND_INTERVAL   = 0.5    # 서버 전송 간격 (초)
CONF_MIN        = 0.6    # 이 신뢰도 이하면 Uncertain 표시
REQUEST_TIMEOUT = 3      # 서버 응답 대기 시간 (초)

# 한글 폰트 경로 (Windows 기본 폰트)
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"   # 맑은 고딕

# 클래스별 색상 (BGR)
CLASS_COLORS = {
    "snack_bag":    (0,   165, 255),
    "glass_bottle": (255, 0,   0  ),
    "paper":        (0,   200, 0  ),
    "can":          (0,   0,   255),
    "pet_bottle":   (200, 0,   200),
}

# 한글 라벨
CLASS_LABEL = {
    "snack_bag":    "과자봉지 / 비닐",
    "glass_bottle": "유리병 / 음료수병",
    "paper":        "종이류 / 포장상자",
    "can":          "캔 / 음료수캔",
    "pet_bottle":   "페트병",
}


def put_korean_text(frame, text, pos, font_size=36, color=(255, 255, 255)):
    """
    OpenCV 프레임에 한글 텍스트 렌더링
    PIL로 텍스트 그린 후 numpy 배열로 변환
    """
    try:
        font     = ImageFont.truetype(FONT_PATH, font_size)
        img_pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw     = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # 폰트 로드 실패 시 영문으로 대체
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return frame


def check_server():
    """서버 연결 및 모델 로드 확인"""
    try:
        res  = requests.get(HEALTH_URL, timeout=2)
        data = res.json()
        if data.get("model_ready"):
            print(f"✅ AI 서버 연결 성공: {SERVER_URL}")
            return True
        print("⚠️  서버는 켜져있지만 모델이 로드되지 않았습니다.")
        return False
    except Exception:
        print(f"❌ AI 서버 연결 실패: {SERVER_URL}")
        print("   step5_api_server.py 를 먼저 실행하세요.")
        return False


def send_frame(frame: np.ndarray):
    """프레임 JPEG 인코딩 후 서버 전송 → 응답 dict 반환"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        res = requests.post(
            PREDICT_URL,
            files   = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout = REQUEST_TIMEOUT,
        )
        return res.json()
    except requests.exceptions.Timeout:
        return {"error": "서버 응답 시간 초과"}
    except Exception as e:
        return {"error": str(e)}


def draw_result(frame: np.ndarray, result: dict, rtt_ms: float):
    """분류 결과를 프레임에 오버레이"""
    h, w = frame.shape[:2]

    if "error" in result:
        frame = put_korean_text(frame, f"오류: {result['error']}", (10, 10),
                                font_size=28, color=(255, 255, 255))
        return frame

    cls_name = result.get("class_name", "unknown")
    conf     = result.get("confidence", 0.0)
    inf_ms   = result.get("inference_ms", 0.0)
    color    = CLASS_COLORS.get(cls_name, (255, 255, 255))
    label    = CLASS_LABEL.get(cls_name, cls_name)

    if conf < CONF_MIN:
        color = (150, 150, 150)
        label = "불확실"

    # 상단 결과 박스
    cv2.rectangle(frame, (0, 0), (w, 65), color, -1)
    frame = put_korean_text(frame, f"{label}  {conf:.0%}", (12, 12),
                            font_size=38, color=(255, 255, 255))

    # 하단 정보 바
    cv2.rectangle(frame, (0, h - 36), (w, h), (40, 40, 40), -1)
    frame = put_korean_text(frame,
                            f"AI: {inf_ms:.0f}ms  RTT: {rtt_ms:.0f}ms  |  Q: 종료  S: 저장",
                            (10, h - 32), font_size=20, color=(200, 200, 200))

    return frame


def run():
    print("=" * 50)
    print("  폐기물 분류 AI - 웹캠 클라이언트")
    print("=" * 50)

    if not check_server():
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ 웹캠 연결 실패 (CAMERA_INDEX={CAMERA_INDEX})")
        return

    WINDOW_TITLE = "Waste Classifier"   # 창 제목 (한글 깨짐 방지용 영문)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    print("웹캠 연결 성공!")
    print("  [Q] 종료  |  [S] 현재 프레임 저장\n")

    last_send   = 0.0
    last_result = {}
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        now = time.time()

        # SEND_INTERVAL 간격으로만 서버 전송
        rtt_ms = 0.0
        if now - last_send >= SEND_INTERVAL:
            t0          = time.time()
            last_result = send_frame(frame)
            rtt_ms      = (time.time() - t0) * 1000
            last_send   = now

        display = draw_result(frame.copy(), last_result, rtt_ms)
        cv2.imshow(WINDOW_TITLE, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("종료합니다.")
            break
        elif key == ord("s"):
            path = f"./capture_{frame_idx:05d}.jpg"
            cv2.imwrite(path, display)
            print(f"저장 완료: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
