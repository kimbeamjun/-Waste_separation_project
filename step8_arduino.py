"""
[8단계] 아두이노 Serial 통신 + 버튼 제어
/latest 엔드포인트로 실시간 다중 감지 결과 수신
→ 감지된 클래스 수만큼 LED 동시 점등

수정 내용:
  - /stats 폴링 방식 → /latest 폴링으로 변경 (실시간 반응)
  - 다중 감지 시 해당 LED 동시 점등
  - 캡처 저장을 collected_data 폴더로 정상화
  - 버튼1 캡처 / 버튼2 일시정지 정상 동작

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step8_arduino.py
설치 필요: pip install pyserial
사전 조건: step5_api_server.py 실행 중, 아두이노 연결
"""

import time
import threading
import requests
import serial
import serial.tools.list_ports
from datetime import datetime

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
SERVER_URL    = "http://localhost:8000"
HEALTH_URL    = f"{SERVER_URL}/health"
LATEST_URL    = f"{SERVER_URL}/latest"         # 최신 감지 결과 (핵심)
PAUSE_URL     = f"{SERVER_URL}/webcam/pause"
RESUME_URL    = f"{SERVER_URL}/webcam/resume"
CAPTURE_URL   = f"{SERVER_URL}/webcam/capture"

BAUD_RATE     = 9600
POLL_INTERVAL = 0.5    # /latest 폴링 간격 (초) — 빠를수록 반응 빠름

# 클래스 → 아두이노 코드
CLASS_CMD = {
    "can":          "0",
    "paper":        "1",
    "pet_bottle":   "2",
    "snack_bag":    "3",
    "glass_bottle": "4",
}

CLASS_KOR = {
    "can":          "캔/음료수캔",
    "paper":        "종이류/포장상자",
    "pet_bottle":   "페트병",
    "snack_bag":    "과자봉지/비닐",
    "glass_bottle": "유리병/음료수병",
}

LED_COLOR = {
    "can":          "🔴 빨간",
    "paper":        "🟢 초록",
    "pet_bottle":   "🔵 파란",
    "snack_bag":    "🟡 노란",
    "glass_bottle": "⚪ 흰색",
}

# ════════════════════════════════════════════════
# 전역 상태
# ════════════════════════════════════════════════
is_paused      = False
ser            = None
last_classes   = []     # 이전 감지 클래스 (변화 감지용)
last_timestamp = ""     # 이전 타임스탬프 (새 감지 여부 판단)


# ════════════════════════════════════════════════
# 아두이노 포트 자동 탐색
# ════════════════════════════════════════════════
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if "arduino" in desc or "ch340" in desc or "usb serial" in desc:
            return port.device
    if ports:
        print("  사용 가능한 포트:")
        for p in ports:
            print(f"    {p.device} - {p.description}")
        return ports[0].device
    return None


# ════════════════════════════════════════════════
# 아두이노 명령 전송
# ════════════════════════════════════════════════
def send_to_arduino(msg: str):
    """아두이노로 명령 전송"""
    global ser
    try:
        if ser and ser.is_open:
            ser.write(f"{msg}\n".encode())
    except Exception as e:
        print(f"  ⚠️  아두이노 전송 오류: {e}")


def send_detected(classes: list, confidence: float = 0.0):
    """
    감지된 클래스 목록 → 아두이노로 전송
    단일 (신뢰도 포함): "S3:72\n"  → LCD에 신뢰도 바 표시
    다중:               "M024\n"
    없음:               "X\n"
    """
    codes = "".join(CLASS_CMD[c] for c in classes if c in CLASS_CMD)
    if not codes:
        send_to_arduino("X")
        return

    if len(codes) == 1:
        # 신뢰도 포함 단일 감지 커맨드 (LCD용)
        conf_int = max(0, min(100, int(confidence * 100)))
        send_to_arduino(f"S{codes}:{conf_int}")
    else:
        send_to_arduino(f"M{codes}")


# ════════════════════════════════════════════════
# 웹캠 제어 (step5 서버에 요청)
# ════════════════════════════════════════════════
def webcam_capture():
    """캡처 저장 요청 → step5 서버 → collected_data 폴더에 저장"""
    try:
        requests.post(CAPTURE_URL, timeout=2)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  📸 [{ts}] 캡처 저장 요청 전송")
    except Exception:
        print("  ⚠️  캡처 요청 실패 (step5 서버 확인)")


def webcam_pause():
    try:
        requests.post(PAUSE_URL, timeout=2)
    except Exception:
        pass
    print("  ⏸  분류 일시정지")


def webcam_resume():
    try:
        requests.post(RESUME_URL, timeout=2)
    except Exception:
        pass
    print("  ▶  분류 재개")


# ════════════════════════════════════════════════
# 버튼 수신 스레드 (아두이노 → PC)
# ════════════════════════════════════════════════
def button_listener():
    """아두이노 버튼 명령 수신"""
    global is_paused, ser

    while True:
        try:
            if ser and ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if line == "BTN:CAPTURE":
                    webcam_capture()

                elif line == "BTN:PAUSE":
                    is_paused = True
                    webcam_pause()

                elif line == "BTN:RESUME":
                    is_paused = False
                    webcam_resume()

                elif line and not line.startswith("LED:"):
                    print(f"  [아두이노] {line}")

        except Exception:
            pass
        time.sleep(0.02)


# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════
def run():
    global ser, is_paused, last_classes, last_timestamp

    print("=" * 55)
    print("  폐기물 분류 AI - 아두이노 LED 제어")
    print("=" * 55)
    print("  버튼1 (D8): 캡처 저장 → collected_data/ 에 저장")
    print("  버튼2 (D9): 일시정지 / 재개 토글")
    print("  다중 감지 시 해당 LED 동시 점등")
    print()

    # ── AI 서버 확인 ─────────────────────────────
    print("[1] AI 서버 연결 확인...")
    try:
        res = requests.get(HEALTH_URL, timeout=3).json()
        if res.get("model_ready"):
            print(f"  ✅ AI 서버 연결: {SERVER_URL}")
        else:
            print("  ⚠️  모델 미로드 — step5_api_server.py 먼저 실행하세요.")
            return
    except Exception:
        print(f"  ❌ AI 서버 연결 실패: {SERVER_URL}")
        return

    # ── 아두이노 연결 ─────────────────────────────
    print("\n[2] 아두이노 포트 탐색...")
    port = find_arduino_port()
    if not port:
        print("  ❌ 아두이노를 찾을 수 없습니다.")
        return

    print(f"  ✅ 아두이노 감지: {port}")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(2)   # 아두이노 리셋 대기
        ready = ser.readline().decode("utf-8", errors="ignore").strip()
        if "READY" in ready:
            print("  ✅ 아두이노 준비 완료")
        else:
            print(f"  ⚠️  응답: {ready}")
    except Exception as e:
        print(f"  ❌ Serial 연결 실패: {e}")
        return

    # ── 버튼 수신 스레드 ──────────────────────────
    t = threading.Thread(target=button_listener, daemon=True)
    t.start()
    print("  ✅ 버튼 수신 스레드 시작")

    # ── 테스트 ───────────────────────────────────
    print("\n[3] 테스트 시퀀스 실행...")
    send_to_arduino("T")
    time.sleep(4)

    # ── 메인 폴링 루프 ────────────────────────────
    print("\n" + "=" * 55)
    print("  실시간 감지 시작! (Ctrl+C로 종료)")
    print("  감지된 클래스 수만큼 LED 동시 점등")
    print("=" * 55)

    error_cnt    = 0
    led_off_tick = 0   # 감지 없음이 몇 번 연속됐는지 카운트

    while True:
        try:
            # 일시정지 중
            if is_paused:
                time.sleep(POLL_INTERVAL)
                continue

            # /latest 로 최신 감지 결과 가져오기
            res     = requests.get(LATEST_URL, timeout=2).json()
            classes = res.get("classes", [])

            if classes:
                led_off_tick = 0

                # ★ 매 폴링마다 무조건 LED 전송 (박스 떠있는 동안 계속 점등)
                confidence = res.get("confidence", 0.0)
                send_detected(classes, confidence)

                # 클래스 바뀐 경우만 콘솔 출력
                if set(classes) != set(last_classes):
                    now      = datetime.now().strftime("%H:%M:%S")
                    kor_list = [CLASS_KOR.get(c, c) for c in classes]
                    clr_list = [LED_COLOR.get(c, "💡") for c in classes]
                    print(f"\n  [{now}] 🗑️  {' + '.join(kor_list)}")
                    print(f"         {' + '.join(clr_list)} LED 점등")
                    last_classes = classes

            else:
                led_off_tick += 1
                # 2회 연속 감지 없음 → LED 끄기 (노이즈 방지)
                if led_off_tick >= 2 and last_classes:
                    send_to_arduino("X")
                    print(f"  ─ 감지 없음 → LED OFF")
                    last_classes = []

            error_cnt = 0
            time.sleep(POLL_INTERVAL)

        except requests.exceptions.ConnectionError:
            error_cnt += 1
            print(f"  ⚠️  서버 연결 끊김 (재시도 {error_cnt}회)")
            send_to_arduino("X")
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break

        except Exception as e:
            print(f"  오류: {e}")
            time.sleep(1)

    # 종료 처리
    send_to_arduino("X")
    time.sleep(0.2)
    ser.close()
    print("Serial 연결 종료.")


if __name__ == "__main__":
    run()
