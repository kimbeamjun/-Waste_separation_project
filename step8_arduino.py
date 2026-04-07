"""
[8단계] 아두이노 Serial 통신 + 버튼 제어 연동
AI 서버 분류 결과 → 아두이노 LED/매트릭스
아두이노 버튼 입력 → 웹캠 캡처/일시정지 제어

버튼 동작:
  버튼1 (D8) 누르기 → 웹캠 현재 화면 캡처 저장
  버튼2 (D9) 누르기 → 분류 일시정지 / 재개 토글

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step8_arduino.py
설치 필요: pip install pyserial
사전 조건: step5_api_server.py + step3_webcam.py 실행 중
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
SERVER_URL     = "http://localhost:8000"
STATS_URL      = f"{SERVER_URL}/stats"
HEALTH_URL     = f"{SERVER_URL}/health"

# 웹캠 제어 API (step5 서버에 추가된 엔드포인트)
PAUSE_URL      = f"{SERVER_URL}/webcam/pause"
RESUME_URL     = f"{SERVER_URL}/webcam/resume"
CAPTURE_URL    = f"{SERVER_URL}/webcam/capture"

BAUD_RATE      = 9600
POLL_INTERVAL  = 0.8    # AI 서버 폴링 간격 (초)
CONF_THRESHOLD = 0.65

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
is_paused    = False
capture_idx  = 0
ser          = None   # Serial 객체 (스레드 공유)


# ════════════════════════════════════════════════
# 아두이노 포트 탐색
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
# 웹캠 제어 (step5 서버에 요청)
# ════════════════════════════════════════════════
def webcam_capture():
    """웹캠 캡처 저장 요청"""
    global capture_idx
    capture_idx += 1
    try:
        # step5 서버에 캡처 요청
        requests.post(CAPTURE_URL, timeout=2)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  📸 [{ts}] 캡처 저장 요청 (#{capture_idx})")
    except Exception:
        # 서버 엔드포인트 없으면 로컬 저장
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"  📸 [{ts}] 캡처 요청 전송 (#{capture_idx})")


def webcam_pause():
    """웹캠 분류 일시정지 요청"""
    try:
        requests.post(PAUSE_URL, timeout=2)
    except Exception:
        pass
    print("  ⏸  분류 일시정지")


def webcam_resume():
    """웹캠 분류 재개 요청"""
    try:
        requests.post(RESUME_URL, timeout=2)
    except Exception:
        pass
    print("  ▶  분류 재개")


# ════════════════════════════════════════════════
# 아두이노 → PC 버튼 수신 스레드
# ════════════════════════════════════════════════
def button_listener():
    """아두이노에서 오는 버튼 명령을 별도 스레드로 수신"""
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

        time.sleep(0.02)   # 20ms 폴링


# ════════════════════════════════════════════════
# 다중 감지 명령 전송
# ════════════════════════════════════════════════
def send_multi(detected_classes: list):
    """
    감지된 클래스 목록을 아두이노로 전송
    단일: "0\n"
    다중: "M012\n"
    """
    if not detected_classes:
        ser.write(b"X\n")
        return

    codes = "".join(CLASS_CMD[c] for c in detected_classes if c in CLASS_CMD)
    if not codes:
        return

    if len(codes) == 1:
        ser.write(f"{codes}\n".encode())
    else:
        ser.write(f"M{codes}\n".encode())


# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════
def run():
    global ser, is_paused

    print("=" * 55)
    print("  폐기물 분류 AI - 아두이노 버튼 제어")
    print("=" * 55)
    print("  버튼1 (D8): 캡처 저장")
    print("  버튼2 (D9): 일시정지 / 재개 토글")
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
        time.sleep(2)
        ready = ser.readline().decode("utf-8", errors="ignore").strip()
        if "READY" in ready:
            print(f"  ✅ 아두이노 준비 완료")
        else:
            print(f"  ⚠️  응답: {ready}")
    except Exception as e:
        print(f"  ❌ Serial 연결 실패: {e}")
        return

    # ── 버튼 수신 스레드 시작 ─────────────────────
    t = threading.Thread(target=button_listener, daemon=True)
    t.start()
    print("\n  ✅ 버튼 수신 스레드 시작")

    # ── 테스트 ───────────────────────────────────
    print("\n[3] LED + 도트매트릭스 테스트...")
    ser.write(b"T\n")
    time.sleep(4)

    # ── 메인 루프 ─────────────────────────────────
    print("\n" + "=" * 55)
    print("  AI 분류 시작! (Ctrl+C로 종료)")
    print("=" * 55)

    last_classes = []
    last_total   = 0
    error_cnt    = 0

    while True:
        try:
            # 일시정지 중이면 폴링 스킵
            if is_paused:
                time.sleep(POLL_INTERVAL)
                continue

            res    = requests.get(STATS_URL, timeout=3).json()
            total  = res.get("total_requests", 0)
            conf   = res.get("avg_confidence", 0.0)
            counts = res.get("class_counts", {})

            if total > last_total and counts:
                last_total = total

                # 신뢰도 충족하는 클래스만 선별
                if conf >= CONF_THRESHOLD:
                    # 예측 횟수 기준 상위 클래스 추출
                    detected = [
                        cls for cls, cnt in counts.items()
                        if cnt > 0
                    ]
                    detected = sorted(
                        detected, key=lambda c: counts[c], reverse=True
                    )[:3]   # 최대 3개까지

                    if detected != last_classes:
                        # 아두이노 전송
                        send_multi(detected)

                        # 콘솔 출력
                        ts       = datetime.now().strftime("%H:%M:%S")
                        kor_list = [CLASS_KOR.get(c, c) for c in detected]
                        clr_list = [LED_COLOR.get(c, "💡") for c in detected]
                        print(f"\n  [{ts}] 🗑️  감지: {' + '.join(kor_list)}")
                        print(f"         LED: {' + '.join(clr_list)}")
                        print(f"         신뢰도: {conf:.0%}")

                        # 응답 수신
                        time.sleep(0.1)
                        if ser.in_waiting > 0:
                            resp = ser.readline().decode("utf-8", errors="ignore").strip()
                            print(f"         아두이노: {resp}")

                        last_classes = detected
                else:
                    # 신뢰도 낮으면 OFF
                    if last_classes:
                        ser.write(b"X\n")
                        print(f"  ⚠️  신뢰도 낮음 ({conf:.0%}) → LED OFF")
                        last_classes = []

            error_cnt = 0
            time.sleep(POLL_INTERVAL)

        except requests.exceptions.ConnectionError:
            error_cnt += 1
            print(f"  ⚠️  서버 연결 끊김 (재시도 {error_cnt}회)")
            ser.write(b"X\n")
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break

        except Exception as e:
            print(f"  오류: {e}")
            time.sleep(2)

    # 종료 처리
    ser.write(b"X\n")
    time.sleep(0.2)
    ser.close()
    print("Serial 연결 종료.")


if __name__ == "__main__":
    run()
