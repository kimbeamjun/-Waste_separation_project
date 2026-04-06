"""
[8단계 - 아두이노 버전] Serial 통신으로 LED 제어
AI 서버 분류 결과 → Serial → 아두이노 → LED 점등

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step8_arduino.py
설치 필요: pip install pyserial

사전 조건:
  1. 아두이노에 arduino_led_control.ino 업로드 완료
  2. step5_api_server.py 실행 중
  3. USB로 아두이노 연결

연결 구조:
  웹캠 클라이언트 → FastAPI 서버 → 이 파일 → Serial → 아두이노 → LED
"""

import time
import requests
import serial
import serial.tools.list_ports

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
SERVER_URL     = "http://localhost:8000"
STATS_URL      = f"{SERVER_URL}/stats"
HEALTH_URL     = f"{SERVER_URL}/health"

BAUD_RATE      = 9600
POLL_INTERVAL  = 1.0    # AI 서버 폴링 간격 (초)
CONF_THRESHOLD = 0.65   # 신뢰도 임계값 (이하면 LED OFF)

# 클래스 → 아두이노 명령 코드
CLASS_CMD = {
    "can":          "0",   # 빨간 LED
    "paper":        "1",   # 초록 LED
    "pet_bottle":   "2",   # 파란 LED
    "snack_bag":    "3",   # 노란 LED
    "glass_bottle": "4",   # 흰색 LED
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
# 아두이노 포트 자동 탐색
# ════════════════════════════════════════════════
def find_arduino_port():
    """연결된 아두이노 포트 자동 탐색"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if "arduino" in desc or "ch340" in desc or "usb serial" in desc:
            return port.device
    # 자동 탐색 실패 시 사용 가능한 포트 출력
    if ports:
        print("  사용 가능한 포트 목록:")
        for p in ports:
            print(f"    {p.device} - {p.description}")
        return ports[0].device   # 첫 번째 포트 사용
    return None


# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════
def run():
    print("=" * 50)
    print("  폐기물 분류 AI - 아두이노 LED 제어")
    print("=" * 50)

    # ── AI 서버 연결 확인 ────────────────────────
    print("\n[1] AI 서버 연결 확인 중...")
    try:
        res = requests.get(HEALTH_URL, timeout=3).json()
        if res.get("model_ready"):
            print(f"  ✅ AI 서버 연결 성공: {SERVER_URL}")
        else:
            print("  ⚠️  모델 미로드 — step5_api_server.py 먼저 실행하세요.")
            return
    except Exception:
        print(f"  ❌ AI 서버 연결 실패: {SERVER_URL}")
        print("  step5_api_server.py 를 먼저 실행하세요.")
        return

    # ── 아두이노 Serial 연결 ──────────────────────
    print("\n[2] 아두이노 포트 탐색 중...")
    port = find_arduino_port()
    if port is None:
        print("  ❌ 아두이노를 찾을 수 없습니다.")
        print("  USB 연결 확인 후 다시 실행하세요.")
        return

    print(f"  ✅ 아두이노 감지: {port}")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(2)   # 아두이노 리셋 대기
        # ARDUINO_READY 수신 확인
        ready_msg = ser.readline().decode("utf-8", errors="ignore").strip()
        if "READY" in ready_msg:
            print(f"  ✅ 아두이노 준비 완료 ({ready_msg})")
        else:
            print(f"  ⚠️  아두이노 응답: {ready_msg}")
    except Exception as e:
        print(f"  ❌ Serial 연결 실패: {e}")
        return

    # ── 테스트 점등 ───────────────────────────────
    print("\n[3] LED 테스트 (전체 순서대로 점등)...")
    ser.write(b"T")
    time.sleep(3)
    print("  ✅ 테스트 완료\n")

    # ── 메인 루프 ─────────────────────────────────
    print("=" * 50)
    print("  AI 분류 시작! (Ctrl+C 로 종료)")
    print("=" * 50)

    last_cls   = None
    last_total = 0
    error_cnt  = 0

    while True:
        try:
            res    = requests.get(STATS_URL, timeout=3).json()
            total  = res.get("total_requests", 0)
            conf   = res.get("avg_confidence", 0.0)
            counts = res.get("class_counts", {})

            # 새 예측이 들어왔을 때만 처리
            if total > last_total and counts:
                last_total = total

                # 가장 최근에 많이 예측된 클래스
                top_cls = max(counts, key=counts.get)

                if conf >= CONF_THRESHOLD:
                    cmd = CLASS_CMD.get(top_cls, "X")

                    if top_cls != last_cls:
                        # 아두이노로 명령 전송
                        ser.write(cmd.encode())
                        time.sleep(0.1)

                        # 아두이노 응답 수신
                        response = ser.readline().decode("utf-8", errors="ignore").strip()

                        kor   = CLASS_KOR.get(top_cls, top_cls)
                        color = LED_COLOR.get(top_cls, "💡")
                        print(f"  🗑️  {kor}  |  {color} LED 점등  |  신뢰도: {conf:.0%}  |  응답: {response}")
                        last_cls = top_cls

                else:
                    # 신뢰도 낮으면 LED 끄기
                    if last_cls is not None:
                        ser.write(b"X")
                        print(f"  ⚠️  신뢰도 낮음 ({conf:.0%}) → LED 전체 OFF")
                        last_cls = None

            error_cnt = 0
            time.sleep(POLL_INTERVAL)

        except requests.exceptions.ConnectionError:
            error_cnt += 1
            print(f"  ⚠️  서버 연결 끊김 (재시도 {error_cnt}회)")
            ser.write(b"X")   # 연결 끊기면 LED 끄기
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break

        except Exception as e:
            print(f"  오류: {e}")
            time.sleep(2)

    # 종료 처리
    ser.write(b"X")    # 전체 LED 끄기
    ser.close()
    print("Serial 연결 종료.")


if __name__ == "__main__":
    run()
