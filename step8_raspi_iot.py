"""
[8단계] 라즈베리파이 IoT 클라이언트
분류 결과를 AI 서버에서 받아 GPIO LED를 점등

실행 환경: 라즈베리파이 (Python 3.x)
실행 방법: python step8_raspi_iot.py

회로 연결:
  GPIO 17 → 빨간 LED  → 220Ω → GND   (캔)
  GPIO 27 → 초록 LED  → 220Ω → GND   (종이류)
  GPIO 22 → 파란 LED  → 220Ω → GND   (페트병)
  GPIO 23 → 노란 LED  → 220Ω → GND   (과자봉지)
  GPIO 24 → 흰색 LED  → 220Ω → GND   (유리병)
  GPIO 25 → 부저      → GND          (분류 완료 알림)

AI 서버(PC)와 같은 네트워크에 연결되어 있어야 함
"""

import time
import requests

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
# ★ AI 서버 PC의 실제 IP로 변경 (같은 와이파이 연결 필요)
# 확인 방법: PC에서 ipconfig 실행 → IPv4 주소 확인
SERVER_IP    = "192.168.0.100"
SERVER_URL   = f"http://{SERVER_IP}:8000"
PREDICT_URL  = f"{SERVER_URL}/predict"
HEALTH_URL   = f"{SERVER_URL}/health"

POLL_INTERVAL   = 1.0    # 서버 폴링 간격 (초)
LED_ON_DURATION = 2.0    # LED 점등 유지 시간 (초)
BUZZER_DURATION = 0.1    # 부저 울림 시간 (초)
CONF_THRESHOLD  = 0.65   # 이 이상 신뢰도일 때만 LED 점등

# GPIO 핀 번호 (BCM 기준)
LED_PINS = {
    "can":          17,   # 빨간
    "paper":        27,   # 초록
    "pet_bottle":   22,   # 파랑
    "snack_bag":    23,   # 노랑
    "glass_bottle": 24,   # 흰색
}
BUZZER_PIN = 25

# 클래스 한글명
CLASS_KOR = {
    "snack_bag":    "과자봉지/비닐",
    "glass_bottle": "유리병/음료수병",
    "paper":        "종이류/포장상자",
    "can":          "캔/음료수캔",
    "pet_bottle":   "페트병",
}


# ════════════════════════════════════════════════
# GPIO 초기화 (라즈베리파이에서만 동작)
# ════════════════════════════════════════════════
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in list(LED_PINS.values()) + [BUZZER_PIN]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    GPIO_AVAILABLE = True
    print("✅ GPIO 초기화 완료")
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️  RPi.GPIO 없음 — 시뮬레이션 모드로 실행합니다.")


def all_leds_off():
    """모든 LED 끄기"""
    if GPIO_AVAILABLE:
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)


def light_led(cls_name: str):
    """해당 클래스 LED 켜기 + 부저 알림"""
    pin = LED_PINS.get(cls_name)
    if pin is None:
        return

    all_leds_off()

    if GPIO_AVAILABLE:
        GPIO.output(pin, GPIO.HIGH)
        # 부저 짧게 울리기
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(BUZZER_DURATION)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        # LED 유지
        time.sleep(LED_ON_DURATION)
        GPIO.output(pin, GPIO.LOW)
    else:
        # 시뮬레이션 (터미널 출력)
        kor = CLASS_KOR.get(cls_name, cls_name)
        icons = {"can":"🔴","paper":"🟢","pet_bottle":"🔵","snack_bag":"🟡","glass_bottle":"⚪"}
        icon  = icons.get(cls_name, "💡")
        print(f"  {icon} LED 점등: {kor} (GPIO {pin}) → {LED_ON_DURATION}초 유지")
        time.sleep(LED_ON_DURATION)


def check_server():
    """AI 서버 연결 확인"""
    try:
        res = requests.get(HEALTH_URL, timeout=3).json()
        return res.get("model_ready", False)
    except Exception:
        return False


def run():
    print("=" * 50)
    print("  폐기물 분류 AI - 라즈베리파이 IoT 클라이언트")
    print("=" * 50)
    print(f"  AI 서버: {SERVER_URL}")
    print(f"  모드: {'실제 GPIO' if GPIO_AVAILABLE else '시뮬레이션'}")
    print()

    # 서버 연결 대기
    print("AI 서버 연결 대기 중...")
    while not check_server():
        print(f"  재시도 중... ({SERVER_URL})")
        time.sleep(3)
    print("✅ AI 서버 연결 완료\n")

    last_cls  = None
    error_cnt = 0

    while True:
        try:
            # /stats API로 최근 예측 클래스 확인
            res   = requests.get(f"{SERVER_URL}/stats", timeout=3)
            stats = res.json()

            # 가장 많이 예측된 클래스 확인
            counts = stats.get("class_counts", {})
            conf   = stats.get("avg_confidence", 0.0)
            total  = stats.get("total_requests", 0)

            if total > 0 and counts:
                top_cls = max(counts, key=counts.get)

                # 이전과 다른 클래스이고 신뢰도 충족 시 LED 점등
                if top_cls != last_cls and conf >= CONF_THRESHOLD:
                    kor = CLASS_KOR.get(top_cls, top_cls)
                    print(f"🗑️  분류 감지: {kor} ({conf:.0%}) → LED 점등")
                    light_led(top_cls)
                    last_cls = top_cls

            error_cnt = 0
            time.sleep(POLL_INTERVAL)

        except requests.exceptions.ConnectionError:
            error_cnt += 1
            print(f"⚠️  서버 연결 끊김 (재시도 {error_cnt}회)")
            all_leds_off()
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

        except Exception as e:
            print(f"오류: {e}")
            time.sleep(2)

    # 종료 처리
    all_leds_off()
    if GPIO_AVAILABLE:
        GPIO.cleanup()
    print("GPIO 정리 완료.")


if __name__ == "__main__":
    run()
