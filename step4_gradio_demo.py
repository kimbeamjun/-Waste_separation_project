"""
[4단계] Gradio 웹 데모 배포
이미지 업로드 → 폐기물 감지 결과 표시하는 웹 인터페이스

실행 환경: PyCharm 로컬 OR Google Colab 둘 다 가능
실행 방법: python step4_gradio_demo.py
접속 주소: http://localhost:7860 (로컬) 또는 Colab share URL
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# ────────────────────────────────────────────────
# 설정값
# ────────────────────────────────────────────────
WEIGHTS_PATH   = "./best.pt"
CONF_THRESHOLD = 0.5
IMG_SIZE       = 640

CLASS_KOR = {
    0: "과자봉지/비닐",
    1: "유리병/음료수병",
    2: "종이류/포장상자",
    3: "캔/음료수캔",
    4: "페트병",
}

CLASS_COLORS = {
    0: (255, 165, 0),    # 주황 (RGB)
    1: (0,   0,   255),  # 파랑
    2: (0,   200, 0),    # 초록
    3: (255, 0,   0),    # 빨강
    4: (200, 0,   200),  # 보라
}

# 모델 로드 (서버 시작 시 한 번만)
print("모델 로딩 중...")
if not Path(WEIGHTS_PATH).exists():
    print(f"❌ {WEIGHTS_PATH} 없음. best.pt를 같은 폴더에 넣어주세요.")
    model = None
else:
    model = YOLO(WEIGHTS_PATH)
    print("✅ 모델 로드 완료")


def predict_image(input_image):
    """
    Gradio 인터페이스 핵심 함수
    입력: PIL Image
    출력: (결과 이미지, 감지 결과 텍스트)
    """
    if model is None:
        return input_image, "❌ 모델 파일(best.pt)이 없습니다."

    # PIL → numpy 변환
    img_np = np.array(input_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 추론
    results = model.predict(
        source  = img_bgr,
        imgsz   = IMG_SIZE,
        conf    = CONF_THRESHOLD,
        verbose = False,
    )

    # 결과 정리
    detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color  = CLASS_COLORS.get(cls_id, (128, 128, 128))
        label  = f"{CLASS_KOR.get(cls_id, cls_id)} ({conf:.0%})"

        # 바운딩박스 그리기 (RGB 기준)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_np, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_np, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detected.append(f"• {label}  위치: ({x1},{y1}) ~ ({x2},{y2})")

    # 결과 텍스트
    if detected:
        result_text = f"🗑️ 감지된 폐기물: {len(detected)}개\n\n" + "\n".join(detected)
    else:
        result_text = "❌ 감지된 폐기물이 없습니다. (신뢰도 임계값: {CONF_THRESHOLD})"

    output_image = Image.fromarray(img_np)
    return output_image, result_text


# ────────────────────────────────────────────────
# Gradio UI 구성
# ────────────────────────────────────────────────
demo = gr.Interface(
    fn          = predict_image,
    inputs      = gr.Image(type="pil", label="폐기물 이미지 업로드"),
    outputs     = [
        gr.Image(type="pil", label="감지 결과"),
        gr.Textbox(label="상세 결과", lines=8),
    ],
    title       = "🗑️ 딥러닝 기반 생활 폐기물 자동 분류 시스템",
    description = "이미지를 업로드하면 비닐/과자봉지, 유리병, 종이류, 캔, 페트병을 자동으로 감지합니다.",
    examples    = [],   # 예시 이미지 있으면 경로 추가
    theme       = gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(
        share  = True,   # True: 외부 공유 URL 생성 (포트폴리오용)
        server_port = 7860,
    )
