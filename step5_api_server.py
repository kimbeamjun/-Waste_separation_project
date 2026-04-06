"""
[5단계] FastAPI AI 서버
클라이언트로부터 이미지를 받아 폐기물 분류 결과를 JSON으로 반환

실행 환경: PyCharm 로컬 (Windows 10) - AI 서버 역할
실행 방법: python step5_api_server.py
접속 주소: http://localhost:8000
API 문서 : http://localhost:8000/docs  (자동 생성)

[구조]
웹캠 클라이언트 (step3_webcam.py)
    ↓ HTTP POST /predict (이미지 전송)
FastAPI AI 서버 (step5_api_server.py)  ← 이 파일
    ↓ JSON 응답 반환
클라이언트에서 결과 화면 표시
"""

import io
import time
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import uvicorn

# ════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════
WEIGHTS_PATH   = "./best.pt"       # 학습된 모델 경로
CONF_THRESHOLD = 0.5               # 분류 신뢰도 임계값
HOST           = "0.0.0.0"        # 모든 IP 허용 (같은 네트워크 PC도 접근 가능)
PORT           = 8000
LOG_DIR        = "./logs"          # 예측 로그 저장 폴더 (피드백 루프용)
SAVE_IMAGES    = True              # 수신 이미지 저장 여부 (재학습 데이터 수집용)
SAVE_IMAGE_DIR = "./collected_data"  # 수집 이미지 저장 폴더

CLASS_KOR = {
    "snack_bag":    "과자봉지/비닐",
    "glass_bottle": "유리병/음료수병",
    "paper":        "종이류/포장상자",
    "can":          "캔/음료수캔",
    "pet_bottle":   "페트병",
}

# ════════════════════════════════════════════════
# 로깅 설정
# ════════════════════════════════════════════════
Path(LOG_DIR).mkdir(exist_ok=True)
Path(SAVE_IMAGE_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/server.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════
# FastAPI 앱 초기화
# ════════════════════════════════════════════════
app = FastAPI(
    title="폐기물 분류 AI 서버",
    description="YOLOv8 Classification 기반 생활 폐기물 자동 분류 API",
    version="1.0.0",
)

# CORS 설정 (다른 PC에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════
# 모델 로드 (서버 시작 시 1회만)
# ════════════════════════════════════════════════
model = None

@app.on_event("startup")
async def load_model():
    global model
    if not Path(WEIGHTS_PATH).exists():
        logger.error(f"모델 파일 없음: {WEIGHTS_PATH}")
        return
    model = YOLO(WEIGHTS_PATH)
    logger.info(f"✅ 모델 로드 완료: {WEIGHTS_PATH}")


# ════════════════════════════════════════════════
# 응답 스키마 (JSON 구조 정의)
# ════════════════════════════════════════════════
class PredictResponse(BaseModel):
    success:      bool
    class_name:   str           # 영문 클래스명
    class_kor:    str           # 한글 클래스명
    confidence:   float         # 신뢰도 (0.0 ~ 1.0)
    inference_ms: float         # 추론 소요 시간 (ms)
    timestamp:    str           # 예측 시각

class HealthResponse(BaseModel):
    status:      str
    model_ready: bool
    uptime:      str

class StatsResponse(BaseModel):
    total_requests:  int
    class_counts:    dict
    avg_confidence:  float
    avg_latency_ms:  float


# ════════════════════════════════════════════════
# 서버 통계 (피드백 루프용 데이터 수집)
# ════════════════════════════════════════════════
stats = {
    "total":       0,
    "class_counts": {name: 0 for name in CLASS_KOR},
    "confidences": [],
    "latencies":   [],
    "start_time":  datetime.now(),
}


# ════════════════════════════════════════════════
# API 엔드포인트
# ════════════════════════════════════════════════

@app.get("/", summary="서버 상태 확인")
async def root():
    return {"message": "폐기물 분류 AI 서버 정상 작동 중", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, summary="헬스체크")
async def health():
    uptime = str(datetime.now() - stats["start_time"]).split(".")[0]
    return HealthResponse(
        status      = "ok",
        model_ready = model is not None,
        uptime      = uptime,
    )


@app.post("/predict", response_model=PredictResponse, summary="폐기물 분류 예측")
async def predict(file: UploadFile = File(..., description="분류할 이미지 파일")):
    """
    이미지를 업로드하면 폐기물 종류를 분류하여 JSON으로 반환합니다.

    - **file**: jpg/png 이미지 파일
    - **반환**: 클래스명, 신뢰도, 추론 시간
    """
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    # 이미지 읽기
    try:
        contents = await file.read()
        img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np   = np.array(img_pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 읽기 실패: {e}")

    # 추론
    start_ms = time.time()
    results  = model.predict(img_np, imgsz=224, verbose=False)
    elapsed  = (time.time() - start_ms) * 1000  # ms 변환

    # 결과 파싱
    probs      = results[0].probs
    top1_idx   = int(probs.top1)
    top1_conf  = float(probs.top1conf)
    class_name = results[0].names[top1_idx]
    class_kor  = CLASS_KOR.get(class_name, class_name)
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 통계 업데이트 (피드백 루프용)
    stats["total"] += 1
    if class_name in stats["class_counts"]:
        stats["class_counts"][class_name] += 1
    stats["confidences"].append(top1_conf)
    stats["latencies"].append(elapsed)

    # 수신 이미지 저장 (재학습 데이터 수집)
    if SAVE_IMAGES:
        save_dir = Path(SAVE_IMAGE_DIR) / class_name
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{timestamp.replace(':', '-')}_{stats['total']:06d}.jpg"
        img_pil.save(save_path)

    logger.info(
        f"예측: {class_kor} ({top1_conf:.1%}) | "
        f"추론: {elapsed:.1f}ms | "
        f"파일: {file.filename}"
    )

    return PredictResponse(
        success      = True,
        class_name   = class_name,
        class_kor    = class_kor,
        confidence   = round(top1_conf, 4),
        inference_ms = round(elapsed, 2),
        timestamp    = timestamp,
    )


@app.get("/stats", response_model=StatsResponse, summary="서버 통계 (피드백 루프용)")
async def get_stats():
    """
    지금까지의 예측 통계를 반환합니다.
    수집된 데이터는 재학습에 활용할 수 있습니다.
    """
    avg_conf    = sum(stats["confidences"]) / len(stats["confidences"]) \
                  if stats["confidences"] else 0.0
    avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) \
                  if stats["latencies"] else 0.0

    return StatsResponse(
        total_requests = stats["total"],
        class_counts   = stats["class_counts"],
        avg_confidence = round(avg_conf, 4),
        avg_latency_ms = round(avg_latency, 2),
    )


@app.get("/collected", summary="수집된 재학습 데이터 현황")
async def collected_data():
    """
    피드백 루프: 서버가 수집한 이미지 현황을 반환합니다.
    충분히 쌓이면 이 데이터로 모델을 재학습할 수 있습니다.
    """
    result = {}
    base   = Path(SAVE_IMAGE_DIR)
    for cls_dir in base.iterdir():
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")))
            result[cls_dir.name] = count
    total = sum(result.values())
    return {
        "collected_per_class": result,
        "total":               total,
        "save_dir":            str(base.resolve()),
        "retrain_ready":       total >= 500,   # 500장 이상이면 재학습 권장
    }


# ════════════════════════════════════════════════
# 서버 실행
# ════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("  폐기물 분류 AI 서버 시작")
    logger.info(f"  주소  : http://{HOST}:{PORT}")
    logger.info(f"  API문서: http://localhost:{PORT}/docs")
    logger.info("=" * 50)
    uvicorn.run("step5_api_server:app", host=HOST, port=PORT, reload=False)
