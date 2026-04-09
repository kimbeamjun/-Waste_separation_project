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
import sqlite3
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
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
DB_PATH        = "./logs/predictions.db"  # SQLite DB 경로

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


# ════════════════════════════════════════════════
# SQLite DB 초기화 및 유틸
# ════════════════════════════════════════════════
_db_lock = threading.Lock()

def init_db():
    """DB 파일 생성 및 테이블 초기화"""
    Path(DB_PATH).parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                class_name   TEXT    NOT NULL,
                class_kor    TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                inference_ms REAL    NOT NULL,
                filename     TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON predictions (timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_class
            ON predictions (class_name)
        """)
        conn.commit()
    logger.info(f"✅ SQLite DB 초기화: {DB_PATH}")

def db_insert(timestamp, class_name, class_kor, confidence, inference_ms, filename):
    """예측 결과를 DB에 비동기 삽입 (메인 루프 블로킹 방지)"""
    def _insert():
        with _db_lock:
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO predictions "
                        "(timestamp, class_name, class_kor, confidence, inference_ms, filename) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (timestamp, class_name, class_kor,
                         round(confidence, 4), round(inference_ms, 2), filename)
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"DB 삽입 오류: {e}")
    threading.Thread(target=_insert, daemon=True).start()

@app.on_event("startup")
async def load_model():
    global model
    if not Path(WEIGHTS_PATH).exists():
        logger.error(f"모델 파일 없음: {WEIGHTS_PATH}")
        return
    init_db()
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
# 서버 통계 + 웹캠 제어 상태
# ════════════════════════════════════════════════
stats = {
    "total":       0,
    "class_counts": {name: 0 for name in CLASS_KOR},
    "confidences": [],
    "latencies":   [],
    "start_time":  datetime.now(),
}

# 최신 감지 결과 (아두이노 LED 제어용)
latest_detection = {
    "classes":    [],
    "timestamp":  "",
    "confidence": 0.0,   # 단일 감지 시 신뢰도 (0.0~1.0)
}

# 웹캠 제어 상태
webcam_state = {
    "paused":           False,
    "capture_total":    0,    # 버튼1 누른 총 횟수 (누적, 절대 줄지 않음)
    "capture_done":     0,    # step3가 처리 완료한 횟수
    "capture_queue": 0,
}


# ════════════════════════════════════════════════
# API 엔드포인트
# ════════════════════════════════════════════════
# ════════════════════════════════════════════════
# 루트 대시보드 HTML
# ════════════════════════════════════════════════
# ════════════════════════════════════════════════
# 루트 대시보드 HTML  (CSS 이스케이프 없는 .replace() 전용 버전)
# ════════════════════════════════════════════════
# ════════════════════════════════════════════════
# 루트 대시보드 HTML — dashboard.html 파일에서 로드
# ════════════════════════════════════════════════
_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"

def _load_dashboard(uptime: str, model_status: str, total: int) -> str:
    """dashboard.html 을 읽어서 플레이스홀더 치환 후 반환"""
    if not _DASHBOARD_PATH.exists():
        return (
            "<h2>dashboard.html 파일이 없습니다.</h2>"
            "<p>step5_api_server.py 와 같은 폴더에 dashboard.html 을 넣어주세요.</p>"
        )
    html = _DASHBOARD_PATH.read_text(encoding="utf-8")
    return (
        html
        .replace("__UPTIME__",       uptime)
        .replace("__MODEL_STATUS__", model_status)
        .replace("__TOTAL__",        str(total))
    )




# ════════════════════════════════════════════════
# 예측 로그 DB 조회 API
# ════════════════════════════════════════════════
@app.get("/logs", summary="예측 로그 조회")
async def get_logs(
    limit:      int = 50,
    class_name: str = None,
    date:       str = None,
):
    """
    SQLite DB에서 예측 로그를 조회합니다.

    - **limit**: 최대 조회 수 (기본 50, 최대 500)
    - **class_name**: 특정 클래스만 필터 (예: can, paper)
    - **date**: 날짜 필터 (예: 2026-04-10)
    """
    limit = min(limit, 500)
    query = "SELECT id, timestamp, class_name, class_kor, confidence, inference_ms, filename FROM predictions"
    params = []
    conditions = []

    if class_name:
        conditions.append("class_name = ?")
        params.append(class_name)
    if date:
        conditions.append("timestamp LIKE ?")
        params.append(f"{date}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return {
            "count": len(rows),
            "limit": limit,
            "logs":  [dict(r) for r in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/summary", summary="예측 로그 통계 요약")
async def get_logs_summary():
    """
    DB 전체 예측 로그 기반 통계를 반환합니다.
    - 총 예측 수, 클래스별 카운트 및 비율
    - 시간대별 예측 분포 (0~23시)
    - 평균/최소/최대 신뢰도 및 추론 시간
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            if total == 0:
                return {"total": 0, "message": "No predictions yet"}

            # 클래스별 카운트
            class_rows = conn.execute(
                "SELECT class_name, class_kor, COUNT(*) as cnt, "
                "AVG(confidence) as avg_conf "
                "FROM predictions GROUP BY class_name ORDER BY cnt DESC"
            ).fetchall()

            # 시간대별 분포
            hour_rows = conn.execute(
                "SELECT SUBSTR(timestamp, 12, 2) as hour, COUNT(*) as cnt "
                "FROM predictions GROUP BY hour ORDER BY hour"
            ).fetchall()

            # 전체 성능 지표
            perf = conn.execute(
                "SELECT AVG(confidence) as avg_conf, MIN(confidence) as min_conf, "
                "MAX(confidence) as max_conf, AVG(inference_ms) as avg_ms, "
                "MIN(inference_ms) as min_ms, MAX(inference_ms) as max_ms "
                "FROM predictions"
            ).fetchone()

            # 날짜별 최근 7일
            daily_rows = conn.execute(
                "SELECT SUBSTR(timestamp, 1, 10) as date, COUNT(*) as cnt "
                "FROM predictions "
                "GROUP BY date ORDER BY date DESC LIMIT 7"
            ).fetchall()

        return {
            "total": total,
            "by_class": [
                {
                    "class_name": r[0],
                    "class_kor":  r[1],
                    "count":      r[2],
                    "ratio":      round(r[2] / total * 100, 1),
                    "avg_conf":   round(r[3] * 100, 1),
                }
                for r in class_rows
            ],
            "by_hour":  {r[0]: r[1] for r in hour_rows},
            "by_date":  {r[0]: r[1] for r in daily_rows},
            "performance": {
                "avg_confidence":  round(perf[0] * 100, 1),
                "min_confidence":  round(perf[1] * 100, 1),
                "max_confidence":  round(perf[2] * 100, 1),
                "avg_latency_ms":  round(perf[3], 2),
                "min_latency_ms":  round(perf[4], 2),
                "max_latency_ms":  round(perf[5], 2),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/logs", summary="예측 로그 전체 삭제")
async def clear_logs():
    """DB의 모든 예측 로그를 삭제합니다. (초기화용)"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM predictions")
            conn.commit()
        return {"ok": True, "message": "All logs cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="서버 상태 확인", response_class=HTMLResponse)
async def root():
    uptime       = str(datetime.now() - stats["start_time"]).split(".")[0]
    model_status = "Model Loaded" if model is not None else "Model Not Loaded"
    return HTMLResponse(content=_load_dashboard(uptime, model_status, stats["total"]))


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

    # 최신 감지 결과 업데이트 (아두이노 LED 제어용)
    latest_detection["classes"]    = [class_name]
    latest_detection["timestamp"]  = timestamp
    latest_detection["confidence"] = round(top1_conf, 4)

    # 수신 이미지 저장 (collected_data 폴더에 저장)
    if SAVE_IMAGES:
        save_dir = Path(SAVE_IMAGE_DIR) / class_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{timestamp.replace(':', '-')}_{stats['total']:06d}.jpg"
        img_pil.save(save_path)

    # DB에 예측 로그 저장
    db_insert(timestamp, class_name, class_kor, top1_conf, elapsed, file.filename)

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


# ── 웹캠 제어 엔드포인트 ─────────────────────────────

@app.post("/webcam/pause", summary="웹캠 분류 일시정지")
async def webcam_pause():
    webcam_state["paused"] = True
    logger.info("웹캠 일시정지")
    return {"paused": True}

@app.post("/webcam/resume", summary="웹캠 분류 재개")
async def webcam_resume():
    webcam_state["paused"] = False
    logger.info("웹캠 재개")
    return {"paused": False}

@app.post("/webcam/capture", summary="웹캠 캡처 저장 요청")
async def webcam_capture():
    webcam_state["capture_total"] += 1
    webcam_state["capture_queue"] += 1
    pending = webcam_state["capture_total"] - webcam_state["capture_done"]
    logger.info(f"캡처 요청 (총: {webcam_state['capture_total']}, 대기: {pending})")
    return {
        "capture_total": webcam_state["capture_total"],
        "capture_done":  webcam_state["capture_done"],
        "pending":       pending,
    }

@app.post("/webcam/capture_done", summary="캡처 완료 처리")
async def webcam_capture_done():
    webcam_state["capture_done"] += 1
    return {
        "capture_total": webcam_state["capture_total"],
        "capture_done":  webcam_state["capture_done"],
    }

@app.get("/webcam/state", summary="웹캠 제어 상태 조회")
async def webcam_get_state():
    return {
        "paused":        webcam_state["paused"],
        "capture_total": webcam_state["capture_total"],
        "capture_done":  webcam_state["capture_done"],
        "pending":       webcam_state["capture_total"] - webcam_state["capture_done"],
    }

@app.get("/latest", summary="최신 감지 결과 (아두이노용)")
async def get_latest():
    """아두이노 step8이 폴링하는 엔드포인트. 최신 감지 클래스 목록 반환."""
    return latest_detection

@app.post("/detect_multi", summary="다중 감지 결과 등록")
async def detect_multi(data: dict):
    """step3(웹캠)이 다중 감지 결과를 등록. 아두이노가 즉시 수신 가능."""
    classes = data.get("classes", [])
    latest_detection["classes"]   = classes
    latest_detection["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 감지된 이미지 저장 (캡처 저장 요청 있을 때)
    if webcam_state["capture_queue"] > 0:
        webcam_state["capture_queue"] -= 1
        img_data = data.get("image")
        if img_data:
            ts       = latest_detection["timestamp"].replace(":", "-")
            save_dir = Path(SAVE_IMAGE_DIR) / "captures"
            save_dir.mkdir(parents=True, exist_ok=True)
            import base64
            with open(save_dir / f"cap_{ts}.jpg", "wb") as f:
                f.write(base64.b64decode(img_data))
            logger.info(f"캡처 저장: cap_{ts}.jpg")

    return {"ok": True, "classes": classes}


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
