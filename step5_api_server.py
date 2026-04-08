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
ROOT_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>폐기물 분류 AI 서버</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Noto+Sans+KR:wght@300;400;600&display=swap');
  :root{{--bg:#0b0f14;--bg2:#111720;--bg3:#181f2a;--bd:#1e2d3d;--green:#00e676;--blue:#40c4ff;--amber:#ffab40;--red:#ff5252;--text:#cdd9e5;--muted:#4a5568;}}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:var(--bg);color:var(--text);font-family:'Noto Sans KR',sans-serif;font-size:14px;line-height:1.6;padding:32px;}}
  h1{{font-size:22px;font-weight:600;letter-spacing:-0.5px;display:flex;align-items:center;gap:12px;margin-bottom:4px;}}
  .sub{{color:var(--muted);font-size:13px;margin-bottom:28px;font-family:'JetBrains Mono',monospace;}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:32px;}}
  .card{{background:var(--bg2);border:1px solid var(--bd);border-radius:10px;padding:18px 16px;}}
  .card .val{{font-size:24px;font-weight:600;color:var(--green);font-family:'JetBrains Mono',monospace;}}
  .card .lbl{{font-size:11px;color:var(--muted);margin-top:4px;}}
  .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);margin-right:6px;}}
  .section-title{{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-bottom:12px;margin-top:28px;}}
  table{{width:100%;border-collapse:collapse;background:var(--bg2);border-radius:10px;overflow:hidden;border:1px solid var(--bd);}}
  th{{background:var(--bg3);padding:10px 16px;text-align:left;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);font-family:'JetBrains Mono',monospace;font-weight:400;}}
  td{{padding:11px 16px;border-top:1px solid var(--bd);font-size:13px;vertical-align:middle;}}
  tr:hover td{{background:var(--bg3);}}
  .badge{{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;padding:3px 8px;border-radius:4px;min-width:44px;text-align:center;}}
  .GET{{background:#1a3a5c;color:var(--blue);}}
  .POST{{background:#1a3d27;color:var(--green);}}
  .ep-path{{font-family:'JetBrains Mono',monospace;font-size:12px;}}
  .try-btn{{background:var(--bg3);border:1px solid var(--bd);color:var(--text);border-radius:6px;padding:5px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;cursor:pointer;transition:all .15s;text-decoration:none;display:inline-block;}}
  .try-btn:hover{{border-color:var(--green);color:var(--green);}}
  .resp-box{{background:var(--bg3);border:1px solid var(--bd);border-radius:8px;padding:14px 16px;font-family:'JetBrains Mono',monospace;font-size:12px;margin-top:8px;display:none;white-space:pre-wrap;word-break:break-all;max-height:220px;overflow-y:auto;line-height:1.7;}}
  .resp-box.show{{display:block;}}
  .json-key{{color:var(--blue);}} .json-str{{color:var(--amber);}} .json-num{{color:#b5e8b0;}} .json-bool{{color:#c792ea;}}
  .run-row{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}}
  input[type=text]{{background:var(--bg3);border:1px solid var(--bd);border-radius:6px;padding:5px 10px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;outline:none;transition:border .15s;}}
  input[type=text]:focus{{border-color:var(--green);}}
  input[type=file]{{color:var(--muted);font-size:12px;font-family:'JetBrains Mono',monospace;}}
  .conf-bar{{height:6px;background:var(--bg);border-radius:3px;overflow:hidden;margin-top:6px;width:180px;display:inline-block;vertical-align:middle;}}
  .conf-fill{{height:100%;border-radius:3px;transition:width .4s;}}
  footer{{margin-top:40px;color:var(--muted);font-size:11px;font-family:'JetBrains Mono',monospace;border-top:1px solid var(--bd);padding-top:16px;}}
  ::-webkit-scrollbar{{width:4px;}} ::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:2px;}}
</style>
</head>
<body>

<h1><span style="font-size:28px">🗑️</span> 폐기물 분류 AI 서버</h1>
<div class="sub">http://localhost:8000 &nbsp;|&nbsp; YOLOv8-cls &nbsp;|&nbsp; FastAPI</div>

<!-- 상태 카드 -->
<div class="cards">
  <div class="card">
    <div class="val" id="c-total">{total}</div>
    <div class="lbl">총 예측 요청 수</div>
  </div>
  <div class="card">
    <div class="val" id="c-uptime">{uptime}</div>
    <div class="lbl">서버 업타임</div>
  </div>
  <div class="card">
    <div class="val" id="c-model" style="font-size:16px">{model_status}</div>
    <div class="lbl">모델 상태</div>
  </div>
  <div class="card">
    <div class="val" id="c-conf">—</div>
    <div class="lbl">평균 신뢰도</div>
  </div>
  <div class="card">
    <div class="val" id="c-lat">—</div>
    <div class="lbl">평균 추론 시간</div>
  </div>
  <div class="card">
    <div class="val" id="c-collected">—</div>
    <div class="lbl">수집 이미지 수</div>
  </div>
</div>

<!-- API 표 -->
<div class="section-title">API 엔드포인트</div>
<table>
  <thead><tr><th>Method</th><th>경로</th><th>설명</th><th>실행</th></tr></thead>
  <tbody>

    <tr>
      <td><span class="badge GET">GET</span></td>
      <td class="ep-path">/health</td>
      <td>헬스체크 — 모델 로드 여부, 업타임</td>
      <td><button class="try-btn" onclick="runGet('/health','r-health')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-health" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge POST">POST</span></td>
      <td class="ep-path">/predict</td>
      <td>이미지 업로드 → 폐기물 분류 예측</td>
      <td>
        <div class="run-row">
          <input type="file" id="f-predict" accept="image/*">
          <button class="try-btn" onclick="runPredict()">▶ 실행</button>
        </div>
      </td>
    </tr>
    <tr><td colspan="4">
      <div id="r-predict" class="resp-box"></div>
      <div id="conf-visual" style="display:none;padding:10px 0 4px;font-size:13px;">
        <span id="conf-class" style="font-weight:600;color:var(--green)"></span>
        <span id="conf-kor" style="color:var(--muted);margin-left:8px;"></span>
        <span class="conf-bar"><span class="conf-fill" id="conf-fill"></span></span>
        <span id="conf-pct" style="font-family:'JetBrains Mono',monospace;font-size:12px;margin-left:8px;"></span>
      </div>
    </td></tr>

    <tr>
      <td><span class="badge POST">POST</span></td>
      <td class="ep-path">/detect_multi</td>
      <td>다중 감지 결과 등록 (아두이노용)</td>
      <td>
        <div class="run-row">
          <input type="text" id="f-multi" value="can,paper" placeholder="can,paper" style="width:130px">
          <button class="try-btn" onclick="runMulti()">▶ 실행</button>
        </div>
      </td>
    </tr>
    <tr><td colspan="4"><div id="r-multi" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge GET">GET</span></td>
      <td class="ep-path">/stats</td>
      <td>서버 통계 — 클래스별 카운트, 신뢰도, 레이턴시</td>
      <td><button class="try-btn" onclick="runGet('/stats','r-stats')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-stats" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge GET">GET</span></td>
      <td class="ep-path">/latest</td>
      <td>최신 감지 결과 (아두이노 step8 폴링용)</td>
      <td><button class="try-btn" onclick="runGet('/latest','r-latest')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-latest" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge GET">GET</span></td>
      <td class="ep-path">/collected</td>
      <td>재학습 수집 데이터 현황 — retrain_ready 포함</td>
      <td><button class="try-btn" onclick="runGet('/collected','r-collected')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-collected" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge POST">POST</span></td>
      <td class="ep-path">/webcam/pause</td>
      <td>웹캠 분류 일시정지</td>
      <td><button class="try-btn" onclick="runPost('/webcam/pause','r-pause')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-pause" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge POST">POST</span></td>
      <td class="ep-path">/webcam/resume</td>
      <td>웹캠 분류 재개</td>
      <td><button class="try-btn" onclick="runPost('/webcam/resume','r-resume')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-resume" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge POST">POST</span></td>
      <td class="ep-path">/webcam/capture</td>
      <td>현재 프레임 캡처 저장 요청</td>
      <td><button class="try-btn" onclick="runPost('/webcam/capture','r-capture')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-capture" class="resp-box"></div></td></tr>

    <tr>
      <td><span class="badge GET">GET</span></td>
      <td class="ep-path">/webcam/state</td>
      <td>웹캠 제어 상태 조회 (paused, capture 카운트)</td>
      <td><button class="try-btn" onclick="runGet('/webcam/state','r-wstate')">▶ 실행</button></td>
    </tr>
    <tr><td colspan="4"><div id="r-wstate" class="resp-box"></div></td></tr>

  </tbody>
</table>

<footer>
  <span class="dot"></span> 서버 실행 중 &nbsp;|&nbsp;
  자동 API 문서: <a href="/docs" style="color:var(--blue)">/docs</a> &nbsp;|&nbsp;
  <a href="/redoc" style="color:var(--blue)">/redoc</a>
</footer>

<script>
function hl(json) {
  return json.replace(/("(\\\\u[a-zA-Z0-9]{{4}}|\\\\[^u]|[^\\\\"])*"(\\s*:)?|\\b(true|false|null)\\b|-?\\d+\\.?\\d*)/g, m => {{
    let c = 'json-num';
    if (/^"/.test(m)) c = /:$/.test(m) ? 'json-key' : 'json-str';
    else if (/true|false/.test(m)) c = 'json-bool';
    return `<span class="${{c}}">${{m}}</span>`;
  }});
}
function show(id, data) {{
  const el = document.getElementById(id);
  el.innerHTML = hl(JSON.stringify(data, null, 2));
  el.classList.add('show');
}}
async function runGet(path, rid) {{
  try {{
    const r = await fetch(path);
    show(rid, await r.json());
  }} catch(e) {{ document.getElementById(rid).textContent = '오류: ' + e.message; document.getElementById(rid).classList.add('show'); }}
}}
async function runPost(path, rid) {{
  try {{
    const r = await fetch(path, {{method:'POST'}});
    show(rid, await r.json());
  }} catch(e) {{ document.getElementById(rid).textContent = '오류: ' + e.message; document.getElementById(rid).classList.add('show'); }}
}}
async function runPredict() {{
  const f = document.getElementById('f-predict').files[0];
  if (!f) {{ alert('이미지를 먼저 선택하세요'); return; }}
  const fd = new FormData(); fd.append('file', f);
  try {{
    const r = await fetch('/predict', {{method:'POST', body:fd}});
    const d = await r.json(); show('r-predict', d);
    if (d.confidence) {{
      const pct = Math.round(d.confidence*100);
      const color = pct>=80?'#00e676':pct>=60?'#ffab40':'#ff5252';
      document.getElementById('conf-class').textContent = d.class_name;
      document.getElementById('conf-class').style.color = color;
      document.getElementById('conf-kor').textContent = d.class_kor||'';
      document.getElementById('conf-fill').style.width = pct+'%';
      document.getElementById('conf-fill').style.background = color;
      document.getElementById('conf-pct').textContent = pct+'%  '+d.inference_ms+'ms';
      document.getElementById('conf-visual').style.display='block';
    }}
  }} catch(e) {{ document.getElementById('r-predict').textContent='오류: '+e.message; document.getElementById('r-predict').classList.add('show'); }}
}}
async function runMulti() {{
  const classes = document.getElementById('f-multi').value.split(',').map(s=>s.trim()).filter(Boolean);
  try {{
    const r = await fetch('/detect_multi',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{classes}})}});
    show('r-multi', await r.json());
  }} catch(e) {{ document.getElementById('r-multi').textContent='오류: '+e.message; document.getElementById('r-multi').classList.add('show'); }}
}}

// 자동 stats 로드
async function loadStats() {{
  try {{
    const s = await (await fetch('/stats')).json();
    if (s.avg_confidence) document.getElementById('c-conf').textContent = (s.avg_confidence*100).toFixed(1)+'%';
    if (s.avg_latency_ms) document.getElementById('c-lat').textContent = s.avg_latency_ms.toFixed(1)+'ms';
    document.getElementById('c-total').textContent = s.total_requests||0;
  }} catch {{}}
  try {{
    const c = await (await fetch('/collected')).json();
    const total = Object.values(c.counts||{{}}).reduce((a,b)=>a+b,0);
    document.getElementById('c-collected').textContent = total;
  }} catch {{}}
  try {{
    const h = await (await fetch('/health')).json();
    document.getElementById('c-uptime').textContent = h.uptime||'—';
  }} catch {{}}
}}
loadStats();
setInterval(loadStats, 10000);
</script>
</body>
</html>"""


@app.get("/", summary="서버 상태 확인", response_class=HTMLResponse)
async def root():
    from fastapi.responses import HTMLResponse
    uptime = str(datetime.now() - stats["start_time"]).split(".")[0]
    model_status = "✅ 로드됨" if model is not None else "❌ 미로드"
    return HTMLResponse(content=ROOT_HTML.format(
        uptime=uptime,
        model_status=model_status,
        total=stats["total"],
    ))


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
