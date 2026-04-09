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
# ════════════════════════════════════════════════
# 루트 대시보드 HTML  (CSS 이스케이프 없는 .replace() 전용 버전)
# ════════════════════════════════════════════════
ROOT_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>폐기물 분류 AI 서버</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,'Segoe UI',sans-serif;font-size:14px;line-height:1.6;min-height:100vh;}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
.hdr{background:#161b22;border-bottom:1px solid #21262d;padding:16px 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:50;}
.hdr-left{display:flex;align-items:center;gap:14px;}
.logo{width:40px;height:40px;background:linear-gradient(135deg,#238636,#1f6feb);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;}
.hdr-title{font-size:16px;font-weight:600;}
.hdr-sub{font-size:11px;color:#6e7681;font-family:monospace;margin-top:1px;}
.srv-pill{display:flex;align-items:center;gap:7px;background:#161b22;border:1px solid #21262d;border-radius:20px;padding:6px 14px;font-size:12px;font-family:monospace;}
.dot{width:8px;height:8px;border-radius:50%;background:#3fb950;box-shadow:0 0 6px #3fb950;animation:pulse 2s infinite;display:inline-block;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.wrap{max-width:1100px;margin:0 auto;padding:28px 32px;}
.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:28px;}
.card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px 14px;}
.card-val{font-size:22px;font-weight:700;font-family:monospace;color:#58a6ff;margin-bottom:3px;}
.card-lbl{font-size:11px;color:#6e7681;}
.card.green .card-val{color:#3fb950;}
.card.amber .card-val{color:#d29922;}
.sec-title{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#6e7681;font-family:monospace;margin-bottom:12px;}
.tbl-wrap{background:#161b22;border:1px solid #21262d;border-radius:12px;overflow:hidden;}
table{width:100%;border-collapse:collapse;}
thead tr{background:#21262d;}
th{padding:10px 16px;text-align:left;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#6e7681;font-family:monospace;font-weight:400;}
td{padding:12px 16px;border-top:1px solid #21262d;vertical-align:middle;}
tr:hover td{background:#1c2128;}
.ep{font-family:monospace;font-size:12px;color:#e6edf3;}
.desc{font-size:12px;color:#8b949e;}
.badge{display:inline-block;font-family:monospace;font-size:10px;font-weight:700;padding:3px 8px;border-radius:5px;letter-spacing:.5px;}
.GET{background:#1f3d6e;color:#58a6ff;border:1px solid #1f6feb55;}
.POST{background:#1a3d27;color:#3fb950;border:1px solid #2ea04355;}
.run-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.btn{background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:6px;padding:6px 14px;font-size:12px;font-family:monospace;cursor:pointer;transition:all .15s;white-space:nowrap;}
.btn:hover{background:#30363d;border-color:#58a6ff;color:#58a6ff;}
.btn.green:hover{border-color:#3fb950;color:#3fb950;}
input[type=text]{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:6px 10px;color:#e6edf3;font-family:monospace;font-size:12px;outline:none;transition:border .15s;}
input[type=text]:focus{border-color:#58a6ff;}
input[type=file]{font-size:11px;color:#8b949e;font-family:monospace;}
.resp{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px 16px;font-family:monospace;font-size:12px;margin-top:8px;display:none;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;line-height:1.8;}
.resp.show{display:block;animation:fadeIn .2s;}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.jk{color:#79c0ff}.js{color:#a5d6ff}.jn{color:#7ee787}.jb{color:#d2a8ff}
.conf-box{display:none;align-items:center;gap:12px;margin-top:10px;padding:12px 16px;background:#0d1117;border:1px solid #21262d;border-radius:8px;}
.conf-box.show{display:flex;}
.conf-name{font-size:18px;font-weight:700;color:#3fb950;}
.conf-kor{font-size:12px;color:#8b949e;margin-top:2px;}
.conf-track{flex:1;height:6px;background:#21262d;border-radius:3px;overflow:hidden;}
.conf-fill{height:100%;border-radius:3px;transition:width .5s,background .3s;}
.conf-pct{font-family:monospace;font-size:13px;font-weight:700;min-width:40px;text-align:right;}
.foot{margin-top:40px;padding-top:16px;border-top:1px solid #21262d;color:#6e7681;font-size:12px;font-family:monospace;display:flex;gap:20px;align-items:center;}
.foot a{color:#58a6ff;text-decoration:none;}
.foot a:hover{text-decoration:underline;}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-left">
    <div class="logo">&#128465;</div>
    <div>
      <div class="hdr-title">폐기물 분류 AI 서버</div>
      <div class="hdr-sub">localhost:8000 &nbsp;&#183;&nbsp; YOLOv8-cls &nbsp;&#183;&nbsp; FastAPI</div>
    </div>
  </div>
  <div class="srv-pill"><span class="dot"></span><span id="srv-txt">__MODEL_STATUS__</span></div>
</div>
<div class="wrap">
  <div class="cards">
    <div class="card green"><div class="card-val" id="c-total">__TOTAL__</div><div class="card-lbl">총 예측 요청</div></div>
    <div class="card"><div class="card-val" id="c-uptime">__UPTIME__</div><div class="card-lbl">업타임</div></div>
    <div class="card amber"><div class="card-val" id="c-conf">&#8212;</div><div class="card-lbl">평균 신뢰도</div></div>
    <div class="card"><div class="card-val" id="c-lat">&#8212;</div><div class="card-lbl">평균 추론시간</div></div>
    <div class="card"><div class="card-val" id="c-coll">&#8212;</div><div class="card-lbl">수집 이미지</div></div>
    <div class="card"><div class="card-val" id="c-ready" style="font-size:14px">&#8212;</div><div class="card-lbl">재학습 준비</div></div>
  </div>
  <div class="sec-title">API 엔드포인트</div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th style="width:70px">Method</th><th style="width:200px">경로</th><th>설명</th><th style="width:320px">실행</th></tr></thead>
      <tbody>
        <tr><td><span class="badge GET">GET</span></td><td class="ep">/health</td><td class="desc">헬스체크 &#8212; 모델 로드 여부, 업타임</td><td><button class="btn" onclick="runGet('/health','r1')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r1" class="resp"></div></td></tr>
        <tr><td><span class="badge POST">POST</span></td><td class="ep">/predict</td><td class="desc">이미지 업로드 &#8594; 폐기물 분류 예측</td><td><div class="run-row"><input type="file" id="fp" accept="image/*"><button class="btn green" onclick="runPredict()">&#9654; 실행</button></div></td></tr>
        <tr><td colspan="4"><div id="r2" class="resp"></div><div id="conf-box" class="conf-box"><div><div class="conf-name" id="conf-cls"></div><div class="conf-kor" id="conf-kor"></div></div><div class="conf-track"><div class="conf-fill" id="conf-fill"></div></div><div class="conf-pct" id="conf-pct"></div></div></td></tr>
        <tr><td><span class="badge POST">POST</span></td><td class="ep">/detect_multi</td><td class="desc">다중 감지 결과 등록 (아두이노 LED용)</td><td><div class="run-row"><input type="text" id="fm" value="can,paper" style="width:130px"><button class="btn green" onclick="runMulti()">&#9654; 실행</button></div></td></tr>
        <tr><td colspan="4"><div id="r3" class="resp"></div></td></tr>
        <tr><td><span class="badge GET">GET</span></td><td class="ep">/stats</td><td class="desc">서버 통계 &#8212; 클래스별 카운트, 신뢰도, 레이턴시</td><td><button class="btn" onclick="runGet('/stats','r4')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r4" class="resp"></div></td></tr>
        <tr><td><span class="badge GET">GET</span></td><td class="ep">/latest</td><td class="desc">최신 감지 결과 &#8212; 아두이노 step8 폴링용</td><td><button class="btn" onclick="runGet('/latest','r5')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r5" class="resp"></div></td></tr>
        <tr><td><span class="badge GET">GET</span></td><td class="ep">/collected</td><td class="desc">재학습 수집 데이터 현황 &#8212; retrain_ready 포함</td><td><button class="btn" onclick="runGet('/collected','r6')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r6" class="resp"></div></td></tr>
        <tr><td><span class="badge POST">POST</span></td><td class="ep">/webcam/pause</td><td class="desc">웹캠 분류 일시정지</td><td><button class="btn" onclick="runPost('/webcam/pause','r7')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r7" class="resp"></div></td></tr>
        <tr><td><span class="badge POST">POST</span></td><td class="ep">/webcam/resume</td><td class="desc">웹캠 분류 재개</td><td><button class="btn" onclick="runPost('/webcam/resume','r8')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r8" class="resp"></div></td></tr>
        <tr><td><span class="badge POST">POST</span></td><td class="ep">/webcam/capture</td><td class="desc">현재 프레임 캡처 저장 요청</td><td><button class="btn" onclick="runPost('/webcam/capture','r9')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r9" class="resp"></div></td></tr>
        <tr><td><span class="badge GET">GET</span></td><td class="ep">/webcam/state</td><td class="desc">웹캠 제어 상태 (paused, capture 카운트)</td><td><button class="btn" onclick="runGet('/webcam/state','r10')">&#9654; 실행</button></td></tr>
        <tr><td colspan="4"><div id="r10" class="resp"></div></td></tr>
      </tbody>
    </table>
  </div>
  <div class="foot">
    <span><span class="dot"></span> 서버 실행 중</span>
    <span>자동 API 문서: <a href="/docs">/docs</a></span>
    <span><a href="/redoc">/redoc</a></span>
  </div>
</div>
<script>
function hl(j){
  return j.replace(/(\"(?:\\\\u[0-9a-fA-F]{4}|\\\\[^u]|[^\\\\\"])*\"(?:\\s*:)?|\\b(?:true|false|null)\\b|-?\\d+\\.?\\d*)/g,function(m){
    var c='jn';
    if(/^"/.test(m)) c=/:$/.test(m)?'jk':'js';
    else if(/true|false/.test(m)) c='jb';
    return '<span class="'+c+'">'+m+'</span>';
  });
}
function show(id,data){var e=document.getElementById(id);e.innerHTML=hl(JSON.stringify(data,null,2));e.classList.add('show');}
async function runGet(p,id){
  try{var r=await fetch(p);show(id,await r.json());}
  catch(e){var el=document.getElementById(id);el.textContent='오류: '+e.message;el.classList.add('show');}
}
async function runPost(p,id){
  try{var r=await fetch(p,{method:'POST'});show(id,await r.json());}
  catch(e){var el=document.getElementById(id);el.textContent='오류: '+e.message;el.classList.add('show');}
}
async function runPredict(){
  var f=document.getElementById('fp').files[0];
  if(!f){alert('이미지를 먼저 선택하세요');return;}
  var fd=new FormData();fd.append('file',f);
  try{
    var r=await fetch('/predict',{method:'POST',body:fd});
    var d=await r.json();show('r2',d);
    if(d.confidence){
      var pct=Math.round(d.confidence*100);
      var color=pct>=80?'#3fb950':pct>=60?'#d29922':'#f85149';
      document.getElementById('conf-cls').textContent=d.class_name||'';
      document.getElementById('conf-cls').style.color=color;
      document.getElementById('conf-kor').textContent=d.class_kor||'';
      document.getElementById('conf-fill').style.width=pct+'%';
      document.getElementById('conf-fill').style.background=color;
      document.getElementById('conf-pct').textContent=pct+'%';
      document.getElementById('conf-pct').style.color=color;
      document.getElementById('conf-box').classList.add('show');
    }
  }catch(e){var el=document.getElementById('r2');el.textContent='오류: '+e.message;el.classList.add('show');}
}
async function runMulti(){
  var cls=document.getElementById('fm').value.split(',').map(function(s){return s.trim();}).filter(Boolean);
  try{
    var r=await fetch('/detect_multi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({classes:cls})});
    show('r3',await r.json());
  }catch(e){var el=document.getElementById('r3');el.textContent='오류: '+e.message;el.classList.add('show');}
}
async function loadCards(){
  try{
    var s=await(await fetch('/stats')).json();
    if(s.avg_confidence)document.getElementById('c-conf').textContent=(s.avg_confidence*100).toFixed(1)+'%';
    if(s.avg_latency_ms)document.getElementById('c-lat').textContent=s.avg_latency_ms.toFixed(1)+'ms';
    document.getElementById('c-total').textContent=s.total_requests||0;
  }catch(e){}
  try{
    var c=await(await fetch('/collected')).json();
    var tot=Object.values(c.counts||{}).reduce(function(a,b){return a+b;},0);
    document.getElementById('c-coll').textContent=tot;
    document.getElementById('c-ready').textContent=c.retrain_ready?'Ready':'Collecting...';
    document.getElementById('c-ready').style.color=c.retrain_ready?'#3fb950':'#d29922';
  }catch(e){}
  try{
    var h=await(await fetch('/health')).json();
    document.getElementById('c-uptime').textContent=h.uptime||'--';
  }catch(e){}
}
loadCards();
setInterval(loadCards,10000);
</script>
</body>
</html>"""


@app.get("/", summary="서버 상태 확인", response_class=HTMLResponse)
async def root():
    uptime = str(datetime.now() - stats["start_time"]).split(".")[0]
    model_status = "Model Loaded" if model is not None else "Model Not Loaded"
    html = ROOT_HTML \
        .replace("__UPTIME__", uptime) \
        .replace("__MODEL_STATUS__", model_status) \
        .replace("__TOTAL__", str(stats["total"]))
    return HTMLResponse(content=html)

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
