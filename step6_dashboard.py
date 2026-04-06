"""
[6단계] 실시간 웹 대시보드
FastAPI 서버 통계를 브라우저에서 실시간으로 시각화

실행 환경: PyCharm 로컬 (Windows 10)
실행 방법: python step6_dashboard.py
접속 주소: http://localhost:8080
사전 조건: step5_api_server.py 가 8000번 포트에서 실행 중이어야 함
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="폐기물 분류 AI 대시보드")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>폐기물 분류 AI 대시보드</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Malgun Gothic', sans-serif;
    background: #0d1117; color: #e6edf3;
    min-height: 100vh; padding: 24px;
  }
  h1 {
    font-size: 1.6rem; font-weight: 700;
    color: #58a6ff; margin-bottom: 4px;
  }
  .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 24px; }

  /* 상단 카드 */
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 24px;
  }
  .card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 20px;
    text-align: center;
  }
  .card .val {
    font-size: 2.2rem; font-weight: 700; color: #58a6ff;
  }
  .card .lbl { color: #8b949e; font-size: 0.85rem; margin-top: 4px; }

  /* 그리드 */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px; margin-bottom: 24px;
  }
  @media(max-width:700px){ .grid{ grid-template-columns:1fr; } }

  .panel {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 20px;
  }
  .panel h2 {
    font-size: 1rem; color: #8b949e;
    margin-bottom: 16px; font-weight: 600;
  }

  /* 바 차트 */
  .bar-row { margin-bottom: 14px; }
  .bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.85rem; margin-bottom: 5px;
  }
  .bar-track {
    background: #21262d; border-radius: 6px;
    height: 22px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 6px;
    transition: width 0.6s ease;
    display: flex; align-items: center;
    padding-left: 8px; font-size: 0.78rem;
    font-weight: 600; color: #fff;
  }

  /* 히스토리 테이블 */
  table { width:100%; border-collapse:collapse; font-size:0.85rem; }
  th {
    background:#21262d; color:#8b949e;
    padding:10px 12px; text-align:left;
    font-weight:600;
  }
  td { padding:9px 12px; border-bottom:1px solid #21262d; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:#21262d44; }

  /* 신뢰도 뱃지 */
  .badge {
    display:inline-block; padding:2px 10px;
    border-radius:20px; font-size:0.78rem; font-weight:700;
  }
  .badge.high   { background:#1f6feb33; color:#58a6ff; }
  .badge.mid    { background:#9a6700aa; color:#e3b341; }
  .badge.low    { background:#da363333; color:#f85149; }

  /* 상태 표시 */
  .status { display:flex; align-items:center; gap:8px; margin-bottom:20px; }
  .dot {
    width:10px; height:10px; border-radius:50%;
    background:#3fb950; animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100%{ opacity:1; } 50%{ opacity:0.4; }
  }
  .dot.offline { background:#f85149; animation:none; }

  .refresh-info { color:#8b949e; font-size:0.78rem; }
</style>
</head>
<body>

<h1>🗑️ 폐기물 분류 AI 대시보드</h1>
<p class="subtitle">YOLOv8 Classification | FastAPI 실시간 연동</p>

<div class="status">
  <div class="dot" id="statusDot"></div>
  <span id="statusText">서버 연결 중...</span>
  <span class="refresh-info" style="margin-left:auto">⟳ 2초마다 자동 갱신</span>
</div>

<!-- 요약 카드 -->
<div class="cards">
  <div class="card">
    <div class="val" id="totalReq">-</div>
    <div class="lbl">총 예측 횟수</div>
  </div>
  <div class="card">
    <div class="val" id="avgConf">-</div>
    <div class="lbl">평균 신뢰도</div>
  </div>
  <div class="card">
    <div class="val" id="avgLatency">-</div>
    <div class="lbl">평균 추론 시간</div>
  </div>
  <div class="card">
    <div class="val" id="uptime">-</div>
    <div class="lbl">서버 가동 시간</div>
  </div>
  <div class="card">
    <div class="val" id="collected">-</div>
    <div class="lbl">수집 이미지 수</div>
  </div>
  <div class="card">
    <div class="val" id="retrainReady" style="font-size:1.4rem">-</div>
    <div class="lbl">재학습 준비</div>
  </div>
</div>

<div class="grid">
  <!-- 클래스별 분포 -->
  <div class="panel">
    <h2>📊 클래스별 예측 분포</h2>
    <div id="barChart"></div>
  </div>

  <!-- 최근 예측 히스토리 -->
  <div class="panel">
    <h2>🕐 최근 예측 히스토리</h2>
    <table>
      <thead>
        <tr>
          <th>시각</th>
          <th>분류 결과</th>
          <th>신뢰도</th>
        </tr>
      </thead>
      <tbody id="historyBody">
        <tr><td colspan="3" style="color:#8b949e;text-align:center">대기 중...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
const API = "http://localhost:8000";

const CLASS_KOR = {
  snack_bag:    "과자봉지/비닐",
  glass_bottle: "유리병/음료수병",
  paper:        "종이류/포장상자",
  can:          "캔/음료수캔",
  pet_bottle:   "페트병",
};

const CLASS_COLOR = {
  snack_bag:    "#f0883e",
  glass_bottle: "#388bfd",
  paper:        "#3fb950",
  can:          "#f85149",
  pet_bottle:   "#bc8cff",
};

// 히스토리 로컬 저장 (최대 50건)
let history = [];

async function fetchStats() {
  try {
    const [statsRes, healthRes, collectedRes] = await Promise.all([
      fetch(`${API}/stats`),
      fetch(`${API}/health`),
      fetch(`${API}/collected`),
    ]);
    const stats     = await statsRes.json();
    const health    = await healthRes.json();
    const collected = await collectedRes.json();

    // 서버 상태
    document.getElementById("statusDot").className  = "dot";
    document.getElementById("statusText").textContent =
      `AI 서버 연결됨 | 모델: ${health.model_ready ? "✅ 로드 완료" : "⚠️ 로드 중"}`;

    // 요약 카드
    document.getElementById("totalReq").textContent   = stats.total_requests.toLocaleString();
    document.getElementById("avgConf").textContent    = (stats.avg_confidence * 100).toFixed(1) + "%";
    document.getElementById("avgLatency").textContent = stats.avg_latency_ms.toFixed(0) + "ms";
    document.getElementById("uptime").textContent     = health.uptime;
    document.getElementById("collected").textContent  = collected.total.toLocaleString();
    const ready = collected.retrain_ready;
    document.getElementById("retrainReady").textContent = ready ? "✅ 준비됨" : "⏳ 수집 중";
    document.getElementById("retrainReady").style.color = ready ? "#3fb950" : "#e3b341";

    // 바 차트
    const counts = stats.class_counts;
    const total  = Object.values(counts).reduce((a,b)=>a+b, 0) || 1;
    const barHtml = Object.entries(counts)
      .sort((a,b)=>b[1]-a[1])
      .map(([cls, cnt]) => {
        const pct   = ((cnt / total) * 100).toFixed(1);
        const color = CLASS_COLOR[cls] || "#58a6ff";
        return `
          <div class="bar-row">
            <div class="bar-label">
              <span>${CLASS_KOR[cls] || cls}</span>
              <span style="color:#8b949e">${cnt}회 (${pct}%)</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct}%;background:${color}">
                ${pct > 8 ? pct + "%" : ""}
              </div>
            </div>
          </div>`;
      }).join("");
    document.getElementById("barChart").innerHTML =
      total === 1 ? '<p style="color:#8b949e">아직 예측 데이터가 없습니다.</p>' : barHtml;

  } catch(e) {
    document.getElementById("statusDot").className   = "dot offline";
    document.getElementById("statusText").textContent = "❌ 서버 연결 실패 — step5_api_server.py 실행 확인";
  }
}

// /predict 실시간 감시 (폴링)
async function pollPredict() {
  try {
    const res  = await fetch(`${API}/stats`);
    const data = await res.json();
    const now  = new Date().toLocaleTimeString("ko-KR");

    // 총 요청이 늘었으면 히스토리에 추가
    const prev = history.length > 0 ? history[0]._total : 0;
    if (data.total_requests > prev) {
      // 클래스별로 가장 많이 증가한 것을 최근 예측으로 추정
      const counts = data.class_counts;
      const topCls = Object.entries(counts).sort((a,b)=>b[1]-a[1])[0];
      if (topCls) {
        history.unshift({
          _total: data.total_requests,
          time:   now,
          cls:    topCls[0],
          conf:   (data.avg_confidence * 100).toFixed(0),
        });
        if (history.length > 20) history.pop();
      }
    }

    // 히스토리 테이블 렌더링
    const tbody = document.getElementById("historyBody");
    if (history.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3" style="color:#8b949e;text-align:center">대기 중...</td></tr>';
    } else {
      tbody.innerHTML = history.map(h => {
        const conf  = parseInt(h.conf);
        const badge = conf >= 80 ? "high" : conf >= 60 ? "mid" : "low";
        return `<tr>
          <td style="color:#8b949e">${h.time}</td>
          <td style="color:${CLASS_COLOR[h.cls]||'#e6edf3'};font-weight:600">
            ${CLASS_KOR[h.cls] || h.cls}
          </td>
          <td><span class="badge ${badge}">${h.conf}%</span></td>
        </tr>`;
      }).join("");
    }
  } catch(e) {}
}

// 2초마다 갱신
setInterval(()=>{ fetchStats(); pollPredict(); }, 2000);
fetchStats();
pollPredict();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML

@app.get("/health")
async def health():
    return {"status": "ok", "dashboard": "running"}

if __name__ == "__main__":
    print("=" * 50)
    print("  폐기물 분류 AI 대시보드")
    print("=" * 50)
    print("  접속 주소: http://localhost:8080")
    print("  사전 조건: step5_api_server.py 가 실행 중이어야 함")
    print("=" * 50)
    uvicorn.run("step6_dashboard:app", host="0.0.0.0", port=8080, reload=False)
