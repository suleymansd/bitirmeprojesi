const fileInput = document.getElementById("fileInput");
const patientIdInput = document.getElementById("patientIdInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const exportBtn = document.getElementById("exportBtn");
const thresholdInput = document.getElementById("thresholdInput");
const thresholdValue = document.getElementById("thresholdValue");

const previewImage = document.getElementById("previewImage");
const previewHint = document.getElementById("previewHint");
const predictionText = document.getElementById("predictionText");
const benignMeter = document.getElementById("benignMeter");
const malignMeter = document.getElementById("malignMeter");
const benignScore = document.getElementById("benignScore");
const malignScore = document.getElementById("malignScore");
const explainText = document.getElementById("explainText");
const modelMeta = document.getElementById("modelMeta");
const healthBadge = document.getElementById("healthBadge");
const clockText = document.getElementById("clockText");

const historyBody = document.getElementById("historyBody");
const historyCount = document.getElementById("historyCount");
const perfRound = document.getElementById("perfRound");
const mAccuracy = document.getElementById("mAccuracy");
const mMacroF1 = document.getElementById("mMacroF1");
const mRecall = document.getElementById("mRecall");
const mPrecision = document.getElementById("mPrecision");
const mTP = document.getElementById("mTP");
const mFP = document.getElementById("mFP");
const mTN = document.getElementById("mTN");
const mFN = document.getElementById("mFN");

let selectedFile = null;
let history = [];

const fmt = (n) => Number(n).toFixed(4);
const nowStr = () => new Date().toLocaleString("tr-TR");

function updateClock() { clockText.textContent = nowStr(); }
setInterval(updateClock, 1000);
updateClock();

function renderHistory() {
  historyCount.textContent = `${history.length} kayıt`;
  if (!history.length) {
    historyBody.innerHTML = '<tr><td colspan="5" class="empty">Henüz kayıt yok.</td></tr>';
    return;
  }
  historyBody.innerHTML = history.map((r) => `
    <tr>
      <td>${r.time}</td>
      <td>${r.caseId}</td>
      <td class="${r.riskClass}">${r.label}</td>
      <td>${r.malign}</td>
      <td>${r.threshold}</td>
    </tr>
  `).join("");
}

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});
thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  selectedFile = file;
  analyzeBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.style.display = "block";
    previewHint.style.display = "none";
  };
  reader.readAsDataURL(file);
});

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    healthBadge.textContent = `Hazır / ${data.device}`;
    modelMeta.textContent = data.checkpoint;
  } catch {
    healthBadge.textContent = "API yok";
  }
}

async function loadPerformance() {
  try {
    const res = await fetch("/api/performance");
    const data = await res.json();
    const t = data.test_metrics || {};
    const cm = t.confusion || {};

    perfRound.textContent = `Round ${data.recommended_round ?? "-"}`;
    mAccuracy.textContent = t.accuracy != null ? fmt(t.accuracy) : "-";
    mMacroF1.textContent = t.macro_f1 != null ? fmt(t.macro_f1) : "-";
    mRecall.textContent = t.malign_recall != null ? fmt(t.malign_recall) : "-";
    mPrecision.textContent = t.malign_precision != null ? fmt(t.malign_precision) : "-";
    mTP.textContent = cm.tp != null ? String(cm.tp) : "-";
    mFP.textContent = cm.fp != null ? String(cm.fp) : "-";
    mTN.textContent = cm.tn != null ? String(cm.tn) : "-";
    mFN.textContent = cm.fn != null ? String(cm.fn) : "-";
  } catch {
    // no-op
  }
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Çalışıyor...";

  const formData = new FormData();
  formData.append("file", selectedFile);
  const threshold = Number(thresholdInput.value);

  try {
    const res = await fetch(`/api/predict?threshold=${threshold}`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "API hatası");

    const isMalign = data.prediction.class_id === 1;
    predictionText.textContent = `${data.prediction.label} / ${data.prediction.risk_level}`;
    predictionText.className = isMalign ? "risk-high" : "risk-low";

    benignMeter.value = data.scores.benign;
    malignMeter.value = data.scores.malign;
    benignScore.textContent = fmt(data.scores.benign);
    malignScore.textContent = fmt(data.scores.malign);

    explainText.textContent = isMalign
      ? "Model eşik üstü malign risk tespit etti. Klinik doğrulama önerilir."
      : "Model benign eğilim gösteriyor. Klinik bağlam ile birlikte değerlendirilmelidir.";

    history.unshift({
      time: nowStr(),
      caseId: patientIdInput.value.trim() || "-",
      label: data.prediction.label,
      malign: fmt(data.scores.malign),
      threshold: Number(data.threshold).toFixed(2),
      riskClass: isMalign ? "risk-high" : "risk-low",
    });
    if (history.length > 25) history = history.slice(0, 25);
    renderHistory();
  } catch (err) {
    predictionText.textContent = `Hata: ${err.message}`;
    predictionText.className = "risk-high";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analizi Başlat";
  }
});

clearBtn.addEventListener("click", () => {
  history = [];
  renderHistory();
});

exportBtn.addEventListener("click", () => {
  const payload = { exported_at: nowStr(), count: history.length, records: history };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `dermai-report-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

renderHistory();
checkHealth();
loadPerformance();
