const modelSelect = document.getElementById("model");
const loadModelBtn = document.getElementById("load-model");
const modelStatus = document.getElementById("model-status");

const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send");

const grid = document.getElementById("grid");
const tokensEl = document.getElementById("tokens");
const tokenMapEl = document.getElementById("token-map");

// This UI is focused on Token → Token (pair) attention visualization.
const viewMode = "pair";

let currentPrompt = "";
let srcIndex = null;
let dstIndex = null;
let allAttentions = null; // { "src_dst": [[layer][head]] }
let currentTokens = null;
let tokenMapMode = "head"; // "head" | "layer" | "overall"
let lastLayerIdx = 0;
let lastHeadIdx = 0;
let activeHeadCell = null;

async function fetchModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  modelSelect.innerHTML = data.models
    .map((m) => `<option value="${m}">${m}</option>`)
    .join("");
}

async function loadModel() {
  loadModelBtn.disabled = true;
  loadModelBtn.innerText = "Loading...";
  modelStatus.textContent = "Loading model...";
  try {
    const res = await fetch("/api/load_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelSelect.value }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      modelStatus.textContent = "Failed to load model";
      return;
    }
    modelStatus.textContent = `Loaded: ${data.model}`;
    addAssistantMessage(`Model loaded: ${data.model}`);
  } catch (err) {
    alert(err);
    modelStatus.textContent = "Failed to load model";
  } finally {
    loadModelBtn.disabled = false;
    loadModelBtn.innerText = "Load model";
  }
}

function addMessage(role, text) {
  if (!chatLog) return;
  const msg = document.createElement("div");
  msg.className = `chat-msg ${role}`;
  msg.innerHTML = `
    <div class="chat-role">${role}</div>
    <div class="chat-bubble"></div>
  `;
  msg.querySelector(".chat-bubble").textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msg;
}

function addUserMessage(text) {
  return addMessage("user", text);
}

function addAssistantMessage(text) {
  return addMessage("assistant", text);
}

async function updateTokens(text) {
  const prompt = (text ?? currentPrompt).trim();
  if (!text) {
    tokensEl.innerHTML = "";
    return;
  }
  // Check if model is loaded
  if (!modelStatus.textContent.includes("Loaded:")) {
    tokensEl.innerHTML =
      "<p style='color:#9aa1b5; font-size:12px; padding:8px;'>모델을 먼저 로드해주세요.</p>";
    return;
  }
  try {
    const res = await fetch("/api/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelSelect.value, text: prompt }),
    });

    if (!res.ok) {
      const errorText = await res.text();
      tokensEl.innerHTML = `<p style='color:#f46b6b; font-size:12px; padding:8px;'>토큰화 실패 (${res.status}): ${errorText.substring(
        0,
        100
      )}</p>`;
      return;
    }

    const contentType = res.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      await res.text();
      tokensEl.innerHTML =
        "<p style='color:#f46b6b; font-size:12px; padding:8px;'>서버 응답 오류: JSON이 아닌 응답을 받았습니다.</p>";
      return;
    }

    const data = await res.json();
    if (data.error) {
      tokensEl.innerHTML = `<p style='color:#f46b6b; font-size:12px; padding:8px;'>Error: ${data.error}</p>`;
      return;
    }
    if (data.tokens && data.tokens.length > 0) {
      renderTokens(data.tokens);
    } else {
      tokensEl.innerHTML = "";
    }
  } catch (err) {
    tokensEl.innerHTML = `<p style='color:#f46b6b; font-size:12px; padding:8px;'>토큰화 실패: ${err.message}</p>`;
  }
}

function colorForValue(v, isPairMode = false) {
  if (isPairMode) {
    // Pair mode: 0..1 where 0=white, 1=green
    const t = Math.max(0, Math.min(1, v));
    const start = { r: 255, g: 255, b: 255 }; // white
    const end = { r: 74, g: 222, b: 128 }; // green
    const r = Math.round(start.r + (end.r - start.r) * t);
    const g = Math.round(start.g + (end.g - start.g) * t);
    const b = Math.round(start.b + (end.b - start.b) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Concept mode: -1 to 1 scale, red to green
    const clamped = Math.max(-1, Math.min(1, v));
    const r = clamped < 0 ? 244 : 74;
    const g = clamped < 0 ? 107 : 222;
    const b = clamped < 0 ? 107 : 128;
    const intensity = Math.min(1, Math.abs(clamped)) * 0.85 + 0.15;
    return `rgba(${r}, ${g}, ${b}, ${intensity})`;
  }
}

function renderGrid(scores) {
  console.log("renderGrid called with scores:", scores);
  if (!scores || !scores.length) {
    grid.innerHTML =
      "<p style='color:#c5c9d6; font-weight: 500;'>No scores yet</p>";
    return;
  }
  // Debug: log first few values to verify they're different
  if (scores.length > 0 && scores[0].length > 0) {
    console.log("First layer, first head value:", scores[0][0]);
    if (scores.length > 1) {
      console.log("Second layer, first head value:", scores[1][0]);
    }
  }
  const heads = scores[0].length;
  const isPairMode = viewMode === "pair";
  let html =
    "<table><thead><tr><th class='head-col' style='min-width: 44px;'>Head</th>";
  scores.forEach((_, idx) => {
    html += `<th style='min-width: 44px;'>L${idx}</th>`;
  });
  html += "</tr></thead><tbody>";
  for (let h = 0; h < heads; h++) {
    html += `<tr><th class='head-label' style='text-align: right; padding-right: 6px;'>H${h}</th>`;
    for (let l = 0; l < scores.length; l++) {
      const v = scores[l][h];
      const displayValue = v.toFixed(2);
      const bgColor = colorForValue(v, isPairMode);
      html += `<td><div class="cell" data-layer="${l}" data-head="${h}" style="background:${bgColor}" title="Layer ${l}, Head ${h}: ${v.toFixed(
        4
      )}">${displayValue}</div></td>`;
    }
    html += "</tr>";
  }
  grid.innerHTML = html;

  // Attach hover handlers to allow per-head token→token maps
  // Use requestAnimationFrame to ensure DOM is ready
  requestAnimationFrame(() => {
    const cells = grid.querySelectorAll(".cell[data-layer][data-head]");
    console.log(`Found ${cells.length} cells to attach hover handlers`);
    cells.forEach((cell) => {
      cell.style.cursor = "pointer";
      let hoverTimeout = null;
      
      cell.addEventListener("mouseenter", function(e) {
        const layer = Number(this.getAttribute("data-layer"));
        const head = Number(this.getAttribute("data-head"));
        if (isNaN(layer) || isNaN(head)) {
          return;
        }
        // Small delay to avoid flickering on quick mouse movements
        hoverTimeout = setTimeout(() => {
          console.log(`Cell hovered: Layer ${layer}, Head ${head}`);
          if (activeHeadCell && activeHeadCell !== this) {
            activeHeadCell.classList.remove("active-head-cell");
          }
          this.classList.add("active-head-cell");
          activeHeadCell = this;
          renderTokenMap(layer, head);
        }, 100);
      });
      
      cell.addEventListener("mouseleave", function() {
        if (hoverTimeout) {
          clearTimeout(hoverTimeout);
          hoverTimeout = null;
        }
      });
    });
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatAxisToken(t) {
  const clean = t.replace("Ġ", " ").replace("▁", " ").trim() || "␣";
  const trimmed = clean.length > 6 ? clean.slice(0, 5) + "…" : clean;
  return trimmed;
}

function renderTokenMap(layerIdx, headIdx) {
  console.log(`renderTokenMap called: Layer ${layerIdx}, Head ${headIdx}`);
  console.log(`tokenMapEl:`, tokenMapEl);
  console.log(`allAttentions:`, allAttentions ? Object.keys(allAttentions).slice(0, 5) : null);
  console.log(`currentTokens:`, currentTokens);
  
  if (!tokenMapEl) {
    console.error("tokenMapEl is null!");
    return;
  }
  if (!allAttentions || !currentTokens) {
    console.warn("Missing data:", { allAttentions: !!allAttentions, currentTokens: !!currentTokens });
    tokenMapEl.innerHTML =
      "<p style='color:#9aa1b5; font-size:12px; padding:4px;'>먼저 프롬프트를 보내 attention을 계산하세요.</p>";
    return;
  }

  // Remember last selection
  lastLayerIdx = layerIdx;
  lastHeadIdx = headIdx;

  const seqLen = currentTokens.length;
  if (!seqLen) {
    console.warn("No tokens available");
    tokenMapEl.innerHTML = "";
    return;
  }
  
  console.log(`Rendering token map for ${seqLen} tokens`);

  // No thead - key labels will be shown above diagonal cells
  let html = "<table><tbody>";

  for (let dst = 0; dst < seqLen; dst++) {
    const rowLabel = escapeHtml(formatAxisToken(currentTokens[dst]));
    html += `<tr><th class="token-axis-row">${rowLabel}</th>`;
    for (let src = 0; src < seqLen; src++) {
      // Upper off-diagonal (src > dst): empty cell, no text, no color
      if (src > dst) {
        html += `<td class="token-map-empty"></td>`;
        continue;
      }
      
      // Diagonal and lower: calculate attention value
      const key = `${src}_${dst}`;
      const perPair = allAttentions ? allAttentions[key] : null;
      let v = 0;
      if (perPair) {
        if (dst === 0 && src === 0) {
          console.log(`Sample perPair[${key}]:`, perPair);
          console.log(`perPair[${layerIdx}]:`, perPair[layerIdx]);
        }
        if (tokenMapMode === "head") {
          if (
            perPair[layerIdx] &&
            typeof perPair[layerIdx][headIdx] === "number"
          ) {
            v = perPair[layerIdx][headIdx];
          } else if (dst === 0 && src === 0) {
            console.warn(`No data for L${layerIdx}H${headIdx} in pair ${key}`);
          }
        } else if (tokenMapMode === "layer") {
          const layerVals = perPair[layerIdx];
          if (Array.isArray(layerVals) && layerVals.length > 0) {
            const sum = layerVals.reduce(
              (acc, val) => acc + (typeof val === "number" ? val : 0),
              0
            );
            v = sum / layerVals.length;
          }
        } else if (tokenMapMode === "overall") {
          // Average over all layers and heads
          let total = 0;
          let count = 0;
          perPair.forEach((layer) => {
            if (Array.isArray(layer)) {
              layer.forEach((val) => {
                if (typeof val === "number") {
                  total += val;
                  count += 1;
                }
              });
            }
          });
          if (count > 0) v = total / count;
        }
      }
      
      // Diagonal cell: show key label above it
      if (src === dst) {
        const keyLabel = escapeHtml(formatAxisToken(currentTokens[src]));
        const bgColor = colorForValue(v, true);
        const displayValue = v.toFixed(2);
        html += `<td class="token-map-diagonal"><div class="token-map-key-label">${keyLabel}</div><div class="token-map-cell" style="background:${bgColor}" title="q=${dst}, k=${src}, L${layerIdx}, H${headIdx}: ${v.toFixed(4)}">${displayValue}</div></td>`;
      } else {
        // Lower diagonal: normal cell
        const bgColor = colorForValue(v, true);
        const displayValue = v.toFixed(2);
        html += `<td><div class="token-map-cell" style="background:${bgColor}" title="q=${dst}, k=${src}, L${layerIdx}, H${headIdx}: ${v.toFixed(4)}">${displayValue}</div></td>`;
      }
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  tokenMapEl.innerHTML = `
    <div class="token-map-layout">
      <div class="token-map-corner"></div>
      <div class="token-map-axis token-map-axis-top">Key (SRC) →</div>
      <div class="token-map-axis token-map-axis-left">Query (DST) ↓</div>
      <div class="token-map-table-wrap">${html}</div>
    </div>
  `;
  console.log(`Token map rendered successfully for L${layerIdx}H${headIdx}`);

  const header = document.querySelector(".token-map-header");
  if (header) {
    const modeLabel =
      tokenMapMode === "overall"
        ? "Overall mean"
        : tokenMapMode === "layer"
          ? `Layer mean (L${layerIdx})`
          : `Layer L${layerIdx}, H${headIdx}`;
    const span = header.querySelector("span");
    if (span) {
      span.textContent = `Token → Token attention (${modeLabel}) – rows=query, columns=key`;
    }
  } else {
    console.warn("token-map-header not found");
  }
}

function renderTokens(tokens) {
  if (!tokens) return;
  const pieces = [];
  tokens.forEach((t, idx) => {
    const clean = t.replace("Ġ", " ").replace("▁", " ");
    const base = clean;
    const isSrc = idx === srcIndex;
    const isDst = idx === dstIndex;
    let borderColor = "#2a2f42";
    let bg = "#1a1d2e";
    let textColor = "#e6e8f0";
    
    if (isSrc) {
      borderColor = "#4ade80";
      bg = "#1a3a2a";
      textColor = "#e6e8f0";
    } else if (isDst) {
      borderColor = "#f97316";
      bg = "#3a2a1a";
      textColor = "#e6e8f0";
    }
    
    pieces.push(
      `<button class="token" data-idx="${idx}" style="display:inline-flex !important; flex-shrink:0 !important; margin:0 !important; border-color:${borderColor};background:${bg};color:${textColor}">${base}</button>`
    );
  });
  tokensEl.innerHTML = pieces.join("");

  document.querySelectorAll(".token").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = Number(el.getAttribute("data-idx"));
      if (srcIndex === null || (srcIndex !== null && dstIndex !== null)) {
        srcIndex = idx;
        dstIndex = null;
      } else if (dstIndex === null && idx !== srcIndex) {
        dstIndex = idx;
      } else {
        srcIndex = idx;
        dstIndex = null;
      }
      renderTokens(tokens);
      if (allAttentions !== null && srcIndex !== null && dstIndex !== null) {
        const key = `${srcIndex}_${dstIndex}`;
        if (allAttentions[key]) renderGrid(allAttentions[key]);
      }
    });
  });
}

async function computeAllPairs(prompt) {
  const res = await fetch("/api/compute_all_pairs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: modelSelect.value, text: prompt }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

async function sendPrompt() {
  const prompt = chatInput.value.trim();
  if (!prompt) return;

  // UI: append messages
  addUserMessage(prompt);
  const assistantMsg = addAssistantMessage("Computing attentions…");

  chatInput.value = "";
  currentPrompt = prompt;

  if (!modelStatus.textContent.includes("Loaded:")) {
    assistantMsg.querySelector(".chat-bubble").textContent =
      "모델을 먼저 Load 해주세요.";
    return;
  }

  try {
    await updateTokens(prompt);
    const data = await computeAllPairs(prompt);
    allAttentions = data.all_scores;
    currentTokens = data.tokens;
    
    console.log("Received allAttentions:", allAttentions ? Object.keys(allAttentions).length : 0, "pairs");
    if (allAttentions && Object.keys(allAttentions).length > 0) {
      const firstKey = Object.keys(allAttentions)[0];
      console.log(`Sample pair ${firstKey}:`, allAttentions[firstKey]);
      console.log(`Number of layers:`, allAttentions[firstKey]?.length);
      if (allAttentions[firstKey] && allAttentions[firstKey][0]) {
        console.log(`Number of heads in layer 0:`, allAttentions[firstKey][0].length);
      }
    }
    console.log("Current tokens:", currentTokens);

    // Auto-select a reasonable default pair: first token -> last token
    srcIndex = currentTokens.length > 0 ? 0 : null;
    dstIndex = currentTokens.length > 1 ? currentTokens.length - 1 : null;
    renderTokens(currentTokens);

    if (srcIndex !== null && dstIndex !== null) {
      const key = `${srcIndex}_${dstIndex}`;
      if (allAttentions[key]) {
        renderGrid(allAttentions[key]);
        // 기본으로 L0, H0 토큰→토큰 맵도 한번 그려줌
        renderTokenMap(0, 0);
      }
    }

    // 클릭 전까지는 안내 문구 유지
    if (tokenMapEl && !tokenMapEl.innerHTML.trim()) {
      tokenMapEl.innerHTML =
        "<p style='color:#9aa1b5; font-size:12px; padding:4px;'>위 레이어/헤드 그리드에서 셀을 클릭하면, 아래에 해당 Head의 토큰→토큰 attention 맵이 표시됩니다.</p>";
    }

    assistantMsg.querySelector(".chat-bubble").textContent =
      "완료. 토큰을 클릭해서 SRC → DST를 선택하면 가운데 그리드가 업데이트됩니다.";
  } catch (err) {
    assistantMsg.querySelector(".chat-bubble").textContent = `Error: ${err.message}`;
  }
}

sendBtn.addEventListener("click", sendPrompt);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendPrompt();
});
loadModelBtn.addEventListener("click", loadModel);

// Mode button handlers
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", function() {
    // Remove active class from all buttons
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    // Add active class to clicked button
    this.classList.add("active");
    tokenMapMode = this.getAttribute("data-mode");
    console.log("tokenMapMode changed to", tokenMapMode);
    // Re-render last selection if available
    if (allAttentions && currentTokens && lastLayerIdx !== null && lastHeadIdx !== null) {
      renderTokenMap(lastLayerIdx, lastHeadIdx);
    }
  });
});

// Set initial active button (run after DOM is ready)
setTimeout(() => {
  const initialBtn = document.querySelector(`.mode-btn[data-mode="${tokenMapMode}"]`);
  if (initialBtn) initialBtn.classList.add("active");
}, 0);

// Initial state
modelStatus.textContent = "Load a model to start.";
addAssistantMessage("왼쪽 입력창에 프롬프트를 보내면 attention을 계산합니다.");
if (tokenMapEl) {
  tokenMapEl.innerHTML = "<p style='color:#9aa1b5; font-size:12px; padding:8px;'>위 레이어/헤드 그리드에서 셀을 클릭하면, 여기에 해당 Head의 토큰→토큰 attention 맵이 표시됩니다.</p>";
}
fetchModels();

