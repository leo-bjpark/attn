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
let headFilterStart = null;
let headFilterEnd = null;
let layerFilterStart = null;
let layerFilterEnd = null;
let currentScores = null; // Store current scores for filtering
let interpretationTokenMode = false; // "interpretation" | false
let interpretationHoverMode = "add"; // "add" | "remove" - determined by hover
let selectedInterpretationTokens = new Set(); // Set of token indices
let lastHoveredTokenIdx = null; // Track last hovered token to prevent repeated triggers
let hoverDebounceTimer = null; // Debounce timer for hover events
let isDragging = false;
let dragStart = null;
let dragSelectionBox = null;
let interpretationQueryIndex = null; // Query (DST) index for interpretation
let interpretationKeyIndex = null; // Key (SRC) index for interpretation

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
  
  // Store scores for filtering
  currentScores = scores;
  
  // Debug: log first few values to verify they're different
  if (scores.length > 0 && scores[0].length > 0) {
    console.log("First layer, first head value:", scores[0][0]);
    if (scores.length > 1) {
      console.log("Second layer, first head value:", scores[1][0]);
    }
  }
  const totalHeads = scores[0].length;
  const totalLayers = scores.length;
  
  // Apply head filter
  let startHead = 0;
  let endHead = totalHeads;
  if (headFilterStart !== null && headFilterStart >= 0) {
    startHead = Math.min(headFilterStart, totalHeads - 1);
  }
  if (headFilterEnd !== null && headFilterEnd >= 0) {
    endHead = Math.min(headFilterEnd + 1, totalHeads);
  }
  if (startHead >= endHead) {
    startHead = 0;
    endHead = totalHeads;
  }
  
  // Apply layer filter
  let startLayer = 0;
  let endLayer = totalLayers;
  if (layerFilterStart !== null && layerFilterStart >= 0) {
    startLayer = Math.min(layerFilterStart, totalLayers - 1);
  }
  if (layerFilterEnd !== null && layerFilterEnd >= 0) {
    endLayer = Math.min(layerFilterEnd + 1, totalLayers);
  }
  if (startLayer >= endLayer) {
    startLayer = 0;
    endLayer = totalLayers;
  }
  
  const isPairMode = viewMode === "pair";
  // Adjust column width based on number of layers
  const numLayers = endLayer - startLayer;
  const colWidth = numLayers > 32 ? "28px" : numLayers > 16 ? "30px" : "32px";
  let html =
    `<table><thead><tr><th class='head-col' style='min-width: 36px;'>Head</th>`;
  for (let l = startLayer; l < endLayer; l++) {
    html += `<th style='min-width: ${colWidth}; max-width: ${colWidth};'>L${l}</th>`;
  }
  html += "</tr></thead><tbody>";
  for (let h = startHead; h < endHead; h++) {
    html += `<tr class="head-row" data-head="${h}"><th class='head-label' style='text-align: right; padding-right: 6px;'>H${h}</th>`;
    for (let l = startLayer; l < endLayer; l++) {
      const v = scores[l][h];
      // Use 1 decimal place for compact display when many layers
      const displayValue = numLayers > 32 ? v.toFixed(1) : v.toFixed(2);
      const bgColor = colorForValue(v, isPairMode);
      html += `<td style='width: ${colWidth};'><div class="cell" data-layer="${l}" data-head="${h}" style="background:${bgColor}" title="Layer ${l}, Head ${h}: ${v.toFixed(
        4
      )}">${displayValue}</div></td>`;
    }
    html += "</tr>";
  }
  grid.innerHTML = html;
  
  // Update head count info
  if (headCountInfo) {
    const showing = endHead - startHead;
    if (showing === totalHeads) {
      headCountInfo.textContent = `(${totalHeads} heads)`;
    } else {
      headCountInfo.textContent = `(showing ${showing} of ${totalHeads} heads)`;
    }
  }
  
  // Update layer count info
  const layerCountInfo = document.getElementById("layer-count-info");
  if (layerCountInfo) {
    const showing = endLayer - startLayer;
    if (showing === totalLayers) {
      layerCountInfo.textContent = `(${totalLayers} layers)`;
    } else {
      layerCountInfo.textContent = `(showing ${showing} of ${totalLayers} layers)`;
    }
  }

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

  // Use filtered tokens if selection is active
  const filteredIndices = selectedInterpretationTokens.size > 0
    ? Array.from(selectedInterpretationTokens).sort((a, b) => a - b)
    : currentTokens.map((_, idx) => idx);
  
  const seqLen = filteredIndices.length;
  if (!seqLen) {
    console.warn("No tokens available");
    tokenMapEl.innerHTML = "<p style='color:#9aa1b5; font-size:12px; padding:4px;'>No tokens selected for interpretation.</p>";
    return;
  }
  
  console.log(`Rendering token map for ${seqLen} tokens (${selectedInterpretationTokens.size > 0 ? 'filtered' : 'all'})`);

  // Build table with tbody and tfoot for key labels at bottom
  let html = "<table><tbody>";

  for (let dstIdx = 0; dstIdx < seqLen; dstIdx++) {
    const dst = filteredIndices[dstIdx];
    const rowLabel = escapeHtml(formatAxisToken(currentTokens[dst]));
    // Use interpretationQueryIndex for row highlighting (Query/DST = orange)
    const isDstRow = interpretationQueryIndex !== null && dst === interpretationQueryIndex;
    const rowClass = isDstRow ? "token-map-row-highlight" : "";
    html += `<tr class="${rowClass}" data-dst="${dst}"><th class="token-axis-row ${isDstRow ? 'token-axis-row-highlight' : ''}">${rowLabel}</th>`;
    for (let srcIdx = 0; srcIdx < seqLen; srcIdx++) {
      const src = filteredIndices[srcIdx];
      // Use interpretationKeyIndex for column highlighting (Key/SRC = green)
      const isSrcCol = interpretationKeyIndex !== null && src === interpretationKeyIndex;
      // Upper off-diagonal (src > dst): empty cell, no text, no color
      if (src > dst) {
        const colClass = isSrcCol ? "token-map-col-highlight" : "";
        html += `<td class="token-map-empty ${colClass}"></td>`;
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
      
      // Check if this is the selected cell (intersection of key and query)
      const isSelectedCell = (interpretationKeyIndex !== null && src === interpretationKeyIndex) && (interpretationQueryIndex !== null && dst === interpretationQueryIndex);
      const colClass = isSrcCol ? "token-map-col-highlight" : "";
      
      // Diagonal cell: no label above, will show in footer
      if (src === dst) {
        const bgColor = colorForValue(v, true);
        const displayValue = v.toFixed(2);
        const cellClass = isSelectedCell ? "token-map-cell-selected" : "";
        html += `<td class="token-map-diagonal ${colClass}"><div class="token-map-cell ${cellClass}" style="background:${bgColor}" title="q=${dst}, k=${src}, L${layerIdx}, H${headIdx}: ${v.toFixed(4)}">${displayValue}</div></td>`;
      } else {
        // Lower diagonal: normal cell
        const bgColor = colorForValue(v, true);
        const displayValue = v.toFixed(2);
        const cellClass = isSelectedCell ? "token-map-cell-selected" : "";
        html += `<td class="${colClass}"><div class="token-map-cell ${cellClass}" style="background:${bgColor}" title="q=${dst}, k=${src}, L${layerIdx}, H${headIdx}: ${v.toFixed(4)}">${displayValue}</div></td>`;
      }
    }
    html += "</tr>";
  }
  html += "</tbody><tfoot><tr><th class='token-map-footer-label'>Key (SRC)</th>";
  
  // Add key labels in footer row
  for (let srcIdx = 0; srcIdx < seqLen; srcIdx++) {
    const src = filteredIndices[srcIdx];
    const keyLabel = escapeHtml(formatAxisToken(currentTokens[src]));
    // Use interpretationKeyIndex for column highlighting (Key/SRC = green)
    const isSrcCol = interpretationKeyIndex !== null && src === interpretationKeyIndex;
    const colClass = isSrcCol ? "token-map-col-highlight" : "";
    html += `<td class="token-map-key-footer ${colClass}"><div class="token-map-key-label-bottom ${isSrcCol ? 'token-map-key-label-highlight' : ''}">${keyLabel}</div></td>`;
  }
  html += "</tr></tfoot></table>";
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
    const isSelected = selectedInterpretationTokens.has(idx);
    const isInterpretationQuery = interpretationQueryIndex === idx;
    const isInterpretationKey = interpretationKeyIndex === idx;
    // Query 이후 토큰은 Query만 선택되고 Key가 아직 선택되지 않았을 때만 비활성화
    const isAfterQuery = interpretationQueryIndex !== null && interpretationKeyIndex === null && idx > interpretationQueryIndex;
    
    let borderColor = "#2a2f42";
    let borderWidth = "1px";
    let bg = "#1a1d2e";
    let textColor = "#e6e8f0";
    let classes = "token";
    let disabled = false;
    let opacity = "1";
    
    // Query/Key 표시 (interpretationQueryIndex/interpretationKeyIndex 사용)
    // Query/Key가 선택되어 있으면 src/dst는 표시하지 않음
    if (isInterpretationQuery) {
      borderColor = "#f97316"; // 주황색 (Query/DST)
      borderWidth = "2px";
    } else if (isInterpretationKey) {
      borderColor = "#4ade80"; // 초록색 (Key/SRC)
      borderWidth = "2px";
    } else if (interpretationQueryIndex === null && interpretationKeyIndex === null) {
      // Query/Key가 선택되지 않았을 때만 src/dst 표시
      if (isSrc) {
        borderColor = "#4ade80"; // 초록색 (Key/SRC)
        borderWidth = "2px";
      } else if (isDst) {
        borderColor = "#f97316"; // 주황색 (Query/DST)
        borderWidth = "2px";
      }
    }
    
    // Query 이후 토큰은 비활성화 (Query가 선택된 경우)
    if (isAfterQuery) {
      disabled = true;
      opacity = "0.3";
      classes += " token-disabled";
    }
    
    // Interpretation mode
    if (interpretationTokenMode === "interpretation") {
      classes += " interpretation-mode";
      // Add hover mode indicator
      if (isSelected) {
        classes += " interpretation-hover-remove";
      } else {
        classes += " interpretation-hover-add";
      }
    }
    
    // 선택된 토큰: 배경색만 (Query/Key가 아닌 경우만, 항상 표시)
    if (isSelected && !isInterpretationQuery && !isInterpretationKey) {
      bg = "rgba(97, 218, 251, 0.2)"; // 파란색 배경
      classes += " interpretation-selected";
    }
    
    const disabledAttr = disabled ? "disabled" : "";
    const tokenHtml = `<button class="${classes}" data-idx="${idx}" ${disabledAttr} style="display:inline-flex !important; flex-shrink:0 !important; margin:0 !important; border:${borderWidth} solid ${borderColor};background:${bg};color:${textColor};opacity:${opacity}">${base}</button>`;
    pieces.push(tokenHtml);
  });
  tokensEl.innerHTML = pieces.join("");

  // Setup hover handlers for interpretation mode - auto add/remove on hover
  if (interpretationTokenMode === "interpretation") {
    document.querySelectorAll(".token").forEach((el) => {
      const idx = Number(el.getAttribute("data-idx"));
      
      // Query/Key tokens should always remain selected, so skip hover for them
      if (idx === interpretationQueryIndex || idx === interpretationKeyIndex) {
        return;
      }
      
      el.addEventListener("mouseenter", () => {
        // Prevent repeated triggers on the same token
        if (lastHoveredTokenIdx === idx) {
          return;
        }
        
        // Clear any pending debounce timer
        if (hoverDebounceTimer) {
          clearTimeout(hoverDebounceTimer);
        }
        
        // Debounce hover events to prevent too frequent updates
        hoverDebounceTimer = setTimeout(() => {
          // Check current state at hover time (not at render time)
          const isSelected = selectedInterpretationTokens.has(idx);
          
          // Skip if this is Query or Key token (they should always be selected)
          if (idx === interpretationQueryIndex || idx === interpretationKeyIndex) {
            return;
          }
          
          // Auto add/remove on hover (only for interpretation tokens, not Query/Key)
          if (isSelected) {
            // Remove if already selected
            selectedInterpretationTokens.delete(idx);
          } else {
            // Add if not selected
            selectedInterpretationTokens.add(idx);
          }
          
          lastHoveredTokenIdx = idx;
          
          // Re-render to update visual state
          renderTokens(tokens);
          updateFilteredVisualizations();
        }, 50); // 50ms debounce
      });
      
      el.addEventListener("mouseleave", () => {
        // Reset last hovered token when leaving
        if (lastHoveredTokenIdx === idx) {
          lastHoveredTokenIdx = null;
        }
      });
    });
  }

  // Update interpretation status
  const queryStatusEl = document.getElementById("query-status");
  const keyStatusEl = document.getElementById("key-status");
  const countEl = document.getElementById("selected-tokens-count");
  
  if (queryStatusEl) {
    if (interpretationQueryIndex !== null && currentTokens) {
      const tokenText = formatAxisToken(currentTokens[interpretationQueryIndex]);
      queryStatusEl.textContent = tokenText;
      queryStatusEl.classList.add("selected");
    } else {
      queryStatusEl.textContent = "Not selected";
      queryStatusEl.classList.remove("selected");
    }
  }
  
  if (keyStatusEl) {
    if (interpretationKeyIndex !== null && currentTokens) {
      const tokenText = formatAxisToken(currentTokens[interpretationKeyIndex]);
      keyStatusEl.textContent = tokenText;
      keyStatusEl.classList.add("selected");
    } else {
      keyStatusEl.textContent = "Not selected";
      keyStatusEl.classList.remove("selected");
    }
  }
  
  if (countEl) {
    // Count all selected tokens (including Query/Key)
    const interpretationCount = selectedInterpretationTokens.size;
    if (interpretationCount > 0) {
      countEl.textContent = `${interpretationCount} tokens selected`;
    } else {
      countEl.textContent = "";
    }
  }

  // Setup click handlers and hover for interpretation mode
  document.querySelectorAll(".token").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (interpretationTokenMode === "interpretation") {
        // In interpretation mode, hover handles add/remove automatically
        // Click is not needed, but we prevent default behavior
        e.stopPropagation();
        return;
      }
      
      // Normal mode: select Query/Key or src/dst
      const idx = Number(el.getAttribute("data-idx"));
      
      // Ignore clicks on disabled tokens (after query, only when key is not selected)
      if (el.disabled || el.classList.contains("token-disabled")) {
        return;
      }
      
      // Query/Key selection (always available)
      // Query 선택: 항상 새로운 Query로 선택 (어떤 위치든)
      if (interpretationQueryIndex === null) {
        // First query selection
        interpretationQueryIndex = idx;
        interpretationKeyIndex = null;
        selectedInterpretationTokens.clear();
        // Add Query to selected tokens
        selectedInterpretationTokens.add(interpretationQueryIndex);
      } else if (interpretationKeyIndex === null) {
        // Query is selected, now select key
        if (idx === interpretationQueryIndex) {
          // Clicking same query again - reset query
          selectedInterpretationTokens.delete(interpretationQueryIndex);
          interpretationQueryIndex = null;
          selectedInterpretationTokens.clear();
        } else if (idx <= interpretationQueryIndex) {
          // Select key (SRC) - must be before or equal to query
          interpretationKeyIndex = idx;
          // Ensure Query and Key are in selected tokens
          selectedInterpretationTokens.clear();
          selectedInterpretationTokens.add(interpretationQueryIndex);
          selectedInterpretationTokens.add(interpretationKeyIndex);
        } else {
          // Cannot select key after query (causal attention)
          alert("Key (SRC) must be at or before Query (DST) position due to causal attention.");
          return;
        }
      } else {
        // Both Query and Key selected
        if (idx === interpretationQueryIndex) {
          // Clicking query again - reset to new query only
          selectedInterpretationTokens.delete(interpretationQueryIndex);
          selectedInterpretationTokens.delete(interpretationKeyIndex);
          interpretationQueryIndex = idx;
          interpretationKeyIndex = null;
          selectedInterpretationTokens.clear();
          selectedInterpretationTokens.add(interpretationQueryIndex);
        } else if (idx === interpretationKeyIndex) {
          // Clicking key again - reset key only
          selectedInterpretationTokens.delete(interpretationKeyIndex);
          interpretationKeyIndex = null;
          // Keep Query in selected tokens
          if (!selectedInterpretationTokens.has(interpretationQueryIndex)) {
            selectedInterpretationTokens.add(interpretationQueryIndex);
          }
        } else if (idx > interpretationQueryIndex) {
          // Clicking after query - select as new query
          selectedInterpretationTokens.delete(interpretationQueryIndex);
          selectedInterpretationTokens.delete(interpretationKeyIndex);
          interpretationQueryIndex = idx;
          interpretationKeyIndex = null;
          selectedInterpretationTokens.clear();
          selectedInterpretationTokens.add(interpretationQueryIndex);
        } else {
          // Query/Key가 모두 선택된 상태에서 다른 토큰 클릭 시
          // Interpretation Mode가 아닌 경우에는 아무것도 하지 않음
          if (interpretationTokenMode !== "interpretation") {
            // Do nothing - just maintain Query/Key selection
            return;
          }
          // Interpretation mode에서는 hover 상태에 따라 토큰 추가/제거
          const isCurrentlySelected = selectedInterpretationTokens.has(idx);
          if (isCurrentlySelected) {
            // Remove if already selected
            selectedInterpretationTokens.delete(idx);
          } else {
            // Add if not selected
            selectedInterpretationTokens.add(idx);
          }
        }
      }
      
      renderTokens(tokens);
      updateFilteredVisualizations();
      
      // Update attention visualization if Query/Key are both selected
      if (interpretationQueryIndex !== null && interpretationKeyIndex !== null) {
        const key = `${interpretationKeyIndex}_${interpretationQueryIndex}`;
        if (allAttentions && allAttentions[key]) {
          renderGrid(allAttentions[key]);
          // Update token map with current layer/head selection
          if (lastLayerIdx !== null && lastHeadIdx !== null) {
            renderTokenMap(lastLayerIdx, lastHeadIdx);
          } else {
            // Default to first layer and head if not set
            renderTokenMap(0, 0);
          }
        }
      }
    });
  });

  // Setup drag selection
  if (interpretationTokenMode === "interpretation") {
    setupDragSelection();
  }
}

function setupDragSelection() {
  if (!tokensEl) return;
  
  tokensEl.addEventListener("mousedown", (e) => {
    if (interpretationTokenMode !== "interpretation" || e.target.closest(".token")) return;
    isDragging = true;
    const rect = tokensEl.getBoundingClientRect();
    dragStart = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
    
    if (!dragSelectionBox) {
      dragSelectionBox = document.createElement("div");
      dragSelectionBox.className = "drag-selection-box";
      tokensEl.style.position = "relative";
      tokensEl.appendChild(dragSelectionBox);
    }
    dragSelectionBox.style.display = "block";
    dragSelectionBox.style.left = dragStart.x + "px";
    dragSelectionBox.style.top = dragStart.y + "px";
    dragSelectionBox.style.width = "0px";
    dragSelectionBox.style.height = "0px";
  });

  tokensEl.addEventListener("mousemove", (e) => {
    if (!isDragging || !dragStart) return;
    const rect = tokensEl.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    const left = Math.min(dragStart.x, currentX);
    const top = Math.min(dragStart.y, currentY);
    const width = Math.abs(currentX - dragStart.x);
    const height = Math.abs(currentY - dragStart.y);
    
    dragSelectionBox.style.left = left + "px";
    dragSelectionBox.style.top = top + "px";
    dragSelectionBox.style.width = width + "px";
    dragSelectionBox.style.height = height + "px";
    
    // Select tokens in the selection box
    document.querySelectorAll(".token").forEach((tokenEl) => {
      const tokenRect = tokenEl.getBoundingClientRect();
      const tokenElRect = tokensEl.getBoundingClientRect();
      const tokenX = tokenRect.left - tokenElRect.left;
      const tokenY = tokenRect.top - tokenElRect.top;
      const tokenWidth = tokenRect.width;
      const tokenHeight = tokenRect.height;
      
      const tokenCenterX = tokenX + tokenWidth / 2;
      const tokenCenterY = tokenY + tokenHeight / 2;
      
      if (tokenCenterX >= left && tokenCenterX <= left + width &&
          tokenCenterY >= top && tokenCenterY <= top + height) {
        const idx = Number(tokenEl.getAttribute("data-idx"));
        
        if (interpretationTokenMode === "interpretation") {
          // In interpretation mode: determine add/remove based on current selection state
          // Skip Query/Key tokens - they should always remain selected
          if (idx === interpretationQueryIndex || idx === interpretationKeyIndex) {
            return;
          }
          
          const isCurrentlySelected = selectedInterpretationTokens.has(idx);
          if (isCurrentlySelected) {
            // Remove if already selected
            selectedInterpretationTokens.delete(idx);
          } else {
            // Add if not selected
            selectedInterpretationTokens.add(idx);
          }
        }
      }
    });
    renderTokens(currentTokens);
  });

  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      if (dragSelectionBox) {
        dragSelectionBox.style.display = "none";
      }
      updateFilteredVisualizations();
    }
  });
}

function updateFilteredVisualizations() {
  if (!currentTokens || !allAttentions) return;
  
  // Filter tokens based on interpretation selection
  const filteredIndices = selectedInterpretationTokens.size > 0
    ? Array.from(selectedInterpretationTokens).sort((a, b) => a - b)
    : currentTokens.map((_, idx) => idx);
  
  // Update grid if src/dst are selected
  if (srcIndex !== null && dstIndex !== null) {
    const key = `${srcIndex}_${dstIndex}`;
    if (allAttentions[key]) {
      renderGrid(allAttentions[key]);
    }
  }
  
  // Update token map with filtered view if interpretation tokens are selected
  if (selectedInterpretationTokens.size > 0 && lastLayerIdx !== null && lastHeadIdx !== null) {
    renderTokenMapFiltered(lastLayerIdx, lastHeadIdx, filteredIndices);
  } else if (lastLayerIdx !== null && lastHeadIdx !== null) {
    // Show all tokens if no interpretation selection
    renderTokenMap(lastLayerIdx, lastHeadIdx);
  }
}

function renderTokenMapFiltered(layerIdx, headIdx, filteredIndices) {
  // Similar to renderTokenMap but only show filtered tokens
  if (!tokenMapEl || !allAttentions || !currentTokens) return;
  
  const seqLen = filteredIndices.length;
  if (seqLen === 0) {
    tokenMapEl.innerHTML = "<p style='color:#9aa1b5; font-size:12px; padding:4px;'>No tokens selected for interpretation.</p>";
    return;
  }
  
  let html = "<table><tbody>";
  
  for (let dstIdx = 0; dstIdx < seqLen; dstIdx++) {
    const dst = filteredIndices[dstIdx];
    const rowLabel = escapeHtml(formatAxisToken(currentTokens[dst]));
    // Use interpretationQueryIndex for row highlighting (Query/DST = orange)
    const isDstRow = interpretationQueryIndex !== null && dst === interpretationQueryIndex;
    const rowClass = isDstRow ? "token-map-row-highlight" : "";
    html += `<tr class="${rowClass}" data-dst="${dst}"><th class="token-axis-row ${isDstRow ? 'token-axis-row-highlight' : ''}">${rowLabel}</th>`;
    
    for (let srcIdx = 0; srcIdx < seqLen; srcIdx++) {
      const src = filteredIndices[srcIdx];
      // Use interpretationKeyIndex for column highlighting (Key/SRC = green)
      const isSrcCol = interpretationKeyIndex !== null && src === interpretationKeyIndex;
      
      if (src > dst) {
        const colClass = isSrcCol ? "token-map-col-highlight" : "";
        html += `<td class="token-map-empty ${colClass}"></td>`;
        continue;
      }
      
      const key = `${src}_${dst}`;
      const perPair = allAttentions[key] || null;
      let v = 0;
      
      if (perPair) {
        if (tokenMapMode === "head") {
          if (perPair[layerIdx] && typeof perPair[layerIdx][headIdx] === "number") {
            v = perPair[layerIdx][headIdx];
          }
        } else if (tokenMapMode === "layer") {
          const layerVals = perPair[layerIdx];
          if (Array.isArray(layerVals) && layerVals.length > 0) {
            const sum = layerVals.reduce((acc, val) => acc + (typeof val === "number" ? val : 0), 0);
            v = sum / layerVals.length;
          }
        } else if (tokenMapMode === "overall") {
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
      
      // Check if this is the selected cell (intersection of key and query)
      const isSelectedCell = (interpretationKeyIndex !== null && src === interpretationKeyIndex) && (interpretationQueryIndex !== null && dst === interpretationQueryIndex);
      const colClass = isSrcCol ? "token-map-col-highlight" : "";
      
      if (src === dst) {
        const bgColor = colorForValue(v, true);
        const displayValue = v.toFixed(2);
        const cellClass = isSelectedCell ? "token-map-cell-selected" : "";
        html += `<td class="token-map-diagonal ${colClass}"><div class="token-map-cell ${cellClass}" style="background:${bgColor}" title="q=${dst}, k=${src}, L${layerIdx}, H${headIdx}: ${v.toFixed(4)}">${displayValue}</div></td>`;
      } else {
        const bgColor = colorForValue(v, true);
        const displayValue = v.toFixed(2);
        const cellClass = isSelectedCell ? "token-map-cell-selected" : "";
        html += `<td class="${colClass}"><div class="token-map-cell ${cellClass}" style="background:${bgColor}" title="q=${dst}, k=${src}, L${layerIdx}, H${headIdx}: ${v.toFixed(4)}">${displayValue}</div></td>`;
      }
    }
    html += "</tr>";
  }
  
  html += "</tbody><tfoot><tr><th class='token-map-footer-label'>Key (SRC)</th>";
  for (let srcIdx = 0; srcIdx < seqLen; srcIdx++) {
    const src = filteredIndices[srcIdx];
    const keyLabel = escapeHtml(formatAxisToken(currentTokens[src]));
    // Use interpretationKeyIndex for column highlighting (Key/SRC = green)
    const isSrcCol = interpretationKeyIndex !== null && src === interpretationKeyIndex;
    const colClass = isSrcCol ? "token-map-col-highlight" : "";
    html += `<td class="token-map-key-footer ${colClass}"><div class="token-map-key-label-bottom ${isSrcCol ? 'token-map-key-label-highlight' : ''}">${keyLabel}</div></td>`;
  }
  html += "</tr></tfoot></table>";
  
  tokenMapEl.innerHTML = `
    <div class="token-map-layout">
      <div class="token-map-corner"></div>
      <div class="token-map-axis token-map-axis-top">Key (SRC) →</div>
      <div class="token-map-axis token-map-axis-left">Query (DST) ↓</div>
      <div class="token-map-table-wrap">${html}</div>
    </div>
  `;
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

// Head filter controls
const headFilterStartInput = document.getElementById("head-filter-start");
const headFilterEndInput = document.getElementById("head-filter-end");
const headFilterApplyBtn = document.getElementById("head-filter-apply");
const headFilterResetBtn = document.getElementById("head-filter-reset");
const headJumpInput = document.getElementById("head-jump-input");
const headJumpBtn = document.getElementById("head-jump-btn");
const headCountInfo = document.getElementById("head-count-info");

function applyHeadFilter() {
  const start = headFilterStartInput.value ? parseInt(headFilterStartInput.value) : null;
  const end = headFilterEndInput.value ? parseInt(headFilterEndInput.value) : null;
  
  if (start !== null && start < 0) {
    alert("Start head must be >= 0");
    return;
  }
  if (end !== null && end < 0) {
    alert("End head must be >= 0");
    return;
  }
  if (start !== null && end !== null && start > end) {
    alert("Start head must be <= end head");
    return;
  }
  
  headFilterStart = start;
  headFilterEnd = end;
  
  if (currentScores) {
    renderGrid(currentScores);
  }
}

function resetHeadFilter() {
  headFilterStart = null;
  headFilterEnd = null;
  headFilterStartInput.value = "0";
  headFilterEndInput.value = "";
  if (currentScores) {
    renderGrid(currentScores);
  }
}

function jumpToHead() {
  const headNum = headJumpInput.value ? parseInt(headJumpInput.value) : null;
  if (headNum === null || headNum < 0) {
    alert("Please enter a valid head number (>= 0)");
    return;
  }
  
  if (!currentScores || !currentScores[0]) {
    alert("No scores available. Please compute attention first.");
    return;
  }
  
  const totalHeads = currentScores[0].length;
  if (headNum >= totalHeads) {
    alert(`Head ${headNum} does not exist. Maximum head is ${totalHeads - 1}.`);
    return;
  }
  
  // Find the row element and scroll to it
  const headRow = grid.querySelector(`tr.head-row[data-head="${headNum}"]`);
  if (headRow) {
    headRow.scrollIntoView({ behavior: "smooth", block: "center" });
    // Highlight the row briefly
    headRow.style.backgroundColor = "rgba(255, 247, 0, 0.2)";
    setTimeout(() => {
      headRow.style.backgroundColor = "";
    }, 1500);
  } else {
    // If row is not visible due to filter, show it
    headFilterStart = headNum;
    headFilterEnd = headNum;
    headFilterStartInput.value = headNum;
    headFilterEndInput.value = headNum;
    renderGrid(currentScores);
    setTimeout(() => {
      const newRow = grid.querySelector(`tr.head-row[data-head="${headNum}"]`);
      if (newRow) {
        newRow.scrollIntoView({ behavior: "smooth", block: "center" });
        newRow.style.backgroundColor = "rgba(255, 247, 0, 0.2)";
        setTimeout(() => {
          newRow.style.backgroundColor = "";
        }, 1500);
      }
    }, 100);
  }
}

if (headFilterApplyBtn) {
  headFilterApplyBtn.addEventListener("click", applyHeadFilter);
}
if (headFilterResetBtn) {
  headFilterResetBtn.addEventListener("click", resetHeadFilter);
}
if (headJumpBtn) {
  headJumpBtn.addEventListener("click", jumpToHead);
}
if (headJumpInput) {
  headJumpInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") jumpToHead();
  });
}

// Layer filter controls
const layerFilterStartInput = document.getElementById("layer-filter-start");
const layerFilterEndInput = document.getElementById("layer-filter-end");
const layerFilterApplyBtn = document.getElementById("layer-filter-apply");
const layerFilterResetBtn = document.getElementById("layer-filter-reset");

function applyLayerFilter() {
  const start = layerFilterStartInput.value ? parseInt(layerFilterStartInput.value) : null;
  const end = layerFilterEndInput.value ? parseInt(layerFilterEndInput.value) : null;
  
  if (start !== null && start < 0) {
    alert("Start layer must be >= 0");
    return;
  }
  if (end !== null && end < 0) {
    alert("End layer must be >= 0");
    return;
  }
  if (start !== null && end !== null && start > end) {
    alert("Start layer must be <= end layer");
    return;
  }
  
  layerFilterStart = start;
  layerFilterEnd = end;
  
  if (currentScores) {
    renderGrid(currentScores);
  }
}

function resetLayerFilter() {
  layerFilterStart = null;
  layerFilterEnd = null;
  layerFilterStartInput.value = "0";
  layerFilterEndInput.value = "";
  if (currentScores) {
    renderGrid(currentScores);
  }
}

if (layerFilterApplyBtn) {
  layerFilterApplyBtn.addEventListener("click", applyLayerFilter);
}
if (layerFilterResetBtn) {
  layerFilterResetBtn.addEventListener("click", resetLayerFilter);
}

// Mode button handlers
const interpretationModeBtn = document.getElementById("interpretation-mode-btn");

function setMode(mode) {
  // Set new mode
  interpretationTokenMode = mode;
  
  // Reset hover tracking when mode changes
  lastHoveredTokenIdx = null;
  if (hoverDebounceTimer) {
    clearTimeout(hoverDebounceTimer);
    hoverDebounceTimer = null;
  }
  
  // Add active class to current mode button
  if (mode === "interpretation" && interpretationModeBtn) {
    interpretationModeBtn.classList.add("active");
  } else if (interpretationModeBtn) {
    interpretationModeBtn.classList.remove("active");
  }
  
  // Don't clear selectedInterpretationTokens when mode is turned off
  // Selected tokens should persist even when mode is off
  
  if (currentTokens) {
    renderTokens(currentTokens);
    updateFilteredVisualizations();
  }
}

if (interpretationModeBtn) {
  interpretationModeBtn.addEventListener("click", () => {
    if (interpretationTokenMode === "interpretation") {
      setMode(false); // Toggle off
    } else {
      setMode("interpretation");
    }
  });
}

// Select All / Remove All buttons
const selectAllBtn = document.getElementById("select-all-btn");
const removeAllBtn = document.getElementById("remove-all-btn");

if (selectAllBtn) {
  selectAllBtn.addEventListener("click", () => {
    if (!currentTokens) return;
    // Select all tokens (including Query/Key)
    currentTokens.forEach((_, idx) => {
      selectedInterpretationTokens.add(idx);
    });
    // Ensure Query/Key are in the set
    if (interpretationQueryIndex !== null) {
      selectedInterpretationTokens.add(interpretationQueryIndex);
    }
    if (interpretationKeyIndex !== null) {
      selectedInterpretationTokens.add(interpretationKeyIndex);
    }
    if (currentTokens) {
      renderTokens(currentTokens);
      updateFilteredVisualizations();
    }
  });
}

if (removeAllBtn) {
  removeAllBtn.addEventListener("click", () => {
    // Remove all selected interpretation tokens, but keep Query/Key if they exist
    const queryIdx = interpretationQueryIndex;
    const keyIdx = interpretationKeyIndex;
    selectedInterpretationTokens.clear();
    // Re-add Query/Key if they exist
    if (queryIdx !== null) {
      selectedInterpretationTokens.add(queryIdx);
    }
    if (keyIdx !== null) {
      selectedInterpretationTokens.add(keyIdx);
    }
    if (currentTokens) {
      renderTokens(currentTokens);
      updateFilteredVisualizations();
    }
  });
}

// Clear button - reset all token selections
const clearSelectionBtn = document.getElementById("clear-selection-btn");
if (clearSelectionBtn) {
  clearSelectionBtn.addEventListener("click", () => {
    // Clear all: Query, Key, and interpretation tokens
    interpretationQueryIndex = null;
    interpretationKeyIndex = null;
    selectedInterpretationTokens.clear();
    srcIndex = null;
    dstIndex = null;
    
    if (currentTokens) {
      renderTokens(currentTokens);
      updateFilteredVisualizations();
    }
  });
}

// Initial state
modelStatus.textContent = "Load a model to start.";
addAssistantMessage("왼쪽 입력창에 프롬프트를 보내면 attention을 계산합니다.");
if (tokenMapEl) {
  tokenMapEl.innerHTML = "<p style='color:#9aa1b5; font-size:12px; padding:8px;'>위 레이어/헤드 그리드에서 셀을 클릭하면, 여기에 해당 Head의 토큰→토큰 attention 맵이 표시됩니다.</p>";
}
fetchModels();

