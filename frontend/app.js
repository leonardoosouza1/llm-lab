// ============================================================================
// IARA LLM Observability Dashboard — app.js
// Architecture: Vue 3 (CDN global build) + D3 v7 + ChartJS 4
// WebSocket: ws://localhost:8080  |  HTTP API: http://localhost:3000
// NO ES module imports — all globals
// ============================================================================

// ── Utility: safe element getter ──
function el(id) { return document.getElementById(id); }

// ── Utility: thermal color mapping ──
// Maps 0→#1a1a2e, 0.5→#a855f7, 1→#00ffc8
function thermalColor(value) {
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.5) {
    // #1a1a2e → #a855f7
    const t = v * 2;
    const r = Math.round(0x1a + (0xa8 - 0x1a) * t);
    const g = Math.round(0x1a + (0x55 - 0x1a) * t);
    const b = Math.round(0x2e + (0xf7 - 0x2e) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    // #a855f7 → #00ffc8
    const t = (v - 0.5) * 2;
    const r = Math.round(0xa8 + (0x00 - 0xa8) * t);
    const g = Math.round(0x55 + (0xff - 0x55) * t);
    const b = Math.round(0xf7 + (0xc8 - 0xf7) * t);
    return `rgb(${r},${g},${b})`;
  }
}

// ── Utility: thermal color as [r,g,b] array ──
function thermalColorRGB(value) {
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.5) {
    const t = v * 2;
    return [
      Math.round(0x1a + (0xa8 - 0x1a) * t),
      Math.round(0x1a + (0x55 - 0x1a) * t),
      Math.round(0x2e + (0xf7 - 0x2e) * t),
    ];
  } else {
    const t = (v - 0.5) * 2;
    return [
      Math.round(0xa8 + (0x00 - 0xa8) * t),
      Math.round(0x55 + (0xff - 0x55) * t),
      Math.round(0xf7 + (0xc8 - 0xf7) * t),
    ];
  }
}

// ── Utility: timestamp HH:MM:SS.mmm ──
function timestamp() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const ms = String(d.getMilliseconds()).padStart(3, '0');
  return `${hh}:${mm}:${ss}.${ms}`;
}

// ── Utility: HTML escape ──
function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = String(text);
  return d.innerHTML;
}

// ── Utility: JSON syntax highlight ──
function syntaxHighlight(json) {
  const str = typeof json === 'string' ? json : JSON.stringify(json, null, 2);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function(match) {
      if (/^"/.test(match)) {
        if (/:$/.test(match)) {
          return `<span style="color:#00d4ff">${match}</span>`;
        }
        return `<span style="color:#00ff88">${match}</span>`;
      }
      if (/true|false/.test(match)) {
        return `<span style="color:#a855f7">${match}</span>`;
      }
      if (/null/.test(match)) {
        return `<span style="color:#ff4466">${match}</span>`;
      }
      return `<span style="color:#ffb347">${match}</span>`;
    });
}

// ============================================================================
// Reactive Store
// ============================================================================
window.store = (function () {
  // Guard: if Vue isn't loaded yet, we'll define a plain object and upgrade later
  const reactive = (typeof Vue !== 'undefined') ? Vue.reactive : function(o) { return o; };
  return reactive({
    // Connection
    wsStatus: 'disconnected',
    httpHealthy: false,
    modelName: 'qwen3-8b',
    modelReady: false,
    loadingMsg: 'Aguardando servidor...',

    // UI
    activeTab: 'live',
    availableLayers: [],
    selectedLayer: null,

    // Live panel
    prompt: '',
    maxTokens: 256,
    temperature: 0.6,
    generating: false,
    responseText: '',
    evalCount: 0,
    totalDurationMs: 0,

    // Per-layer data keyed by layer_idx
    layerData: {},

    // Pipeline result
    topTokens: [],
    pipelineComplete: false,

    // Current WS frame
    currentFrame: null,

    // Trace
    trace: null,
    anomalyCount: 0,

    // Performance
    tokensPerSec: 0,
    msPerLayer: 0,

    // Console logs
    logs: [],
    logFilter: 'ALL',

    // Brain map
    brainMap: null,
  });
})();

// ============================================================================
// Log helper — adds to store.logs AND legacy DOM log container
// ============================================================================
let _logId = 0;
function addLog(typeOrLevel, raw, text) {
  // Support old-style addLog(message, level) and new-style addLog(type, raw, text)
  let type, rawObj, textStr;
  if (text === undefined) {
    // Legacy call: addLog(message, level)
    type = (raw === 'error') ? 'error' : (raw === 'warning' ? 'error' : 'http');
    rawObj = typeOrLevel;
    textStr = String(typeOrLevel);
  } else {
    type = typeOrLevel;
    rawObj = raw;
    textStr = text || (typeof raw === 'string' ? raw : JSON.stringify(raw));
  }

  const entry = {
    id: ++_logId,
    ts: timestamp(),
    type,
    raw: rawObj,
    text: textStr,
  };

  store.logs.push(entry);
  if (store.logs.length > 500) {
    store.logs.splice(0, store.logs.length - 500);
  }

  // Also update legacy DOM log container if present
  const logContainer = el('logContainer');
  if (logContainer) {
    const div = document.createElement('div');
    div.className = 'log-entry ' + (type === 'error' ? 'error' : type === 'ws-in' ? 'info' : type === 'ws-out' ? 'success' : 'info');
    div.textContent = `[${entry.ts}] ${textStr}`;
    logContainer.appendChild(div);
    logContainer.scrollTop = logContainer.scrollHeight;
    while (logContainer.children.length > 100) {
      logContainer.removeChild(logContainer.firstChild);
    }
  }
}

// ============================================================================
// WebSocket Manager
// ============================================================================
window.wsManager = {
  socket: null,
  reconnectDelay: 1000,
  _reconnectAttempt: 0,
  _backoffDelays: [1000, 2000, 4000, 8000, 16000, 30000],

  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    store.wsStatus = 'connecting';
    this._updateConnectionUI();
    addLog('ws-out', 'connect', `Connecting to ws://localhost:8080...`);

    try {
      this.socket = new WebSocket('ws://localhost:8080');
    } catch (e) {
      addLog('error', e.message, `WS constructor error: ${e.message}`);
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      store.wsStatus = 'connected';
      this._reconnectAttempt = 0;
      this.reconnectDelay = 1000;
      this._updateConnectionUI();
      addLog('ws-in', { type: 'connected' }, 'Connected to IARA engine');

      // Hide overlay if present
      const overlay = el('canvasOverlay');
      if (overlay) overlay.classList.add('hidden');

      // Request layer list
      this.listLayers();
    };

    this.socket.onclose = () => {
      store.wsStatus = 'disconnected';
      this._updateConnectionUI();
      addLog('error', { type: 'disconnected' }, `WS disconnected — reconnecting in ${this.reconnectDelay}ms`);
      const overlay = el('canvasOverlay');
      if (overlay) {
        overlay.classList.remove('hidden');
        const ot = overlay.querySelector('.overlay-text');
        if (ot) ot.textContent = 'Connection lost. Reconnecting...';
      }
      this._scheduleReconnect();
    };

    this.socket.onerror = () => {
      addLog('error', { type: 'ws-error' }, 'WebSocket error');
    };

    this.socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.dispatch(msg);
      } catch (e) {
        addLog('error', event.data, `Parse error: ${e.message}`);
      }
    };
  },

  _scheduleReconnect() {
    const delays = this._backoffDelays;
    const idx = Math.min(this._reconnectAttempt, delays.length - 1);
    this.reconnectDelay = delays[idx];
    this._reconnectAttempt++;
    setTimeout(() => this.connect(), this.reconnectDelay);
  },

  _updateConnectionUI() {
    const indicator = el('connectionIndicator');
    const text = el('connectionText');
    if (indicator) {
      indicator.className = 'status-dot ' + store.wsStatus;
    }
    if (text) {
      const labels = {
        connected: 'Connected',
        connecting: 'Connecting...',
        disconnected: 'Disconnected',
      };
      text.textContent = labels[store.wsStatus] || store.wsStatus;
    }
  },

  send(obj) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      addLog('error', obj, 'WS not connected — cannot send');
      return;
    }
    const str = JSON.stringify(obj);
    this.socket.send(str);
    addLog('ws-out', obj, `→ ${obj.type}${obj.text ? ': ' + obj.text : ''}`);
  },

  stimulus(text) { this.send({ type: 'stimulus', text }); },
  resonate(text) { this.send({ type: 'resonate', text }); },
  selectLayer(name) { this.send({ type: 'select_layer', name }); },
  listLayers() { this.send({ type: 'list_layers' }); },
  reset() { this.send({ type: 'reset' }); },

  dispatch(msg) {
    addLog('ws-in', msg, `← ${msg.type}`);

    switch (msg.type) {
      case 'status':
        addLog('ws-in', msg, `Status: ${msg.message}`);
        {
          const overlay = el('canvasOverlay');
          if (overlay) overlay.classList.add('hidden');
        }
        break;

      case 'layers':
        store.availableLayers = msg.names || [];
        addLog('ws-in', msg, `${store.availableLayers.length} layers available`);
        renderLayerList(store.availableLayers);
        break;

      case 'frame':
        store.currentFrame = msg;
        triggerFrameHeatmap();
        updateLegacyFrameUI(msg);
        break;

      case 'pipeline_layer':
        store.layerData[msg.layer_idx] = {
          attention_energy: msg.attention_energy,
          mlp_energy: msg.mlp_energy,
          residual_norm: msg.residual_norm,
          max_activation: msg.max_activation,
          hot_neurons: msg.hot_neurons || [],
          time_us: msg.time_us,
          spikes: msg.spikes,
          phase: msg.phase,
          total_layers: msg.total_layers,
        };
        updatePipelineUI(msg);
        break;

      case 'pipeline_result':
        store.topTokens = msg.top_tokens || [];
        store.pipelineComplete = true;
        updatePipelineResultUI(msg);
        updateLogitsChart();
        // Auto-fetch trace after 500ms
        setTimeout(() => api.fetchTrace(), 500);
        break;

      case 'brain_map':
        store.brainMap = msg;
        renderBrainMap(msg);
        break;

      case 'loading':
        {
          const overlay = el('canvasOverlay');
          if (overlay) {
            overlay.classList.remove('hidden');
            const ot = overlay.querySelector('.overlay-text');
            if (ot) ot.textContent = msg.progress || 'Loading...';
          }
        }
        break;

      case 'generation_update':
        if (msg.data && msg.data.full_text) {
          store.responseText = msg.data.full_text;
        }
        break;

      case 'status':
        {
          const message = msg.message || '';
          store.loadingMsg = message;
          // Detect when model is ready via WS status message
          if (message.toLowerCase().includes('carregado') || message.toLowerCase().includes('pronto')) {
            store.modelReady = true;
            updateModelLoadingBanner();
          }
          addLog('ws-in', msg, `Status: ${message}`);
          updateModelLoadingBanner();
        }
        break;

      default:
        addLog('ws-in', msg, `Unknown message type: ${msg.type}`);
        break;
    }
  },
};

// ============================================================================
// HTTP API
// ============================================================================
window.api = {
  async health() {
    try {
      const res = await fetch('http://localhost:3000/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      store.httpHealthy = true;
      store.modelName = data.model || store.modelName;
      store.modelReady = data.status === 'ok';
      if (!store.modelReady) {
        store.loadingMsg = 'Carregando modelo ' + (data.model || '') + '...';
      }
      updateModelLoadingBanner();
      addLog('http', data, `Health: ${data.status} — model: ${data.model}`);
      return data;
    } catch (e) {
      store.httpHealthy = false;
      addLog('error', { error: e.message }, `Health check failed: ${e.message}`);
      return null;
    }
  },

  async generate(prompt, maxTokens, temperature) {
    store.generating = true;
    store.responseText = '';
    store.evalCount = 0;
    store.totalDurationMs = 0;
    store.pipelineComplete = false;

    const body = {
      model: store.modelName,
      prompt,
      max_tokens: maxTokens,
      temperature,
    };

    addLog('http', body, `POST /api/generate — prompt: "${prompt}"`);

    try {
      const res = await fetch('http://localhost:3000/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      store.responseText = data.response || '';
      store.evalCount = data.eval_count || 0;
      store.totalDurationMs = data.total_duration ? data.total_duration / 1e6 : 0;
      store.generating = false;

      // Performance metrics
      if (store.evalCount > 0 && store.totalDurationMs > 0) {
        store.tokensPerSec = store.evalCount / (store.totalDurationMs / 1000);
      }
      store.msPerLayer = Object.values(store.layerData)
        .reduce((s, l) => s + (l.time_us || 0), 0) / 1000 / 36;

      addLog('http', data, `Generate done — ${store.evalCount} tokens in ${store.totalDurationMs.toFixed(0)}ms`);

      updateGenerateResultUI(data);
      return data;
    } catch (e) {
      store.generating = false;
      addLog('error', { error: e.message }, `Generate failed: ${e.message}`);
      return null;
    }
  },

  async fetchTrace() {
    try {
      const res = await fetch('http://localhost:3000/api/debug/last-trace');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      store.trace = data;
      store.anomalyCount = (data.anomalies || []).length;
      addLog('http', data, `Trace fetched — resonance: ${data.iara_resonance_score}`);
      renderTracePanel();
      return data;
    } catch (e) {
      addLog('error', { error: e.message }, `Fetch trace failed: ${e.message}`);
      return null;
    }
  },
};

// ============================================================================
// Legacy DOM helpers (for existing index.html elements)
// ============================================================================

function updateLegacyFrameUI(msg) {
  const metricTick   = el('metricTick');
  const metricEnergy = el('metricEnergy');
  const metricActive = el('metricActive');
  const metricTotal  = el('metricTotal');
  const metricGrid   = el('metricGrid');
  const currentLayer = el('currentLayer');
  const overlay      = el('canvasOverlay');

  if (overlay) overlay.classList.add('hidden');
  if (metricTick)   metricTick.textContent   = (msg.tick || 0).toLocaleString();
  if (metricEnergy) metricEnergy.textContent = (msg.energy || 0).toFixed(2);
  if (metricActive) metricActive.textContent = (msg.active_nodes || 0).toLocaleString();
  if (metricTotal)  metricTotal.textContent  = (msg.total_nodes || 0).toLocaleString();
  if (metricGrid && msg.grid_size) metricGrid.textContent = `${msg.grid_size}×${msg.grid_size}`;
  if (currentLayer && msg.layer)   currentLayer.textContent = msg.layer;
}

function updatePipelineUI(msg) {
  const pipelinePanel    = el('pipelinePanel');
  const pipelinePhase    = el('pipelinePhase');
  const pipelineProgress = el('pipelineProgress');
  const metricActive     = el('metricActive');

  if (pipelinePanel) pipelinePanel.style.display = '';
  if (pipelinePhase) pipelinePhase.textContent = msg.phase || `Layer ${msg.layer_idx + 1}/${msg.total_layers}`;
  if (pipelineProgress) {
    const pct = ((msg.layer_idx + 1) / (msg.total_layers || 36) * 100).toFixed(0);
    pipelineProgress.style.width = pct + '%';
  }
  if (metricActive && msg.spikes !== undefined) {
    metricActive.textContent = msg.spikes.toLocaleString();
  }

  // Also update layer grid if in layers tab
  if (store.activeTab === 'layers') {
    renderLayerGrid();
  }
}

function updatePipelineResultUI(msg) {
  const pipelinePanel    = el('pipelinePanel');
  const pipelineProgress = el('pipelineProgress');
  const responsePanel    = el('responsePanel');
  const responseTime     = el('responseTime');
  const responseTokens   = el('responseTokens');

  if (pipelinePanel)    pipelinePanel.style.display = 'none';
  if (pipelineProgress) pipelineProgress.style.width = '0%';
  if (responsePanel)    responsePanel.style.display = '';
  if (responseTime)     responseTime.textContent = `${msg.total_time_ms}ms | ${msg.layers_processed} layers | "${msg.prompt}"`;
  if (responseTokens)   renderResponseTokens(msg.top_tokens || []);

  addLog('ws-in', msg, `Pipeline complete: ${msg.total_time_ms}ms — top: ${msg.top_tokens && msg.top_tokens[0] ? msg.top_tokens[0].text : '?'}`);
}

function updateGenerateResultUI(data) {
  const responsePanel  = el('responsePanel');
  const responseTime   = el('responseTime');
  const responseTokens = el('responseTokens');

  if (responsePanel) responsePanel.style.display = '';
  if (responseTime) {
    responseTime.textContent = `${store.totalDurationMs.toFixed(0)}ms | ${store.evalCount} tokens | ${store.tokensPerSec.toFixed(1)} tok/s`;
  }
  if (responseTokens && data.response) {
    responseTokens.innerHTML = `<div class="log-entry info" style="white-space:pre-wrap">${escapeHtml(data.response)}</div>`;
  }
}

function renderResponseTokens(tokens) {
  const responseTokens = el('responseTokens');
  if (!responseTokens) return;
  responseTokens.innerHTML = '';
  if (!tokens || tokens.length === 0) return;

  const maxScore = Math.max(...tokens.map(t => Math.abs(t.score)), 1);
  tokens.forEach(function(tok, i) {
    const row = document.createElement('div');
    row.className = 'token-row';
    const pct = (Math.abs(tok.score) / maxScore * 100).toFixed(0);
    row.innerHTML =
      `<span class="token-rank">#${i+1}</span>` +
      `<span class="token-text">${escapeHtml(tok.text)}</span>` +
      `<span class="token-id">ID:${tok.token_id}</span>` +
      `<div class="token-score-bar"><div class="token-score-fill" style="width:${pct}%"></div></div>` +
      `<span class="token-score-value">${tok.score.toFixed(4)}</span>`;
    responseTokens.appendChild(row);
  });
}

function renderLayerList(names) {
  const layerList = el('layerList');
  if (!layerList) return;
  layerList.innerHTML = '';

  const grouped = {};
  names.forEach(function(name) {
    const match = name.match(/layers\.(\d+)\./);
    const key = match ? 'Layer ' + match[1] : 'Other';
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(name);
  });

  Object.keys(grouped).sort(function(a, b) {
    const na = parseInt(a.replace(/\D/g, '')) || 999;
    const nb = parseInt(b.replace(/\D/g, '')) || 999;
    return na - nb;
  }).forEach(function(group) {
    const header = document.createElement('div');
    header.className = 'layer-item';
    header.style.cssText = 'color:var(--accent-purple);font-weight:600;font-size:11px;margin-top:6px;cursor:default';
    header.textContent = '▸ ' + group;
    layerList.appendChild(header);

    grouped[group].forEach(function(name) {
      const item = document.createElement('div');
      item.className = 'layer-item';
      item.textContent = '  ' + name.split('.').slice(-2).join('.');
      item.title = name;
      item.addEventListener('click', function() {
        wsManager.selectLayer(name);
        store.selectedLayer = name;
        const cur = el('currentLayer');
        if (cur) cur.textContent = name;
      });
      layerList.appendChild(item);
    });
  });

  layerList.classList.remove('hidden');
}

// ============================================================================
// Frame Heatmap (D3 on canvas or #frame-canvas)
// ============================================================================

let _frameHeatmapCanvas = null;
let _frameHeatmapCtx = null;
let _frameImageData = null;
let _frameDirty = false;
let _frameRafId = null;
let _frameFpsCount = 0;
let _frameFpsLast = performance.now();

function initFrameHeatmap() {
  // Prefer #frame-canvas (new layout), fall back to #heatmapCanvas (legacy)
  _frameHeatmapCanvas = el('frame-canvas') || el('heatmapCanvas');
  if (!_frameHeatmapCanvas) return;

  _frameHeatmapCtx = _frameHeatmapCanvas.getContext('2d');
  _startFrameLoop();
}

function _startFrameLoop() {
  function loop() {
    _frameRafId = requestAnimationFrame(loop);
    if (_frameDirty && store.currentFrame) {
      _drawFrame(store.currentFrame);
      _frameDirty = false;
    }
  }
  loop();
}

function triggerFrameHeatmap() {
  _frameDirty = true;
}

function _drawFrame(frame) {
  if (!_frameHeatmapCtx || !frame || !frame.data) return;

  const size = frame.grid_size || Math.floor(Math.sqrt(frame.data.length));
  const canvas = _frameHeatmapCanvas;

  if (canvas.width !== size || canvas.height !== size) {
    canvas.width = size;
    canvas.height = size;
    _frameImageData = _frameHeatmapCtx.createImageData(size, size);
    const metricGrid = el('metricGrid');
    if (metricGrid) metricGrid.textContent = `${size}×${size}`;
  }

  if (!_frameImageData) {
    _frameImageData = _frameHeatmapCtx.createImageData(size, size);
  }

  const data = frame.data;
  const pixels = _frameImageData.data;
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < minVal) minVal = data[i];
    if (data[i] > maxVal) maxVal = data[i];
  }
  const range = maxVal - minVal || 1;

  for (let i = 0; i < data.length && i < size * size; i++) {
    const norm = (data[i] - minVal) / range;
    const [r, g, b] = thermalColorRGB(norm);
    const idx = i * 4;
    pixels[idx]     = r;
    pixels[idx + 1] = g;
    pixels[idx + 2] = b;
    pixels[idx + 3] = 255;
  }

  _frameHeatmapCtx.putImageData(_frameImageData, 0, 0);

  // FPS tracking
  _frameFpsCount++;
  const now = performance.now();
  if (now - _frameFpsLast >= 1000) {
    const fps = Math.round(_frameFpsCount * 1000 / (now - _frameFpsLast));
    const metricFps = el('metricFps');
    if (metricFps) metricFps.textContent = fps;
    _frameFpsCount = 0;
    _frameFpsLast = now;
  }
}

// ============================================================================
// Brain Map (Canvas 2D)
// ============================================================================

function renderBrainMap(snap) {
  const brainCanvas = el('brainCanvas');
  if (!brainCanvas || !snap) return;
  const bctx = brainCanvas.getContext('2d');
  const w = brainCanvas.width || brainCanvas.clientWidth || 600;
  const h = brainCanvas.height || brainCanvas.clientHeight || 400;

  bctx.clearRect(0, 0, w, h);

  // Update band bars
  if (snap.bands) {
    const barFocus   = el('barFocus');
    const barNovelty = el('barNovelty');
    const barDrive   = el('barDrive');
    if (barFocus)   barFocus.style.setProperty('--progress',   `${(snap.bands[0] ? snap.bands[0].energy * 100 : 0)}%`);
    if (barNovelty) barNovelty.style.setProperty('--progress', `${(snap.bands[1] ? snap.bands[1].energy * 100 : 0)}%`);
    if (barDrive)   barDrive.style.setProperty('--progress',   `${(snap.bands[2] ? snap.bands[2].energy * 100 : 0)}%`);
  }

  // Draw synapses
  if (snap.synapses && snap.nodes) {
    snap.synapses.forEach(function(syn) {
      const from = snap.nodes[syn.from];
      const to   = snap.nodes[syn.to];
      if (!from || !to) return;
      bctx.beginPath();
      bctx.moveTo(from.x * w, from.y * h);
      bctx.lineTo(to.x * w, to.y * h);
      const alpha = Math.min(1, syn.strength * 0.5);
      bctx.strokeStyle = syn.excitatory
        ? `rgba(0,255,136,${alpha})`
        : `rgba(255,68,102,${alpha})`;
      bctx.lineWidth = Math.max(0.5, syn.strength * 2);
      bctx.stroke();
    });
  }

  // Draw nodes
  if (snap.nodes) {
    snap.nodes.forEach(function(nd) {
      const nx = nd.x * w;
      const ny = nd.y * h;
      const size = 3 + (nd.amplitude || 0) * 4;
      const glow = 5 + Math.abs(nd.activation || 0) * 10;

      const colors = ['#00d4ff', '#00ff88', '#ff4466'];
      const color  = colors[nd.band] || '#a855f7';

      bctx.beginPath();
      bctx.arc(nx, ny, size, 0, Math.PI * 2);
      bctx.fillStyle = color;
      bctx.shadowBlur  = glow;
      bctx.shadowColor = color;
      bctx.fill();

      if (Math.abs(nd.activation || 0) > 0.8) {
        bctx.beginPath();
        bctx.arc(nx, ny, size + 2, 0, Math.PI * 2);
        bctx.strokeStyle = 'white';
        bctx.lineWidth = 1;
        bctx.stroke();
      }
    });
    bctx.shadowBlur = 0;
  }
}

// ============================================================================
// Logits Chart (ChartJS on #logits-canvas)
// ============================================================================

let _logitsChart = null;

function initLogitsChart() {
  const canvas = el('logits-canvas');
  if (!canvas || typeof Chart === 'undefined') return;

  _logitsChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Logit Score',
        data: [],
        backgroundColor: [],
        borderWidth: 0,
        barThickness: 14,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: '#6b7fa3' },
          grid:  { color: '#1a2035' },
        },
        y: {
          ticks: { color: '#6b7fa3', font: { family: 'JetBrains Mono, monospace', size: 11 } },
          grid:  { color: '#1a2035' },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              return ` Score: ${ctx.raw.toFixed(4)}`;
            },
          },
        },
      },
    },
  });
}

function updateLogitsChart() {
  if (!_logitsChart) return;
  const tokens = store.topTokens.slice(0, 40);
  if (!tokens.length) return;

  _logitsChart.data.labels  = tokens.map(t => t.text.slice(0, 12));
  _logitsChart.data.datasets[0].data = tokens.map(t => t.score);
  _logitsChart.data.datasets[0].backgroundColor = tokens.map((_, i) =>
    i === 0 ? '#00ffc8' : 'rgba(168,85,247,0.6)'
  );
  _logitsChart.update();
}

// ============================================================================
// Layer Grid (Layers Panel — #layer-grid-container)
// ============================================================================

let _layerSortBy = 'energy'; // 'energy'|'time'|'index'

function renderLayerGrid() {
  const container = el('layer-grid-container');
  if (!container) return;

  const data = store.layerData;
  const layers = Array.from({ length: 36 }, (_, i) => ({
    idx: i,
    ...(data[i] || {
      attention_energy: 0,
      mlp_energy: 0,
      residual_norm: 0,
      max_activation: 0,
      hot_neurons: [],
      time_us: 0,
    }),
  }));

  const maxAttn = Math.max(...layers.map(l => l.attention_energy || 0), 1);
  const maxMlp  = Math.max(...layers.map(l => l.mlp_energy      || 0), 1);
  const maxTime = Math.max(...layers.map(l => l.time_us         || 0), 1);

  // Sort
  if (_layerSortBy === 'energy') {
    layers.sort((a, b) => ((b.attention_energy || 0) + (b.mlp_energy || 0)) - ((a.attention_energy || 0) + (a.mlp_energy || 0)));
  } else if (_layerSortBy === 'time') {
    layers.sort((a, b) => (b.time_us || 0) - (a.time_us || 0));
  } else {
    layers.sort((a, b) => a.idx - b.idx);
  }

  container.innerHTML = '';

  layers.forEach(function(layer) {
    const row = document.createElement('div');
    row.className = 'layer-row';
    row.style.cssText = `
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    `;

    const attnPct = ((layer.attention_energy || 0) / maxAttn * 100).toFixed(1);
    const mlpPct  = ((layer.mlp_energy       || 0) / maxMlp  * 100).toFixed(1);
    const timePct = ((layer.time_us           || 0) / maxTime * 100).toFixed(1);

    // 10 hot-neuron dots (indices 0-9 representative)
    let dotHTML = '';
    for (let d = 0; d < 10; d++) {
      const isHot = (layer.hot_neurons || []).includes(d);
      dotHTML += `<span style="
        display:inline-block;width:6px;height:6px;border-radius:50%;
        background:${isHot ? '#ff7700' : '#1a1a2e'};
        border:1px solid ${isHot ? '#ff7700' : '#333'};
        opacity:${isHot ? 1 : 0.4};
      "></span>`;
    }

    row.innerHTML = `
      <span style="color:#6b7fa3;width:28px;flex-shrink:0">L${layer.idx}</span>
      <div style="flex:1;display:flex;flex-direction:column;gap:2px">
        <div style="display:flex;align-items:center;gap:4px">
          <span style="color:#6b7fa3;width:32px;font-size:10px">Attn</span>
          <div style="flex:1;height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden">
            <div style="width:${attnPct}%;height:100%;background:var(--accent-cyan,#00d4ff);border-radius:3px"></div>
          </div>
          <span style="color:#6b7fa3;width:36px;text-align:right">${(layer.attention_energy||0).toFixed(2)}</span>
        </div>
        <div style="display:flex;align-items:center;gap:4px">
          <span style="color:#6b7fa3;width:32px;font-size:10px">MLP</span>
          <div style="flex:1;height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden">
            <div style="width:${mlpPct}%;height:100%;background:#a855f7;border-radius:3px"></div>
          </div>
          <span style="color:#6b7fa3;width:36px;text-align:right">${(layer.mlp_energy||0).toFixed(2)}</span>
        </div>
      </div>
      <span style="
        background:rgba(0,212,255,0.1);color:#00d4ff;
        padding:1px 5px;border-radius:4px;width:36px;text-align:center;flex-shrink:0
      ">${(layer.residual_norm||0).toFixed(1)}</span>
      <div style="width:50px;height:8px;background:#1a1a2e;border-radius:4px;overflow:hidden;flex-shrink:0">
        <div style="width:${timePct}%;height:100%;background:#3b82f6;border-radius:4px"></div>
      </div>
      <div style="display:flex;gap:2px;flex-shrink:0">${dotHTML}</div>
    `;

    row.title = `Layer ${layer.idx}: attn=${(layer.attention_energy||0).toFixed(3)} mlp=${(layer.mlp_energy||0).toFixed(3)} time=${(layer.time_us||0)}µs`;
    container.appendChild(row);
  });
}

// ============================================================================
// Layer Heatmap (36 cells in #layer-heatmap-cells)
// ============================================================================

function renderLayerHeatmapCells() {
  const container = el('layer-heatmap-cells');
  if (!container) return;

  // Create cells if not yet populated
  if (container.children.length !== 36) {
    container.innerHTML = '';
    for (let i = 0; i < 36; i++) {
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.dataset.idx = i;
      cell.style.cssText = `
        width: 32px; height: 32px;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.06);
        cursor: pointer;
        transition: transform 0.1s;
        position: relative;
      `;
      cell.addEventListener('mouseover', function() {
        const d = store.layerData[i] || {};
        this.title = `L${i} | attn:${(d.attention_energy||0).toFixed(3)} mlp:${(d.mlp_energy||0).toFixed(3)}`;
      });
      container.appendChild(cell);
    }
  }

  const cells = container.querySelectorAll('.heatmap-cell');
  const maxE = Math.max(
    ...Array.from({ length: 36 }, (_, i) => (store.layerData[i] || {}).attention_energy || 0),
    1
  );

  cells.forEach(function(cell) {
    const idx = parseInt(cell.dataset.idx);
    const d = store.layerData[idx] || {};
    const energy = (d.attention_energy || 0) / maxE;
    cell.style.background = thermalColor(energy);
  });
}

// ============================================================================
// Force Graph (D3 on #force-graph-svg)
// ============================================================================

let _forceSimulation = null;
let _forceNodes = [];
let _forceLinks = [];
let _forceSvgG = null;
let _forceNodeElements = null;
let _forceLinkElements = null;
let _forceInitialized = false;

function initForceGraph() {
  const svgEl = el('force-graph-svg');
  if (!svgEl || typeof d3 === 'undefined' || _forceInitialized) return;
  _forceInitialized = true;

  const width  = svgEl.clientWidth  || 600;
  const height = svgEl.clientHeight || 400;

  const svg = d3.select('#force-graph-svg');
  svg.selectAll('*').remove();

  // Zoom + pan
  const g = svg.append('g').attr('class', 'force-root');
  _forceSvgG = g;

  svg.call(d3.zoom()
    .scaleExtent([0.3, 3])
    .on('zoom', function(event) {
      g.attr('transform', event.transform);
    })
  );

  // Defs: glow filter
  const defs = svg.append('defs');
  const filter = defs.append('filter').attr('id', 'glow');
  filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
  const feMerge = filter.append('feMerge');
  feMerge.append('feMergeNode').attr('in', 'coloredBlur');
  feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

  // Nodes: 36 transformer layers
  _forceNodes = Array.from({ length: 36 }, (_, i) => ({
    id: i,
    label: `L${i}`,
    radius: 10,
    energy: 0,
    hot: false,
  }));

  // Links: sequential residual stream
  _forceLinks = Array.from({ length: 35 }, (_, i) => ({
    source: i,
    target: i + 1,
    strength: 0.5,
  }));

  const colorScale = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);

  _forceSimulation = d3.forceSimulation(_forceNodes)
    .force('link', d3.forceLink(_forceLinks).id(d => d.id).distance(60))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collide', d3.forceCollide(d => d.radius + 6));

  // Draw links
  _forceLinkElements = g.append('g').attr('class', 'links')
    .selectAll('line')
    .data(_forceLinks)
    .enter()
    .append('line')
    .attr('stroke', '#1a2035')
    .attr('stroke-width', 1.5)
    .attr('stroke-opacity', 0.6);

  // Draw node groups
  const nodeG = g.append('g').attr('class', 'nodes')
    .selectAll('g')
    .data(_forceNodes)
    .enter()
    .append('g')
    .attr('class', 'node-group')
    .call(d3.drag()
      .on('start', function(event, d) {
        if (!event.active) _forceSimulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', function(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', function(event, d) {
        if (!event.active) _forceSimulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      })
    );

  // Hot glow ring (initially hidden)
  nodeG.append('circle')
    .attr('class', 'glow-ring')
    .attr('r', d => d.radius + 5)
    .attr('fill', 'none')
    .attr('stroke', '#ff7700')
    .attr('stroke-width', 2)
    .attr('opacity', 0);

  // Main circle
  nodeG.append('circle')
    .attr('class', 'node-circle')
    .attr('r', d => d.radius)
    .attr('fill', colorScale(0))
    .attr('filter', 'url(#glow)');

  // Label
  nodeG.append('text')
    .attr('class', 'node-label')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', 9)
    .attr('font-family', 'JetBrains Mono, monospace')
    .attr('fill', '#e8e8f0')
    .attr('pointer-events', 'none')
    .text(d => d.label);

  _forceNodeElements = nodeG;

  _forceSimulation.on('tick', function() {
    _forceLinkElements
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    _forceNodeElements
      .attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

function updateForceGraph(layerData) {
  if (!_forceInitialized || !_forceNodeElements) return;

  const colorScale = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);

  const maxAttn = Math.max(
    ...Object.values(layerData).map(l => l.attention_energy || 0),
    1
  );
  const maxAct = Math.max(
    ...Object.values(layerData).map(l => l.max_activation || 0),
    1
  );

  _forceNodes.forEach(function(node) {
    const d = layerData[node.id] || {};
    node.energy  = (d.attention_energy || 0) / maxAttn;
    node.hot     = (d.hot_neurons || []).length > 5;
    node.radius  = 8 + (d.max_activation || 0) / maxAct * 15;
  });

  _forceNodeElements.select('.node-circle')
    .attr('r', d => d.radius)
    .attr('fill', d => colorScale(d.energy));

  _forceNodeElements.select('.glow-ring')
    .attr('r', d => d.radius + 5)
    .attr('opacity', d => d.hot ? 0.8 : 0);

  _forceSimulation
    .force('collide', d3.forceCollide(d => d.radius + 6))
    .alpha(0.3)
    .restart();
}

// ============================================================================
// Trace Panel
// ============================================================================

let _energyChart = null;

function renderTracePanel() {
  const trace = store.trace;
  if (!trace) return;

  renderEnergyChart(trace);
  renderTokenTimeline(trace);
  renderAnomalyCards(trace);
  renderResonanceScore(trace);
}

function renderEnergyChart(trace) {
  const canvas = el('energy-chart-canvas');
  if (!canvas || typeof Chart === 'undefined') return;

  const energyData = trace.energy_progression || [];
  const anomalySteps = (trace.anomalies || []).map(a => a.step);

  if (_energyChart) {
    _energyChart.data.labels   = energyData.map((_, i) => `step ${i}`);
    _energyChart.data.datasets[0].data = energyData;
    _energyChart.update();
    return;
  }

  _energyChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: energyData.map((_, i) => `step ${i}`),
      datasets: [{
        label: 'Energy',
        data: energyData,
        borderColor: '#00ffc8',
        backgroundColor: 'rgba(0,255,200,0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: energyData.map((_, i) => anomalySteps.includes(i) ? 5 : 2),
        pointBackgroundColor: energyData.map((_, i) => anomalySteps.includes(i) ? '#ff4466' : '#00ffc8'),
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: '#6b7fa3', maxTicksLimit: 20 },
          grid:  { color: '#1a2035' },
        },
        y: {
          ticks: { color: '#6b7fa3' },
          grid:  { color: '#1a2035' },
        },
      },
      plugins: {
        legend: { labels: { color: '#6b7fa3' } },
      },
    },
  });
}

function renderTokenTimeline(trace) {
  const tbody = el('token-timeline-body');
  if (!tbody) return;

  tbody.innerHTML = '';
  const timeline = trace.layer_timeline || [];
  const anomalySteps = new Set((trace.anomalies || []).map(a => a.step));

  timeline.forEach(function(step) {
    const isAnomaly = anomalySteps.has(step.step_num);
    const tr = document.createElement('tr');
    if (isAnomaly) tr.style.background = 'rgba(255,68,102,0.08)';

    const activePills = (step.active_layers || []).slice(0, 5).map(idx =>
      `<span style="
        background:rgba(168,85,247,0.2);color:#a855f7;
        padding:1px 6px;border-radius:10px;font-size:10px;margin:1px
      ">${idx}</span>`
    ).join('');

    tr.innerHTML = `
      <td style="padding:4px 8px;color:#6b7fa3">${step.step_num}${isAnomaly ? ' <span style="color:#ff4466">⚠</span>' : ''}</td>
      <td style="padding:4px 8px;color:#00ffc8;font-family:monospace">${escapeHtml(step.token || '')}</td>
      <td style="padding:4px 8px">${activePills}</td>
      <td style="padding:4px 8px;color:#ffb347">${(step.total_energy || 0).toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAnomalyCards(trace) {
  const list = el('anomaly-list');
  if (!list) return;

  list.innerHTML = '';
  (trace.anomalies || []).forEach(function(anomaly) {
    const card = document.createElement('div');
    card.style.cssText = `
      background: rgba(255,68,102,0.06);
      border: 1px solid rgba(255,68,102,0.2);
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 8px;
    `;

    const kindColor = anomaly.kind === 'SPIKE' ? '#ff7700' : '#ff4466';
    const sevWidth  = Math.min(100, (anomaly.severity || 0) * 100).toFixed(0);

    card.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="
          background:${kindColor};color:#000;
          font-size:10px;font-weight:700;
          padding:2px 7px;border-radius:4px;
          font-family:monospace;letter-spacing:0.05em
        ">${escapeHtml(anomaly.kind || 'UNKNOWN')}</span>
        <span style="color:#6b7fa3;font-size:11px">step ${anomaly.step}</span>
      </div>
      <div style="height:4px;background:#1a1a2e;border-radius:2px;margin-bottom:8px">
        <div style="width:${sevWidth}%;height:100%;background:${kindColor};border-radius:2px"></div>
      </div>
      <p style="color:#e8e8f0;font-size:12px">${escapeHtml(anomaly.description || '')}</p>
    `;
    list.appendChild(card);
  });
}

function renderResonanceScore(trace) {
  const bar = el('resonance-score');
  if (!bar) return;
  const score = trace.iara_resonance_score || 0;
  bar.style.width = `${(score * 100).toFixed(1)}%`;
  bar.textContent = `${(score * 100).toFixed(1)}%`;
}

// ============================================================================
// Console Panel
// ============================================================================

let _consoleAutoScroll = true;
let _selectedLogId = null;

function renderConsoleLogs() {
  const container = el('console-log-list');
  if (!container) return;

  const filter = store.logFilter;
  const entries = store.logs.filter(function(e) {
    if (filter === 'ALL') return true;
    if (filter === 'WS')   return e.type === 'ws-in' || e.type === 'ws-out';
    if (filter === 'HTTP') return e.type === 'http';
    if (filter === 'ERROR') return e.type === 'error';
    return true;
  });

  container.innerHTML = '';
  entries.forEach(function(entry) {
    const div = document.createElement('div');
    div.style.cssText = `
      display:flex;align-items:flex-start;gap:8px;
      padding:4px 8px;border-bottom:1px solid rgba(255,255,255,0.03);
      cursor:pointer;font-family:monospace;font-size:11px;
    `;
    div.addEventListener('click', function() {
      _selectedLogId = entry.id;
      renderJsonInspector(entry.raw);
    });
    div.addEventListener('mouseover', function() {
      this.style.background = 'rgba(255,255,255,0.03)';
    });
    div.addEventListener('mouseout', function() {
      this.style.background = '';
    });

    const typeColors = {
      'ws-in':  '#00d4ff',
      'ws-out': '#00ff88',
      'http':   '#a855f7',
      'error':  '#ff4466',
    };
    const color = typeColors[entry.type] || '#6b7fa3';

    const rawStr = typeof entry.raw === 'string'
      ? entry.raw
      : JSON.stringify(entry.raw);
    const truncated = rawStr.length > 80 ? rawStr.slice(0, 80) + '…' : rawStr;

    div.innerHTML = `
      <span style="color:#555568;flex-shrink:0">${entry.ts}</span>
      <span style="color:${color};flex-shrink:0;width:52px">${entry.type}</span>
      <span style="color:#8888a0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(truncated)}</span>
    `;

    container.appendChild(div);
  });

  if (_consoleAutoScroll) {
    container.scrollTop = container.scrollHeight;
  }
}

function renderJsonInspector(raw) {
  const inspector = el('json-inspector');
  if (!inspector) return;

  const str = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
  inspector.innerHTML = `<pre style="margin:0;white-space:pre-wrap;word-break:break-all">${syntaxHighlight(str)}</pre>`;
}

// ============================================================================
// Vue Application
// ============================================================================

function mountVueApp() {
  if (typeof Vue === 'undefined') {
    // Vue not loaded — fallback to pure DOM mode
    addLog('http', 'vue-missing', 'Vue not loaded — running in DOM-only mode');
    return;
  }

  const { createApp, reactive, ref, computed, onMounted, watch, nextTick } = Vue;

  const app = createApp({
    setup() {
      const s = window.store;

      // Tab switching
      function switchTab(tab) {
        s.activeTab = tab;
        nextTick(function() {
          if (tab === 'graph' && !_forceInitialized) {
            initForceGraph();
          }
          if (tab === 'layers') {
            renderLayerGrid();
            renderLayerHeatmapCells();
          }
          if (tab === 'trace' && s.trace) {
            renderTracePanel();
          }
          if (tab === 'console') {
            renderConsoleLogs();
          }
        });
      }

      // Generate via HTTP
      async function generate() {
        if (s.generating || !s.prompt.trim()) return;
        await api.generate(s.prompt, s.maxTokens, s.temperature);
      }

      // Resonate via WS
      function resonate() {
        if (!s.prompt.trim()) return;
        wsManager.resonate(s.prompt);
        s.pipelineComplete = false;
        s.layerData = {};
      }

      // Send stimulus via WS
      function sendStimulus() {
        if (!s.prompt.trim()) return;
        wsManager.stimulus(s.prompt);
      }

      // Reset
      function reset() {
        wsManager.reset();
        s.layerData = {};
        s.topTokens = [];
        s.pipelineComplete = false;
        s.responseText = '';
        s.trace = null;
      }

      // List layers
      function listLayers() {
        wsManager.listLayers();
      }

      // Select layer
      function selectLayer(name) {
        wsManager.selectLayer(name);
        s.selectedLayer = name;
      }

      // Fetch trace
      function fetchTrace() {
        api.fetchTrace();
      }

      // Console filter
      function setLogFilter(f) {
        s.logFilter = f;
        renderConsoleLogs();
      }

      // Layer sort
      function setLayerSort(by) {
        _layerSortBy = by;
        renderLayerGrid();
      }

      // Watch layerData changes
      watch(
        () => Object.keys(s.layerData).length,
        function() {
          if (s.activeTab === 'layers') {
            renderLayerGrid();
            renderLayerHeatmapCells();
          }
          if (s.activeTab === 'graph' && _forceInitialized) {
            updateForceGraph(s.layerData);
          }
        }
      );

      // Watch topTokens
      watch(
        () => s.topTokens.length,
        function() {
          updateLogitsChart();
        }
      );

      // Watch logs
      watch(
        () => s.logs.length,
        function() {
          if (s.activeTab === 'console') {
            renderConsoleLogs();
          }
        }
      );

      // Watch trace
      watch(
        () => s.trace,
        function(t) {
          if (t && s.activeTab === 'trace') {
            renderTracePanel();
          }
        }
      );

      // Computed
      const wsStatusClass = computed(function() {
        return {
          connected:    s.wsStatus === 'connected',
          disconnected: s.wsStatus === 'disconnected',
          connecting:   s.wsStatus === 'connecting',
        };
      });

      const filteredLogs = computed(function() {
        const f = s.logFilter;
        if (f === 'ALL') return s.logs;
        return s.logs.filter(function(e) {
          if (f === 'WS')    return e.type === 'ws-in' || e.type === 'ws-out';
          if (f === 'HTTP')  return e.type === 'http';
          if (f === 'ERROR') return e.type === 'error';
          return true;
        });
      });

      onMounted(function() {
        // Resize canvases for proper display
        nextTick(function() {
          initFrameHeatmap();
          initLogitsChart();
        });
      });

      return {
        store: s,
        wsStatusClass,
        filteredLogs,
        switchTab,
        generate,
        resonate,
        sendStimulus,
        reset,
        listLayers,
        selectLayer,
        fetchTrace,
        setLogFilter,
        setLayerSort,
      };
    },
  });

  // Only mount if #app exists (new layout)
  const appEl = el('app');
  if (appEl) {
    app.mount('#app');
  }
}

// ============================================================================
// Legacy DOM event bindings (for existing index.html without Vue mount)
// ============================================================================

function bindLegacyEvents() {
  const btnSend      = el('btnSend');
  const btnResonate  = el('btnResonate');
  const btnReset     = el('btnReset');
  const btnListLayers = el('btnListLayers');
  const promptInput  = el('promptInput');
  const layerList    = el('layerList');

  if (promptInput) {
    promptInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        if (e.shiftKey) {
          wsManager.resonate(promptInput.value.trim());
        } else {
          wsManager.stimulus(promptInput.value.trim());
        }
      }
    });
  }

  if (btnSend) {
    btnSend.addEventListener('click', function() {
      const text = promptInput ? promptInput.value.trim() : '';
      if (text) wsManager.stimulus(text);
    });
  }

  if (btnResonate) {
    btnResonate.addEventListener('click', function() {
      const text = promptInput ? promptInput.value.trim() : '';
      if (!text) return;
      wsManager.resonate(text);
      store.layerData = {};
      store.pipelineComplete = false;
      const pipelinePanel = el('pipelinePanel');
      if (pipelinePanel) pipelinePanel.style.display = '';
      const pipelinePhase = el('pipelinePhase');
      if (pipelinePhase) pipelinePhase.textContent = 'Starting...';
      const pipelineProgress = el('pipelineProgress');
      if (pipelineProgress) pipelineProgress.style.width = '0%';
      const responsePanel = el('responsePanel');
      if (responsePanel) responsePanel.style.display = 'none';
    });
  }

  if (btnReset) {
    btnReset.addEventListener('click', function() {
      wsManager.reset();
      store.layerData = {};
      store.topTokens = [];
    });
  }

  if (btnListLayers) {
    btnListLayers.addEventListener('click', function() {
      if (layerList && !layerList.classList.contains('hidden')) {
        layerList.classList.add('hidden');
      } else {
        wsManager.listLayers();
      }
    });
  }
}

// ============================================================================
// Performance Metrics periodic update
// ============================================================================

function updatePerfMetrics() {
  const tpsEl  = el('metric-tokens-per-sec');
  const mplEl  = el('metric-ms-per-layer');
  const resonEl = el('metric-resonance-score');

  if (tpsEl)   tpsEl.textContent  = store.tokensPerSec.toFixed(1);
  if (mplEl)   mplEl.textContent  = store.msPerLayer.toFixed(1);
  if (resonEl && store.trace) {
    resonEl.textContent = (store.trace.iara_resonance_score * 100).toFixed(1) + '%';
  }

  // Also update anomaly count badge
  const anomEl = el('anomaly-count');
  if (anomEl) anomEl.textContent = store.anomalyCount;
}

// ============================================================================
// Full Initialization
// ============================================================================

// ============================================================================
// New HTML Bridge — wires new-layout IDs to store/wsManager
// ============================================================================

function updateModelLoadingBanner() {
  const banner = el('model-loading-banner');
  if (!banner) return;
  if (store.modelReady) {
    banner.style.display = 'none';
  } else {
    banner.style.display = 'flex';
    const msgEl = banner.querySelector('#loading-msg');
    if (msgEl) msgEl.textContent = store.loadingMsg;
  }
}

function switchActiveTab(tab) {
  // Update active button
  document.querySelectorAll('#tab-nav [data-tab]').forEach(function(b) {
    b.classList.remove('tab-btn--active');
    b.setAttribute('aria-selected', 'false');
  });
  const activeBtn = document.querySelector('#tab-nav [data-tab="' + tab + '"]');
  if (activeBtn) {
    activeBtn.classList.add('tab-btn--active');
    activeBtn.setAttribute('aria-selected', 'true');
  }
  // Show/hide panels
  document.querySelectorAll('.tab-panel').forEach(function(p) {
    p.classList.add('hidden');
  });
  const panel = el('panel-' + tab);
  if (panel) panel.classList.remove('hidden');
  // Side effects
  store.activeTab = tab;
  if (tab === 'graph' && !_forceInitialized) initForceGraph();
  if (tab === 'layers') { renderLayerGrid(); renderLayerHeatmapCells(); }
  if (tab === 'trace' && store.trace) renderTracePanel();
  if (tab === 'console') renderConsoleLogs();
}

// Document-level delegation — survives Vue DOM patching
document.addEventListener('click', function(e) {
  const btn = e.target.closest('#tab-nav [data-tab]');
  if (btn) switchActiveTab(btn.dataset.tab);
});

function bindNewLayoutEvents() {

  // ── Prompt input ────────────────────────────────────────────────────────
  const promptInput = el('prompt-input');
  if (promptInput) {
    promptInput.addEventListener('input', function() { store.prompt = promptInput.value; });
    promptInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        wsManager.resonate(store.prompt.trim());
        store.layerData = {};
      }
    });
  }

  // ── Send / Resonate buttons ─────────────────────────────────────────────
  const sendBtn = el('send-btn');
  if (sendBtn) {
    sendBtn.addEventListener('click', function() {
      const text = (promptInput ? promptInput.value : store.prompt).trim();
      if (!text) return;
      store.prompt = text;
      store.layerData = {};
      store.pipelineComplete = false;
      wsManager.resonate(text);
    });
  }

  // ── Status indicators (updated by updateNewLayoutStatus) ────────────────
  // Called periodically and on WS status change
}

function updateNewLayoutStatus() {
  // WS status badge
  const wsEl = el('ws-status');
  if (wsEl) {
    wsEl.className = 'ws-status ws-status--' + store.wsStatus;
    wsEl.setAttribute('aria-label', 'WebSocket: ' + store.wsStatus);
  }

  // HTTP status
  const httpEl = el('http-status');
  if (httpEl) {
    httpEl.className = 'http-status http-status--' + (store.httpHealthy ? 'ok' : 'err');
  }

  // Tokens/sec and ms/layer in status bar
  const tpsEl = el('tokens-per-sec');
  if (tpsEl) tpsEl.textContent = store.tokensPerSec > 0 ? store.tokensPerSec.toFixed(1) : '—';

  const mplEl = el('ms-per-layer');
  if (mplEl) mplEl.textContent = store.msPerLayer > 0 ? store.msPerLayer.toFixed(1) + 'ms' : '—';

  // Anomaly count
  const anomEl = el('anomaly-count');
  if (anomEl) anomEl.textContent = store.anomalyCount || '0';

  // Response text (from HTTP generate)
  const respEl = el('response-text');
  if (respEl && store.responseText) {
    respEl.textContent = store.responseText;
  }

  // Layer heatmap cells (36-cell row in Live tab)
  const heatmapEl = el('layer-heatmap');
  if (heatmapEl && Object.keys(store.layerData).length > 0) {
    renderLayerHeatmapInLive(heatmapEl);
  }
}

function renderLayerHeatmapInLive(container) {
  const cells = container.querySelectorAll('.heatmap-cell');
  if (cells.length === 0) {
    // Build cells if not present
    container.innerHTML = '';
    for (let i = 0; i < 36; i++) {
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.dataset.layer = i;
      cell.title = 'Layer ' + i;
      container.appendChild(cell);
    }
  }
  const maxAttn = Math.max(1, ...Object.values(store.layerData).map(function(d) { return d.attention_energy || 0; }));
  container.querySelectorAll('.heatmap-cell').forEach(function(cell) {
    const li = parseInt(cell.dataset.layer);
    const d = store.layerData[li];
    if (d) {
      const v = (d.attention_energy || 0) / maxAttn;
      cell.style.background = thermalColor(v);
      cell.title = 'L' + li + ' attn=' + (d.attention_energy || 0).toFixed(2) + ' mlp=' + (d.mlp_energy || 0).toFixed(2);
    }
  });
}

document.addEventListener('DOMContentLoaded', function() {
  addLog('http', 'init', 'IARA Observability Dashboard initializing...');

  // Start WebSocket
  wsManager.connect();

  // Health check — poll every 2s while model is loading, then every 10s
  api.health();
  setInterval(function() {
    api.health();
  }, 2000);
  // Slow down to 10s once model is ready (avoid spam)
  // (2s timer keeps running but health() is fast and cheap)

  // Perf metrics refresh
  setInterval(updatePerfMetrics, 2000);
  setInterval(updateNewLayoutStatus, 1000);

  // Bind legacy DOM events (for existing index.html layout)
  bindLegacyEvents();

  // Mount Vue app FIRST — Vue may patch DOM, so bind listeners after
  mountVueApp();

  // Bind new layout events (after Vue mounts, so DOM is stable)
  bindNewLayoutEvents();

  // Init loading banner
  updateModelLoadingBanner();

  // Initialize visualizations after mount
  if (typeof Vue !== 'undefined') {
    Vue.nextTick(function() {
      initFrameHeatmap();
      initLogitsChart();
    });
  } else {
    requestAnimationFrame(function() {
      initFrameHeatmap();
      initLogitsChart();
    });
  }

  addLog('http', 'init', 'Dashboard ready');
});
