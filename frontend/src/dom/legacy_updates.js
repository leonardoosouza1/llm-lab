// frontend/src/dom/legacy_updates.js

import { el, escapeHtml, createLogEntry } from '../utils.js';
import { store } from '../store/index.js';
import { renderLayerGrid, renderLayerHeatmapCells } from '../components/layer_grid.js'; // Will be created later

let _logId = 0; // Reset for modularization purposes. Actual logs will be stored in store.logs.

/**
 * Log helper — adds to store.logs AND legacy DOM log container
 * This function is designed to be passed to modules like wsManager and api
 * so they don't directly depend on the store.logs array.
 * @param {string} typeOrLevel
 * @param {any} raw
 * @param {string} [text]
 */
export function addLog(typeOrLevel, raw, text) {
  const entry = createLogEntry(typeOrLevel, raw, text);
  store.logs.push(entry);
  if (store.logs.length > 500) {
    store.logs.splice(0, store.logs.length - 500);
  }

  // Also update legacy DOM log container if present
  const logContainer = el('logContainer');
  if (logContainer) {
    const div = document.createElement('div');
    div.className = 'log-entry ' + (entry.type === 'error' ? 'error' : entry.type === 'ws-in' ? 'info' : entry.type === 'ws-out' ? 'success' : 'info');
    div.textContent = `[${entry.ts}] ${entry.text}`;
    logContainer.appendChild(div);
    logContainer.scrollTop = logContainer.scrollHeight;
    while (logContainer.children.length > 100) {
      logContainer.removeChild(logContainer.firstChild);
    }
  }
}

/**
 * Updates legacy UI elements based on a frame message.
 * @param {object} msg - The frame message from WebSocket.
 */
export function updateLegacyFrameUI(msg) {
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

/**
 * Updates pipeline-related UI elements.
 * @param {object} msg - The pipeline message.
 */
export function updatePipelineUI(msg) {
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
    renderLayerGrid(); // renderLayerGrid needs to be exported from components/layer_grid.js
  }
}

/**
 * Updates pipeline result UI elements.
 * @param {object} msg - The pipeline result message.
 */
export function updatePipelineResultUI(msg) {
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

/**
 * Updates generate result UI elements.
 * @param {object} data - The generate API response data.
 */
export function updateGenerateResultUI(data) {
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

/**
 * Renders response tokens in the UI.
 * @param {Array<object>} tokens - Array of token objects.
 */
export function renderResponseTokens(tokens) {
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

/**
 * Renders the list of available layers.
 * @param {string[]} names - Array of layer names.
 */
export function renderLayerList(names) {
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
        // wsManager will be imported by main.js and then passed to app.js
        // so this will need to be changed to emit an event or call a prop
        // for now, we'll keep it as a direct call, assuming wsManager is global or imported by main
        if (window.wsManager) {
          window.wsManager.selectLayer(name);
          store.selectedLayer = name;
          const cur = el('currentLayer');
          if (cur) cur.textContent = name;
        }
      });
      layerList.appendChild(item);
    });
  });

  layerList.classList.remove('hidden');
}

/**
 * Updates performance metrics in the UI.
 */
export function updatePerfMetrics() {
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

/**
 * Binds new layout events (prompt input, send/resonate, model loader, image upload).
 * This will now be called from main.js after Vue app is mounted.
 * @param {object} wsManager - The WebSocket manager instance.
 * @param {object} api - The HTTP API manager instance.
 */
export function bindNewLayoutEvents(wsManager, api) {
  // ── Prompt input ────────────────────────────────────────────────────────
  const promptInput = el('prompt-input');
  if (promptInput) {
    promptInput.addEventListener('input', function() { store.prompt = promptInput.value; });
    promptInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = store.prompt.trim();
        if (!text) return;
        store.generating = true;
        store.responseText = '';
        store.pipelineComplete = false;
        store.layerData = {};
        store.esnHistory = [];
        store.gravityHistory = [];
        store.topTokens = [];
        wsManager.resonate(text);
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
      store.generating = true;
      store.responseText = '';
      store.pipelineComplete = false;
      store.layerData = {};
      store.esnHistory = [];
      store.gravityHistory = [];
      store.topTokens = [];
      wsManager.resonate(text);
    });
  }

  // ── Image upload (VLM) ────────────────────────────────────────────────────
  const imageFileInput = el('image-file-input');
  const imageClearBtn  = el('image-clear-btn');
  const imagePreview   = el('image-preview');
  const imagePreviewContainer = el('image-preview-container');

  if (imageFileInput) {
    imageFileInput.addEventListener('change', function() {
      const file = imageFileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(e) {
        const dataUrl = e.target.result;
        store.selectedImageBase64 = dataUrl.split(',')[1] || null;
        if (imagePreview) {
          imagePreview.src = dataUrl;
        }
        if (imagePreviewContainer) imagePreviewContainer.style.display = 'block';
        if (imageClearBtn) imageClearBtn.style.display = 'inline-flex';
        addLog('http', null, `Imagem selecionada: ${file.name} (${(file.size/1024).toFixed(1)}KB)`);
      };
      reader.readAsDataURL(file);
    });
  }

  if (imageClearBtn) {
    imageClearBtn.addEventListener('click', function() {
      store.selectedImageBase64 = null;
      if (imageFileInput) imageFileInput.value = '';
      if (imagePreview) imagePreview.src = '';
      if (imagePreviewContainer) imagePreviewContainer.style.display = 'none';
      imageClearBtn.style.display = 'none';
    });
  }

  // Model select and load logic (will be handled by model_loader.js)
  // These event listeners will be removed from here and placed in model_loader.js
}

/**
 * Binds legacy events from the original app.js.
 * This will now be called from main.js after Vue app is mounted.
 * @param {object} wsManager - The WebSocket manager instance.
 */
export function bindLegacyEvents(wsManager) {
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

/**
 * Switches the active tab in the UI.
 * This function will be called from the Vue app.
 * @param {string} tab - The name of the tab to switch to.
 * @param {object} initFunctions - Object containing initialization functions for charts/graphs.
 */
export function switchActiveTab(tab, initFunctions) {
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

  // Initialize specific components when their tab is activated
  if (tab === 'graph' && !initFunctions.forceGraphInitialized()) {
    initFunctions.initForceGraph();
  }
  if (tab === 'layers') {
    renderLayerGrid();
    initFunctions.renderLayerHeatmapCells();
  }
  if (tab === 'trace' && store.trace) {
    initFunctions.renderTracePanel(); // From trace_panel.js, to be created
  }
  if (tab === 'console') {
    initFunctions.renderConsoleLogs(); // From log_console.js, to be created
  }
}
