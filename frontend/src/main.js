// frontend/src/main.js — IARA Dashboard v3
// Arquitetura: funções de render no nível de módulo (sem closures escondidas),
// persistência via sessionStorage, watchers reativos Vue3.

import { store } from './store/index.js';
import { api } from './api/index.js';
import { wsManager } from './api/websocket.js';
import { Charts } from './utils/charts.js';
import { thermalColor, escapeHtml } from './utils.js';
import {
  populateModelSelect, updateModelUI, updateModelLoadingBanner,
  bindModelEvents, updateVramBar,
} from './components/model_loader.js';
import { bindNewLayoutEvents } from './dom/legacy_updates.js';
import { renderLayerGrid } from './components/layer_grid.js';
import { renderConsoleLogs, setConsoleAutoScroll } from './components/log_console.js';
import { initForceGraph, updateForceGraph } from './charts/force_graph.js';

const { watch, createApp, nextTick } = Vue;

// ═══════════════════════════════════════════════════════════════════════════
// PERSISTÊNCIA DE SESSÃO (sobrevive a F5)
// ═══════════════════════════════════════════════════════════════════════════

const SESSION_KEY = 'iara_session_v1';

function saveSession() {
  try {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({
      modelName:    store.modelName,
      modelReady:   store.modelReady,
      responseText: store.responseText,
      evalCount:    store.evalCount,
      totalDurationMs: store.totalDurationMs,
      tokensPerSec: store.tokensPerSec,
      backend:      store.backend,
      quantization: store.quantization,
      numLayers:    store.numLayers,
      traceTokens:  store.traceTokens.slice(-100), // últimos 100 tokens
      layerData:    store.layerData,
      activeTab:    store.activeTab,
    }));
  } catch (e) { /* sessionStorage cheia ou bloqueada */ }
}

function restoreSession() {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (s.modelName)    store.modelName    = s.modelName;
    if (s.modelReady)   store.modelReady   = s.modelReady;
    if (s.responseText) store.responseText = s.responseText;
    if (s.evalCount)    store.evalCount    = s.evalCount;
    if (s.totalDurationMs) store.totalDurationMs = s.totalDurationMs;
    if (s.tokensPerSec) store.tokensPerSec = s.tokensPerSec;
    if (s.backend)      store.backend      = s.backend;
    if (s.quantization) store.quantization = s.quantization;
    if (s.numLayers)    store.numLayers    = s.numLayers;
    if (s.traceTokens)  store.traceTokens  = s.traceTokens;
    if (s.layerData)    store.layerData    = s.layerData;
    if (s.activeTab)    store.activeTab    = s.activeTab;
  } catch (e) { /* dados corrompidos */ }
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS DOM
// ═══════════════════════════════════════════════════════════════════════════

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ═══════════════════════════════════════════════════════════════════════════
// RENDER — TRACE TAB
// Lê store.traceTokens (populado pelo websocket.js a cada pipeline_result)
// ═══════════════════════════════════════════════════════════════════════════

let _traceEnergyChart = null;

export function renderTraceFromTokens() {
  const tokens = store.traceTokens;
  if (!tokens) return;

  // ── Token Timeline ───────────────────────────────────────────────────────
  const tbody = document.getElementById('token-timeline-body');
  if (tbody) {
    tbody.innerHTML = '';
    tokens.forEach(function(t) {
      const tr = document.createElement('tr');
      if (t.anomaly) tr.style.background = 'rgba(255,68,102,0.08)';
      const logStr  = t.top_logit != null ? t.top_logit.toFixed(2) : '—';
      const probStr = t.prob      != null
        ? (Math.min(1, Math.max(0, t.prob)) * 100).toFixed(1) + '%' : '—';
      const entStr  = t.entropy   != null ? t.entropy.toFixed(3)   : '—';
      const msStr   = t.ms_token  != null ? t.ms_token.toFixed(0)  : '—';
      const tok     = (t.token || '').replace(/<\|[^|]+\|>/g, ''); // strip special tokens
      tr.innerHTML = `
        <td style="padding:4px 8px;color:#6b7fa3;font-family:monospace">${t.step}${t.anomaly ? ' <span style="color:#ff4466">⚠</span>' : ''}</td>
        <td style="padding:4px 8px;color:#00ffc8;font-family:monospace;max-width:120px;overflow:hidden;text-overflow:ellipsis" title="${escapeHtml(tok)}">${escapeHtml(tok || '∅')}</td>
        <td style="padding:4px 8px;color:#6b7fa3;font-family:monospace">${t.token_id != null ? t.token_id : '—'}</td>
        <td style="padding:4px 8px;color:#a855f7;font-family:monospace">${logStr}</td>
        <td style="padding:4px 8px;color:#fbbf24;font-family:monospace">${probStr}</td>
        <td style="padding:4px 8px;color:#38bdf8;font-family:monospace">${msStr}</td>
        <td style="padding:4px 8px;color:#34d399;font-family:monospace">${entStr}</td>
        <td style="padding:4px 8px">${t.anomaly ? '<span style="color:#ff4466;font-weight:700">SPIKE</span>' : ''}</td>
      `;
      tbody.appendChild(tr);
    });
    // Scroll to bottom
    const wrapper = tbody.closest('.table-scroll-wrapper');
    if (wrapper) wrapper.scrollTop = wrapper.scrollHeight;
  }

  // ── Energy Chart ─────────────────────────────────────────────────────────
  const energyCanvas = document.getElementById('energy-chart-canvas');
  if (energyCanvas && tokens.length > 1 && typeof Chart !== 'undefined') {
    const labels     = tokens.map(t => `#${t.step}`);
    const energyData = tokens.map(t => +(t.energy || 0).toFixed(4));
    const anomalyPts = tokens.map(t => t.anomaly ? 5 : 2);
    const anomalyClr = tokens.map(t => t.anomaly ? '#ff4466' : '#00ffc8');

    if (_traceEnergyChart) {
      _traceEnergyChart.data.labels = labels;
      _traceEnergyChart.data.datasets[0].data = energyData;
      _traceEnergyChart.data.datasets[0].pointRadius = anomalyPts;
      _traceEnergyChart.data.datasets[0].pointBackgroundColor = anomalyClr;
      _traceEnergyChart.update('none');
    } else {
      _traceEnergyChart = new Chart(energyCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Avg Layer Energy',
            data: energyData,
            borderColor: '#00ffc8',
            backgroundColor: 'rgba(0,255,200,0.08)',
            fill: true,
            tension: 0.4,
            pointRadius: anomalyPts,
            pointBackgroundColor: anomalyClr,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: { ticks: { color: '#6b7fa3', maxTicksLimit: 20 }, grid: { color: '#1a2035' } },
            y: { ticks: { color: '#6b7fa3' }, grid: { color: '#1a2035' } },
          },
          plugins: { legend: { labels: { color: '#6b7fa3' } } },
        },
      });
    }
  }

  // ── Resonance Score (heurística: entropia baixa = alta coerência) ─────────
  if (tokens.length > 0) {
    const entsWithData = tokens.filter(t => t.entropy != null);
    const avgEnt = entsWithData.length > 0
      ? entsWithData.reduce((s, t) => s + t.entropy, 0) / entsWithData.length
      : 2.0;
    const score = Math.max(0, Math.min(1, 1 - avgEnt / 5));
    const bar = document.getElementById('resonance-score');
    const val = document.getElementById('resonance-value');
    if (bar) { bar.value = score * 100; }
    if (val)  { val.textContent = score.toFixed(2); }
  }

  // ── Anomaly List ──────────────────────────────────────────────────────────
  const anomalies = tokens.filter(t => t.anomaly);
  const anomalyCount = document.getElementById('anomaly-count-trace');
  if (anomalyCount) anomalyCount.textContent = anomalies.length;

  const anomalyList = document.getElementById('anomaly-list');
  if (anomalyList) {
    if (anomalies.length === 0) {
      anomalyList.innerHTML = '<div class="anomaly-empty" style="color:#6b7fa3;padding:16px;text-align:center">Nenhuma anomalia detectada</div>';
    } else {
      anomalyList.innerHTML = anomalies.map(a => {
        const tok = (a.token || '').replace(/<\|[^|]+\|>/g, '');
        return `<div style="background:rgba(255,68,102,0.06);border:1px solid rgba(255,68,102,0.2);border-radius:8px;padding:10px 14px;margin-bottom:8px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="background:#ff4466;color:#000;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px">SPIKE</span>
            <span style="color:#6b7fa3;font-size:11px">step ${a.step}</span>
          </div>
          <p style="color:#e8e8f0;font-size:12px;margin:0">Energy: <b>${(a.energy||0).toFixed(3)}</b> — token: <code style="color:#00ffc8">${escapeHtml(tok||'∅')}</code></p>
        </div>`;
      }).join('');
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// RENDER — STATUS HELPERS
// ═══════════════════════════════════════════════════════════════════════════

function updateWsDot(status) {
  const el = document.getElementById('ws-status');
  if (!el) return;
  const dot = el.querySelector('.ws-dot');
  if (dot) dot.style.background =
    status === 'connected'  ? '#00ffc8' :
    status === 'connecting' ? '#ffb347' : '#ff4466';
}

function updateHttpDot(healthy) {
  const el = document.getElementById('http-status');
  if (!el) return;
  const dot = el.querySelector('.http-dot');
  if (dot) dot.style.background = healthy ? '#00ffc8' : '#ff4466';
}

function updateHeatmap() {
  const cells = document.querySelectorAll('#layer-heatmap .heatmap-cell');
  if (!cells.length) return;
  const max = Math.max(...Object.values(store.layerData).map(d => d.attention_energy || 0), 1);
  cells.forEach(cell => {
    const idx = parseInt(cell.dataset.layer);
    const d = store.layerData[idx] || {};
    cell.style.background = thermalColor((d.attention_energy || 0) / max);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════════════

function switchTab(tab) {
  store.activeTab = tab;
  saveSession();

  document.querySelectorAll('#tab-nav [data-tab]').forEach(btn => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle('tab-btn--active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.add('hidden'));
  const panel = document.getElementById('panel-' + tab);
  if (panel) panel.classList.remove('hidden');

  // Renderização específica de cada aba ao ativar
  if (tab === 'layers') {
    renderLayerGrid();
  }
  if (tab === 'graph') {
    initForceGraph();
    updateForceGraph(store.layerData);
  }
  if (tab === 'trace') {
    renderTraceFromTokens();
  }
  if (tab === 'console') {
    renderConsoleLogs();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// HEALTH POLLING
// ═══════════════════════════════════════════════════════════════════════════

async function pollHealth() {
  const data = await api.health();
  if (data) updateVramBar(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// SPARKLINES (ESN)
// ═══════════════════════════════════════════════════════════════════════════

function drawSparkline(canvasId, points, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || points.length < 2) return;
  canvas.width = canvas.offsetWidth || 200;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const max = Math.max(...points, 0.01);
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  points.forEach((v, i) => {
    const x = (i / (points.length - 1)) * W;
    const y = H - (v / max) * (H - 4) - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════════════════
// APP
// ═══════════════════════════════════════════════════════════════════════════

createApp({
  mounted() {
    // Restaura sessão antes de qualquer outra coisa
    restoreSession();

    // Init WS + fetch model list
    api.init();
    api.fetchModels();

    // Charts
    nextTick(() => {
      Charts.initLogitsChart('logits-canvas');
    });

    // Bind DOM events
    bindNewLayoutEvents(wsManager, api);
    bindModelEvents(api);

    // Tab nav
    const tabNav = document.getElementById('tab-nav');
    if (tabNav) {
      tabNav.addEventListener('click', e => {
        const btn = e.target.closest('[data-tab]');
        if (btn) switchTab(btn.dataset.tab);
      });
    }

    // Activar a aba correta restaurada da sessão
    switchTab(store.activeTab || 'live');

    // Health polling
    pollHealth();
    setInterval(pollHealth, 5000);

    // Salva sessão periodicamente e antes de fechar
    setInterval(saveSession, 3000);
    window.addEventListener('beforeunload', saveSession);

    // ─────────────────────────────────────────────────────────────────────
    // WATCHERS REATIVOS
    // ─────────────────────────────────────────────────────────────────────

    watch(() => store.wsStatus, updateWsDot, { immediate: true });
    watch(() => store.httpHealthy, updateHttpDot, { immediate: true });

    watch(() => [store.modelLoading, store.modelReady, store.loadingMsg], () => {
      updateModelUI();
      updateModelLoadingBanner();
      const msgEl = document.getElementById('loading-msg');
      if (msgEl) msgEl.textContent = store.loadingMsg;
    }, { immediate: true });

    watch(() => store.availableModels, populateModelSelect, { deep: true });

    watch(() => store.modelName, name => {
      setText('active-model', name || '—');
      saveSession();
    }, { immediate: true });

    watch(() => store.backend, b => {
      setText('backend-url', b || '—');
    }, { immediate: true });

    // Resposta gerada
    watch(() => store.responseText, text => {
      const el = document.getElementById('response-text');
      if (!el) return;
      el.innerHTML = text
        ? `<span style="white-space:pre-wrap">${escapeHtml(text)}</span>`
        : '<span class="response-placeholder">Response will appear here…</span>';
    }, { immediate: true });

    // Logits chart
    watch(() => store.topTokens, tokens => {
      Charts.updateLogitsChart(tokens);
    }, { deep: true });

    // Heatmap + Layers tab
    watch(() => store.layerData, () => {
      updateHeatmap();
      if (store.activeTab === 'layers') renderLayerGrid();
      if (store.activeTab === 'graph')  updateForceGraph(store.layerData);
    }, { deep: true });

    // TPS e ms/layer
    watch(() => store.tokensPerSec, tps => {
      const v = tps > 0 ? tps.toFixed(1) : '—';
      setText('tokens-per-sec', v);
      setText('topbar-tps', v);
    });

    watch(() => store.msPerLayer, ms => {
      setText('ms-per-layer', ms > 0 ? ms.toFixed(1) : '—');
    });

    watch(() => store.anomalyCount, n => {
      setText('anomaly-count', n);
    });

    // Footer: latency and token count
    watch(() => store.totalDurationMs, ms => {
      setText('gen-latency', ms > 0 ? `${ms.toFixed(0)}ms` : '—');
    });
    watch(() => store.evalCount, n => {
      setText('gen-tokens', n > 0 ? String(n) : '—');
    });

    watch(() => [store.evalCount, store.totalDurationMs, store.tokensPerSec], () => {
      const el = document.getElementById('eval-stats');
      if (!el) return;
      el.textContent = store.evalCount > 0
        ? `${store.evalCount} tok · ${store.totalDurationMs.toFixed(0)}ms · ${store.tokensPerSec.toFixed(1)} t/s`
        : '—';
    }, { immediate: true });

    // ── ESN ───────────────────────────────────────────────────────────────
    let esnEntropyPoints = [];
    let esnTop1Points    = [];
    let esnStateNormPts  = [];

    watch(() => store.esnLatest, esn => {
      if (!esn) return;
      setText('esn-entropy',    esn.entropy    != null ? esn.entropy.toFixed(3)               : '—');
      setText('esn-top1',       esn.top1_prob  != null ? (esn.top1_prob * 100).toFixed(1)+'%' : '—');
      setText('esn-state-norm', esn.state_norm != null ? esn.state_norm.toFixed(3)            : '—');
      setText('esn-temp-delta', esn.temp_delta != null ? esn.temp_delta.toFixed(3)            : '—');
      setText('esn-step',       esn.step != null ? String(esn.step)                           : '—');

      const badge = document.getElementById('esn-trained-badge');
      if (badge) {
        badge.textContent = esn.readout_trained ? 'readout: treinado ✓' : 'readout: zero (observando)';
        badge.style.color = esn.readout_trained ? '#00ffc8' : '#6b7280';
        badge.style.background = esn.readout_trained ? 'rgba(0,255,200,.1)' : 'rgba(255,255,255,.06)';
      }

      esnEntropyPoints.push(esn.entropy  || 0); if (esnEntropyPoints.length > 200) esnEntropyPoints.shift();
      esnTop1Points.push(esn.top1_prob   || 0); if (esnTop1Points.length    > 200) esnTop1Points.shift();
      esnStateNormPts.push(esn.state_norm|| 0); if (esnStateNormPts.length  > 200) esnStateNormPts.shift();

      drawSparkline('esn-entropy-canvas',    esnEntropyPoints, '#a855f7');
      drawSparkline('esn-top1-canvas',       esnTop1Points,    '#00ffc8');
      drawSparkline('esn-state-norm-canvas', esnStateNormPts,  '#fbbf24');
    });

    // ── Gravity ───────────────────────────────────────────────────────────
    let gravThreshPts = [];
    let gravMeanPts   = [];

    watch(() => store.gravityLatest, grav => {
      if (!grav) return;
      const panel = document.getElementById('gravity-panel');
      if (panel) panel.style.display = '';

      setText('gravity-mean', grav.mean_gravity != null ? grav.mean_gravity.toFixed(3) : '—');
      setText('gravity-max',  grav.max_gravity  != null ? grav.max_gravity.toFixed(3)  : '—');
      const highCount = Math.max(0, Math.round((grav.history_len || 0) * (grav.max_gravity || 0)));
      setText('gravity-high-count', highCount > 0 ? String(highCount) : '—');

      const badge = document.getElementById('gravity-badge');
      if (badge) { badge.textContent = 'active'; badge.style.color = '#fbbf24'; badge.style.background = 'rgba(251,191,36,.1)'; }

      gravThreshPts.push(grav.threshold    || 0); if (gravThreshPts.length > 200) gravThreshPts.shift();
      gravMeanPts.push(grav.mean_gravity   || 0); if (gravMeanPts.length   > 200) gravMeanPts.shift();

      const canvas = document.getElementById('gravity-threshold-canvas');
      if (!canvas || gravThreshPts.length < 2) return;
      canvas.width = canvas.offsetWidth || 200;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const maxVal = Math.max(...gravThreshPts, ...gravMeanPts, 0.01);
      [[gravThreshPts, '#fbbf24'], [gravMeanPts, 'rgba(56,189,248,0.6)']].forEach(([pts, clr]) => {
        ctx.beginPath(); ctx.strokeStyle = clr; ctx.lineWidth = 1.5;
        pts.forEach((v, i) => {
          const x = (i / (pts.length - 1)) * W;
          const y = H - (v / maxVal) * (H - 4) - 2;
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
    });

    // ── CfC ───────────────────────────────────────────────────────────────
    let cfcCorrPts = [];

    watch(() => store.cfcLatest, cfc => {
      if (!cfc) return;
      const panel = document.getElementById('cfc-panel');
      if (panel) panel.style.display = '';

      setText('cfc-state-norm',      cfc.state_norm      != null ? cfc.state_norm.toFixed(3)      : '—');
      setText('cfc-correction-norm', cfc.correction_norm  != null ? cfc.correction_norm.toFixed(3) : '—');
      setText('cfc-params', cfc.param_count != null
        ? (cfc.param_count > 1e6 ? (cfc.param_count/1e6).toFixed(1)+'M' : (cfc.param_count/1e3).toFixed(1)+'K') : '—');
      setText('cfc-vram',  cfc.vram_mb != null ? cfc.vram_mb.toFixed(1)+' MB' : '—');

      const badge = document.getElementById('cfc-badge');
      if (badge) { badge.textContent = 'active'; badge.style.color = '#38bdf8'; badge.style.background = 'rgba(56,189,248,.1)'; }

      if (cfc.correction_norm != null) {
        cfcCorrPts.push(cfc.correction_norm);
        if (cfcCorrPts.length > 200) cfcCorrPts.shift();
        drawSparkline('cfc-correction-canvas', cfcCorrPts, '#818cf8');
      }
    });

    // ── Q4 ────────────────────────────────────────────────────────────────
    watch(() => store.q4Latest, q4 => {
      if (!q4) return;
      const panel = document.getElementById('q4-panel');
      if (panel) panel.style.display = '';
      setText('q4-vram', q4.vram_mb != null ? q4.vram_mb.toFixed(0)+' MB' : '—');
      const badge = document.getElementById('q4-badge');
      if (badge) { badge.textContent = 'Q4 active'; badge.style.color = '#00ffc8'; badge.style.background = 'rgba(0,255,200,.1)'; }
    });

    // ── Subsystem badges ──────────────────────────────────────────────────
    const subSys = [
      ['subsys-backend',  () => store.backend,      v => v === 'unknown' ? '#6b7280' : '#00ffc8',   v => v.toUpperCase()],
      ['subsys-quant',    () => store.quantization,  v => v === 'q4' ? '#f472b6' : '#00ffc8',       v => v.toUpperCase()],
      ['subsys-esn',      () => store.esnActive,     v => v ? '#a855f7' : '#6b7280',                v => v ? 'ON' : 'OFF'],
      ['subsys-gravity',  () => store.gravityActive, v => v ? '#fbbf24' : '#6b7280',                v => v ? 'ON' : 'OFF'],
      ['subsys-cfc',      () => store.cfcActive,     v => v ? '#38bdf8' : '#6b7280',                v => v ? 'ON' : 'OFF'],
    ];
    subSys.forEach(([id, getter, colorFn, textFn]) => {
      watch(getter, v => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = textFn(v);
        el.style.color = colorFn(v);
      }, { immediate: true });
    });

    // ── Console ───────────────────────────────────────────────────────────
    watch(() => store.logFilter, () => {
      if (store.activeTab === 'console') renderConsoleLogs();
    });

    let _consoleRafPending = false;
    watch(() => store.logs.length, () => {
      if (store.activeTab !== 'console') return;
      if (_consoleRafPending) return;
      _consoleRafPending = true;
      requestAnimationFrame(() => { renderConsoleLogs(); _consoleRafPending = false; });
    });

    // ── Trace — atualiza quando novos tokens chegam ───────────────────────
    let _traceRafPending = false;
    watch(() => store.traceTokens.length, () => {
      if (store.activeTab !== 'trace') return;
      if (_traceRafPending) return;
      _traceRafPending = true;
      requestAnimationFrame(() => { renderTraceFromTokens(); _traceRafPending = false; });
    });

    // ── Geração: reset + UI ───────────────────────────────────────────────
    watch(() => store.generating, gen => {
      const sendBtn  = document.getElementById('send-btn');
      const evalStats = document.getElementById('eval-stats');

      if (gen) {
        if (sendBtn) {
          sendBtn.disabled = true;
          sendBtn.innerHTML = '<span class="spinner-inline"></span> Gerando...';
        }
        if (evalStats) evalStats.textContent = 'Gerando...';

        // Reset telemetria para nova geração
        store.traceTokens  = [];
        store.layerData    = {};
        store.esnHistory   = [];
        store.esnLatest    = null;
        store.gravityLatest = null;
        store.gravityHistory = [];
        store.cfcLatest    = null;
        store.q4Latest     = null;
        if (_traceEnergyChart) {
          _traceEnergyChart.data.labels = [];
          _traceEnergyChart.data.datasets[0].data = [];
          _traceEnergyChart.update('none');
        }
        ['esn-entropy','esn-top1','esn-state-norm','esn-temp-delta','esn-step',
         'gravity-mean','gravity-max','gravity-high-count',
         'cfc-state-norm','cfc-correction-norm','cfc-params','cfc-vram','q4-vram'
        ].forEach(id => setText(id, '—'));

      } else {
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.innerHTML = '<span aria-hidden="true">▶</span> Generate';
        }
        if (evalStats && store.evalCount > 0) {
          const tps = store.tokensPerSec > 0 ? store.tokensPerSec.toFixed(1) : '?';
          evalStats.textContent = `${store.evalCount} tokens · ${store.totalDurationMs.toFixed(0)}ms · ${tps} t/s`;
        }
        saveSession();
      }
    });

    // ── Console buttons ───────────────────────────────────────────────────
    document.getElementById('console-clear-btn')?.addEventListener('click', () => {
      store.logs = [];
      renderConsoleLogs();
    });

    document.getElementById('console-pause-btn')?.addEventListener('click', function() {
      const paused = this.getAttribute('aria-pressed') === 'true';
      this.setAttribute('aria-pressed', String(!paused));
      this.classList.toggle('ctrl-btn--active', !paused);
      setConsoleAutoScroll(paused);
    });

    document.getElementById('inspector-copy-btn')?.addEventListener('click', () => {
      const pre = document.getElementById('json-inspector');
      if (pre && navigator.clipboard) navigator.clipboard.writeText(pre.textContent);
    });

    document.querySelectorAll('.console-filter-inner [data-filter]').forEach(btn =>
      btn.addEventListener('click', () => { store.logFilter = btn.dataset.filter; })
    );

    // ── Layers sort/filter ────────────────────────────────────────────────
    document.querySelectorAll('.ctrl-btn[data-sort]').forEach(btn =>
      btn.addEventListener('click', function() {
        document.querySelectorAll('.ctrl-btn[data-sort]').forEach(b => b.classList.remove('ctrl-btn--active'));
        this.classList.add('ctrl-btn--active');
        import('./components/layer_grid.js').then(m => {
          m.setLayerSort(this.dataset.sort);
        });
      })
    );

    // Graph controls now handled internally by force_graph.js bindControls()

    console.log('✅ IARA Dashboard v3 iniciado');
  },
}).mount('#app');
