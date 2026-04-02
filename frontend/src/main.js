// frontend/src/main.js
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
import { renderTracePanel } from './charts/energy_chart.js';
import { renderConsoleLogs, renderJsonInspector, setConsoleAutoScroll } from './components/log_console.js';
import { initForceGraph, updateForceGraph } from './charts/force_graph.js';

const { watch, createApp, nextTick } = Vue;

// ── Tab switching ─────────────────────────────────────────────────────────────

function switchTab(tab) {
  store.activeTab = tab;
  document.querySelectorAll('#tab-nav [data-tab]').forEach(btn => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle('tab-btn--active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.add('hidden'));
  const panel = document.getElementById('panel-' + tab);
  if (panel) panel.classList.remove('hidden');

  if (tab === 'layers') renderLayerGrid();
  if (tab === 'graph')  { initForceGraph(); updateForceGraph(store.layerData); }
  if (tab === 'trace')  renderTraceFromTokens();
  if (tab === 'console') renderConsoleLogs();
}

// ── Status helpers ────────────────────────────────────────────────────────────

function updateWsDot(status) {
  const el = document.getElementById('ws-status');
  if (!el) return;
  const dot = el.querySelector('.ws-dot');
  if (dot) dot.style.background =
    status === 'connected'   ? '#00ffc8' :
    status === 'connecting'  ? '#ffb347' : '#ff4466';
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
  const max = Math.max(
    ...Object.values(store.layerData).map(d => d.attention_energy || 0), 1
  );
  cells.forEach(cell => {
    const idx = parseInt(cell.dataset.layer);
    const d = store.layerData[idx] || {};
    cell.style.background = thermalColor((d.attention_energy || 0) / max);
  });
}

// ── Health polling ────────────────────────────────────────────────────────────

async function pollHealth() {
  const data = await api.health();
  if (data) updateVramBar(data);
}

// ── App ───────────────────────────────────────────────────────────────────────

createApp({
  mounted() {
    // Init WS + fetch model list
    api.init();
    api.fetchModels();

    // Init charts after DOM is ready
    nextTick(() => {
      Charts.initLogitsChart('logits-canvas');
    });

    // Bind DOM event handlers
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

    // Health polling
    pollHealth();
    setInterval(pollHealth, 5000);

    // ── Reactive watchers ─────────────────────────────────────────────────────

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
      const el = document.getElementById('active-model');
      if (el) el.textContent = name || '—';
    }, { immediate: true });

    watch(() => store.responseText, text => {
      const el = document.getElementById('response-text');
      if (!el) return;
      el.innerHTML = text
        ? `<span style="white-space:pre-wrap">${escapeHtml(text)}</span>`
        : '<span class="response-placeholder">Response will appear here…</span>';
    });

    watch(() => store.topTokens, tokens => {
      Charts.updateLogitsChart(tokens);
    }, { deep: true });

    watch(() => store.layerData, () => {
      updateHeatmap();
      if (store.activeTab === 'layers') renderLayerGrid();
    }, { deep: true });

    watch(() => store.tokensPerSec, tps => {
      const v = tps > 0 ? tps.toFixed(1) : '—';
      const el1 = document.getElementById('tokens-per-sec');
      const el2 = document.getElementById('topbar-tps');
      if (el1) el1.textContent = v;
      if (el2) el2.textContent = v;
    });

    watch(() => store.msPerLayer, ms => {
      const el = document.getElementById('ms-per-layer');
      if (el) el.textContent = ms > 0 ? ms.toFixed(1) : '—';
    });

    watch(() => store.anomalyCount, n => {
      const el = document.getElementById('anomaly-count');
      if (el) el.textContent = n;
    });

    watch(() => [store.evalCount, store.totalDurationMs, store.tokensPerSec], () => {
      const el = document.getElementById('eval-stats');
      if (!el) return;
      el.textContent = store.evalCount > 0
        ? `${store.evalCount} tok · ${store.totalDurationMs.toFixed(0)}ms · ${store.tokensPerSec.toFixed(1)} t/s`
        : '—';
    });

    watch(() => store.trace, trace => {
      if (trace && store.activeTab === 'trace') renderTracePanel();
    });

    // ── ESN Observer watcher ──────────────────────────────────────────────
    let esnEntropyPoints = [];
    let esnTop1Points = [];
    let esnStateNormPoints = [];

    function drawSparkline(canvasId, points, color) {
      const canvas = document.getElementById(canvasId);
      if (!canvas || points.length < 2) return;
      canvas.width = canvas.offsetWidth || 600;
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

    watch(() => store.esnLatest, esn => {
      if (!esn) return;

      // Update stat numbers
      const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setText('esn-entropy',     esn.entropy    != null ? esn.entropy.toFixed(3)     : '—');
      setText('esn-top1',        esn.top1_prob  != null ? (esn.top1_prob * 100).toFixed(1) + '%' : '—');
      setText('esn-state-norm',  esn.state_norm != null ? esn.state_norm.toFixed(3)  : '—');
      setText('esn-temp-delta',  esn.temp_delta != null ? esn.temp_delta.toFixed(3)  : '—');
      setText('esn-step',        esn.step != null ? String(esn.step) : '—');

      // Trained badge
      const badge = document.getElementById('esn-trained-badge');
      if (badge) {
        if (esn.readout_trained) {
          badge.textContent = 'readout: treinado ✓';
          badge.style.color = '#00ffc8';
          badge.style.background = 'rgba(0,255,200,.1)';
        } else {
          badge.textContent = 'readout: zero (observando)';
          badge.style.color = '#6b7280';
          badge.style.background = 'rgba(255,255,255,.06)';
        }
      }

      // Sparklines for entropy, top1_prob, state_norm
      esnEntropyPoints.push(esn.entropy || 0);
      if (esnEntropyPoints.length > 200) esnEntropyPoints.shift();

      esnTop1Points.push(esn.top1_prob || 0);
      if (esnTop1Points.length > 200) esnTop1Points.shift();

      esnStateNormPoints.push(esn.state_norm || 0);
      if (esnStateNormPoints.length > 200) esnStateNormPoints.shift();

      drawSparkline('esn-entropy-canvas', esnEntropyPoints, '#a855f7');
      drawSparkline('esn-top1-canvas', esnTop1Points, '#fbbf24');
      drawSparkline('esn-state-norm-canvas', esnStateNormPoints, '#fbbf24');
    });

    // ── Gravity watcher ────────────────────────────────────────────────
    let gravityThresholdPoints = [];
    let gravityMeanPoints = [];
    let gravityHighCountHistory = [];

    watch(() => store.gravityLatest, grav => {
      if (!grav) return;
      const panel = document.getElementById('gravity-panel');
      if (panel) panel.style.display = '';

      const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setText('gravity-mean',      grav.mean_gravity != null ? grav.mean_gravity.toFixed(3) : '—');
      setText('gravity-max',       grav.max_gravity != null ? grav.max_gravity.toFixed(3) : '—');

      // Count high-gravity tokens (e.g., gravity > threshold)
      let highCount = 0;
      if (grav.history_len != null && grav.threshold != null) {
        // Estimate based on semantic significance threshold
        highCount = Math.max(0, Math.round(grav.history_len * (grav.max_gravity || 0.5)));
      }
      setText('gravity-high-count', highCount > 0 ? String(highCount) : '—');

      const badge = document.getElementById('gravity-badge');
      if (badge) { badge.textContent = 'active'; badge.style.color = '#fbbf24'; badge.style.background = 'rgba(251,191,36,.1)'; }

      // Sparkline for threshold and mean_gravity (dual line)
      gravityThresholdPoints.push(grav.threshold || 0);
      if (gravityThresholdPoints.length > 200) gravityThresholdPoints.shift();

      gravityMeanPoints.push(grav.mean_gravity || 0);
      if (gravityMeanPoints.length > 200) gravityMeanPoints.shift();

      gravityHighCountHistory.push(highCount);
      if (gravityHighCountHistory.length > 200) gravityHighCountHistory.shift();

      const canvas = document.getElementById('gravity-threshold-canvas');
      if (!canvas || gravityThresholdPoints.length < 2) return;
      canvas.width = canvas.offsetWidth || 600;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const maxThreshold = Math.max(...gravityThresholdPoints, 0.01);
      const maxMean = Math.max(...gravityMeanPoints, 0.01);
      const maxVal = Math.max(maxThreshold, maxMean);

      // Draw threshold line (amber)
      ctx.beginPath();
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 1.5;
      gravityThresholdPoints.forEach((v, i) => {
        const x = (i / (gravityThresholdPoints.length - 1)) * W;
        const y = H - (v / maxVal) * (H - 4) - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Draw mean line (faded cyan)
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(56,189,248,0.6)';
      ctx.lineWidth = 1.5;
      gravityMeanPoints.forEach((v, i) => {
        const x = (i / (gravityMeanPoints.length - 1)) * W;
        const y = H - (v / maxVal) * (H - 4) - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // ── CfC watcher ──────────────────────────────────────────────────
    let cfcCorrectionNormPoints = [];

    watch(() => store.cfcLatest, cfc => {
      if (!cfc) return;
      const panel = document.getElementById('cfc-panel');
      if (panel) panel.style.display = '';

      const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setText('cfc-state-norm',     cfc.state_norm != null ? cfc.state_norm.toFixed(3) : '—');
      setText('cfc-correction-norm', cfc.correction_norm != null ? cfc.correction_norm.toFixed(3) : '—');
      setText('cfc-params', cfc.param_count != null ? (cfc.param_count > 1e6 ? (cfc.param_count / 1e6).toFixed(1) + 'M' : (cfc.param_count / 1e3).toFixed(1) + 'K') : '—');
      setText('cfc-vram',   cfc.vram_mb != null ? cfc.vram_mb.toFixed(1) + ' MB' : '—');

      const badge = document.getElementById('cfc-badge');
      if (badge) { badge.textContent = 'active'; badge.style.color = '#38bdf8'; badge.style.background = 'rgba(56,189,248,.1)'; }

      // Sparkline for correction_norm
      if (cfc.correction_norm != null) {
        cfcCorrectionNormPoints.push(cfc.correction_norm);
        if (cfcCorrectionNormPoints.length > 200) cfcCorrectionNormPoints.shift();
        drawSparkline('cfc-correction-canvas', cfcCorrectionNormPoints, '#818cf8');
      }
    });

    // ── Q4 watcher ───────────────────────────────────────────────────
    watch(() => store.q4Latest, q4 => {
      if (!q4) return;
      const panel = document.getElementById('q4-panel');
      if (panel) panel.style.display = '';

      const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setText('q4-vram', q4.vram_mb != null ? q4.vram_mb.toFixed(0) + ' MB' : '—');

      const badge = document.getElementById('q4-badge');
      if (badge) { badge.textContent = 'Q4 active'; badge.style.color = '#00ffc8'; badge.style.background = 'rgba(0,255,200,.1)'; }
    });

    // ── Subsystem Status bar ──────────────────────────────────────────
    watch(() => store.backend, backend => {
      const el = document.getElementById('subsys-backend');
      if (!el) return;
      const statusText = backend === 'unknown' ? 'unknown' : backend.toUpperCase();
      const color = backend === 'unknown' ? '#6b7280' : '#00ffc8';
      el.textContent = statusText;
      el.style.color = color;
      el.style.background = backend === 'unknown' ? 'rgba(255,255,255,.08)' : 'rgba(0,255,200,.1)';
    }, { immediate: true });

    watch(() => store.quantization, quant => {
      const el = document.getElementById('subsys-quant');
      if (!el) return;
      const statusText = quant.toUpperCase();
      const color = quant === 'bf16' ? '#00ffc8' : quant === 'q4' ? '#f472b6' : '#9ca3af';
      el.textContent = statusText;
      el.style.color = color;
      el.style.background = quant !== 'unknown' ? 'rgba(255,255,255,.1)' : 'rgba(255,255,255,.08)';
    }, { immediate: true });

    watch(() => store.esnActive, active => {
      const el = document.getElementById('subsys-esn');
      if (!el) return;
      el.textContent = active ? 'ON' : 'OFF';
      el.style.color = active ? '#a855f7' : '#6b7280';
      el.style.background = active ? 'rgba(168,85,247,.1)' : 'rgba(255,255,255,.08)';
    }, { immediate: true });

    watch(() => store.gravityActive, active => {
      const el = document.getElementById('subsys-gravity');
      if (!el) return;
      el.textContent = active ? 'ON' : 'OFF';
      el.style.color = active ? '#fbbf24' : '#6b7280';
      el.style.background = active ? 'rgba(251,191,36,.1)' : 'rgba(255,255,255,.08)';
    }, { immediate: true });

    watch(() => store.cfcActive, active => {
      const el = document.getElementById('subsys-cfc');
      if (!el) return;
      el.textContent = active ? 'ON' : 'OFF';
      el.style.color = active ? '#38bdf8' : '#6b7280';
      el.style.background = active ? 'rgba(56,189,248,.1)' : 'rgba(255,255,255,.08)';
    }, { immediate: true });

    // ── Console filter watcher ────────────────────────────────────────
    watch(() => store.logFilter, filter => {
      const buttons = document.querySelectorAll('.console-filter-inner [data-filter]');
      buttons.forEach(btn => {
        const isActive = btn.dataset.filter === filter;
        btn.classList.toggle('filter-btn--active', isActive);
        btn.setAttribute('aria-pressed', String(isActive));
      });
    });

    // Reseta ESN + Gravity + CfC ao iniciar nova geração + gerencia UI
    watch(() => store.generating, gen => {
      const sendBtn = document.getElementById('send-btn');
      const evalStats = document.getElementById('eval-stats');

      if (gen) {
        // Desabilita botão e mostra estado de geração
        if (sendBtn) {
          sendBtn.disabled = true;
          sendBtn.innerHTML = '<span class="spinner-inline" aria-hidden="true"></span> Gerando...';
        }
        if (evalStats) evalStats.textContent = 'Gerando...';

        // Reset dados de telemetria
        esnEntropyPoints = [];
        esnTop1Points = [];
        esnStateNormPoints = [];
        gravityThresholdPoints = [];
        gravityMeanPoints = [];
        gravityHighCountHistory = [];
        cfcCorrectionNormPoints = [];
        store.esnHistory = [];
        store.esnLatest = null;
        store.gravityLatest = null;
        store.gravityHistory = [];
        store.cfcLatest = null;
        store.q4Latest = null;
        const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setText('esn-entropy', '—');
        setText('esn-top1', '—');
        setText('esn-state-norm', '—');
        setText('esn-temp-delta', '—');
        setText('esn-step', '—');
        setText('gravity-mean', '—');
        setText('gravity-max', '—');
        setText('gravity-high-count', '—');
        setText('cfc-state-norm', '—');
        setText('cfc-correction-norm', '—');
        setText('cfc-params', '—');
        setText('cfc-vram', '—');
        setText('q4-vram', '—');
      } else {
        // Geração finalizada: reabilita botão
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.innerHTML = '<span aria-hidden="true">▶</span> Generate';
        }
        // Atualiza métricas
        if (evalStats && store.evalCount > 0) {
          const tps = store.tokensPerSec > 0 ? store.tokensPerSec.toFixed(1) : '?';
          evalStats.textContent = `${store.evalCount} tokens · ${store.totalDurationMs.toFixed(0)}ms · ${tps} t/s`;
        }
      }
    });

    // Mostra painel Gravity quando subsistema ativa
    watch(() => store.gravityActive, active => {
      const panel = document.getElementById('gravity-panel');
      if (panel) panel.style.display = active ? 'flex' : 'none';
    });

    // Mostra painel CfC quando subsistema ativa
    watch(() => store.cfcActive, active => {
      const panel = document.getElementById('cfc-panel');
      if (panel) panel.style.display = active ? 'flex' : 'none';
    });

    // ── Console watcher — renderiza log-list quando store.logs cresce ────────
    let _consoleRenderPending = false;
    watch(() => store.logs.length, () => {
      if (store.activeTab !== 'console') return;
      if (_consoleRenderPending) return;
      _consoleRenderPending = true;
      requestAnimationFrame(() => { renderConsoleLogs(); _consoleRenderPending = false; });
    });

    watch(() => store.logFilter, () => {
      if (store.activeTab === 'console') renderConsoleLogs();
    });

    // ── Graph watcher — atualiza D3 force graph com dados reais das camadas ──
    watch(() => store.layerData, data => {
      if (store.activeTab === 'graph') updateForceGraph(data);
    }, { deep: true });

    // ── Trace tab — renderiza token timeline a partir de store.traceTokens ───
    function renderTraceFromTokens() {
      const tbody = document.getElementById('token-timeline-body');
      if (!tbody) return;

      const tokens = store.traceTokens;
      tbody.innerHTML = '';
      tokens.forEach(function(t) {
        const tr = document.createElement('tr');
        if (t.anomaly) tr.style.background = 'rgba(255,68,102,0.08)';
        const probStr = t.prob != null ? (Math.min(1, Math.max(0, t.prob)) * 100).toFixed(1) + '%' : '—';
        const entStr  = t.entropy != null ? t.entropy.toFixed(3) : '—';
        const logStr  = t.top_logit != null ? t.top_logit.toFixed(2) : '—';
        const msStr   = t.ms_token != null ? t.ms_token.toFixed(0) : '—';
        tr.innerHTML = `
          <td style="padding:4px 8px;color:#6b7fa3">${t.step}${t.anomaly ? ' <span style="color:#ff4466">⚠</span>' : ''}</td>
          <td style="padding:4px 8px;color:#00ffc8;font-family:monospace">${escapeHtml(t.token)}</td>
          <td style="padding:4px 8px;color:#6b7fa3;font-family:monospace">${t.token_id != null ? t.token_id : '—'}</td>
          <td style="padding:4px 8px;color:#a855f7;font-family:monospace">${logStr}</td>
          <td style="padding:4px 8px;color:#fbbf24;font-family:monospace">${probStr}</td>
          <td style="padding:4px 8px;color:#38bdf8;font-family:monospace">${msStr}</td>
          <td style="padding:4px 8px;color:#34d399;font-family:monospace">${entStr}</td>
          <td style="padding:4px 8px">${t.anomaly ? '<span style="color:#ff4466;font-weight:700">SPIKE</span>' : ''}</td>
        `;
        tbody.appendChild(tr);
      });

      // Energy chart from traceTokens
      const energyCanvas = document.getElementById('energy-chart-canvas');
      if (energyCanvas && tokens.length > 1 && typeof Chart !== 'undefined') {
        const labels = tokens.map(t => `s${t.step}`);
        const energyData = tokens.map(t => t.energy || 0);
        if (window._traceEnergyChart) {
          window._traceEnergyChart.data.labels = labels;
          window._traceEnergyChart.data.datasets[0].data = energyData;
          window._traceEnergyChart.update('none');
        } else {
          window._traceEnergyChart = new Chart(energyCanvas.getContext('2d'), {
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
                pointRadius: tokens.map(t => t.anomaly ? 5 : 2),
                pointBackgroundColor: tokens.map(t => t.anomaly ? '#ff4466' : '#00ffc8'),
              }],
            },
            options: {
              responsive: true, maintainAspectRatio: false,
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

      // Resonance score = entropy-weighted coherence (simple heuristic)
      if (tokens.length > 0) {
        const avgEnt = tokens.reduce((s, t) => s + (t.entropy || 1), 0) / tokens.length;
        const score = Math.max(0, Math.min(1, 1 - (avgEnt / 5)));
        const bar = document.getElementById('resonance-score');
        const val = document.getElementById('resonance-value');
        if (bar) { bar.value = score * 100; }
        if (val)  { val.textContent = score.toFixed(2); }
      }

      // Anomaly list
      const anomalies = tokens.filter(t => t.anomaly);
      const anomalyList = document.getElementById('anomaly-list');
      const anomalyCount = document.getElementById('anomaly-count-trace');
      if (anomalyCount) anomalyCount.textContent = anomalies.length;
      if (anomalyList) {
        if (anomalies.length === 0) {
          anomalyList.innerHTML = '<div class="anomaly-empty">No anomalies detected</div>';
        } else {
          anomalyList.innerHTML = anomalies.map(a =>
            `<div style="background:rgba(255,68,102,0.06);border:1px solid rgba(255,68,102,0.2);border-radius:8px;padding:10px 14px;margin-bottom:8px">
              <span style="background:#ff4466;color:#000;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;margin-right:8px">SPIKE</span>
              <span style="color:#6b7fa3;font-size:11px">step ${a.step}</span>
              <p style="color:#e8e8f0;font-size:12px;margin-top:6px">Energy spike detected: ${(a.energy || 0).toFixed(3)} (token: <code>${escapeHtml(a.token)}</code>)</p>
            </div>`
          ).join('');
        }
      }
    }

    watch(() => store.traceTokens.length, () => {
      if (store.activeTab === 'trace') renderTraceFromTokens();
    });

    // Reseta traceTokens ao iniciar nova geração
    watch(() => store.generating, gen => {
      if (gen) {
        store.traceTokens = [];
        store.layerData = {};
        if (window._traceEnergyChart) {
          window._traceEnergyChart.data.labels = [];
          window._traceEnergyChart.data.datasets[0].data = [];
          window._traceEnergyChart.update('none');
        }
      }
    });

    // Console — wire up buttons
    const clearBtn = document.getElementById('console-clear-btn');
    if (clearBtn) clearBtn.addEventListener('click', () => { store.logs = []; renderConsoleLogs(); });

    const pauseBtn = document.getElementById('console-pause-btn');
    if (pauseBtn) pauseBtn.addEventListener('click', function() {
      const paused = this.getAttribute('aria-pressed') === 'true';
      this.setAttribute('aria-pressed', String(!paused));
      this.classList.toggle('ctrl-btn--active', !paused);
      setConsoleAutoScroll(paused); // toggle: if was paused, resume
    });

    const copyBtn = document.getElementById('inspector-copy-btn');
    if (copyBtn) copyBtn.addEventListener('click', () => {
      const pre = document.getElementById('json-inspector');
      if (pre && navigator.clipboard) navigator.clipboard.writeText(pre.textContent);
    });

    const consoleSearch = document.getElementById('console-search');
    if (consoleSearch) consoleSearch.addEventListener('input', () => renderConsoleLogs());

    const filterBtns = document.querySelectorAll('.console-filter-inner [data-filter]');
    filterBtns.forEach(btn => btn.addEventListener('click', () => {
      store.logFilter = btn.dataset.filter;
    }));

    // Graph tab — controls wiring
    const graphResetBtn = document.getElementById('graph-reset-btn');
    if (graphResetBtn) graphResetBtn.addEventListener('click', () => {
      const svg = document.getElementById('force-graph-svg');
      if (svg && typeof d3 !== 'undefined') {
        d3.select(svg).transition().duration(500).call(
          d3.zoom().transform, d3.zoomIdentity
        );
      }
    });

    console.log('IARA Neural Dashboard started');
  },
}).mount('#app');
