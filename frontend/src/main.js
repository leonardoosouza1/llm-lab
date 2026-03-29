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
  if (tab === 'trace' && store.trace) renderTracePanel();
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
    let esnEntropyCtx = null;
    let esnEntropyPoints = [];

    watch(() => store.esnLatest, esn => {
      if (!esn) return;

      // Update stat numbers
      const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setText('esn-entropy',     esn.entropy    != null ? esn.entropy.toFixed(3)     : '—');
      setText('esn-top1',        esn.top1_prob  != null ? (esn.top1_prob * 100).toFixed(1) + '%' : '—');
      setText('esn-state-norm',  esn.state_norm != null ? esn.state_norm.toFixed(3)  : '—');
      setText('esn-temperature', esn.temperature != null ? esn.temperature.toFixed(3) : '—');
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

      // Mini sparkline de entropia
      esnEntropyPoints.push(esn.entropy || 0);
      if (esnEntropyPoints.length > 200) esnEntropyPoints.shift();

      const canvas = document.getElementById('esn-entropy-canvas');
      if (!canvas) return;
      canvas.width = canvas.offsetWidth || 600;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      if (esnEntropyPoints.length < 2) return;
      const max = Math.max(...esnEntropyPoints, 0.01);
      ctx.beginPath();
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1.5;
      esnEntropyPoints.forEach((v, i) => {
        const x = (i / (esnEntropyPoints.length - 1)) * W;
        const y = H - (v / max) * (H - 4) - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Reseta ESN ao iniciar nova geração
    watch(() => store.generating, gen => {
      if (gen) {
        esnEntropyPoints = [];
        store.esnHistory = [];
        store.esnLatest = null;
        const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setText('esn-entropy', '—');
        setText('esn-top1', '—');
        setText('esn-state-norm', '—');
        setText('esn-temperature', '—');
        setText('esn-step', '—');
      }
    });

    console.log('IARA Neural Dashboard started');
  },
}).mount('#app');
