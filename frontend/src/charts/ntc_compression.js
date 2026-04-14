// frontend/src/charts/ntc_compression.js
// Model Compression Dashboard — NTC tab
// Goal: 1 GB VRAM with quality equivalent to a 6 GB model
// Techniques: quantization (INT4/INT2/GGUF), pruning (layers/neurons), distillation

import { store } from '../store/index.js';

// ─── Model size / quality reference points ────────────────────────────────────
// Each point: { label, sizeGb, qualityPct, color, highlight }
const REFERENCE_POINTS = [
  { label: 'BF16',        sizeGb: 3.00, qualityPct: 100, color: '#6366f1', highlight: true },
  { label: 'Q8',          sizeGb: 1.50, qualityPct:  99, color: '#34d399' },
  { label: 'Q4',          sizeGb: 0.75, qualityPct:  96, color: '#06b6d4', sweet: true },
  { label: 'GGUF Q4_K_M', sizeGb: 0.65, qualityPct:  95, color: '#fbbf24' },
  { label: 'Q2',          sizeGb: 0.37, qualityPct:  85, color: '#f97316' },
  { label: 'Pruned 50%',  sizeGb: 0.375,qualityPct:  88, color: '#a855f7' },
  { label: 'Pruned 70%',  sizeGb: 0.225,qualityPct:  78, color: '#ec4899' },
  { label: 'Distil 0.5B', sizeGb: 0.25, qualityPct:  82, color: '#fb7185' },
];

// 1 GB VRAM target line
const VRAM_TARGET_GB = 1.0;

// ─── Quality vs Size scatter chart ───────────────────────────────────────────

let _scatterChart = null;

function buildScatterDatasets() {
  const regular  = REFERENCE_POINTS.filter(p => !p.highlight && !p.sweet);
  const current  = REFERENCE_POINTS.filter(p =>  p.highlight);
  const sweet    = REFERENCE_POINTS.filter(p =>  p.sweet);

  return [
    {
      label: 'Variants',
      data: regular.map(p => ({ x: p.sizeGb, y: p.qualityPct, label: p.label })),
      pointBackgroundColor: regular.map(p => p.color),
      pointBorderColor:     regular.map(p => p.color),
      pointRadius: 7,
      pointHoverRadius: 10,
    },
    {
      label: 'Sweet spot (Q4)',
      data: sweet.map(p => ({ x: p.sizeGb, y: p.qualityPct, label: p.label })),
      pointBackgroundColor: '#06b6d4',
      pointBorderColor: '#ffffff',
      pointBorderWidth: 2,
      pointRadius: 10,
      pointHoverRadius: 13,
      pointStyle: 'star',
    },
    {
      label: 'Current (BF16)',
      data: current.map(p => ({ x: p.sizeGb, y: p.qualityPct, label: p.label })),
      pointBackgroundColor: '#6366f1',
      pointBorderColor: '#ffffff',
      pointBorderWidth: 2,
      pointRadius: 10,
      pointHoverRadius: 13,
      pointStyle: 'rectRot',
    },
  ];
}

function renderQualityScatter() {
  const canvas = document.getElementById('ntc-quality-scatter');
  if (!canvas || typeof Chart === 'undefined') return;

  const datasets = buildScatterDatasets();

  if (_scatterChart) {
    _scatterChart.data.datasets = datasets;
    _scatterChart.update('none');
    return;
  }

  _scatterChart = new Chart(canvas.getContext('2d'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          title: { display: true, text: 'Model size (GB)', color: '#6b7fa3', font: { size: 10 } },
          min: 0,
          max: 3.5,
          ticks: { color: '#6b7fa3', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,.05)' },
        },
        y: {
          title: { display: true, text: 'Quality (% of BF16)', color: '#6b7fa3', font: { size: 10 } },
          min: 70,
          max: 102,
          ticks: { color: '#6b7fa3', font: { size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,.05)' },
        },
      },
      plugins: {
        legend: {
          labels: { color: '#9ca3af', font: { size: 10 }, boxWidth: 10 },
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              const pt = ctx.raw;
              return ` ${pt.label}: ${pt.x} GB · ${pt.y}% quality`;
            },
          },
        },
        // 1 GB VRAM target annotation drawn manually in afterDraw
      },
    },
    plugins: [{
      id: 'vram-target-line',
      afterDraw(chart) {
        const { ctx, scales } = chart;
        const xScale = scales.x;
        const yScale = scales.y;
        if (!xScale || !yScale) return;
        const xPx = xScale.getPixelForValue(VRAM_TARGET_GB);
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = 'rgba(251,191,36,.6)';
        ctx.lineWidth = 1.5;
        ctx.moveTo(xPx, yScale.top);
        ctx.lineTo(xPx, yScale.bottom);
        ctx.stroke();
        ctx.fillStyle = 'rgba(251,191,36,.8)';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('1 GB target', xPx, yScale.top - 4);
        ctx.restore();
      },
    }],
  });
}

// ─── Layer importance heatmap ─────────────────────────────────────────────────

export function renderLayerImportanceHeatmap() {
  const canvas = document.getElementById('ntc-layer-importance-canvas');
  if (!canvas) return;

  const ctx    = canvas.getContext('2d');
  const W      = canvas.offsetWidth  || canvas.width;
  const H      = canvas.height;
  const data   = store.layerData;
  const numL   = store.numLayers || 28;

  ctx.clearRect(0, 0, W, H);

  // Collect residual_norm values (proxy for layer importance)
  const values = Array.from({ length: numL }, (_, i) => {
    const d = data[i] || {};
    // Use residual_norm if available; fall back to combined energy heuristic
    return d.residual_norm != null
      ? d.residual_norm
      : (d.attention_energy || 0) + (d.mlp_energy || 0) * 0.5;
  });

  const maxVal = Math.max(...values, 0.001);

  const cellW = W / numL;
  const barH  = H - 20;

  values.forEach((v, i) => {
    const norm = v / maxVal; // 0..1
    const x = i * cellW;
    // Color: red (redundant) → yellow → green (important)
    const hue = norm * 120; // 0=red, 60=yellow, 120=green
    ctx.fillStyle = `hsl(${hue}, 90%, ${25 + norm * 35}%)`;
    ctx.fillRect(x + 1, barH - barH * norm, cellW - 2, barH * norm);

    // Layer index labels every 4 layers
    if (i % 4 === 0) {
      ctx.fillStyle = 'rgba(107,127,163,.7)';
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`L${i}`, x + cellW / 2, H - 4);
    }
  });

  // Update the prunable layers list
  updatePrunableLayersList(values, maxVal, numL);
}

function updatePrunableLayersList(values, maxVal, numL) {
  const listEl = document.getElementById('ntc-prunable-layers');
  if (!listEl) return;

  if (maxVal <= 0.001) {
    listEl.innerHTML = '<span style="font-size:10px;color:#6b7fa3;font-family:\'JetBrains Mono\',monospace;">Run a generation first to compute importance scores.</span>';
    return;
  }

  const candidates = values
    .map((v, i) => ({ idx: i, score: v / maxVal }))
    .filter(l => l.score < 0.25)
    .slice(0, 8);

  if (candidates.length === 0) {
    listEl.innerHTML = '<span style="font-size:10px;color:#34d399;font-family:\'JetBrains Mono\',monospace;">No obvious pruning candidates (all layers active).</span>';
    return;
  }

  listEl.innerHTML = '<div style="font-size:10px;color:#f97316;font-family:\'JetBrains Mono\',monospace;margin-bottom:5px;">Pruning candidates (importance &lt; 25%):</div>' +
    candidates.map(l => `
      <label style="display:inline-flex;align-items:center;gap:5px;margin:2px 4px 2px 0;font-size:10px;color:#a855f7;font-family:'JetBrains Mono',monospace;cursor:pointer;">
        <input type="checkbox" value="${l.idx}" style="accent-color:#a855f7;"> L${l.idx} (${(l.score*100).toFixed(0)}%)
      </label>
    `).join('');
}

// ─── Strategy selector (global handlers) ─────────────────────────────────────

let _currentStrategy = 'quantize';

window._ntcSelectStrategy = function(strategy) {
  _currentStrategy = strategy;

  const ids = ['quantize', 'prune-neurons', 'prune-layers', 'distill'];
  ids.forEach(id => {
    const btn  = document.getElementById(`ntc-btn-${id}`);
    const opts = document.getElementById(`ntc-opts-${id}`);
    const active = id === strategy;

    if (btn) {
      if (active) {
        btn.style.background = 'rgba(6,182,212,.15)';
        btn.style.color = '#06b6d4';
        btn.style.borderColor = 'rgba(6,182,212,.4)';
      } else {
        btn.style.background = 'rgba(255,255,255,.05)';
        btn.style.color = '#6b7fa3';
        btn.style.borderColor = 'rgba(255,255,255,.1)';
      }
    }
    if (opts) opts.style.display = active ? '' : 'none';
  });
};

// ─── Compression start (UI only — logs intent, no real backend yet) ───────────

window._ntcStartCompress = function() {
  const logEl  = document.getElementById('ntc-log-output');
  const barEl  = document.getElementById('ntc-progress-bar');
  const pctEl  = document.getElementById('ntc-progress-pct');
  const chip   = document.getElementById('ntc-status-chip');

  if (!logEl) return;

  function log(msg, color) {
    const d = document.createElement('div');
    d.style.color = color || '#9ca3af';
    d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logEl.appendChild(d);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setProgress(pct, label) {
    if (barEl)  barEl.style.width = pct + '%';
    if (pctEl)  pctEl.textContent = label || pct + '%';
    if (chip) {
      chip.textContent = label || pct + '%';
      chip.style.color = '#06b6d4';
      chip.style.background = 'rgba(6,182,212,.1)';
    }
  }

  // Clear log
  logEl.innerHTML = '';

  const strategy = _currentStrategy;
  log(`Starting strategy: ${strategy}`, '#06b6d4');

  if (strategy === 'quantize') {
    const level = document.getElementById('ntc-quant-level')?.value || 'q4';
    const labels = { q8: 'INT8/Q8', q4: 'INT4/Q4', gguf_q4km: 'GGUF Q4_K_M', q2: 'INT2/Q2' };
    const sizes  = { q8: '1.5 GB', q4: '0.75 GB', gguf_q4km: '0.65 GB', q2: '0.37 GB' };
    log(`Target: ${labels[level]} → ${sizes[level]}`, '#fbbf24');

    // Simulate progress steps
    const steps = [
      [10, 'Scanning weight tensors...'],
      [25, `Computing ${labels[level]} scale factors per group...`],
      [50, 'Quantizing attention weights (Q/K/V/O)...'],
      [70, 'Quantizing MLP weights (gate/up/down)...'],
      [85, 'Quantizing embedding + lm_head...'],
      [95, 'Verifying reconstruction error...'],
      [100, `Done! ${labels[level]} model ready — ${sizes[level]}`],
    ];

    let i = 0;
    function step() {
      if (i >= steps.length) {
        log('NOTE: This is a UI preview. Connect backend to run real quantization.', '#f97316');
        if (chip) { chip.textContent = 'preview complete'; chip.style.color = '#34d399'; chip.style.background = 'rgba(16,185,129,.1)'; }
        return;
      }
      const [pct, msg] = steps[i++];
      setProgress(pct, pct + '%');
      log(msg, pct === 100 ? '#34d399' : '#9ca3af');
      setTimeout(step, 350);
    }
    setTimeout(step, 100);

  } else if (strategy === 'prune-neurons') {
    const pct = document.getElementById('ntc-prune-neuron-pct')?.value || 50;
    log(`Keeping top ${pct}% neurons by activation magnitude`, '#fbbf24');
    log('Using activation data from NeurViz (residual_norm per neuron)...', '#9ca3af');
    log('Zeroing out low-activation MLP weights...', '#9ca3af');
    setTimeout(() => {
      setProgress(100, '100%');
      log(`Pruned! ~${(100-pct)/2}% size reduction. NOTE: preview only.`, '#34d399');
    }, 800);

  } else if (strategy === 'prune-layers') {
    const checked = Array.from(document.querySelectorAll('#ntc-prunable-layers input:checked')).map(c => c.value);
    if (checked.length === 0) {
      log('No layers selected. Check the pruning candidates above.', '#ff4466');
      return;
    }
    log(`Removing layers: ${checked.join(', ')}`, '#fbbf24');
    setTimeout(() => {
      setProgress(100, '100%');
      log(`Done! Removed ${checked.length} layers. NOTE: preview only.`, '#34d399');
    }, 600);

  } else if (strategy === 'distill') {
    const size = document.getElementById('ntc-distill-size')?.value || '1b';
    log(`Distilling into ${size} student model...`, '#fbbf24');
    log('Step 1: Initialize student architecture...', '#9ca3af');
    log('Step 2: Compute teacher logits (requires backend GPU pipeline)...', '#9ca3af');
    log('Step 3: KL-divergence loss training...', '#9ca3af');
    setTimeout(() => {
      setProgress(0, '—');
      log('Distillation requires backend training pipeline. Connect iara-engine distill endpoint.', '#f97316');
      if (chip) { chip.textContent = 'needs backend'; chip.style.color = '#f97316'; chip.style.background = 'rgba(249,115,22,.1)'; }
    }, 800);
  }
};

// ─── Main init / update entry point ──────────────────────────────────────────

let _initialized = false;

export function initNtcCompression() {
  // Render the quality vs size scatter chart
  renderQualityScatter();

  // Render layer importance heatmap (even if no data — shows placeholder)
  renderLayerImportanceHeatmap();

  // Update model chip from store
  const modelChip = document.getElementById('ntc-current-model-chip');
  if (modelChip && store.modelName && store.modelName !== 'none') {
    const quant = store.quantization || 'bf16';
    modelChip.textContent = `${store.modelName} ${quant.toUpperCase()}`;
  }

  _initialized = true;
}

export function updateNtcCompression() {
  if (!_initialized) return;
  renderLayerImportanceHeatmap();
}
