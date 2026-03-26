// frontend/src/charts/energy_chart.js

import { el, escapeHtml } from '../utils.js';
import { store } from '../store/index.js';

let _energyChart = null;

export function renderTracePanel() {
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

  const valueEl = el('resonance-value');
  if (valueEl) {
    valueEl.textContent = score.toFixed(2);
  }
}