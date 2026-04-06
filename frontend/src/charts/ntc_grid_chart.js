// frontend/src/charts/ntc_grid_chart.js

import { el } from '../utils.js';
import { store } from '../store/index.js';

let _ntcGridChart = null;

export function renderNtcGridHeatmap() {
  const canvas = el('ntc-grid-heatmap');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const gridSize = store.ntc.gridSize || 64;
  const latentDim = store.ntc.latentDim || 32;

  if (gridSize === 0 || latentDim === 0) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#6b7fa3';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Module not active', canvas.width / 2, canvas.height / 2);
    return;
  }

  const cellWidth = canvas.width / latentDim;
  const cellHeight = canvas.height / gridSize;

  // Draw background
  ctx.fillStyle = '#0a0e27';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw heatmap: grid_size (rows) × latent_dim (columns)
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < latentDim; j++) {
      const x = j * cellWidth;
      const y = i * cellHeight;

      // Mock intensity based on position (would be actual latent values)
      const intensity = Math.sin(i / gridSize * Math.PI) * Math.sin(j / latentDim * Math.PI);
      const hue = 180; // Cyan
      const saturation = 100 * Math.max(0, intensity);
      const lightness = 25 + 50 * Math.max(0, intensity);

      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.fillRect(x, y, cellWidth, cellHeight);
    }
  }
}

export function renderNtcReconstructionLoss() {
  const canvas = el('ntc-loss-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const history = store.ntc.history || [];
  if (history.length === 0) return;

  const labels = history.map((_, i) => `step ${i}`);
  const data = history.map(h => h.reconstructionLoss || 0);

  if (_ntcGridChart) {
    _ntcGridChart.data.labels = labels;
    _ntcGridChart.data.datasets[0].data = data;
    _ntcGridChart.update();
    return;
  }

  _ntcGridChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Reconstruction Loss',
        data: data,
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6,182,212,0.1)',
        fill: true,
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
        pointBackgroundColor: '#06b6d4',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: '#6b7fa3', maxTicksLimit: 20 },
          grid: { color: '#1a2035' },
        },
        y: {
          ticks: { color: '#6b7fa3' },
          grid: { color: '#1a2035' },
        },
      },
      plugins: {
        legend: { labels: { color: '#6b7fa3' } },
      },
    },
  });
}

export function renderNtcVramComparison() {
  const canvas = el('ntc-vram-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const vramNtc = store.ntc.vramMb || 0;
  const labels = ['NTC', 'F32', 'INT8'];
  const data = [
    vramNtc,
    vramNtc * 2.5,  // F32 baseline
    vramNtc * 1.4   // INT8
  ];

  new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'VRAM (MB)',
        data: data,
        backgroundColor: ['#06b6d4', '#f59e0b', '#ef4444'],
        borderRadius: 4,
        borderWidth: 1,
        borderColor: ['#0891b2', '#d97706', '#dc2626'],
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: '#6b7fa3' },
          grid: { display: false },
        },
        y: {
          ticks: { color: '#6b7fa3' },
          grid: { color: '#1a2035' },
        },
      },
      plugins: {
        legend: { labels: { color: '#6b7fa3' } },
      },
    },
  });
}

export function updateNtcCharts() {
  renderNtcGridHeatmap();
  renderNtcReconstructionLoss();
  renderNtcVramComparison();
}
