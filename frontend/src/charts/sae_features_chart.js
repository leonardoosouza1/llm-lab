// frontend/src/charts/sae_features_chart.js

import { el } from '../utils.js';
import { store } from '../store/index.js';

let _saeChart = null;

export function renderSaeFeatures() {
  const canvas = el('sae-features-canvas');
  if (!canvas || typeof Chart === 'undefined') return;

  const topFeatures = store.sae.topFeatures.slice(0, 20);
  if (topFeatures.length === 0) return;

  const labels = topFeatures.map((f, i) => `F${f[0]}`);
  const data = topFeatures.map(f => f[1]);

  if (_saeChart) {
    _saeChart.data.labels = labels;
    _saeChart.data.datasets[0].data = data;
    _saeChart.update();
    return;
  }

  _saeChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Activation Count',
        data: data,
        backgroundColor: 'rgba(34,197,94,0.6)',
        borderColor: '#22c55e',
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: '#6b7fa3' },
          grid: { color: '#1a2035' },
        },
        y: {
          ticks: { color: '#6b7fa3' },
          grid: { display: false },
        },
      },
      plugins: {
        legend: { labels: { color: '#6b7fa3' } },
      },
    },
  });
}

export function renderSaeActivationHeatmap() {
  const canvas = el('sae-activation-heatmap');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const topFeatures = store.sae.topFeatures.slice(0, 20);
  const numFeatures = topFeatures.length;
  const numTokens = 50; // Last 50 tokens

  const cellWidth = canvas.width / numFeatures;
  const cellHeight = canvas.height / numTokens;

  // Draw background
  ctx.fillStyle = '#0a0e27';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw heatmap cells (mock data for now — would be populated from actual history)
  for (let i = 0; i < numTokens; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const x = j * cellWidth;
      const y = i * cellHeight;

      // Mock intensity based on feature index and token position
      const intensity = Math.random() * 0.5;
      const hue = 120; // Green
      const saturation = 100 * intensity;
      const lightness = 25 + 50 * (1 - intensity);

      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.fillRect(x, y, cellWidth, cellHeight);
    }
  }
}

export function updateSaeChart() {
  renderSaeFeatures();
  renderSaeActivationHeatmap();
}
