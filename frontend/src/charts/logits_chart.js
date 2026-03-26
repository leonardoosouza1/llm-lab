// frontend/src/charts/logits_chart.js

import { el } from '../utils.js';
import { store } from '../store/index.js';

let _logitsChart = null;

export function initLogitsChart() {
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

export function updateLogitsChart() {
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