// frontend/src/utils/charts.js

let _logitsChart = null;

export const Charts = {
  initLogitsChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;

    _logitsChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Logit Score',
          data: [],
          backgroundColor: 'rgba(168, 85, 247, 0.4)',
          borderColor: '#a855f7',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
          x: { grid: { display: false } }
        }
      }
    });
  },

  updateLogitsChart(tokens) {
    if (!_logitsChart || !tokens) return;
    _logitsChart.data.labels = tokens.map(t => t.text);
    _logitsChart.data.datasets[0].data = tokens.map(t => t.score);
    _logitsChart.update();
  }
};
