// frontend/src/components/layer_grid.js

import { el, thermalColor } from '../utils.js';
import { store } from '../store/index.js';

let _layerSortBy = 'energy'; // 'energy'|'time'|'index'

export function renderLayerGrid() {
  const container = el('layer-grid-container');
  if (!container) return;

  const data = store.layerData;
  const layers = Array.from({ length: 36 }, (_, i) => ({
    idx: i,
    ...(data[i] || {
      attention_energy: 0,
      mlp_energy: 0,
      residual_norm: 0,
      max_activation: 0,
      hot_neurons: [],
      time_us: 0,
    }),
  }));

  const maxAttn = Math.max(...layers.map(l => l.attention_energy || 0), 1);
  const maxMlp  = Math.max(...layers.map(l => l.mlp_energy      || 0), 1);
  const maxTime = Math.max(...layers.map(l => l.time_us         || 0), 1);

  // Sort
  if (_layerSortBy === 'energy') {
    layers.sort((a, b) => ((b.attention_energy || 0) + (b.mlp_energy || 0)) - ((a.attention_energy || 0) + (a.mlp_energy || 0)));
  } else if (_layerSortBy === 'time') {
    layers.sort((a, b) => (b.time_us || 0) - (a.time_us || 0));
  } else {
    layers.sort((a, b) => a.idx - b.idx);
  }

  container.innerHTML = '';

  layers.forEach(function(layer) {
    const row = document.createElement('div');
    row.className = 'layer-row';
    row.style.cssText = `
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
    `;

    const attnPct = ((layer.attention_energy || 0) / maxAttn * 100).toFixed(1);
    const mlpPct  = ((layer.mlp_energy       || 0) / maxMlp  * 100).toFixed(1);
    const timePct = ((layer.time_us           || 0) / maxTime * 100).toFixed(1);

    // 10 hot-neuron dots (indices 0-9 representative)
    let dotHTML = '';
    for (let d = 0; d < 10; d++) {
      const isHot = (layer.hot_neurons || []).includes(d);
      dotHTML += `<span style="
        display:inline-block;width:6px;height:6px;border-radius:50%;
        background:${isHot ? '#ff7700' : '#1a1a2e'};
        border:1px solid ${isHot ? '#ff7700' : '#333'};
        opacity:${isHot ? 1 : 0.4};
      "></span>`;
    }

    row.innerHTML = `
      <span style="color:#6b7fa3;width:28px;flex-shrink:0">L${layer.idx}</span>
      <div style="flex:1;display:flex;flex-direction:column;gap:2px">
        <div style="display:flex;align-items:center;gap:4px">
          <span style="color:#6b7fa3;width:32px;font-size:10px">Attn</span>
          <div style="flex:1;height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden">
            <div style="width:${attnPct}%;height:100%;background:var(--accent-cyan,#00d4ff);border-radius:3px"></div>
          </div>
          <span style="color:#6b7fa3;width:36px;text-align:right">${(layer.attention_energy||0).toFixed(2)}</span>
        </div>
        <div style="display:flex;align-items:center;gap:4px">
          <span style="color:#6b7fa3;width:32px;font-size:10px">MLP</span>
          <div style="flex:1;height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden">
            <div style="width:${mlpPct}%;height:100%;background:#a855f7;border-radius:3px"></div>
          </div>
          <span style="color:#6b7fa3;width:36px;text-align:right">${(layer.mlp_energy||0).toFixed(2)}</span>
        </div>
      </div>
      <span style="
        background:rgba(0,212,255,0.1);color:#00d4ff;
        padding:1px 5px;border-radius:4px;width:36px;text-align:center;flex-shrink:0
      ">${(layer.residual_norm||0).toFixed(1)}</span>
      <div style="width:50px;height:8px;background:#1a1a2e;border-radius:4px;overflow:hidden;flex-shrink:0">
        <div style="width:${timePct}%;height:100%;background:#3b82f6;border-radius:4px"></div>
      </div>
      <div style="display:flex;gap:2px;flex-shrink:0">${dotHTML}</div>
    `;

    row.title = `Layer ${layer.idx}: attn=${(layer.attention_energy||0).toFixed(3)} mlp=${(layer.mlp_energy||0).toFixed(3)} time=${(layer.time_us||0)}µs`;
    container.appendChild(row);
  });
}

export function renderLayerHeatmapCells() {
  const container = el('layer-heatmap-cells');
  if (!container) return;

  // Create cells if not yet populated
  if (container.children.length !== 36) {
    container.innerHTML = '';
    for (let i = 0; i < 36; i++) {
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.dataset.idx = i;
      cell.style.cssText = `
        width: 32px; height: 32px;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.06);
        cursor: pointer;
        transition: transform 0.1s;
        position: relative;
      `;
      cell.addEventListener('mouseover', function() {
        const d = store.layerData[i] || {};
        this.title = `L${i} | attn:${(d.attention_energy||0).toFixed(3)} mlp:${(d.mlp_energy||0).toFixed(3)}`;
      });
      container.appendChild(cell);
    }
  }

  const cells = container.querySelectorAll('.heatmap-cell');
  const maxE = Math.max(
    ...Array.from({ length: 36 }, (_, i) => (store.layerData[i] || {}).attention_energy || 0),
    1
  );

  cells.forEach(function(cell) {
    const idx = parseInt(cell.dataset.idx);
    const d = store.layerData[idx] || {};
    const energy = (d.attention_energy || 0) / maxE;
    cell.style.background = thermalColor(energy);
  });
}

export function setLayerSort(by) {
  _layerSortBy = by;
  renderLayerGrid();
}