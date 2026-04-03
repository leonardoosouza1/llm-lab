// frontend/src/charts/force_graph.js — Neural Network Topology Visualization
// Mostra a arquitetura do transformer como rede neural com:
//   - Camadas como blocos (Embed → Layers → Norm → LM Head)
//   - Conexões com fluxo de dados (residual stream)
//   - Pesos ativos por camada (Attn, MLP, Norm)
//   - Cores baseadas em energia/ativação em tempo real
//   - Tooltip com detalhes ao hover
//   - Zoom e pan D3

import { store } from '../store/index.js';

let _initialized = false;
let _svg = null;
let _g = null;
let _colorMode = 'attention'; // attention | activation | energy
let _frozen = false;
let _tooltip = null;
let _layerBlocks = null;
let _connections = null;
let _subComponents = null;

// Paletas
const PALETTES = {
  attention: d3.scaleSequential(d3.interpolateInferno).domain([0, 1]),
  activation: d3.scaleSequential(d3.interpolateViridis).domain([0, 1]),
  energy: d3.scaleSequential(d3.interpolatePlasma).domain([0, 1]),
};

export function isForceGraphInitialized() {
  return _initialized;
}

export function initForceGraph() {
  const svgEl = document.getElementById('force-graph-svg');
  if (!svgEl || typeof d3 === 'undefined' || _initialized) return;
  _initialized = true;

  const width  = svgEl.clientWidth  || 900;
  const height = svgEl.clientHeight || 600;

  _svg = d3.select('#force-graph-svg');
  _svg.selectAll('*').remove();

  // ── Defs: gradientes, filtros, markers ──
  const defs = _svg.append('defs');

  // Glow filter
  const glow = defs.append('filter').attr('id', 'glow')
    .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
  glow.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
  const merge = glow.append('feMerge');
  merge.append('feMergeNode').attr('in', 'blur');
  merge.append('feMergeNode').attr('in', 'SourceGraphic');

  // Hot glow (orange)
  const hotGlow = defs.append('filter').attr('id', 'hot-glow')
    .attr('x', '-100%').attr('y', '-100%').attr('width', '300%').attr('height', '300%');
  hotGlow.append('feGaussianBlur').attr('stdDeviation', '6').attr('result', 'blur');
  hotGlow.append('feFlood').attr('flood-color', '#ff6600').attr('flood-opacity', '0.6').attr('result', 'color');
  hotGlow.append('feComposite').attr('in', 'color').attr('in2', 'blur').attr('operator', 'in').attr('result', 'glow');
  const hm = hotGlow.append('feMerge');
  hm.append('feMergeNode').attr('in', 'glow');
  hm.append('feMergeNode').attr('in', 'SourceGraphic');

  // Arrow marker for connections
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 10).attr('refY', 5)
    .attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 Z').attr('fill', '#1a4a44');

  // Background gradient
  const bgGrad = defs.append('linearGradient').attr('id', 'bg-grad')
    .attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%');
  bgGrad.append('stop').attr('offset', '0%').attr('stop-color', '#060f0e');
  bgGrad.append('stop').attr('offset', '100%').attr('stop-color', '#0a1c1a');

  // Background
  _svg.append('rect').attr('width', width).attr('height', height).attr('fill', 'url(#bg-grad)');

  // ── Zoom + Pan ──
  _g = _svg.append('g').attr('class', 'graph-root');
  _svg.call(d3.zoom()
    .scaleExtent([0.2, 4])
    .on('zoom', (event) => _g.attr('transform', event.transform))
  );

  // ── Tooltip ──
  _tooltip = d3.select('body').append('div')
    .attr('class', 'graph-tooltip')
    .style('position', 'absolute')
    .style('pointer-events', 'none')
    .style('background', 'rgba(10,28,26,0.95)')
    .style('border', '1px solid #00e6ff')
    .style('border-radius', '4px')
    .style('padding', '8px 12px')
    .style('font-family', 'JetBrains Mono, monospace')
    .style('font-size', '11px')
    .style('color', '#e0ffff')
    .style('z-index', '9999')
    .style('opacity', 0)
    .style('box-shadow', '0 0 15px rgba(0,230,255,0.2)');

  buildTopology(width, height);
  bindControls();
}

function buildTopology(width, height) {
  const numL = store.numLayers || 28;
  const padding = { top: 60, bottom: 40, left: 80, right: 80 };

  // Layout: vertical cascade (embed → layers → norm → lm_head)
  // Layers organized in 2 columns if > 20 layers for readability
  const useTwoColumns = numL > 20;
  const cols = useTwoColumns ? 2 : 1;
  const rowsPerCol = Math.ceil(numL / cols);

  const colWidth = (width - padding.left - padding.right) / (cols + 1);
  const blockW = 140;
  const blockH = 36;
  const subW = 38;
  const subH = 16;
  const layerSpacing = Math.min(48, (height - padding.top - padding.bottom - 120) / rowsPerCol);

  // ── Special nodes (Embed, Final Norm, LM Head) ──
  const specialNodes = [
    { id: 'embed', label: 'Embed', x: width / 2, y: padding.top, type: 'special' },
    { id: 'norm', label: 'Final Norm', x: width / 2, y: 0, type: 'special' }, // y set later
    { id: 'lm_head', label: 'LM Head', x: width / 2, y: 0, type: 'special' }, // y set later
  ];

  // ── Layer nodes ──
  const layerNodes = [];
  for (let i = 0; i < numL; i++) {
    const col = useTwoColumns ? Math.floor(i / rowsPerCol) : 0;
    const row = useTwoColumns ? (i % rowsPerCol) : i;
    const x = padding.left + colWidth * (col + 0.5 + (useTwoColumns ? 0 : 0.5));
    const y = padding.top + 60 + row * layerSpacing;

    layerNodes.push({
      id: `layer_${i}`,
      idx: i,
      label: `Layer ${i}`,
      x, y,
      type: 'layer',
      attn_energy: 0,
      mlp_energy: 0,
      residual_norm: 0,
      max_activation: 0,
      hot_neurons: [],
      time_us: 0,
    });
  }

  // Update positions of final norm and lm_head
  const lastLayerY = Math.max(...layerNodes.map(n => n.y));
  specialNodes[1].y = lastLayerY + 60;
  specialNodes[2].y = lastLayerY + 110;

  const allNodes = [...specialNodes, ...layerNodes];

  // ── Connections ──
  const connections = [];

  // Embed → first layer(s)
  if (useTwoColumns) {
    connections.push({ source: 'embed', target: 'layer_0', type: 'residual' });
    connections.push({ source: 'embed', target: `layer_${rowsPerCol}`, type: 'residual' });
  } else {
    connections.push({ source: 'embed', target: 'layer_0', type: 'residual' });
  }

  // Layer → Layer (sequential)
  for (let i = 0; i < numL - 1; i++) {
    const samCol = useTwoColumns && Math.floor(i / rowsPerCol) === Math.floor((i+1) / rowsPerCol);
    const nextIsFirstOfCol2 = useTwoColumns && i === rowsPerCol - 1;
    if (!useTwoColumns || samCol) {
      connections.push({ source: `layer_${i}`, target: `layer_${i+1}`, type: 'residual' });
    }
    if (nextIsFirstOfCol2) {
      // Last of col1 → norm (col1 ends here)
    }
  }

  // Last layer(s) → Final Norm
  if (useTwoColumns) {
    connections.push({ source: `layer_${rowsPerCol - 1}`, target: 'norm', type: 'residual' });
    connections.push({ source: `layer_${numL - 1}`, target: 'norm', type: 'residual' });
  } else {
    connections.push({ source: `layer_${numL - 1}`, target: 'norm', type: 'residual' });
  }

  // Final Norm → LM Head
  connections.push({ source: 'norm', target: 'lm_head', type: 'output' });

  // ── Draw connections ──
  const nodeMap = {};
  allNodes.forEach(n => nodeMap[n.id] = n);

  _connections = _g.append('g').attr('class', 'connections')
    .selectAll('path')
    .data(connections)
    .enter()
    .append('path')
    .attr('d', d => {
      const s = nodeMap[d.source];
      const t = nodeMap[d.target];
      if (!s || !t) return '';
      const midY = (s.y + t.y) / 2;
      return `M${s.x},${s.y + blockH/2} C${s.x},${midY} ${t.x},${midY} ${t.x},${t.y - blockH/2}`;
    })
    .attr('stroke', d => d.type === 'output' ? '#00e6ff' : '#1a4a44')
    .attr('stroke-width', d => d.type === 'output' ? 2 : 1.5)
    .attr('fill', 'none')
    .attr('stroke-dasharray', d => d.type === 'output' ? '' : '4,3')
    .attr('opacity', 0.6)
    .attr('marker-end', 'url(#arrow)');

  // ── Draw special nodes ──
  specialNodes.forEach(node => {
    const grp = _g.append('g').attr('class', `special-node ${node.id}`)
      .attr('transform', `translate(${node.x - blockW/2}, ${node.y - blockH/2})`);

    grp.append('rect')
      .attr('width', blockW).attr('height', blockH)
      .attr('rx', 4)
      .attr('fill', node.id === 'embed' ? '#0d2f2a' : node.id === 'lm_head' ? '#1a0d2f' : '#0d1f2f')
      .attr('stroke', node.id === 'embed' ? '#00ffb0' : node.id === 'lm_head' ? '#b080ff' : '#00e6ff')
      .attr('stroke-width', 1.5);

    grp.append('text')
      .attr('x', blockW/2).attr('y', blockH/2 + 1)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
      .attr('font-family', 'Inter, sans-serif').attr('font-size', 12)
      .attr('font-weight', 600).attr('fill', '#e0ffff')
      .text(node.label);
  });

  // ── Draw layer blocks ──
  _layerBlocks = _g.selectAll('.layer-block')
    .data(layerNodes)
    .enter()
    .append('g')
    .attr('class', 'layer-block')
    .attr('transform', d => `translate(${d.x - blockW/2}, ${d.y - blockH/2})`)
    .style('cursor', 'pointer');

  // Main rect
  _layerBlocks.append('rect')
    .attr('class', 'layer-rect')
    .attr('width', blockW).attr('height', blockH)
    .attr('rx', 3)
    .attr('fill', '#0c1f1c')
    .attr('stroke', '#1a3a35')
    .attr('stroke-width', 1);

  // Layer label
  _layerBlocks.append('text')
    .attr('class', 'layer-label')
    .attr('x', 8).attr('y', blockH/2 + 1)
    .attr('dominant-baseline', 'middle')
    .attr('font-family', 'JetBrains Mono, monospace').attr('font-size', 9)
    .attr('fill', '#668f8c')
    .text(d => `L${d.idx}`);

  // Sub-component blocks (Attn, MLP, Norm) inside each layer
  const subComponents = [
    { key: 'attn', label: 'Attn', offset: 32 },
    { key: 'mlp', label: 'MLP', offset: 72 },
    { key: 'norm', label: 'N', offset: 112 },
  ];

  subComponents.forEach(sub => {
    _layerBlocks.append('rect')
      .attr('class', `sub-${sub.key}`)
      .attr('x', sub.offset).attr('y', 6)
      .attr('width', sub.key === 'norm' ? 20 : subW).attr('height', blockH - 12)
      .attr('rx', 2)
      .attr('fill', '#0a1816')
      .attr('stroke', '#1a3530')
      .attr('stroke-width', 0.5);

    _layerBlocks.append('text')
      .attr('x', sub.offset + (sub.key === 'norm' ? 10 : subW/2))
      .attr('y', blockH/2 + 1)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
      .attr('font-family', 'JetBrains Mono, monospace').attr('font-size', 7)
      .attr('fill', '#4a7a75')
      .text(sub.label);
  });

  // ── Hover tooltip ──
  _layerBlocks
    .on('mouseenter', function(event, d) {
      const data = store.layerData[d.idx] || {};
      _tooltip.html(`
        <div style="margin-bottom:4px;font-weight:bold;color:#00e6ff">Layer ${d.idx}</div>
        <div>Attention: <span style="color:#ffb300">${(data.attention_energy || 0).toFixed(2)}</span></div>
        <div>MLP: <span style="color:#00ffb0">${(data.mlp_energy || 0).toFixed(2)}</span></div>
        <div>Residual: <span style="color:#e0ffff">${(data.residual_norm || 0).toFixed(2)}</span></div>
        <div>Max Act: <span style="color:#ff6666">${(data.max_activation || 0).toFixed(2)}</span></div>
        <div>Hot neurons: <span style="color:#ff7700">${(data.hot_neurons || []).length}</span></div>
        <div>Time: <span style="color:#b2d8d8">${((data.time_us || 0) / 1000).toFixed(1)}ms</span></div>
      `)
      .style('opacity', 1)
      .style('left', (event.pageX + 12) + 'px')
      .style('top', (event.pageY - 10) + 'px');

      d3.select(this).select('.layer-rect')
        .attr('stroke', '#00e6ff').attr('stroke-width', 2);
    })
    .on('mousemove', function(event) {
      _tooltip
        .style('left', (event.pageX + 12) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseleave', function() {
      _tooltip.style('opacity', 0);
      d3.select(this).select('.layer-rect')
        .attr('stroke', '#1a3a35').attr('stroke-width', 1);
    });
}

function bindControls() {
  // ── Color mode buttons ──
  document.querySelectorAll('[data-color]').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('[data-color]').forEach(b => {
        b.classList.remove('ctrl-btn--active');
        b.setAttribute('aria-pressed', 'false');
      });
      this.classList.add('ctrl-btn--active');
      this.setAttribute('aria-pressed', 'true');
      _colorMode = this.dataset.color;
      updateForceGraph(store.layerData);
    });
  });

  // ── Reset View ──
  document.getElementById('graph-reset-btn')?.addEventListener('click', () => {
    if (_svg) _svg.call(d3.zoom().transform, d3.zoomIdentity);
  });

  // ── Freeze ──
  document.getElementById('graph-freeze-btn')?.addEventListener('click', function() {
    _frozen = !_frozen;
    this.setAttribute('aria-pressed', _frozen ? 'true' : 'false');
    this.classList.toggle('ctrl-btn--active', _frozen);
    this.textContent = _frozen ? 'Frozen' : 'Freeze';
  });

  // ── Force toggles (for the topology-based view, these are visual only) ──
  document.querySelectorAll('[data-force]').forEach(btn => {
    btn.addEventListener('click', function() {
      const active = this.getAttribute('aria-pressed') === 'true';
      this.setAttribute('aria-pressed', !active ? 'true' : 'false');
      this.classList.toggle('ctrl-btn--active', !active);
      // In topology mode, force toggles control connection visibility
      const force = this.dataset.force;
      if (force === 'links') {
        _g?.selectAll('.connections path').attr('opacity', !active ? 0.6 : 0.08);
      }
    });
  });
}

export function updateForceGraph(layerData) {
  if (!_initialized || !_layerBlocks || _frozen) return;

  const palette = PALETTES[_colorMode] || PALETTES.attention;

  // Compute maximums for normalization
  const values = Object.values(layerData);
  const maxAttn = Math.max(...values.map(l => l.attention_energy || 0), 0.01);
  const maxMlp  = Math.max(...values.map(l => l.mlp_energy || 0), 0.01);
  const maxAct  = Math.max(...values.map(l => l.max_activation || 0), 0.01);
  const maxRes  = Math.max(...values.map(l => l.residual_norm || 0), 0.01);

  // Energy palette for Attention sub-blocks
  const attnScale = d3.scaleSequential(d3.interpolateInferno).domain([0, maxAttn]);
  const mlpScale  = d3.scaleSequential(d3.interpolateViridis).domain([0, maxMlp]);

  _layerBlocks.each(function(d) {
    const data = layerData[d.idx] || {};
    const el = d3.select(this);

    // ── Main block color based on mode ──
    let mainVal = 0;
    switch (_colorMode) {
      case 'attention':  mainVal = (data.attention_energy || 0) / maxAttn; break;
      case 'activation': mainVal = (data.max_activation || 0) / maxAct; break;
      case 'energy':     mainVal = (data.residual_norm || 0) / maxRes; break;
    }

    const mainColor = palette(mainVal);
    el.select('.layer-rect')
      .attr('fill', d3.color(mainColor).darker(2).toString())
      .attr('stroke', mainVal > 0.7 ? '#ff7700' : '#1a3a35')
      .attr('stroke-width', mainVal > 0.7 ? 1.5 : 1);

    // ── Sub-component fills ──
    const attnE = data.attention_energy || 0;
    const mlpE  = data.mlp_energy || 0;
    el.select('.sub-attn').attr('fill', attnE > 0 ? attnScale(attnE) : '#0a1816');
    el.select('.sub-mlp').attr('fill', mlpE > 0 ? mlpScale(mlpE) : '#0a1816');

    // Norm sub-block: bright if hot neurons
    const hotCount = (data.hot_neurons || []).length;
    el.select('.sub-norm').attr('fill', hotCount > 3 ? '#3a2000' : '#0a1816')
      .attr('stroke', hotCount > 5 ? '#ff7700' : '#1a3530');

    // ── Hot glow on the whole block ──
    if (hotCount > 5) {
      el.select('.layer-rect').attr('filter', 'url(#hot-glow)');
    } else {
      el.select('.layer-rect').attr('filter', mainVal > 0.5 ? 'url(#glow)' : null);
    }
  });

  // ── Update connection colors based on energy flow ──
  if (_connections) {
    _connections.attr('stroke', function(d) {
      if (d.type === 'output') return '#00e6ff';
      const targetIdx = parseInt(d.target.replace('layer_', ''), 10);
      const data = layerData[targetIdx];
      if (!data) return '#1a4a44';
      const norm = (data.residual_norm || 0) / maxRes;
      return d3.interpolateRgb('#1a4a44', '#00ffb0')(Math.min(norm, 1));
    })
    .attr('stroke-width', function(d) {
      if (d.type === 'output') return 2;
      const targetIdx = parseInt(d.target.replace('layer_', ''), 10);
      const data = layerData[targetIdx];
      if (!data) return 1.5;
      const norm = (data.residual_norm || 0) / maxRes;
      return 1.5 + norm * 2;
    })
    .attr('opacity', function(d) {
      if (d.type === 'output') return 0.9;
      const targetIdx = parseInt(d.target.replace('layer_', ''), 10);
      const data = layerData[targetIdx];
      if (!data) return 0.3;
      const norm = (data.residual_norm || 0) / maxRes;
      return 0.3 + norm * 0.5;
    });
  }
}
