// frontend/src/charts/force_graph.js

import { el } from '../utils.js';
import { store } from '../store/index.js';

let _forceSimulation = null;
let _forceNodes = [];
let _forceLinks = [];
let _forceSvgG = null;
let _forceNodeElements = null;
let _forceLinkElements = null;
let _forceInitialized = false;

export function isForceGraphInitialized() {
  return _forceInitialized;
}

export function initForceGraph() {
  const svgEl = el('force-graph-svg');
  if (!svgEl || typeof d3 === 'undefined' || _forceInitialized) return;
  _forceInitialized = true;

  const width  = svgEl.clientWidth  || 600;
  const height = svgEl.clientHeight || 400;

  const svg = d3.select('#force-graph-svg');
  svg.selectAll('*').remove();

  // Zoom + pan
  const g = svg.append('g').attr('class', 'force-root');
  _forceSvgG = g;

  svg.call(d3.zoom()
    .scaleExtent([0.3, 3])
    .on('zoom', function(event) {
      g.attr('transform', event.transform);
    })
  );

  // Defs: glow filter
  const defs = svg.append('defs');
  const filter = defs.append('filter').attr('id', 'glow');
  filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
  const feMerge = filter.append('feMerge');
  feMerge.append('feMergeNode').attr('in', 'coloredBlur');
  feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

  // Nodes: 36 transformer layers
  _forceNodes = Array.from({ length: 36 }, (_, i) => ({
    id: i,
    label: `L${i}`,
    radius: 10,
    energy: 0,
    hot: false,
  }));

  // Links: sequential residual stream
  _forceLinks = Array.from({ length: 35 }, (_, i) => ({
    source: i,
    target: i + 1,
    strength: 0.5,
  }));

  const colorScale = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);

  _forceSimulation = d3.forceSimulation(_forceNodes)
    .force('link', d3.forceLink(_forceLinks).id(d => d.id).distance(60))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collide', d3.forceCollide(d => d.radius + 6));

  // Draw links
  _forceLinkElements = g.append('g').attr('class', 'links')
    .selectAll('line')
    .data(_forceLinks)
    .enter()
    .append('line')
    .attr('stroke', '#1a2035')
    .attr('stroke-width', 1.5)
    .attr('stroke-opacity', 0.6);

  // Draw node groups
  const nodeG = g.append('g').attr('class', 'nodes')
    .selectAll('g')
    .data(_forceNodes)
    .enter()
    .append('g')
    .attr('class', 'node-group')
    .call(d3.drag()
      .on('start', function(event, d) {
        if (!event.active) _forceSimulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', function(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', function(event, d) {
        if (!event.active) _forceSimulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      })
    );

  // Hot glow ring (initially hidden)
  nodeG.append('circle')
    .attr('class', 'glow-ring')
    .attr('r', d => d.radius + 5)
    .attr('fill', 'none')
    .attr('stroke', '#ff7700')
    .attr('stroke-width', 2)
    .attr('opacity', 0);

  // Main circle
  nodeG.append('circle')
    .attr('class', 'node-circle')
    .attr('r', d => d.radius)
    .attr('fill', colorScale(0))
    .attr('filter', 'url(#glow)');

  // Label
  nodeG.append('text')
    .attr('class', 'node-label')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', 9)
    .attr('font-family', 'JetBrains Mono, monospace')
    .attr('fill', '#e8e8f0')
    .attr('pointer-events', 'none')
    .text(d => d.label);

  _forceNodeElements = nodeG;

  _forceSimulation.on('tick', function() {
    _forceLinkElements
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    _forceNodeElements
      .attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

export function updateForceGraph(layerData) {
  if (!_forceInitialized || !_forceNodeElements) return;

  const colorScale = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);

  const maxAttn = Math.max(
    ...Object.values(layerData).map(l => l.attention_energy || 0),
    1
  );
  const maxAct = Math.max(
    ...Object.values(layerData).map(l => l.max_activation || 0),
    1
  );

  _forceNodes.forEach(function(node) {
    const d = layerData[node.id] || {};
    node.energy  = (d.attention_energy || 0) / maxAttn;
    node.hot     = (d.hot_neurons || []).length > 5;
    node.radius  = 8 + (d.max_activation || 0) / maxAct * 15;
  });

  _forceNodeElements.select('.node-circle')
    .attr('r', d => d.radius)
    .attr('fill', d => colorScale(d.energy));

  _forceNodeElements.select('.glow-ring')
    .attr('r', d => d.radius + 5)
    .attr('opacity', d => d.hot ? 0.8 : 0);

  _forceSimulation
    .force('collide', d3.forceCollide(d => d.radius + 6))
    .alpha(0.3)
    .restart();
}