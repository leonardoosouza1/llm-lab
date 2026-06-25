/**
 * NeurViz 3D — Neural activation visualizer integrated into the IARA dashboard.
 * Adapted from neuroviz.html — uses Three.js r128 (loaded globally via CDN in index.html).
 * Connects to POST /api/neuroviz for real data; falls back gracefully when model not loaded.
 */
'use strict';

// ── Module-level state ────────────────────────────────────────────────────────
let _initialized = false;
let vizData = null;
let selectedTokenIdx = 0;
let selectedLayerIdx = -1;
let highlightedNeuronIdx = null;
let charpathMode = false;

let renderer, scene, camera, controls;
let neuronMeshes = [];
let layerPlanes = [];
let residualLines = [];
let highlightLines = [];
let _animFrameId = null;

// ── Color helpers (IARA palette: #00E6FF = accent-blue, #22c55e = green) ─────
function actColor(v, maxV) {
  const t = Math.min(1, v / (maxV + 1e-6));
  // blue → cyan → green → yellow — matches IARA thermalColor spirit
  const r = Math.min(1, t * 2.5);
  const g = Math.max(0, Math.min(1, t * 2.0 - 0.3));
  const b = Math.max(0, 1.0 - t * 2.0);
  return new THREE.Color(r, g, b);
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Three.js init ─────────────────────────────────────────────────────────────
function initThree() {
  const canvas = document.getElementById('nv-three-canvas');
  const wrapper = document.getElementById('nv-canvas-wrapper');
  if (!canvas || !wrapper) return;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0a0e1a, 1);

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  camera.position.set(0, 8, 20);

  controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0x1a2a3a, 2.0));
  const dir = new THREE.DirectionalLight(0x00e6ff, 0.7);
  dir.position.set(5, 10, 5);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0x22c55e, 0.3);
  fill.position.set(-3, -5, 3);
  scene.add(fill);

  function resize() {
    const w = wrapper.clientWidth, h = wrapper.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  // Store resize handler so we can remove it if needed
  wrapper._nvResize = resize;
  window.addEventListener('resize', resize);
  resize();

  // Raycasting for click / hover
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const meshes = neuronMeshes.map(n => n.mesh);
    const hits = raycaster.intersectObjects(meshes);
    if (hits.length > 0) {
      const found = neuronMeshes.find(n => n.mesh === hits[0].object);
      if (found) onNeuronClick(found);
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const meshes = neuronMeshes.map(n => n.mesh);
    const hits = raycaster.intersectObjects(meshes);
    const tip = document.getElementById('nv-tooltip');
    if (!tip) return;
    if (hits.length > 0) {
      const found = neuronMeshes.find(n => n.mesh === hits[0].object);
      if (found) {
        tip.style.display = 'block';
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top = (e.clientY - 22) + 'px';
        const tokData = vizData && vizData.tokens[selectedTokenIdx];
        const lyr = tokData && tokData.layers.find(l => l.layer_idx === found.layerIdx);
        tip.innerHTML =
          `<b>Layer ${found.layerIdx}</b><br>Neuron #${found.neuronIdx}<br>` +
          `Activation: <span style="color:#00e6ff">${found.activation.toFixed(4)}</span>` +
          (lyr && lyr.mlp_activation_rate !== undefined
            ? `<br>Act.rate: <span style="color:#22c55e">${lyr.mlp_activation_rate}%</span>` +
              `<br>Phase: <span style="color:#818cf8">${lyr.phase_coherence}</span>` : '');
      }
    } else {
      tip.style.display = 'none';
    }
  });

  function animate() {
    _animFrameId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

// ── Scene helpers ─────────────────────────────────────────────────────────────
function clearScene() {
  neuronMeshes.forEach(n => { scene.remove(n.mesh); n.mesh.geometry.dispose(); n.mesh.material.dispose(); });
  layerPlanes.forEach(p => { scene.remove(p); p.geometry.dispose(); p.material.dispose(); });
  residualLines.forEach(l => { scene.remove(l); l.geometry.dispose(); l.material.dispose(); });
  clearHighlightLines();
  neuronMeshes = []; layerPlanes = []; residualLines = [];
}

function clearHighlightLines() {
  highlightLines.forEach(l => { scene.remove(l); l.geometry.dispose(); l.material.dispose(); });
  highlightLines = [];
}

// ── Build 3D scene from token data ───────────────────────────────────────────
function buildScene(tokenData) {
  clearScene();
  if (!tokenData || !tokenData.layers || tokenData.layers.length === 0) return;

  const layers = tokenData.layers;
  const N = layers.length;
  const LAYER_SPACING = 0.55;
  const COLS = 16;
  const DOT_SIZE = 0.08;
  const DOT_SPACING = 0.22;

  const geo = new THREE.SphereGeometry(DOT_SIZE, 7, 7);
  const planeGeo = new THREE.PlaneGeometry(COLS * DOT_SPACING + 0.2, 2.8);
  const planeMat = new THREE.MeshBasicMaterial({
    color: 0x0a1c2e, transparent: true, opacity: 0.22, side: THREE.DoubleSide
  });

  const labelsDiv = document.getElementById('nv-layer-labels');
  if (labelsDiv) labelsDiv.innerHTML = '';

  layers.forEach((layer, li) => {
    const y = (li - N / 2) * LAYER_SPACING;

    const plane = new THREE.Mesh(planeGeo, planeMat.clone());
    plane.position.set(0, y, 0);
    plane.rotation.x = -Math.PI / 2;
    scene.add(plane);
    layerPlanes.push(plane);

    if (li > 0) {
      const prevY = ((li - 1) - N / 2) * LAYER_SPACING;
      const norm = layer.residual_norm !== undefined ? layer.residual_norm : 1;
      const opacity = Math.min(0.55, 0.12 + norm / 12);
      const pts = [new THREE.Vector3(0, prevY, 0), new THREE.Vector3(0, y, 0)];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lineMat = new THREE.LineBasicMaterial({ color: 0x00e6ff, transparent: true, opacity });
      const line = new THREE.Line(lineGeo, lineMat);
      scene.add(line);
      residualLines.push(line);
    }

    const neuronMap = new Map();
    if (layer.hot_neurons) layer.hot_neurons.forEach((idx, k) => neuronMap.set(idx, (layer.hot_neuron_values && layer.hot_neuron_values[k]) || layer.max_activation || 1.0));
    if (layer.mlp_top) layer.mlp_top.forEach(([idx, v]) => neuronMap.set(idx, v));

    const displayCount = 48;
    const activatedSet = new Set(neuronMap.keys());
    for (let f = 0; neuronMap.size < displayCount && f < 2000; f++) {
      const randIdx = Math.floor(Math.random() * 10000);
      if (!activatedSet.has(randIdx)) neuronMap.set(randIdx, 0.005 + Math.random() * 0.05);
    }

    const neurons = [...neuronMap.entries()].sort((a, b) => b[1] - a[1]);
    const maxInLayer = neurons[0] ? neurons[0][1] : 1;

    neurons.slice(0, displayCount).forEach(([idx, val], ni) => {
      const col = ni % COLS;
      const row = Math.floor(ni / COLS);
      const x = (col - COLS / 2) * DOT_SPACING;
      const z = (row - 1) * DOT_SPACING;

      const color = actColor(val, maxInLayer);
      const mat = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.4, shininess: 60 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, y, z);
      mesh.scale.setScalar(0.45 + (val / (maxInLayer + 1e-6)) * 1.6);

      scene.add(mesh);
      neuronMeshes.push({ mesh, layerIdx: layer.layer_idx, neuronIdx: idx, activation: val });
    });

    if (labelsDiv) {
      const lbl = document.createElement('div');
      lbl.className = 'nv-layer-label';
      lbl.textContent = li % 4 === 0 ? `L${layer.layer_idx}` : '';
      labelsDiv.appendChild(lbl);
    }
  });
}

// ── Highlight neuron path across layers ───────────────────────────────────────
function drawNeuronPath(neuronIdx) {
  clearHighlightLines();
  const positions = [];
  neuronMeshes.forEach(nm => {
    if (nm.neuronIdx === neuronIdx) {
      positions.push({ y: nm.mesh.position.y, x: nm.mesh.position.x, z: nm.mesh.position.z });
    }
  });
  positions.sort((a, b) => a.y - b.y);
  if (positions.length < 2) return;

  const pts = positions.map(p => new THREE.Vector3(p.x, p.y, p.z));
  const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.9 });
  const line = new THREE.Line(lineGeo, lineMat);
  scene.add(line);
  highlightLines.push(line);
}

// ── Stats overlay ─────────────────────────────────────────────────────────────
function updateStats(layer) {
  const overlay = document.getElementById('nv-stats-overlay');
  if (!overlay) return;
  if (!layer) { overlay.style.display = 'none'; return; }
  overlay.style.display = 'block';
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set('nv-stat-layer', `L${layer.layer_idx}`);
  set('nv-stat-actrate', layer.mlp_activation_rate !== undefined ? `${layer.mlp_activation_rate}%` : 'n/a');
  set('nv-stat-phase', layer.phase_coherence !== undefined ? layer.phase_coherence : 'n/a');
  set('nv-stat-resnorm', layer.residual_norm !== undefined ? layer.residual_norm : 'n/a');
  set('nv-stat-entropy', layer.attn_entropy !== undefined ? layer.attn_entropy : 'n/a');
  set('nv-stat-maxact', layer.max_activation !== undefined ? layer.max_activation.toFixed(4) : 'n/a');
}

// ── Bottom bar: neuron chips + logit chips ─────────────────────────────────────
function renderBottom(tokenData, layerIdx) {
  const chipsDiv = document.getElementById('nv-neuron-chips');
  if (chipsDiv) chipsDiv.innerHTML = '';
  const layerInfo = document.getElementById('nv-layer-info');
  if (layerInfo) layerInfo.textContent = layerIdx >= 0 ? `Layer ${layerIdx}` : 'hover a layer';

  let layer = null;
  if (tokenData && layerIdx >= 0) layer = tokenData.layers.find(l => l.layer_idx === layerIdx);
  if (!layer && tokenData && tokenData.layers.length > 0) layer = tokenData.layers[tokenData.layers.length - 1];

  if (layer && chipsDiv) {
    const top = layer.mlp_top && layer.mlp_top.length > 0
      ? layer.mlp_top
      : (layer.hot_neurons || []).map((i, k) => [i, (layer.hot_neuron_values && layer.hot_neuron_values[k]) || layer.max_activation || 1.0]);
    top.slice(0, 20).forEach(([idx, val]) => {
      const chip = document.createElement('div');
      chip.className = 'nv-neuron-chip' + (idx === highlightedNeuronIdx ? ' highlighted' : '');
      chip.innerHTML = `#${idx}<span class="nv-chip-val">${val.toFixed(2)}</span>`;
      chip.title = `Click to trace neuron #${idx}`;
      chip.addEventListener('click', () => {
        highlightedNeuronIdx = (highlightedNeuronIdx === idx) ? null : idx;
        if (highlightedNeuronIdx !== null) drawNeuronPath(highlightedNeuronIdx);
        else clearHighlightLines();
        renderBottom(tokenData, layerIdx);
      });
      chipsDiv.appendChild(chip);
    });
  }

  const logitsDiv = document.getElementById('nv-logit-chips');
  if (logitsDiv) {
    logitsDiv.innerHTML = '';
    if (tokenData && tokenData.top_logits) {
      tokenData.top_logits.slice(0, 12).forEach(([id, prob, str]) => {
        const chip = document.createElement('div');
        chip.className = 'nv-logit-chip';
        const display = (str || `[${id}]`).replace(/\n/g, '↵').replace(/ /g, '·');
        chip.innerHTML = `"${escHtml(display)}"<span class="nv-logit-pct">${(prob * 100).toFixed(1)}%</span>`;
        logitsDiv.appendChild(chip);
      });
    }
  }
}

// ── Character path overlay ────────────────────────────────────────────────────
function showCharpathOverlay(tokenData) {
  if (!tokenData) return;
  const overlay = document.getElementById('nv-charpath-overlay');
  const body = document.getElementById('nv-charpath-body');
  if (!overlay || !body) return;
  overlay.style.display = 'block';

  const tok = tokenData.token_str;
  const chars = [...tok].filter(c => c.trim());
  const N = tokenData.layers.length;
  const anchorLayers = [0, Math.floor(N / 3), Math.floor(2 * N / 3), N - 1]
    .filter((v, i, a) => a.indexOf(v) === i);

  let html = `<b>Token: "${escHtml(tok)}"</b>&nbsp;`;
  chars.forEach(c => {
    html += `<span style="background:#0a2a0a;border:1px solid #22c55e;padding:1px 5px;border-radius:4px;margin:0 1px;">${escHtml(c)}</span>`;
  });
  html += '<br><br><b>Top neurons at anchor layers:</b><br>';
  anchorLayers.forEach(li => {
    const lyr = tokenData.layers[li];
    if (!lyr || !lyr.mlp_top) return;
    const topN = lyr.mlp_top.slice(0, 3).map(([idx, v]) => `#${idx}(${v.toFixed(2)})`).join(', ');
    html += `<span style="color:#22c55e">L${li}</span> → <span style="color:#00e6ff">${topN}</span><br>`;
  });
  body.innerHTML = html;

  const lastLayer = tokenData.layers[N - 1];
  if (lastLayer && lastLayer.mlp_top && lastLayer.mlp_top.length > 0) {
    highlightedNeuronIdx = lastLayer.mlp_top[0][0];
    drawNeuronPath(highlightedNeuronIdx);
  }
}

// ── Neuron click ──────────────────────────────────────────────────────────────
function onNeuronClick(neuronObj) {
  selectedLayerIdx = neuronObj.layerIdx;
  highlightedNeuronIdx = neuronObj.neuronIdx;
  drawNeuronPath(neuronObj.neuronIdx);
  const tokenData = vizData && vizData.tokens[selectedTokenIdx];
  const layer = tokenData && tokenData.layers.find(l => l.layer_idx === neuronObj.layerIdx);
  if (layer) {
    renderAttentionD3(layer);
    renderBottom(tokenData, neuronObj.layerIdx);
    updateStats(layer);
  }
}

// ── Token selection ───────────────────────────────────────────────────────────
function selectToken(idx) {
  selectedTokenIdx = idx;
  highlightedNeuronIdx = null;
  document.querySelectorAll('.nv-token-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });
  if (!vizData) return;
  const tokenData = vizData.tokens[idx];
  if (!tokenData) return;

  // If this token has no layer data (prompt token), try to find nearest generated token
  if (!tokenData.layers || tokenData.layers.length === 0) {
    const status = document.getElementById('nv-status');
    if (status) status.textContent = 'prompt token — sem layer data (clique num token gerado)';
    clearScene();
    updateStats(null);
    return;
  }

  buildScene(tokenData);
  const layer = tokenData.layers[tokenData.layers.length - 1];
  renderAttentionD3(layer);
  renderBottom(tokenData, layer ? layer.layer_idx : -1);
  updateStats(layer || null);
  if (charpathMode) showCharpathOverlay(tokenData);
}

// ── Attention D3 bars ─────────────────────────────────────────────────────────
function renderAttentionD3(layer) {
  const svg = d3.select('#nv-attn-svg');
  if (!svg || !svg.node()) return;
  svg.selectAll('*').remove();

  if (!layer || !layer.attn_top || layer.attn_top.length === 0) {
    svg.append('text').attr('x', 5).attr('y', 20).attr('fill', '#334155').attr('font-size', 11).text('no data');
    return;
  }

  const barH = 16, pad = 4, maxW = 160;
  svg.attr('height', layer.attn_top.length * (barH + pad) + 24);
  const maxWeight = layer.attn_top[0][1] || 1;

  layer.attn_top.forEach(([pos, weight], i) => {
    const g = svg.append('g').attr('transform', `translate(0,${i * (barH + pad) + 4})`);
    g.append('rect').attr('width', maxW).attr('height', barH).attr('rx', 2).attr('fill', '#0f172a').attr('opacity', 0.7);
    g.append('rect').attr('width', (weight / maxWeight) * maxW).attr('height', barH).attr('rx', 2)
      .attr('fill', `hsl(${200 + i * 15},70%,50%)`).attr('opacity', 0.85);
    const tokStr = vizData && vizData.tokens[pos] ? escHtml(vizData.tokens[pos].token_str) : `pos ${pos}`;
    g.append('text').attr('x', Math.max((weight / maxWeight) * maxW + 4, 60)).attr('y', 12)
      .attr('fill', '#475569').attr('font-size', 10).text(tokStr.slice(0, 9));
    g.append('text').attr('x', 3).attr('y', 12).attr('fill', '#e2e8f0').attr('font-size', 10)
      .text((weight * 100).toFixed(1) + '%');
  });

  const infoEl = document.getElementById('nv-attn-info');
  if (infoEl) {
    const ent = layer.attn_entropy !== undefined ? layer.attn_entropy.toFixed(3) : 'n/a';
    let html = `entropy: <span style="color:#00e6ff">${ent}</span>`;
    if (layer.residual_delta !== undefined) {
      html += `&nbsp;&nbsp;delta: <span style="color:#22c55e">${layer.residual_delta.toFixed(3)}</span>`;
    }
    infoEl.innerHTML = html;
  }
}

// ── API call ──────────────────────────────────────────────────────────────────
export async function runNeurovizAnalysis() {
  const promptEl = document.getElementById('nv-prompt-input');
  if (!promptEl) return;
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  const btn = document.getElementById('nv-run-btn');
  const status = document.getElementById('nv-status');
  const loading = document.getElementById('nv-loading');

  if (btn) btn.disabled = true;
  if (status) status.textContent = 'analyzing…';
  if (loading) loading.style.display = 'flex';

  charpathMode = false;
  const cpBtn = document.getElementById('nv-charpath-btn');
  if (cpBtn) cpBtn.classList.remove('active');
  const cpOverlay = document.getElementById('nv-charpath-overlay');
  if (cpOverlay) cpOverlay.style.display = 'none';

  try {
    const resp = await fetch(`${window.location.origin}/api/neuroviz`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_tokens: 30 })
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || resp.statusText);
    }

    vizData = await resp.json();
    const info = vizData.model_info || {};
    const genText = vizData.generated_text || '';
    if (status) status.textContent =
      `${vizData.tokens.length} tokens · ${info.num_layers || '?'}L · ${info.hidden_size || '?'}h`;

    // Show generated response prominently
    const responseEl = document.getElementById('nv-response-text');
    if (responseEl) {
      responseEl.textContent = genText || '(no response)';
      responseEl.style.color = genText ? '#00e6ff' : '#334155';
    }

    const listDiv = document.getElementById('nv-token-list');
    if (listDiv) {
      listDiv.innerHTML = '<div class="nv-token-header">TOKENS</div>';
      vizData.tokens.forEach((tok, i) => {
        const el = document.createElement('div');
        // Mark prompt vs generated tokens differently
        const isGenerated = tok.layers && tok.layers.length > 0;
        el.className = 'nv-token-item' + (isGenerated ? ' nv-token-generated' : '');
        el.title = isGenerated ? 'generated token (has layer data)' : 'prompt token';
        el.innerHTML =
          `<div class="nv-token-pos">pos ${tok.position}</div>` +
          `<div>${escHtml(tok.token_str || `[${tok.token_id}]`)}</div>`;
        el.addEventListener('click', () => selectToken(i));
        listDiv.appendChild(el);
      });
    }

    // Auto-select first generated token (has layer data), not first prompt token
    const firstGenIdx = vizData.tokens.findIndex(t => t.layers && t.layers.length > 0);
    selectToken(firstGenIdx >= 0 ? firstGenIdx : 0);

  } catch (e) {
    if (status) status.textContent = 'error — ' + e.message.slice(0, 50);
    console.error('[neuroviz] fetch error:', e);
  } finally {
    if (btn) btn.disabled = false;
    if (loading) loading.style.display = 'none';
  }
}

// ── Public: init tab (called once when tab first activated) ───────────────────
export function initNeurovizTab() {
  if (_initialized) {
    // Already initialized — just resize renderer to fit current panel size
    if (renderer) {
      const wrapper = document.getElementById('nv-canvas-wrapper');
      if (wrapper) {
        const w = wrapper.clientWidth, h = wrapper.clientHeight;
        if (w > 0 && h > 0) {
          renderer.setSize(w, h, false);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        }
      }
    }
    return;
  }

  if (typeof THREE === 'undefined') {
    console.warn('[neuroviz] THREE.js not loaded yet');
    return;
  }

  _initialized = true;
  initThree();

  // Wire up buttons
  const runBtn = document.getElementById('nv-run-btn');
  if (runBtn) runBtn.addEventListener('click', runNeurovizAnalysis);

  const promptEl = document.getElementById('nv-prompt-input');
  if (promptEl) promptEl.addEventListener('keydown', e => { if (e.key === 'Enter') runNeurovizAnalysis(); });

  const cpBtn = document.getElementById('nv-charpath-btn');
  if (cpBtn) cpBtn.addEventListener('click', () => {
    charpathMode = !charpathMode;
    cpBtn.classList.toggle('active', charpathMode);
    const overlay = document.getElementById('nv-charpath-overlay');
    if (charpathMode) {
      const tokenData = vizData && vizData.tokens[selectedTokenIdx];
      if (tokenData) showCharpathOverlay(tokenData);
    } else {
      if (overlay) overlay.style.display = 'none';
      highlightedNeuronIdx = null;
      clearHighlightLines();
    }
  });
}
