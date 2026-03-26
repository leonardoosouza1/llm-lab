// frontend/src/charts/frame_heatmap.js

import { el, thermalColorRGB } from '../utils.js';
import { store } from '../store/index.js';

let _frameHeatmapCanvas = null;
let _frameHeatmapCtx = null;
let _frameImageData = null;
let _frameDirty = false;
let _frameRafId = null;
let _frameFpsCount = 0;
let _frameFpsLast = performance.now();

export function initFrameHeatmap() {
  // Prefer #frame-canvas (new layout), fall back to #heatmapCanvas (legacy)
  _frameHeatmapCanvas = el('frame-canvas') || el('heatmapCanvas');
  if (!_frameHeatmapCanvas) return;

  _frameHeatmapCtx = _frameHeatmapCanvas.getContext('2d');
  _startFrameLoop();
}

function _startFrameLoop() {
  function loop() {
    _frameRafId = requestAnimationFrame(loop);
    if (_frameDirty && store.currentFrame) {
      _drawFrame(store.currentFrame);
      _frameDirty = false;
    }
  }
  loop();
}

export function triggerFrameHeatmap() {
  _frameDirty = true;
}

function _drawFrame(frame) {
  if (!_frameHeatmapCtx || !frame || !frame.data) return;

  const size = frame.grid_size || Math.floor(Math.sqrt(frame.data.length));
  const canvas = _frameHeatmapCanvas;

  if (canvas.width !== size || canvas.height !== size) {
    canvas.width = size;
    canvas.height = size;
    _frameImageData = _frameHeatmapCtx.createImageData(size, size);
    const metricGrid = el('metricGrid');
    if (metricGrid) metricGrid.textContent = `${size}×${size}`;
  }

  if (!_frameImageData) {
    _frameImageData = _frameHeatmapCtx.createImageData(size, size);
  }

  const data = frame.data;
  const pixels = _frameImageData.data;
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < minVal) minVal = data[i];
    if (data[i] > maxVal) maxVal = data[i];
  }
  const range = maxVal - minVal || 1;

  for (let i = 0; i < data.length && i < size * size; i++) {
    const norm = (data[i] - minVal) / range;
    const [r, g, b] = thermalColorRGB(norm);
    const idx = i * 4;
    pixels[idx]     = r;
    pixels[idx + 1] = g;
    pixels[idx + 2] = b;
    pixels[idx + 3] = 255;
  }

  _frameHeatmapCtx.putImageData(_frameImageData, 0, 0);

  // FPS tracking
  _frameFpsCount++;
  const now = performance.now();
  if (now - _frameFpsLast >= 1000) {
    const fps = Math.round(_frameFpsCount * 1000 / (now - _frameFpsLast));
    const metricFps = el('metricFps');
    if (metricFps) metricFps.textContent = fps;
    _frameFpsCount = 0;
    _frameFpsLast = now;
  }
}