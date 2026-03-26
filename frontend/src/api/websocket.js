// frontend/src/api/websocket.js

import { store } from '../store/index.js';
import { addLog, updateLegacyFrameUI, updatePipelineUI, renderLayerList } from '../dom/legacy_updates.js';
import { el } from '../utils.js';
import { triggerFrameHeatmap } from '../charts/frame_heatmap.js'; // Will be created later
import { updateModelUI, updateModelLoadingBanner, updateImageUploadArea } from '../components/model_loader.js'; // Will be created later

// Forward declaration for API to avoid circular dependency
let api = {};
export function setApiInstance(apiInstance) {
  api = apiInstance;
}

export const wsManager = {
  socket: null,
  reconnectDelay: 1000,
  _reconnectAttempt: 0,
  _backoffDelays: [1000, 2000, 4000, 8000, 16000, 30000],

  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    store.wsStatus = 'connecting';
    this._updateConnectionUI();
    addLog('ws-out', 'connect', `Connecting to ws://localhost:8080...`);

    try {
      this.socket = new WebSocket('ws://localhost:8080');
    } catch (e) {
      addLog('error', e.message, `WS constructor error: ${e.message}`);
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      store.wsStatus = 'connected';
      this._reconnectAttempt = 0;
      this.reconnectDelay = 1000;
      this._updateConnectionUI();
      addLog('ws-in', { type: 'connected' }, 'Connected to IARA engine');

      // Hide overlay if present
      const overlay = el('canvasOverlay');
      if (overlay) overlay.classList.add('hidden');

      // Request layer list
      this.listLayers();
    };

    this.socket.onclose = () => {
      store.wsStatus = 'disconnected';
      this._updateConnectionUI();
      addLog('error', { type: 'disconnected' }, `WS disconnected — reconnecting in ${this.reconnectDelay}ms`);
      const overlay = el('canvasOverlay');
      if (overlay) {
        overlay.classList.remove('hidden');
        const ot = overlay.querySelector('.overlay-text');
        if (ot) ot.textContent = 'Connection lost. Reconnecting...';
      }
      this._scheduleReconnect();
    };

    this.socket.onerror = () => {
      addLog('error', { type: 'ws-error' }, 'WebSocket error');
    };

    this.socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.dispatch(msg);
      } catch (e) {
        addLog('error', event.data, `Parse error: ${e.message}`);
      }
    };
  },

  _scheduleReconnect() {
    const delays = this._backoffDelays;
    const idx = Math.min(this._reconnectAttempt, delays.length - 1);
    this.reconnectDelay = delays[idx];
    this._reconnectAttempt++;
    setTimeout(() => this.connect(), this.reconnectDelay);
  },

  _updateConnectionUI() {
    const indicator = el('connectionIndicator');
    const text = el('connectionText');
    if (indicator) {
      indicator.className = 'status-dot ' + store.wsStatus;
    }
    if (text) {
      const labels = {
        connected: 'Connected',
        connecting: 'Connecting...',
        disconnected: 'Disconnected',
      };
      text.textContent = labels[store.wsStatus] || store.wsStatus;
    }
  },

  send(obj) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      addLog('error', obj, 'WS not connected — cannot send');
      return;
    }
    const str = JSON.stringify(obj);
    this.socket.send(str);
    addLog('ws-out', obj, `→ ${obj.type}${obj.text ? ': ' + obj.text : ''}`);
  },

  stimulus(text) { this.send({ type: 'stimulus', text }); },
  resonate(text) { this.send({ type: 'resonate', text }); },
  selectLayer(name) { this.send({ type: 'select_layer', name }); },
  listLayers() { this.send({ type: 'list_layers' }); },
  reset() { this.send({ type: 'reset' }); },

  dispatch(msg) {
    addLog('ws-in', msg, `← ${msg.type}`);

    switch (msg.type) {
      case 'status':
        addLog('ws-in', msg, `Status: ${msg.message}`);
        {
          const overlay = el('canvasOverlay');
          if (overlay) overlay.classList.add('hidden');
        }
        break;

      case 'layers':
        store.availableLayers = msg.names || [];
        addLog('ws-in', msg, `${store.availableLayers.length} layers available`);
        renderLayerList(store.availableLayers);
        break;

      case 'frame':
        store.currentFrame = msg;
        triggerFrameHeatmap();
        updateLegacyFrameUI(msg);
        break;

      case 'pipeline_layer':
        store.layerData[msg.layer_idx] = {
          attention_energy: msg.attention_energy,
          mlp_energy: msg.mlp_energy,
          residual_norm: msg.residual_norm,
          max_activation: msg.max_activation,
          hot_neurons: msg.hot_neurons || [],
          time_us: msg.time_us,
          spikes: msg.spikes,
          phase: msg.phase,
          total_layers: msg.total_layers,
        };
        updatePipelineUI(msg);
        break;

      case 'pipeline_result':
        store.topTokens = msg.top_tokens || [];
        store.pipelineComplete = true;
        // updatePipelineResultUI(msg); // This is Vue managed now via watch(store.topTokens)
        // Auto-fetch trace after 500ms
        setTimeout(() => api.fetchTrace(), 500);
        break;

      case 'brain_map':
        store.brainMap = msg;
        // renderBrainMap(msg); // This will be handled by Vue component later
        break;

      case 'loading':
        {
          const overlay = el('canvasOverlay');
          if (overlay) {
            overlay.classList.remove('hidden');
            const ot = overlay.querySelector('.overlay-text');
            if (ot) ot.textContent = msg.progress || 'Loading...';
          }
        }
        break;

      case 'generation_update':
        if (msg.data && msg.data.full_text) {
          store.responseText = msg.data.full_text;
        }
        break;

      case 'status':
        {
          const message = msg.message || '';
          store.loadingMsg = message;
          const msgLow = message.toLowerCase();
          if (msgLow.includes('carregad') || msgLow.includes('pronto')) {
            store.modelReady = true;
            store.modelLoading = false;
          } else if (msgLow.includes('carregando') || msgLow.includes('loading')) {
            store.modelLoading = true;
          }
          updateModelLoadingBanner();
          updateModelUI();
          addLog('ws-in', msg, `Status: ${message}`);
        }
        break;

      case 'model_loaded':
        {
          store.modelReady = true;
          store.modelLoading = false;
          store.modelName = msg.name || store.selectedModelName;
          store.loadingMsg = msg.message || `✅ ${store.modelName} pronto!`;

          const loadedModel = store.availableModels.find(function(m) { return m.name === msg.name; });
          if (loadedModel) {
            store.modelKind = loadedModel.model_kind || 'unknown';
            store.hasVision = loadedModel.has_vision || false;
          }
          updateImageUploadArea();

          updateModelUI();
          updateModelLoadingBanner();
          // Enable prompt input
          const promptEl = el('prompt-input');
          const sendEl = el('send-btn');
          if (promptEl) { promptEl.disabled = false; promptEl.placeholder = 'Digite seu prompt...'; }
          if (sendEl) sendEl.disabled = false;
          addLog('ws-in', msg, `Model loaded: ${msg.name}`);
        }
        break;

      default:
        addLog('ws-in', msg, `Unknown message type: ${msg.type}`);
        break;
    }
  },
};
