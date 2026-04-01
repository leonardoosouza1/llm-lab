// frontend/src/api/websocket.js
import { store } from '../store/index.js';

/**
 * WebSocket Manager for IARA Engine
 * Handles real-time traffic, reconnection, and dispatching.
 */
export const wsManager = {
  socket: null,
  reconnectDelay: 1000,
  _reconnectAttempt: 0,
  _backoffDelays: [1000, 2000, 4000, 8000, 16000, 30000],
  url: 'ws://localhost:8080',

  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    store.wsStatus = 'connecting';
    console.log(`🔌 Connecting to ${this.url}...`);

    try {
      this.socket = new WebSocket(this.url);
    } catch (e) {
      console.error('WS Connection error:', e);
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      store.wsStatus = 'connected';
      this._reconnectAttempt = 0;
      this.reconnectDelay = 1000;
      console.log('✅ Connected to IARA Engine');
      this.listLayers();
    };

    this.socket.onclose = () => {
      store.wsStatus = 'disconnected';
      console.log(`❌ Disconnected. Reconnecting in ${this.reconnectDelay}ms...`);
      this._scheduleReconnect();
    };

    this.socket.onerror = (err) => {
      console.error('WS Error:', err);
    };

    this.socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.dispatch(msg);
      } catch (e) {
        console.error('WS Parse Error:', e, event.data);
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

  send(obj) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('WS not connected. Cannot send:', obj);
      return;
    }
    this.socket.send(JSON.stringify(obj));
  },

  // Actions
  stimulus(text) { this.send({ type: 'stimulus', text }); },
  resonate(text) { this.send({ type: 'resonate', text }); },
  selectLayer(name) { this.send({ type: 'select_layer', name }); },
  listLayers() { this.send({ type: 'list_layers' }); },
  reset() { this.send({ type: 'reset' }); },

  /**
   * Dispatch messages to the global store
   */
  dispatch(msg) {
    switch (msg.type) {
      case 'status':
        store.loadingMsg = msg.message || '';
        break;

      case 'layers':
        store.availableLayers = msg.names || [];
        break;

      case 'frame':
        store.currentFrame = msg;
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
        break;

      case 'pipeline_result':
        store.topTokens = msg.top_tokens || [];
        store.pipelineComplete = true;
        // The UI will react via watchers in main.js
        break;

      case 'brain_map':
        store.brainMap = msg;
        break;

      case 'model_loaded':
        store.modelReady = true;
        store.modelLoading = false;
        store.modelName = msg.name || store.selectedModelName;
        break;

      case 'esn_state':
        store.esnLatest = msg;
        store.esnHistory.push({
          step:        msg.step,
          entropy:     msg.entropy,
          top1_prob:   msg.top1_prob,
          state_norm:  msg.state_norm,
          temp_delta:  msg.temp_delta,
          temperature: msg.temperature,
        });
        // Mantém histórico em 512 entradas (uma por token gerado)
        if (store.esnHistory.length > 512) store.esnHistory.shift();
        break;

      case 'gravity_state':
        store.gravityLatest = msg;
        store.gravityHistory.push({
          step:        msg.step,
          history_len: msg.history_len,
          threshold:   msg.threshold,
        });
        if (store.gravityHistory.length > 512) store.gravityHistory.shift();
        break;

      case 'cfc_state':
        store.cfcLatest = msg;
        break;

      case 'q4_status':
        store.q4Latest = msg;
        break;

      default:
        // console.log('Unhandled WS message type:', msg.type);
        break;
    }
  }
};
