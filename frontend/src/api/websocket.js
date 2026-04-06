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
   * Log a message to the console
   */
  _logMessage(type, summary, raw) {
    const now = new Date();
    const ts = now.toTimeString().split(' ')[0]; // HH:MM:SS
    const entry = {
      id: `${Date.now()}-${Math.random()}`,
      ts: ts,
      type: type,
      summary: summary,
      raw: raw,
    };
    store.logs.push(entry);
    // Keep log buffer bounded (5000 entries)
    if (store.logs.length > 5000) store.logs.shift();
  },

  /**
   * Dispatch messages to the global store
   */
  dispatch(msg) {
    // Log ALL incoming messages
    const summary = msg.type + (msg.step !== undefined ? ` #${msg.step}` : '');
    this._logMessage('ws-in', summary, msg);

    switch (msg.type) {
      case 'status':
        store.loadingMsg = msg.message || '';
        // Detecta conclusão ou erro da geração via WS para resetar store.generating
        if (msg.message && (msg.message.includes('Geração completa') || msg.message.startsWith('❌'))) {
          store.generating = false;
        }
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
        // Actualiza contagem real de camadas para evitar hardcode de 36
        if (msg.total_layers && msg.total_layers > 0) {
          store.numLayers = msg.total_layers;
        }
        break;

      case 'pipeline_result': {
        store.topTokens = msg.top_tokens || [];
        store.pipelineComplete = true;
        // Atualiza texto gerado em tempo real (streaming WS)
        if (msg.generated_so_far) {
          store.responseText = msg.generated_so_far;
        }
        // Métricas de performance
        if (msg.total_time_ms) {
          store.totalDurationMs = msg.total_time_ms;
        }
        store.evalCount = (msg.step || 0) + 1;
        if (store.evalCount > 0 && store.totalDurationMs > 0) {
          store.tokensPerSec = store.evalCount / (store.totalDurationMs / 1000);
        }
        // Compute ms/layer from layer telemetry
        const layerTimes = Object.values(store.layerData).map(l => l.time_us || 0).filter(t => t > 0);
        if (layerTimes.length > 0) {
          store.msPerLayer = (layerTimes.reduce((a, b) => a + b, 0) / layerTimes.length) / 1000;
        }

        // Constrói trace token-by-token para aba Trace
        const avgLayerEnergy = Object.keys(store.layerData).length > 0
          ? Object.values(store.layerData).reduce((s, l) => s + (l.attention_energy || 0) + (l.mlp_energy || 0), 0)
            / Object.keys(store.layerData).length
          : 0;
        const topLogit = msg.top_tokens && msg.top_tokens[0] ? msg.top_tokens[0].score : null;
        const topProb  = msg.top_tokens && msg.top_tokens[0] ? Math.exp(topLogit || 0) : null; // approx
        const esnEnt   = store.esnLatest ? store.esnLatest.entropy : null;
        const isAnomaly = avgLayerEnergy > 0
          && store.traceTokens.length > 2
          && avgLayerEnergy > (store.traceTokens.slice(-3).reduce((s, t) => s + (t.energy || 0), 0) / 3) * 2.5;

        store.traceTokens.push({
          step:      msg.step || 0,
          token:     msg.token_str || '',
          energy:    avgLayerEnergy,
          top_logit: topLogit,
          prob:      topProb,
          ms_token:  msg.total_time_ms || 0,
          entropy:   esnEnt,
          anomaly:   isAnomaly,
        });
        // Mantém no máximo 512 tokens no trace
        if (store.traceTokens.length > 512) store.traceTokens.shift();
        break;
      }

      case 'brain_map':
        store.brainMap = msg;
        break;

      case 'model_loaded':
        store.modelReady = true;
        store.modelLoading = false;
        store.modelName = msg.name || store.selectedModelName;
        // Update backend and quantization info if available
        if (msg.backend) store.backend = msg.backend;
        if (msg.quantization) store.quantization = msg.quantization;
        break;

      case 'esn_state':
        store.esnLatest = msg;
        store.esnActive = true;
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
        store.gravityActive = true;
        store.gravityHistory.push({
          step:        msg.step,
          history_len: msg.history_len,
          threshold:   msg.threshold,
          mean_gravity: msg.mean_gravity,
          max_gravity: msg.max_gravity,
        });
        if (store.gravityHistory.length > 512) store.gravityHistory.shift();
        break;

      case 'cfc_state':
        store.cfcLatest = msg;
        store.cfcActive = true;
        break;

      case 'q4_status':
        store.q4Latest = msg;
        if (msg.active) store.quantization = 'q4';
        break;

      case 'sae_state':
        store.sae.active = true;
        store.sae.layer = msg.layer;
        store.sae.activeFeatures = msg.active_features || 0;
        store.sae.topFeatures = msg.top_features || [];
        store.sae.sparsity = msg.sparsity || 0;
        store.sae.loss = msg.loss || 0;
        store.sae.step = msg.step || 0;
        store.sae.history.push({
          step: msg.step,
          loss: msg.loss,
          sparsity: msg.sparsity,
          activeFeatures: msg.active_features,
        });
        if (store.sae.history.length > 512) store.sae.history.shift();
        break;

      case 'ntc_state':
        store.ntc.active = true;
        store.ntc.gridSize = msg.grid_size || 0;
        store.ntc.latentDim = msg.latent_dim || 0;
        store.ntc.vramMb = msg.memory_bytes ? Math.round(msg.memory_bytes / 1024 / 1024) : 0;
        store.ntc.compressedTokens = msg.current_len || 0;
        store.ntc.recentTokens = msg.recent_window || 0;
        store.ntc.reconstructionLoss = msg.reconstruction_loss || 0;
        store.ntc.history.push({
          step: msg.step,
          reconstructionLoss: msg.reconstruction_loss,
          vramMb: store.ntc.vramMb,
          compressionRatio: msg.compression_ratio,
        });
        if (store.ntc.history.length > 512) store.ntc.history.shift();
        break;

      case 'quant_status':
        store.quant.mode = msg.mode || 'bf16';
        store.quant.vramMb = msg.vram_mb || 0;
        store.quant.baselineVramMb = msg.baseline_vram_mb || 0;
        store.quant.nrmse = msg.nrmse || 0;
        store.quant.perLayerError = msg.per_layer_error || [];
        break;

      default:
        // Still log unhandled types
        break;
    }
  }
};
