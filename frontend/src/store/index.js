// frontend/src/store/index.js

// Import Vue reactive if available globally (assuming CDN load)
const reactive = (typeof Vue !== 'undefined') ? Vue.reactive : function(o) { return o; };

export const store = reactive({
  // Connection
  wsStatus: 'disconnected',
  httpHealthy: false,
  modelName: 'none',
  modelReady: false,
  modelLoading: false,
  loadingMsg: 'Aguardando servidor...',
  availableModels: [],     // [{name, hidden_size, num_layers, ...}]
  selectedModelName: '',   // currently selected in dropdown (not yet loaded)
  modelKind: 'unknown',    // 'causal_lm' | 'vlm' | 'unknown'
  hasVision: false,
  selectedImageBase64: null,

  // UI
  activeTab: 'live',
  availableLayers: [],
  selectedLayer: null,

  // Live panel
  prompt: '',
  maxTokens: 256,
  temperature: 0.6,
  generating: false,
  responseText: '',
  evalCount: 0,
  totalDurationMs: 0,

  // Per-layer data keyed by layer_idx
  layerData: {},

  // Pipeline result
  topTokens: [],
  pipelineComplete: false,

  // Current WS frame
  currentFrame: null,

  // Trace
  trace: null,
  anomalyCount: 0,

  // Performance
  tokensPerSec: 0,
  msPerLayer: 0,

  // Console logs
  logs: [],
  logFilter: 'ALL',

  // Brain map
  brainMap: null,

  // ESN (Fase 1) — observador do stream de tokens
  esnHistory: [],   // [{step, entropy, top1_prob, state_norm, temp_delta, temperature}]
  esnLatest: null,  // último frame ESN

  // Gravity (Fase 2) — gravidade semântica
  gravityLatest: null,   // último frame gravity_state
  gravityHistory: [],    // [{step, history_len, threshold}]

  // CfC (Fase 2) — Liquid Correction Layer
  cfcLatest: null,       // último frame cfc_state

  // Q4 Quantization status
  q4Latest: null,        // último frame q4_status

  // Subsystem Status
  backend: 'unknown',    // 'rocm' | 'opencl' | 'cpu'
  quantization: 'bf16',  // 'bf16' | 'q4' | 'f32'
  esnActive: false,
  gravityActive: false,
  cfcActive: false,

  // Dynamic model info (set from pipeline_layer.total_layers)
  numLayers: 28,

  // Trace tab — token-by-token timeline built from pipeline_result
  traceTokens: [],  // [{step, token, energy, top_logit, prob, ms_token, entropy, anomaly}]

  // SAE (Sparse Autoencoder) interpretability
  sae: {
    active: false,
    layer: null,
    activeFeatures: 0,
    topFeatures: [],  // [[feature_idx, activation_count, description], ...]
    sparsity: 0,
    loss: 0,
    step: 0,
    history: []  // [{step, loss, sparsity, activeFeatures}]
  },

  // NTC KV-Cache compression
  ntc: {
    active: false,
    gridSize: 0,
    latentDim: 0,
    vramMb: 0,
    reconstructionLoss: 0,
    compressedTokens: 0,
    recentTokens: 0,
    history: []  // [{step, reconstructionLoss, vramMb, compressionRatio}]
  },

  // Quantization status
  quant: {
    mode: 'bf16',  // 'bf16' | 'q4' | 'fp8' | 'bitnet'
    vramMb: 0,
    baselineVramMb: 0,
    nrmse: 0,
    perLayerError: []
  }
});