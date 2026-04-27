// ============================================================================
// model/mod.rs — Pipeline Universal de Inferência: GPU via ComputeBackend
// ============================================================================
// Suporta múltiplas famílias de modelos (Qwen, Llama, Mistral, Phi, Gemma, etc.)
// via detecção automática de arquitetura a partir de config.json.
//
// Todas as operações no device (GPU ou CPU-backend):
//   - Pesos: AnyTensor BF16 (metade da memória)
//   - Ativações: AnyTensor F32 (scratchpad pré-alocado, zero-alloc no forward)
//   - KV-cache: GpuKvCache (AnyTensor F32 pré-alocado, append via backend)
//   - Projeções: gemm_bf16_f32 (BF16 pesos × F32 ativações → F32 saída)
//   - Norm/RoPE/Activation/Softmax: kernels via ComputeBackend
//   - Único download_f32(): logits finais [vocab_size] por token gerado
// ============================================================================

pub mod architecture;
pub use architecture::{ModelArchitecture, NormType, Activation};

mod kv_cache;
pub mod ntc_cache;
pub(crate) use kv_cache::GpuKvCache;

use std::sync::Arc;
use compute_core::{
    ComputeBackend, AnyTensor,
    GemmBf16Params, SgemmStridedBatchedParams, EmbedParams, RmsNormParams, LayerNormParams,
    RopeParams, SoftmaxParams, SiluMulParams, GeluMulParams, KvCacheParams,
    AddBiasParams, GemvQ4Params, GemvQ4KParams, GemvQ6KParams, GemvQ6KF32Params, FusedAttentionDecodeParams,
    GemvW8A16Params,
};
use half::bf16;
use std::collections::HashMap;
use std::path::Path;
use crate::storage::quantized::{QuantizedTensor, quantize_bf16_to_q4, upload_quantized, QuantizationCpuResult, save_quantized_cache, load_quantized_cache};
use crate::storage::fp8::fp8_e4m3_to_f32;
use crate::inference::forward::LayerTrace;
use crate::lnn::esn::LayerBudget;

// ── Tipos de erro ────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Backend error: {0}")]
    Backend(#[from] compute_core::BackendError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Model error: {0}")]
    Model(String),
}

pub type ModelResult<T> = std::result::Result<T, ModelError>;

// ── Phase Coherence Computation ──────────────────────────────────────────────

/// Delegates to compute_core::math::phase_coherence.
/// Kuramoto order parameter: treats hidden pairs as complex numbers.
pub fn phase_coherence(hidden: &[f32]) -> f32 {
    compute_core::math::phase_coherence(hidden)
}

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelCfg {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl From<compute_core::config::ModelConfig> for ModelCfg {
    fn from(c: compute_core::config::ModelConfig) -> Self {
        Self {
            hidden_size: c.hidden_size,
            num_layers: c.num_hidden_layers,
            intermediate_size: c.intermediate_size,
            vocab_size: c.vocab_size,
            num_heads: c.num_attention_heads,
            num_kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            rms_eps: c.rms_norm_eps as f32,
            rope_theta: c.rope_theta as f32,
            max_seq_len: c.max_position_embeddings,
        }
    }
}

impl ModelCfg {
    /// Parse config.json via compute_core::config::ModelConfig.
    pub fn from_config_json(config_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mc = compute_core::config::ModelConfig::from_json(config_path)?;
        Ok(Self::from(mc))
    }

    /// Constrói ModelCfg a partir dos metadados de um arquivo GGUF.
    /// Os keys seguem o padrão: "{arch}.embedding_length", "{arch}.block_count", etc.
    pub fn from_gguf(gguf: &crate::storage::gguf::GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        let arch_name = gguf.get_str("general.architecture")
            .unwrap_or("qwen2");

        let get = |suffix: &str| -> Result<usize, Box<dyn std::error::Error>> {
            let key = format!("{}.{}", arch_name, suffix);
            gguf.get_u32(&key)
                .map(|v| v as usize)
                .ok_or_else(|| format!("GGUF: key '{}' ausente", key).into())
        };

        let hidden_size = get("embedding_length")?;
        let num_heads = get("attention.head_count")?;
        let num_kv_heads = get("attention.head_count_kv").unwrap_or(num_heads);
        let head_dim = hidden_size / num_heads;

        let rms_eps = {
            let key = format!("{}.attention.layer_norm_rms_epsilon", arch_name);
            gguf.get_f32(&key).unwrap_or(1e-6)
        };

        let rope_theta = {
            let key = format!("{}.rope.freq_base", arch_name);
            gguf.get_f32(&key).unwrap_or(1_000_000.0)
        };

        let max_seq = {
            let key = format!("{}.context_length", arch_name);
            let raw = gguf.get_u32(&key).unwrap_or(2048) as usize;
            raw.min(1024)
        };

        Ok(Self {
            hidden_size,
            num_layers: get("block_count")?,
            intermediate_size: get("feed_forward_length")?,
            vocab_size: gguf.get_u32(&format!("{}.vocab_size", arch_name))
                .or_else(|| gguf.get_u32("tokenizer.ggml.vocab_size"))
                .map(|v| v as usize)
                .unwrap_or(151936), // fallback Qwen vocab
            num_heads,
            num_kv_heads,
            head_dim,
            rms_eps,
            rope_theta,
            max_seq_len: max_seq,
        })
    }

    #[allow(dead_code)]
    pub fn qwen_8b() -> Self {
        Self {
            hidden_size: 4096, num_layers: 36, intermediate_size: 12288,
            vocab_size: 151936, num_heads: 32, num_kv_heads: 8,
            head_dim: 128, rms_eps: 1e-6, rope_theta: 1_000_000.0, max_seq_len: 2048,
        }
    }
}

// ── Scratchpad GPU ────────────────────────────────────────────────────────────

/// Buffers pré-alocados para o forward pass. Zero alocação durante geração.
pub struct ForwardScratchpad {
    pub x:        AnyTensor,  // [max_seq, hidden]
    pub normed:   AnyTensor,  // [max_seq, hidden]
    pub q:        AnyTensor,  // [max_seq, q_dim]
    pub k:        AnyTensor,  // [max_seq, kv_dim]
    pub v:        AnyTensor,  // [max_seq, kv_dim]
    pub attn_out: AnyTensor,  // [max_seq, q_dim]
    pub gate:     AnyTensor,  // [max_seq, inter]
    pub up:       AnyTensor,  // [max_seq, inter]
    pub scores:   AnyTensor,  // [n_heads * max_seq, max_seq] — scores de atenção
    pub logits:   AnyTensor,  // [vocab_size]
    /// Buffers para dequantização INT8 (se modo INT8 está ativo)
    pub kv_k_deq: Option<AnyTensor>,  // [max_seq, kv_dim] F32 — dequant temp
    pub kv_v_deq: Option<AnyTensor>,  // [max_seq, kv_dim] F32 — dequant temp
    /// Q8_1 scratch buffer for dp4a GEMV (quantized activations)
    /// Size: (max_k / 32) * 36 bytes, where max_k = max(hidden, intermediate)
    pub q8_scratch: Option<AnyTensor>,
}

impl ForwardScratchpad {
    fn new(cfg: &ModelCfg, backend: &Arc<dyn ComputeBackend>) -> Self {
        let ms = cfg.max_seq_len;
        let h  = cfg.hidden_size;
        let qd = cfg.num_heads * cfg.head_dim;
        let kd = cfg.num_kv_heads * cfg.head_dim;
        let it = cfg.intermediate_size;
        let nh = cfg.num_heads;
        let v  = cfg.vocab_size;

        let mode = kv_cache::KvCacheMode::detect();
        let (kv_k_deq, kv_v_deq) = match mode {
            kv_cache::KvCacheMode::F32 => (None, None),
            kv_cache::KvCacheMode::Int8 | kv_cache::KvCacheMode::Wht => {
                // Aloca buffers para dequantização INT8/WHT → F32 durante atenção
                (
                    Some(backend.alloc_f32(&[ms, kd])),
                    Some(backend.alloc_f32(&[ms, kd])),
                )
            }
        };

        // Q8_1 scratch for dp4a GEMV: needs (max_k / 32) * 36 bytes
        // max_k = max(hidden, intermediate) — covers all projection sizes
        let max_k = h.max(it);
        let q8_bytes = (max_k / 32) * 36;
        // Always allocate Q8 scratch when native GGUF is active (default)
        let native_mode = std::env::var("IARA_GGUF_NATIVE").unwrap_or_else(|_| "1".to_string());
        let q8_scratch = if native_mode != "0" {
            Some(backend.alloc_q4(&[q8_bytes]))
        } else {
            None
        };

        Self {
            x:        backend.alloc_f32(&[ms, h]),
            normed:   backend.alloc_f32(&[ms, h]),
            q:        backend.alloc_f32(&[ms, qd]),
            k:        backend.alloc_f32(&[ms, kd]),
            v:        backend.alloc_f32(&[ms, kd]),
            attn_out: backend.alloc_f32(&[ms, qd]),
            gate:     backend.alloc_f32(&[ms, it]),
            up:       backend.alloc_f32(&[ms, it]),
            scores:   backend.alloc_f32(&[nh * ms, ms]),
            logits:   backend.alloc_f32(&[v]),
            kv_k_deq,
            kv_v_deq,
            q8_scratch,
        }
    }
}

// ── Sparse MLP Scratchpad (Kuramoto Router) ─────────────────────────────────

/// Pre-allocated GPU buffers for sparse MLP forward pass.
/// Sized for max_active_neurons (e.g. 25% of intermediate_size).
pub struct SparseMlpScratch {
    /// Sparse gate output [max_seq, max_active] F32
    pub sparse_gate: AnyTensor,
    /// Sparse up output [max_seq, max_active] F32
    pub sparse_up: AnyTensor,
    /// Max active neurons this scratchpad supports
    pub max_active: usize,
    /// Hidden size
    pub hidden: usize,
    /// Intermediate size (total neurons per layer)
    pub intermediate_size: usize,
    /// Cached n_active per layer
    pub cached_n_active: Vec<usize>,
    /// Token counter for logging
    pub token_counter: usize,
    /// Pre-gathered weight matrices per layer (persistent across all tokens)
    /// Format: (gathered_gate, gathered_up, gathered_down) per layer
    pub cached_weights: Vec<Option<(AnyTensor, AnyTensor, AnyTensor)>>,
    /// Whether cached weights are valid
    pub weights_valid: Vec<bool>,
    /// GPU-side tuning forks per layer [n_neurons, phase_dim] F32
    pub gpu_tuning_forks: Vec<Option<AnyTensor>>,
    /// GPU coherence buffer [n_neurons] F32 — reused across layers
    pub gpu_coherences: Option<AnyTensor>,
    /// GPU indices buffer [max_active] I32 — reused across layers
    pub gpu_indices: Option<AnyTensor>,
    /// Static hot neuron indices per layer (computed from calibration)
    pub static_hot_indices: Vec<Option<AnyTensor>>,
    /// Calibration state: accumulates |activation| per neuron during prefill
    /// Vec<Vec<f32>> — [n_layers][intermediate_size], sum of |SiLU(gate)*up| per neuron
    pub calibration_accum: Option<Vec<Vec<f32>>>,
    /// Number of tokens used for calibration
    pub calibration_tokens: usize,
    /// Whether calibration is complete and sparse mode is ready
    pub calibrated: bool,
}

impl SparseMlpScratch {
    fn new(cfg: &ModelCfg, backend: &Arc<dyn ComputeBackend>, active_ratio: f32) -> Self {
        let ms = cfg.max_seq_len;
        let h = cfg.hidden_size;
        let it = cfg.intermediate_size;
        let n_layers = cfg.num_layers;
        let max_active = (it as f32 * active_ratio).ceil() as usize;

        // Initialize calibration accumulators (zeroed)
        let calibration_accum: Vec<Vec<f32>> = (0..n_layers)
            .map(|_| vec![0.0f32; it])
            .collect();

        Self {
            sparse_gate:   backend.alloc_f32(&[ms, max_active]),
            sparse_up:     backend.alloc_f32(&[ms, max_active]),
            max_active,
            hidden: h,
            intermediate_size: it,
            cached_n_active: vec![0; n_layers],
            token_counter: 0,
            cached_weights: (0..n_layers).map(|_| None).collect(),
            weights_valid: vec![false; n_layers],
            gpu_tuning_forks: (0..n_layers).map(|_| None).collect(),
            gpu_coherences: Some(backend.alloc_f32(&[it])),
            gpu_indices: Some(backend.upload_i32(&vec![0i32; max_active], &[max_active])),
            static_hot_indices: (0..n_layers).map(|_| None).collect(),
            calibration_accum: Some(calibration_accum),
            calibration_tokens: 0,
            calibrated: false,
        }
    }

    /// Set sequential indices (0..max_active) for all layers and mark as calibrated.
    /// Used for debugging to verify the gather/GEMM pipeline works correctly.
    fn set_sequential_indices(
        &mut self,
        n_layers: usize,
        backend: &Arc<dyn ComputeBackend>,
    ) {
        let top_k = self.max_active;
        let indices: Vec<i32> = (0..top_k as i32).collect();
        for layer_idx in 0..n_layers {
            let gpu_idx = backend.upload_i32(&indices, &[top_k]);
            self.static_hot_indices[layer_idx] = Some(gpu_idx);
            self.cached_n_active[layer_idx] = top_k;
        }
        self.calibration_accum = None;
        self.calibrated = true;
        log::info!(
            "🔧 Sequential indices set: 0..{} for {} layers (debug mode)",
            top_k, n_layers,
        );
    }

    /// Accumulate activation statistics for one layer during calibration.
    /// Called from the dense MLP path after SiLU*mul (gate tensor contains activated values).
    /// Downloads gate activations and adds |value| per neuron to the accumulator.
    fn accumulate_calibration(
        &mut self,
        layer_idx: usize,
        gate_tensor: &AnyTensor,
        backend: &Arc<dyn ComputeBackend>,
        n_tokens: usize,
        intermediate_size: usize,
    ) {
        if let Some(ref mut accum) = self.calibration_accum {
            let gate_data = backend.download_f32(gate_tensor);
            let layer_accum = &mut accum[layer_idx];

            // For each token, add |activation| per neuron
            for tok in 0..n_tokens {
                let offset = tok * intermediate_size;
                for neuron in 0..intermediate_size {
                    if offset + neuron < gate_data.len() {
                        layer_accum[neuron] += gate_data[offset + neuron].abs();
                    }
                }
            }
        }
    }

    /// Finalize calibration: select top-K neurons per layer based on accumulated
    /// activation magnitudes, upload indices to GPU.
    fn finalize_calibration(
        &mut self,
        backend: &Arc<dyn ComputeBackend>,
    ) {
        let accum = match self.calibration_accum.take() {
            Some(a) => a,
            None => return,
        };

        let n_tokens = self.calibration_tokens.max(1) as f32;

        for (layer_idx, layer_accum) in accum.iter().enumerate() {
            let it = layer_accum.len();
            let top_k = self.max_active.min(it);

            // Average activation magnitude per neuron
            let mut scored: Vec<(usize, f32)> = layer_accum.iter()
                .enumerate()
                .map(|(i, &sum)| (i, sum / n_tokens))
                .collect();

            // Sort by activation magnitude descending
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-K, then sort by index for memory locality in gather
            let mut hot_indices: Vec<i32> = scored[..top_k]
                .iter()
                .map(|(idx, _)| *idx as i32)
                .collect();
            hot_indices.sort();

            let gpu_idx = backend.upload_i32(&hot_indices, &[top_k]);
            self.static_hot_indices[layer_idx] = Some(gpu_idx);
            self.cached_n_active[layer_idx] = top_k;

            if layer_idx == 0 {
                let active_pct = scored[..top_k].iter().map(|(_, v)| *v).sum::<f32>()
                    / scored.iter().map(|(_, v)| *v).sum::<f32>().max(1e-8) * 100.0;
                log::info!(
                    "🔥 Calibrated hot neurons: top-{} of {} cover {:.1}% of total activation energy \
                     (avg magnitude range: {:.6}..{:.6}, calibrated on {} tokens)",
                    top_k, it, active_pct,
                    scored[top_k - 1].1, scored[0].1,
                    self.calibration_tokens,
                );
            }
        }

        self.calibrated = true;
        log::info!(
            "✅ Calibration complete: hot neuron indices computed for {} layers from {} prefill tokens",
            accum.len(), self.calibration_tokens,
        );
    }
}

// ── Online Layer Router (adaptive BI-based skip) ────────────────────────────

/// Dynamic layer skip via online Block Influence (BI) measurement.
/// Measures cos_sim(input, output) per layer in real-time and skips layers
/// where BI stays consistently low (the layer barely changes the hidden state).
///
/// Phase 1 (calibration, first `warmup_tokens` tokens): run ALL layers, measure BI
/// Phase 2 (routing): skip layers where moving_avg(BI) < threshold
/// Phase 3 (re-probe): every `reprobe_interval` tokens, re-measure a skipped layer
pub struct OnlineLayerRouter {
    /// Moving average of BI per layer [num_layers]
    pub bi_ema: Vec<f32>,
    /// GPU buffer to snapshot hidden state before a layer [hidden_size]
    pub x_snapshot: AnyTensor,
    /// GPU buffer for cosine similarity result [1]
    pub cos_sim_buf: AnyTensor,
    /// Number of tokens seen so far
    pub tokens_seen: usize,
    /// Warmup phase: run all layers, measure BI
    pub warmup_tokens: usize,
    /// BI threshold: layers below this are skipped
    pub bi_threshold: f32,
    /// EMA decay factor (0.9 = slow, 0.5 = fast adaptation)
    pub ema_alpha: f32,
    /// How often to re-probe a skipped layer (in tokens)
    pub reprobe_interval: usize,
    /// Number of layers
    pub n_layers: usize,
    /// Hidden size
    pub hidden: usize,
    /// Whether routing is enabled
    pub enabled: bool,
}

impl OnlineLayerRouter {
    pub fn new(
        backend: &Arc<dyn ComputeBackend>,
        n_layers: usize,
        hidden: usize,
    ) -> Self {
        let warmup = std::env::var("IARA_ROUTER_WARMUP")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(5);
        let threshold = std::env::var("IARA_ROUTER_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0.15);
        let alpha = std::env::var("IARA_ROUTER_ALPHA")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0.7);
        let reprobe = std::env::var("IARA_ROUTER_REPROBE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(50);
        let enabled = std::env::var("IARA_ROUTER").map(|v| v == "1").unwrap_or(false);

        Self {
            bi_ema: vec![1.0; n_layers],  // start high (assume all layers are needed)
            x_snapshot: backend.alloc_f32(&[hidden]),
            cos_sim_buf: backend.alloc_f32(&[1]),
            tokens_seen: 0,
            warmup_tokens: warmup,
            bi_threshold: threshold,
            ema_alpha: alpha,
            reprobe_interval: reprobe,
            n_layers,
            hidden,
            enabled,
        }
    }

    /// Should we skip this layer during decode?
    /// Returns (skip, should_measure) — if should_measure is true, caller must
    /// snapshot x before the layer and compute BI after.
    pub fn should_skip(&self, layer_idx: usize) -> (bool, bool) {
        if !self.enabled {
            return (false, false);
        }
        // During warmup: never skip, always measure
        if self.tokens_seen < self.warmup_tokens {
            return (false, true);
        }
        // After warmup: skip if BI is low
        let bi = self.bi_ema[layer_idx];
        if bi < self.bi_threshold {
            // Re-probe periodically to catch layers that become important
            let should_reprobe = (self.tokens_seen % self.reprobe_interval) == 0;
            if should_reprobe {
                return (false, true);  // don't skip, but measure
            }
            return (true, false);  // skip
        }
        // Layer is important — run it, measure occasionally
        let measure = self.tokens_seen < self.warmup_tokens + 10
            || (self.tokens_seen % 20) == 0;
        (false, measure)
    }

    /// Update BI for a layer after measuring cos_sim.
    /// bi = 1 - cos_sim (0 = identity, 1 = orthogonal)
    pub fn update_bi(&mut self, layer_idx: usize, cos_sim: f32) {
        let bi = (1.0 - cos_sim).max(0.0);
        let alpha = self.ema_alpha;
        self.bi_ema[layer_idx] = alpha * self.bi_ema[layer_idx] + (1.0 - alpha) * bi;
    }

    /// Called once per token to advance the token counter.
    pub fn advance_token(&mut self) {
        self.tokens_seen += 1;
        // Log after warmup completes
        if self.tokens_seen == self.warmup_tokens && self.enabled {
            let skipped: Vec<usize> = self.bi_ema.iter().enumerate()
                .filter(|(_, &bi)| bi < self.bi_threshold)
                .map(|(i, _)| i)
                .collect();
            let kept: Vec<usize> = self.bi_ema.iter().enumerate()
                .filter(|(_, &bi)| bi >= self.bi_threshold)
                .map(|(i, _)| i)
                .collect();
            log::info!(
                "🧭 OnlineLayerRouter warmup done: {}/{} layers active, skip {:?}",
                kept.len(), self.n_layers, skipped,
            );
            log::info!(
                "   BI scores: {:?}",
                self.bi_ema.iter().map(|b| format!("{:.3}", b)).collect::<Vec<_>>()
            );
        }
    }
}

// ── Variante de peso para suportar BF16 ou INT4 ───────────────────────────────

/// Peso de uma camada: pode ser BF16 original (padrão) ou Q4 (quantizado INT4).
/// Permite transição gradual para quantização sem refatoração massiva do forward pass.
pub enum LayerWeight {
    /// Tensor BF16 padrão (no device)
    Bf16(AnyTensor),
    /// Tensor INT4 quantizado (packed + scales + zeros no device)
    Q4(QuantizedTensor),
    /// GGML Q4_K: raw quantized blocks on GPU (144 bytes per 256 values).
    /// Stores (raw_data_tensor, n_rows, k_dim).
    Q4K { data: AnyTensor, n: usize, k: usize },
    /// GGML Q6_K: raw quantized blocks on GPU (210 bytes per 256 values).
    /// Stores (raw_data_tensor, n_rows, k_dim).
    Q6K { data: AnyTensor, n: usize, k: usize },
    /// BitNet ternary {-1, 0, +1}: 2 bits/weight, 4 weights/byte.
    /// Encoding: -1=0b00, 0=0b01, +1=0b10. HIP kernel: pure add/sub, zero multiplies.
    /// Stores (packed_data, n_rows, k_dim, per_tensor_scale).
    Ternary { data: AnyTensor, n: usize, k: usize, scale: f32 },
    /// W8A32 quantization: INT8 weights (symmetric per-row) × F32 activations.
    /// Halves weight memory bandwidth vs BF16: 1 byte/weight → ~2× decode speed.
    /// Quantized from BF16 at load time via quantize_w8_to_gpu().
    W8 { data: AnyTensor, scales: AnyTensor, n: usize, k: usize },
    /// DNA V4 SVD factorization: W ≈ A·B, never materialize W in VRAM.
    /// `a`: [n, rank] BF16, `b`: [rank, k] BF16, `scratch`: [rank] F32 device buffer.
    /// Compute: Y = A·(B·X) via two chained gemm_bf16_f32 calls. ~95% memory savings.
    Svd { a: AnyTensor, b: AnyTensor, scratch: AnyTensor, n: usize, k: usize, rank: usize },
}

impl LayerWeight {
    /// Estimated VRAM usage in bytes for this weight tensor.
    pub fn vram_bytes(&self) -> usize {
        match self {
            LayerWeight::Bf16(t) => t.numel() * 2,
            LayerWeight::Q4(qt) => {
                let [rows, cols] = qt.shape;
                rows * cols / 2 + rows * (cols / qt.group_size) * 8
            }
            LayerWeight::Q4K { n, k, .. } => n * (k / 256) * 144,
            LayerWeight::Q6K { n, k, .. } => n * (k / 256) * 210,
            LayerWeight::Ternary { n, k, .. } => n * ((k + 3) / 4), // 2 bits/weight
            LayerWeight::W8 { n, k, .. } => n * k + n * 4,          // 1 byte/weight + 4 bytes scale/row
            LayerWeight::Svd { n, k, rank, .. } => (n + k) * rank * 2 + rank * 4, // A + B + scratch
        }
    }
}

// ── Camada Transformer ────────────────────────────────────────────────────────

/// Pesos e estado de uma camada Qwen — tensores no device via AnyTensor ou QuantizedTensor.
pub struct TransformerLayer {
    // Pesos matriciais: BF16 ou Q4 conforme modo
    pub wq:    LayerWeight,   // BF16 ou Q4
    pub wk:    LayerWeight,   // BF16 ou Q4
    pub wv:    LayerWeight,   // BF16 ou Q4
    pub wo:    LayerWeight,   // BF16 ou Q4
    pub wgate: LayerWeight,   // BF16 ou Q4
    pub wup:   LayerWeight,   // BF16 ou Q4
    pub wdown: LayerWeight,   // BF16 ou Q4
    // Biases opcionais (Qwen2/2.5 têm bias em Q/K/V; Qwen3 não tem)
    pub q_bias: Option<AnyTensor>,  // F32 opcional
    pub k_bias: Option<AnyTensor>,  // F32 opcional
    pub v_bias: Option<AnyTensor>,  // F32 opcional
    // QK Norm weights (Qwen3 aplica RMSNorm per-head em Q e K antes do RoPE)
    pub q_norm_w: Option<AnyTensor>,  // F32 [head_dim] — Qwen3 only
    pub k_norm_w: Option<AnyTensor>,  // F32 [head_dim] — Qwen3 only
    // Norm weights no device (pequenos, mas evita round-trip host→device por token)
    pub input_norm_w:    AnyTensor, // F32
    pub post_attn_norm_w: AnyTensor, // F32
    // KV-cache no device (janela deslizante via copy_to_kv_cache)
    pub kv_k: GpuKvCache,
    pub kv_v: GpuKvCache,
}

// ── Modelo completo ───────────────────────────────────────────────────────────

pub struct CausalLM {
    pub embed:   AnyTensor,     // BF16 [vocab, hidden] — sempre (sensível a quantização)
    pub layers:  Vec<TransformerLayer>,
    pub norm_w:  AnyTensor,     // F32 [hidden]
    pub norm_b:  Option<AnyTensor>, // F32 [hidden] — bias (LayerNorm only)
    pub lm_head: LayerWeight,   // BF16 ou Q4 [vocab, hidden]
    pub cfg:     ModelCfg,
    /// Descritor da arquitetura (família, norm, ativação, tensor map, EOS tokens)
    pub arch: ModelArchitecture,
    /// Bytes estimados no device: pesos BF16/Q4 + scratchpad + KV-cache.
    pub vram_bytes: usize,
    pub backend: Arc<dyn ComputeBackend>,
    pub scratch: ForwardScratchpad,
    // ────────────────────────────────────────────────────────────────────────
    // Campos para CfC e Gravity (quantização + correção adaptativa)
    // ────────────────────────────────────────────────────────────────────────
    /// CfC Liquid Correction Layer (opcional).
    pub cfc: Option<crate::lnn::cfc::CfcLayer>,
    /// Sistema de Gravidade Semântica (opcional).
    pub gravity: Option<crate::lnn::gravity::GravitySystem>,
    /// Flag: se true, pesos estão em Q4 (quantizados INT4).
    pub quantized: bool,
    /// Kuramoto Phase Router for sparse MLP neuron selection (opcional).
    /// If Some, enabled via IARA_ROUTING=1 env var during model load.
    pub kuramoto_router: Option<crate::inference::KuramotoRouter>,
    /// Pre-allocated GPU buffers for sparse MLP computation.
    pub sparse_scratch: Option<SparseMlpScratch>,
    /// Optional channel to send layer stimulus signals to the orchestrator.
    #[cfg(feature = "experimental")]
    pub stimulus_tx: Option<tokio::sync::mpsc::Sender<crate::experimental::orchestrator::LayerStimulus>>,
    /// When false (default), per-layer telemetry download_f32 calls are skipped entirely.
    /// Set to true only when the dashboard/debug UI needs live layer traces.
    pub telemetry_enabled: bool,
    /// Online adaptive layer skip router (IARA_ROUTER=1).
    /// Measures BI per layer in real-time and dynamically skips low-BI layers.
    pub layer_router: OnlineLayerRouter,
    /// HIP graph capture-once-replay mode active (IARA_GRAPH=2).
    /// When true, decode forward dispatches `_dyn` kernel variants reading
    /// seq_pos from a device-resident i32 buffer.
    pub graph_replay_mode: bool,
}

impl CausalLM {
    /// Helper: faz projeção com pesos BF16 ou Q4 quantizados.
    /// Despacha para backend.gemm_bf16_f32 ou backend.gemv_q4 conforme necessário.
    /// Quantize a BF16 weight matrix to INT8 symmetric per-row on the CPU,
    /// then upload both the INT8 weights and per-row F32 scales to the GPU.
    ///
    /// Formula: scale[r] = max_abs(row[r]) / 127.0
    ///          q[r][c]  = round(w[r][c] / scale[r]).clamp(-127, 127)
    ///
    /// Returns a `LayerWeight::W8` ready for decode.
    pub fn quantize_w8_to_gpu(
        backend: &Arc<dyn ComputeBackend>,
        w_bf16: &[half::bf16],
        rows: usize,
        cols: usize,
    ) -> LayerWeight {
        let mut q = vec![0u8; rows * cols]; // stored as u8, interpreted as i8 by kernel
        let mut scales = vec![0f32; rows];
        for r in 0..rows {
            let row = &w_bf16[r * cols..(r + 1) * cols];
            let max_abs = row.iter()
                .map(|x| x.to_f32().abs())
                .fold(0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0f32 } else { 1.0f32 };
            scales[r] = scale;
            let inv_scale = 1.0f32 / scale;
            for (c, &v) in row.iter().enumerate() {
                let quantized = (v.to_f32() * inv_scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8 as u8;
                q[r * cols + c] = quantized;
            }
        }
        let data_tensor = backend.upload_raw(&q, &[rows * cols]);
        let scales_tensor = backend.upload_f32(&scales, &[rows]);
        LayerWeight::W8 { data: data_tensor, scales: scales_tensor, n: rows, k: cols }
    }

    /// Quantize input activations to Q8_1 format for dp4a GEMV.
    /// Call once before a batch of project_q8() calls that share the same input.
    fn quantize_input_q8(
        backend: &Arc<dyn ComputeBackend>,
        input: &AnyTensor,
        a_offset: usize,
        q8_scratch: &AnyTensor,
        k: usize,
    ) -> ModelResult<()> {
        backend.quantize_q8(&compute_core::QuantizeQ8Params {
            x: input, x_offset: a_offset, out: q8_scratch, k,
        })?;
        Ok(())
    }

    fn project(
        backend: &Arc<dyn ComputeBackend>,
        weight: &LayerWeight,
        input: &AnyTensor,
        output: &AnyTensor,
        m: usize,
        n: usize,
        k: usize,
        a_offset: usize,
        beta: f32,
        q8_scratch: Option<&AnyTensor>,
        q8_precomputed: bool,
    ) -> ModelResult<()> {
        {
            static PROJ_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !PROJ_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let wtype = match weight {
                    LayerWeight::Bf16(_) => "BF16",
                    LayerWeight::Q4(_) => "Q4",
                    LayerWeight::Q4K { .. } => "Q4K",
                    LayerWeight::Q6K { .. } => "Q6K",
                    LayerWeight::Ternary { .. } => "Ternary",
                    LayerWeight::W8 { .. } => "W8",
                    LayerWeight::Svd { .. } => "SVD",
                };
                log::warn!("project() first call: weight={}, m={}, n={}, k={}, q8_scratch={}, precomputed={}",
                    wtype, m, n, k, q8_scratch.is_some(), q8_precomputed);
            }
        }
        match weight {
            LayerWeight::Bf16(w) => {
                // For decode (m=1, beta=0): custom GEMV avoids hipBLAS overhead.
                // hipBLAS SGEMM for M=1 has ~3.5ms latency vs ~1ms theoretical BW.
                // Only use GEMV when beta=0 (lm_head always 0; wout is 1.0 → SGEMM).
                if m == 1 && beta == 0.0 {
                    let gemv_result = backend.gemv_bf16_f32(&compute_core::GemvBf16F32Params {
                        n, k, w, x: input, x_offset: a_offset, out: output,
                    });
                    if gemv_result.is_ok() {
                        static BF16_GEMV_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !BF16_GEMV_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("BF16 GEMV: using custom kernel (m={}, n={}, k={}) — bypasses hipBLAS", m, n, k);
                        }
                        return Ok(());
                    }
                    // Fallthrough to hipBLAS on error
                }
                backend.gemm_bf16_f32(&GemmBf16Params {
                    m,
                    n,
                    k,
                    alpha: 1.0,
                    beta,
                    a: input,
                    a_offset,
                    b: w,
                    c: output,
                })?;
            }
            LayerWeight::Q4(qt) => {
                backend.gemv_q4(&GemvQ4Params {
                    m,
                    n,
                    k,
                    packed: &qt.packed,
                    scales: &qt.scales,
                    zeros: &qt.zeros,
                    group_size: qt.group_size,
                    x: input,
                    x_offset: a_offset,
                    out: output,
                    beta,
                })?;
            }
            LayerWeight::Q4K { data, n: _, k: _ } => {
                // ── Fast path: Q4K × F32 GEMV (decode, m=1) ──────────────────────
                // Eliminates the intermediate Q8_1 quantize step (saves 1 kernel launch).
                // gemv_q4k_f32_multi dequants Q4K on-the-fly with direct F32 dot product.
                // Skipped by default (Q8 dp4a wave64 is faster). Set IARA_NO_Q8_MLP=1 to re-enable.
                if m == 1 && std::env::var("IARA_NO_Q8_MLP").is_ok() {
                    let f32_result = backend.gemv_q4k_f32(&compute_core::GemvQ4KF32Params {
                        m, n, k, data, x: input, x_offset: a_offset, out: output, beta,
                    });
                    if f32_result.is_ok() {
                        static Q4K_F32_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !Q4K_F32_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("Q4K F32: using direct F32 GEMV (m={}, n={}, k={}) — no Q8 quantize", m, n, k);
                        }
                        return Ok(());
                    }
                    // Fallthrough to Q8 path if gemv_q4k_f32 not supported
                }
                if let (Some(q8), true) = (q8_scratch, m == 1) {
                    static Q4K_Q8_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !Q4K_Q8_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        log::warn!("Q4K path: using Q8 dp4a (m={}, n={}, k={}, precomputed={})", m, n, k, q8_precomputed);
                    }
                    if !q8_precomputed {
                        backend.quantize_q8(&compute_core::QuantizeQ8Params {
                            x: input, x_offset: a_offset, out: q8, k,
                        })?;
                    }
                    backend.gemv_q4k_q8(&compute_core::GemvQ4KQ8Params {
                        m, n, k, data, x_q8: q8, out: output, beta,
                    })?;
                } else {
                    // SLOW path: used during prefill (m>1, no batched Q4K GEMM) or when
                    // q8_scratch is unavailable (IARA_GGUF_NATIVE=0).
                    // Only warn on decode (m==1) with missing q8_scratch — genuine fallback.
                    if m == 1 && q8_scratch.is_none() {
                        static Q4K_SLOW_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !Q4K_SLOW_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("Q4K path: SLOW dequant_gemv during decode (m=1, q8_scratch=None). \
                                Set IARA_GGUF_NATIVE=1 (default) to enable fast Wave64 Q8 dp4a path. \
                                (n={}, k={})", n, k);
                        }
                    }
                    backend.gemv_q4k(&GemvQ4KParams {
                        m,
                        n,
                        k,
                        data,
                        x: input,
                        x_offset: a_offset,
                        out: output,
                        beta,
                    })?;
                }
            }
            LayerWeight::Q6K { data, n: _, k: _ } => {
                // Fast path: direct F32 GEMV — eliminates quantize_q8 step (saves 1 launch).
                if m == 1 && std::env::var("IARA_NO_Q8_MLP").is_ok() && (!std::env::var("IARA_UNFUSED").is_ok() || std::env::var("IARA_FUSE_Q6KF32").is_ok()) {
                    let f32_result = backend.gemv_q6k_f32(&GemvQ6KF32Params {
                        m, n, k, data, x: input, x_offset: a_offset, out: output, beta,
                    });
                    if f32_result.is_ok() {
                        static Q6K_F32_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !Q6K_F32_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("Q6K F32: using direct F32 GEMV (m={}, n={}, k={}) — no Q8 quantize", m, n, k);
                        }
                        return Ok(());
                    }
                    // Fallthrough to Q8 path if gemv_q6k_f32 not supported
                }
                if let (Some(q8), true) = (q8_scratch, m == 1) {
                    if !q8_precomputed {
                        backend.quantize_q8(&compute_core::QuantizeQ8Params {
                            x: input, x_offset: a_offset, out: q8, k,
                        })?;
                    }
                    backend.gemv_q6k_q8(&compute_core::GemvQ6KQ8Params {
                        m, n, k, data, x_q8: q8, out: output, beta,
                    })?;
                } else {
                    backend.gemv_q6k(&GemvQ6KParams {
                        m,
                        n,
                        k,
                        data,
                        x: input,
                        x_offset: a_offset,
                        out: output,
                        beta,
                    })?;
                }
            }
            LayerWeight::Ternary { data, n: _, k: _, scale } => {
                backend.gemv_ternary_f32(&compute_core::GemvTernaryF32Params {
                    m, n, k,
                    packed: data,
                    x: input,
                    x_offset: a_offset,
                    out: output,
                    scale: *scale,
                    beta,
                })?;
            }
            LayerWeight::W8 { data, scales, n: _, k: _ } => {
                // W8A32 GEMV: INT8 weights × F32 activations → F32.
                // Supports decode (m=1) with any beta (0=overwrite, 1=residual accumulate).
                // For prefill (m>1): W8 batched GEMM not yet implemented — return error.
                if m != 1 {
                    static W8_PREFILL_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !W8_PREFILL_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        log::warn!("W8 prefill (m={}) not supported — W8 models require short prompts (m=1 decode only)", m);
                    }
                    return Err(ModelError::Model(format!(
                        "W8 weight does not support prefill GEMM (m={}) — use BF16 or Q4K models for long prompts", m
                    )));
                }
                static W8_GEMV_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !W8_GEMV_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::warn!("W8 GEMV: INT8 weights × F32 activations (m={}, n={}, k={}, beta={}) — ~2× BW vs BF16", m, n, k, beta);
                }
                backend.gemv_w8a16(&GemvW8A16Params {
                    n, k,
                    w: data,
                    scales,
                    x: input,
                    x_offset: a_offset,
                    out: output,
                    beta,
                })?;
            }
            LayerWeight::Svd { a, b, scratch, n: _, k: _, rank } => {
                // DNA V4 factorized projection: Y = X · Wᵀ = (X · Bᵀ) · Aᵀ via 2 chained gemm_bf16_f32.
                // Default trait impl handles the chaining — any backend with gemm_bf16_f32 works.
                // Supports both decode (m=1) and prefill (m>1). Scratch must be sized for
                // `max_m * rank` F32 elements (caller's responsibility at allocation time).
                backend.gemv_svd_bf16(&compute_core::GemvSvdBf16Params {
                    m,
                    rows: n,
                    cols: k,
                    rank: *rank,
                    a, b, scratch,
                    x: input,
                    x_offset: a_offset,
                    out: output,
                    beta,
                })?;
            }
        }
        Ok(())
    }

    /// Aplica normalização baseada no tipo da arquitetura (RMSNorm ou LayerNorm).
    /// `norm_bias` é obrigatório para LayerNorm, ignorado para RMSNorm.
    fn apply_norm(
        backend: &Arc<dyn ComputeBackend>,
        norm_type: NormType,
        input: &AnyTensor,
        weight: &AnyTensor,
        bias: Option<&AnyTensor>,
        output: &AnyTensor,
        n_tokens: usize,
        hidden_size: usize,
        eps: f32,
    ) -> ModelResult<()> {
        match norm_type {
            NormType::RMSNorm => {
                backend.rms_norm(&RmsNormParams {
                    input, weight, output, n_tokens, hidden_size, eps,
                })?;
            }
            NormType::LayerNorm => {
                let bias = bias.ok_or_else(|| ModelError::Model(
                    "LayerNorm requer bias, mas nenhum foi carregado".into()
                ))?;
                backend.layer_norm(&LayerNormParams {
                    input, weight, bias, output, n_tokens, hidden_size, eps,
                })?;
            }
        }
        Ok(())
    }

    /// Aplica ativação + multiplicação baseada no tipo da arquitetura (SiLU ou GELU).
    fn apply_activation(
        backend: &Arc<dyn ComputeBackend>,
        activation: Activation,
        gate: &AnyTensor,
        up: &AnyTensor,
        n_tokens: usize,
        intermediate_size: usize,
    ) -> ModelResult<()> {
        match activation {
            Activation::SiLU => {
                backend.silu_mul_inplace(&SiluMulParams {
                    gate, up, n_tokens, intermediate_size,
                })?;
            }
            Activation::GELU => {
                backend.gelu_mul_inplace(&GeluMulParams {
                    gate, up, n_tokens, intermediate_size,
                })?;
            }
        }
        Ok(())
    }

    /// Carrega pesos do diretório. Suporta single-shard e multi-shard.
    /// Usa ModelArchitecture para determinar nomes de tensores e variantes.
    pub fn load(model_dir: &Path, cfg: ModelCfg, arch: ModelArchitecture, backend: Arc<dyn ComputeBackend>) -> ModelResult<Self> {
        log::info!(
            "⏳ Carregando modelo [{}]: {} camadas, hidden={}, vocab={}",
            arch.family, cfg.num_layers, cfg.hidden_size, cfg.vocab_size
        );

        // ── Mmaps ───────────────────────────────────────────────────────────
        let mut shards: HashMap<String, memmap2::Mmap> = HashMap::new();
        let tensor_shard_map: Option<HashMap<String, String>>;

        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let weight_map = index["weight_map"].as_object()
                .ok_or_else(|| ModelError::Model("weight_map inválido".into()))?;
            let mut tsm: HashMap<String, String> = HashMap::new();
            for (k, v) in weight_map {
                if let Some(s) = v.as_str() { tsm.insert(k.clone(), s.to_string()); }
            }
            for shard_name in tsm.values() {
                if shards.contains_key(shard_name) { continue; }
                let file = std::fs::File::open(model_dir.join(shard_name))?;
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                shards.insert(shard_name.clone(), mmap);
            }
            tensor_shard_map = Some(tsm);
            log::info!("  → multi-shard ({} shards)", shards.len());
        } else {
            let single = model_dir.join("model.safetensors");
            if !single.exists() {
                return Err(ModelError::Model(
                    "Nenhum safetensors encontrado (sem index.json nem model.safetensors)".into()
                ));
            }
            let file = std::fs::File::open(&single)?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            log::info!("  → single-shard ({:.2} GB)", mmap.len() as f64 / 1e9);
            shards.insert("model.safetensors".to_string(), mmap);
            tensor_shard_map = None;
        }

        // ── Helpers ──────────────────────────────────────────────────────────
        let find_mmap = |name: &str| -> Option<&memmap2::Mmap> {
            match &tensor_shard_map {
                Some(tsm) => tsm.get(name).and_then(|s| shards.get(s)),
                None => shards.get("model.safetensors"),
            }
        };
        let tensor_exists = |name: &str| -> bool {
            match &tensor_shard_map {
                Some(tsm) => tsm.contains_key(name),
                None => shards.get("model.safetensors")
                    .map(|m| shard_has_tensor(m, name))
                    .unwrap_or(false),
            }
        };

        let load_f32 = |name: &str| -> ModelResult<(Vec<f32>, Vec<usize>)> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            read_bf16_f32(mmap, name)
        };

        // Upload Zero-Copy: Casting de &[u8] direto para &[u16] no buffer Mmap
        // Seguro e ultrarrápido pois os Safetensors já exigem alinhamento de 8 bytes,
        // garantindo total validade para casting O(1).
        let gpu = |name: &str| -> ModelResult<AnyTensor> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            
            let (raw_bytes, shape) = shard_tensor_bytes(mmap, name)?;
            
            // Garantia Rust-Pro de alinhamento e corretude em little-endian.
            let bits = unsafe {
                std::slice::from_raw_parts(
                    raw_bytes.as_ptr() as *const u16, 
                    raw_bytes.len() / 2
                )
            };
            
            Ok(backend.upload_bf16(bits, &shape))
        };
        // Faz upload F32 para o device via backend
        let gpu_f32 = |data: &[f32]| -> AnyTensor {
            backend.upload_f32(data, &[data.len()])
        };

        // ── W8 quantization mode ─────────────────────────────────────────────
        // IARA_W8=1: quantize BF16 weights to INT8 symmetric per-row during load.
        // Halves weight memory bandwidth → ~2× decode tok/s vs BF16.
        // Only for BF16 safetensors models; GGUF Q4K/Q6K models already quantized.
        let use_w8 = std::env::var("IARA_W8").map(|v| v == "1").unwrap_or(false);
        if use_w8 {
            log::warn!("W8 quantization enabled (IARA_W8=1): BF16 weights → INT8 per-row on load");
        }

        // Helper: load tensor as BF16 and return as LayerWeight (W8 or BF16 depending on mode)
        let load_layer_weight = |name: &str, rows: usize, cols: usize| -> ModelResult<LayerWeight> {
            if use_w8 {
                // Load BF16 tensor and quantize to INT8 on CPU, then upload both to GPU
                let mmap = find_mmap(name)
                    .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
                let (raw_bytes, _shape) = shard_tensor_bytes(mmap, name)?;
                let w_bits = unsafe {
                    std::slice::from_raw_parts(raw_bytes.as_ptr() as *const u16, raw_bytes.len() / 2)
                };
                // Convert u16 BF16 bits → half::bf16 slice
                let w_bf16: Vec<bf16> = w_bits.iter().map(|&b| bf16::from_bits(b)).collect();
                Ok(CausalLM::quantize_w8_to_gpu(&backend, &w_bf16, rows, cols))
            } else {
                // Standard BF16 upload
                let mmap = find_mmap(name)
                    .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
                let (raw_bytes, shape) = shard_tensor_bytes(mmap, name)?;
                let bits = unsafe {
                    std::slice::from_raw_parts(raw_bytes.as_ptr() as *const u16, raw_bytes.len() / 2)
                };
                Ok(LayerWeight::Bf16(backend.upload_bf16(bits, &shape)))
            }
        };

        // ── Nomes de tensores (via architecture descriptor) ─────────────────
        let tmap = &arch.tensor_map;

        // ── Embedding ────────────────────────────────────────────────────────
        log::info!("  → embed_tokens");
        let embed = gpu(&tmap.embed)?;

        // ── Camadas ──────────────────────────────────────────────────────────
        let kv_dim  = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;

        // Weight shapes for W8 quantization
        let h = cfg.hidden_size;
        let q_dim   = cfg.num_heads * cfg.head_dim;
        let kv_dim_w = kv_dim; // num_kv_heads * head_dim
        let inter   = cfg.intermediate_size;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 { log::info!("  → layers {}/{}", i, cfg.num_layers); }

            let (in_w, _)  = load_f32(&tmap.layer_tensor(i, &tmap.input_norm))?;
            let (pa_w, _)  = load_f32(&tmap.layer_tensor(i, &tmap.post_attn_norm))?;

            // Tensor names for optional fields
            let qbn = tmap.layer_tensor_opt(i, &tmap.attn_q_bias);
            let kbn = tmap.layer_tensor_opt(i, &tmap.attn_k_bias);
            let vbn = tmap.layer_tensor_opt(i, &tmap.attn_v_bias);
            let qnn = tmap.layer_tensor_opt(i, &tmap.q_norm);
            let knn = tmap.layer_tensor_opt(i, &tmap.k_norm);

            layers.push(TransformerLayer {
                wq:    load_layer_weight(&tmap.layer_tensor(i, &tmap.attn_q),    q_dim,    h)?,
                wk:    load_layer_weight(&tmap.layer_tensor(i, &tmap.attn_k),    kv_dim_w, h)?,
                wv:    load_layer_weight(&tmap.layer_tensor(i, &tmap.attn_v),    kv_dim_w, h)?,
                wo:    load_layer_weight(&tmap.layer_tensor(i, &tmap.attn_o),    h,        q_dim)?,
                wgate: load_layer_weight(&tmap.layer_tensor(i, &tmap.mlp_gate),  inter,    h)?,
                wup:   load_layer_weight(&tmap.layer_tensor(i, &tmap.mlp_up),    inter,    h)?,
                wdown: load_layer_weight(&tmap.layer_tensor(i, &tmap.mlp_down),  h,        inter)?,
                q_bias: match &qbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                k_bias: match &kbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                v_bias: match &vbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                q_norm_w: match &qnn {
                    Some(name) if tensor_exists(name) => {
                        log::debug!("  → q_norm layer {}", i);
                        Some(gpu_f32(&load_f32(name)?.0))
                    },
                    _ => None,
                },
                k_norm_w: match &knn {
                    Some(name) if tensor_exists(name) => {
                        log::debug!("  → k_norm layer {}", i);
                        Some(gpu_f32(&load_f32(name)?.0))
                    },
                    _ => None,
                },
                input_norm_w:    gpu_f32(&in_w),
                post_attn_norm_w: gpu_f32(&pa_w),
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
            });
        }

        // Log QK norm detection
        if layers.first().map_or(false, |l| l.q_norm_w.is_some()) {
            log::info!("  → QK Normalization detectada: q_norm + k_norm em todas as camadas");
        }

        let (nw, _) = load_f32(&tmap.final_norm)?;
        let norm_w = gpu_f32(&nw);

        // Norm bias (para LayerNorm)
        let norm_b = match &tmap.final_norm_bias {
            Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
            _ => None,
        };

        let lm_head = LayerWeight::Bf16(if !arch.tie_word_embeddings && tensor_exists(&tmap.lm_head) {
            gpu(&tmap.lm_head)?
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            gpu(&tmap.embed)?
        });

        // ── Contabilidade de VRAM/memória ────────────────────────────────────
        // Estimativa: BF16 = 2 bytes/elem, F32 = 4 bytes/elem
        let bf16_bytes = |t: &AnyTensor| t.numel() * 2;
        let f32_bytes  = |t: &AnyTensor| t.numel() * 4;
        let layer_weight_bytes = |w: &LayerWeight| w.vram_bytes();

        let mut vram_bytes = bf16_bytes(&embed) + layer_weight_bytes(&lm_head);
        for l in &layers {
            vram_bytes += layer_weight_bytes(&l.wq) + layer_weight_bytes(&l.wk) + layer_weight_bytes(&l.wv)
                        + layer_weight_bytes(&l.wo) + layer_weight_bytes(&l.wgate)
                        + layer_weight_bytes(&l.wup) + layer_weight_bytes(&l.wdown);
            vram_bytes += f32_bytes(&l.input_norm_w) + f32_bytes(&l.post_attn_norm_w);
            vram_bytes += l.kv_k.data.numel() * 4 + l.kv_v.data.numel() * 4;
        }

        log::info!("✅ Modelo [{}] carregado — memória estimada: {:.2} GB", arch.family, vram_bytes as f64 / 1e9);

        let scratch = ForwardScratchpad::new(&cfg, &backend);

        // Initialize calibration-based sparse MLP if enabled via IARA_ROUTING=1
        let (kuramoto_router, sparse_scratch) = if std::env::var("IARA_ROUTING").unwrap_or_default() == "1" {
            let active_ratio = std::env::var("IARA_ROUTING_RATIO")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.25);

            log::info!("🔥 Sparse MLP enabled (active_ratio={:.0}%), will calibrate during first prefill...", active_ratio * 100.0);
            let ss = SparseMlpScratch::new(&cfg, &backend, active_ratio);

            (None::<crate::inference::KuramotoRouter>, Some(ss))
        } else {
            (None, None)
        };

        let layer_router = OnlineLayerRouter::new(&backend, cfg.num_layers, cfg.hidden_size);

        Ok(Self {
            embed,
            layers,
            norm_w,
            norm_b,
            lm_head,
            cfg,
            arch,
            vram_bytes,
            backend,
            scratch,
            cfc: None,
            gravity: None,
            quantized: false,
            kuramoto_router,
            sparse_scratch,
            #[cfg(feature = "experimental")]
            stimulus_tx: None,
            telemetry_enabled: false,
            layer_router,
            graph_replay_mode: std::env::var("IARA_GRAPH").map(|v| v == "2").unwrap_or(false),
        })
    }

    /// Carrega modelo a partir de arquivo `.dna` (DNA V4 — SVD factorization).
    ///
    /// Projection weights arrive as `DnaTensor::Svd { a, b, rows, cols, rank }` →
    /// wrapped as `LayerWeight::Svd` with a per-projection `[rank]` F32 scratch buffer.
    /// Non-factorized tensors (embed, lm_head, norms, biases) arrive as
    /// `DnaTensor::Full` and become `LayerWeight::Bf16` / direct `AnyTensor`.
    pub fn load_dna(
        dna_path: &Path,
        cfg: ModelCfg,
        arch: ModelArchitecture,
        backend: Arc<dyn ComputeBackend>,
    ) -> ModelResult<Self> {
        use crate::storage::dna::{DnaModel, DnaTensor};

        log::info!(
            "⏳ Carregando modelo DNA [{}]: {} camadas, hidden={}, vocab={}, path={}",
            arch.family, cfg.num_layers, cfg.hidden_size, cfg.vocab_size, dna_path.display()
        );

        let mut dna = DnaModel::load(dna_path, &backend)
            .map_err(|e| ModelError::Model(format!("Falha ao carregar .dna: {e}")))?;
        log::info!(
            "  → .dna carregado: {} tensores, max_rank={}, version={}",
            dna.tensors.len(), dna.max_rank, dna.version
        );

        let tmap = &arch.tensor_map;

        // Helper: take a Full tensor by name, returning the underlying AnyTensor.
        fn take_full(dna: &mut crate::storage::dna::DnaModel, name: &str) -> ModelResult<AnyTensor> {
            match dna.tensors.remove(name) {
                Some(DnaTensor::Full { tensor, .. }) => Ok(tensor),
                Some(DnaTensor::Svd { .. }) => Err(ModelError::Model(format!(
                    "Tensor '{}' está como SVD mas esperado Full (norm/bias/embed)", name
                ))),
                None => Err(ModelError::Model(format!("Tensor '{}' não encontrado no .dna", name))),
            }
        }

        // Helper: take any tensor by name and convert to LayerWeight.
        // SVD → LayerWeight::Svd (scratch [max_m * rank] F32 — must fit prefill batch).
        // Full → LayerWeight::Bf16.
        fn take_layer_weight(
            dna: &mut crate::storage::dna::DnaModel,
            backend: &Arc<dyn ComputeBackend>,
            name: &str,
            max_m: usize,
        ) -> ModelResult<LayerWeight> {
            match dna.tensors.remove(name) {
                Some(DnaTensor::Svd { a, b, rows, cols, rank }) => {
                    let scratch = backend.alloc_f32(&[max_m * rank]);
                    Ok(LayerWeight::Svd { a, b, scratch, n: rows, k: cols, rank })
                }
                Some(DnaTensor::Full { tensor, .. }) => Ok(LayerWeight::Bf16(tensor)),
                None => Err(ModelError::Model(format!("Tensor '{}' não encontrado no .dna", name))),
            }
        }

        // Optional take helper: None if absent, Error if present-but-wrong-kind.
        fn take_full_opt(
            dna: &mut crate::storage::dna::DnaModel,
            name_opt: Option<&String>,
        ) -> ModelResult<Option<AnyTensor>> {
            let Some(name) = name_opt else { return Ok(None); };
            if !dna.tensors.contains_key(name) { return Ok(None); }
            Ok(Some(take_full(dna, name)?))
        }

        // ── Embedding ───────────────────────────────────────────────────────
        log::info!("  → embed_tokens");
        let embed = take_full(&mut dna, &tmap.embed)?;

        // ── Camadas ─────────────────────────────────────────────────────────
        let kv_dim  = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;
        // SVD scratch sizing — cap prefill batch to 1024 to bound VRAM.
        // gemv_svd_bf16 fallback writes [m * rank] F32 to scratch on prefill (m>1).
        let svd_max_m = max_seq.min(1024);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 { log::info!("  → layers {}/{} (DNA SVD)", i, cfg.num_layers); }

            let wq_name    = tmap.layer_tensor(i, &tmap.attn_q);
            let wk_name    = tmap.layer_tensor(i, &tmap.attn_k);
            let wv_name    = tmap.layer_tensor(i, &tmap.attn_v);
            let wo_name    = tmap.layer_tensor(i, &tmap.attn_o);
            let wgate_name = tmap.layer_tensor(i, &tmap.mlp_gate);
            let wup_name   = tmap.layer_tensor(i, &tmap.mlp_up);
            let wdown_name = tmap.layer_tensor(i, &tmap.mlp_down);
            let in_norm    = tmap.layer_tensor(i, &tmap.input_norm);
            let pa_norm    = tmap.layer_tensor(i, &tmap.post_attn_norm);

            let qbn = tmap.layer_tensor_opt(i, &tmap.attn_q_bias);
            let kbn = tmap.layer_tensor_opt(i, &tmap.attn_k_bias);
            let vbn = tmap.layer_tensor_opt(i, &tmap.attn_v_bias);
            let qnn = tmap.layer_tensor_opt(i, &tmap.q_norm);
            let knn = tmap.layer_tensor_opt(i, &tmap.k_norm);

            let wq    = take_layer_weight(&mut dna, &backend, &wq_name, svd_max_m)?;
            let wk    = take_layer_weight(&mut dna, &backend, &wk_name, svd_max_m)?;
            let wv    = take_layer_weight(&mut dna, &backend, &wv_name, svd_max_m)?;
            let wo    = take_layer_weight(&mut dna, &backend, &wo_name, svd_max_m)?;
            let wgate = take_layer_weight(&mut dna, &backend, &wgate_name, svd_max_m)?;
            let wup   = take_layer_weight(&mut dna, &backend, &wup_name, svd_max_m)?;
            let wdown = take_layer_weight(&mut dna, &backend, &wdown_name, svd_max_m)?;

            let input_norm_w     = take_full(&mut dna, &in_norm)?;
            let post_attn_norm_w = take_full(&mut dna, &pa_norm)?;

            let q_bias   = take_full_opt(&mut dna, qbn.as_ref())?;
            let k_bias   = take_full_opt(&mut dna, kbn.as_ref())?;
            let v_bias   = take_full_opt(&mut dna, vbn.as_ref())?;
            let q_norm_w = take_full_opt(&mut dna, qnn.as_ref())?;
            let k_norm_w = take_full_opt(&mut dna, knn.as_ref())?;

            layers.push(TransformerLayer {
                wq, wk, wv, wo, wgate, wup, wdown,
                q_bias, k_bias, v_bias,
                q_norm_w, k_norm_w,
                input_norm_w,
                post_attn_norm_w,
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
            });
        }

        if layers.first().map_or(false, |l| l.q_norm_w.is_some()) {
            log::info!("  → QK Normalization detectada (DNA): q_norm + k_norm em todas as camadas");
        }

        // ── Final norm ──────────────────────────────────────────────────────
        let norm_w = take_full(&mut dna, &tmap.final_norm)?;
        let norm_b = take_full_opt(&mut dna, tmap.final_norm_bias.as_ref())?;

        // ── LM head ─────────────────────────────────────────────────────────
        // Tied weights path exige `lm_head` explícito no .dna (não reutilizamos `embed`
        // pois `AnyTensor` do device não é clonável). Writers devem gravar ambos.
        let lm_head = if dna.tensors.contains_key(&tmap.lm_head) {
            take_layer_weight(&mut dna, &backend, &tmap.lm_head, svd_max_m)?
        } else if arch.tie_word_embeddings {
            return Err(ModelError::Model(
                "tied word embeddings no .dna exige 'lm_head' presente como tensor separado. \
                 Rebuild do .dna incluindo lm_head explícito.".into()
            ));
        } else {
            return Err(ModelError::Model(format!(
                "Tensor lm_head '{}' não encontrado no .dna", tmap.lm_head
            )));
        };

        // ── VRAM accounting ────────────────────────────────────────────────
        let bf16_bytes = |t: &AnyTensor| t.numel() * 2;
        let f32_bytes  = |t: &AnyTensor| t.numel() * 4;
        let layer_weight_bytes = |w: &LayerWeight| w.vram_bytes();

        let mut vram_bytes = bf16_bytes(&embed) + layer_weight_bytes(&lm_head);
        for l in &layers {
            vram_bytes += layer_weight_bytes(&l.wq) + layer_weight_bytes(&l.wk) + layer_weight_bytes(&l.wv)
                        + layer_weight_bytes(&l.wo) + layer_weight_bytes(&l.wgate)
                        + layer_weight_bytes(&l.wup) + layer_weight_bytes(&l.wdown);
            vram_bytes += f32_bytes(&l.input_norm_w) + f32_bytes(&l.post_attn_norm_w);
            vram_bytes += l.kv_k.data.numel() * 4 + l.kv_v.data.numel() * 4;
        }

        log::info!("✅ Modelo DNA [{}] carregado — memória estimada: {:.2} GB",
            arch.family, vram_bytes as f64 / 1e9);

        let scratch = ForwardScratchpad::new(&cfg, &backend);
        let layer_router = OnlineLayerRouter::new(&backend, cfg.num_layers, cfg.hidden_size);

        Ok(Self {
            embed,
            layers,
            norm_w,
            norm_b,
            lm_head,
            cfg,
            arch,
            vram_bytes,
            backend,
            scratch,
            cfc: None,
            gravity: None,
            quantized: false,
            kuramoto_router: None,
            sparse_scratch: None,
            #[cfg(feature = "experimental")]
            stimulus_tx: None,
            telemetry_enabled: false,
            layer_router,
            graph_replay_mode: std::env::var("IARA_GRAPH").map(|v| v == "2").unwrap_or(false),
        })
    }

    /// Carrega modelo a partir de arquivo GGUF (formato Ollama/llama.cpp).
    /// Usa kernels nativos Q4_K/Q6_K GEMV: upload dos blocos raw quantizados
    /// direto para GPU, dequantização on-the-fly em F32 durante dot product.
    /// Sem intermediário BF16, sem compound error entre camadas.
    pub fn load_gguf(
        gguf_path: &Path,
        cfg: ModelCfg,
        arch: ModelArchitecture,
        backend: Arc<dyn ComputeBackend>,
    ) -> ModelResult<Self> {
        use crate::storage::gguf::GgufFile;

        log::info!("⏳ Carregando modelo GGUF [{}]: {}", arch.family, gguf_path.display());

        let gguf = GgufFile::open(gguf_path)
            .map_err(|e| ModelError::Model(format!("Falha ao abrir GGUF: {}", e)))?;

        // Log model info from GGUF metadata
        if let Some(name) = gguf.get_str("general.name") {
            log::info!("  → Nome: {}", name);
        }
        if let Some(arch_str) = gguf.get_str("general.architecture") {
            log::info!("  → Arquitetura GGUF: {}", arch_str);
        }

        // Count tensor types
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for t in gguf.tensors.values() {
            *type_counts.entry(format!("{:?}", t.dtype)).or_insert(0) += 1;
        }
        log::info!("  → Tensores por tipo: {:?}", type_counts);

        // ── Helper: upload GGUF tensor as native quantized LayerWeight ──
        // For Q4_K/Q6_K: upload raw block bytes directly → native GEMV kernel
        // For F32/BF16/other: dequantize → BF16 on GPU (for embed_lookup, norms)
        //
        // GGUF shape convention: [ne0, ne1] = [cols(k), rows(n)]
        // PyTorch convention: [rows(n), cols(k)]

        // Upload quantized weight as native LayerWeight (Q4K, Q6K, or fallback BF16)
        let gpu_weight_from_gguf = |gguf_name: &str| -> ModelResult<LayerWeight> {
            use crate::storage::gguf::GgmlType;
            let info = gguf.tensors.get(gguf_name)
                .ok_or_else(|| ModelError::Model(format!("Tensor '{}' não encontrado", gguf_name)))?;
            // For 2D weights: GGUF shape [ne0=k, ne1=n] → n=ne1, k=ne0
            let (n, k) = if info.shape.len() == 2 {
                (info.shape[1], info.shape[0])
            } else {
                // 1D tensors: treat as [1, k]
                (1, info.shape[0])
            };

            // Debug: IARA_GGUF_NATIVE=0 → fallback all to BF16 (old behavior)
            //        IARA_GGUF_NATIVE=q4k → only Q4_K native, Q6_K→BF16
            //        IARA_GGUF_NATIVE=1 or unset → all native
            let native_mode = std::env::var("IARA_GGUF_NATIVE").unwrap_or_else(|_| "1".to_string());

            let use_native_q4k = native_mode != "0";
            let use_native_q6k = native_mode == "1" || native_mode == "q6k";

            match info.dtype {
                GgmlType::Q4_K if use_native_q4k => {
                    let raw = gguf.tensor_bytes(gguf_name)
                        .ok_or_else(|| ModelError::Model(format!("Raw bytes de '{}' não encontrados", gguf_name)))?;
                    let tensor = backend.upload_raw(raw, &[n, k]);
                    Ok(LayerWeight::Q4K { data: tensor, n, k })
                }
                GgmlType::Q6_K if use_native_q6k => {
                    let raw = gguf.tensor_bytes(gguf_name)
                        .ok_or_else(|| ModelError::Model(format!("Raw bytes de '{}' não encontrados", gguf_name)))?;
                    let tensor = backend.upload_raw(raw, &[n, k]);
                    Ok(LayerWeight::Q6K { data: tensor, n, k })
                }
                _ => {
                    // Fallback: dequantize → BF16
                    let bf16_data = gguf.dequantize_bf16(gguf_name)
                        .ok_or_else(|| ModelError::Model(
                            format!("Falha ao dequantizar tensor GGUF '{}'", gguf_name)))?;
                    let shape: Vec<usize> = info.shape.iter().rev().cloned().collect();
                    Ok(LayerWeight::Bf16(backend.upload_bf16(&bf16_data, &shape)))
                }
            }
        };

        // Upload as BF16 (for embedding, lm_head — used by embed_lookup kernel)
        let gpu_bf16_from_gguf = |gguf_name: &str| -> ModelResult<AnyTensor> {
            let bf16_data = gguf.dequantize_bf16(gguf_name)
                .ok_or_else(|| ModelError::Model(
                    format!("Falha ao dequantizar tensor GGUF '{}'", gguf_name)))?;
            let info = gguf.tensors.get(gguf_name)
                .ok_or_else(|| ModelError::Model(format!("Tensor '{}' não encontrado", gguf_name)))?;
            let shape: Vec<usize> = info.shape.iter().rev().cloned().collect();
            Ok(backend.upload_bf16(&bf16_data, &shape))
        };

        // Upload as F32 (for norms, biases)
        let gpu_f32_from_gguf = |gguf_name: &str| -> ModelResult<AnyTensor> {
            let f32_data = gguf.dequantize_f32(gguf_name)
                .ok_or_else(|| ModelError::Model(
                    format!("Falha ao dequantizar tensor GGUF '{}'", gguf_name)))?;
            let info = gguf.tensors.get(gguf_name)
                .ok_or_else(|| ModelError::Model(format!("Tensor '{}' não encontrado", gguf_name)))?;
            let shape: Vec<usize> = info.shape.iter().rev().cloned().collect();
            Ok(backend.upload_f32(&f32_data, &shape))
        };

        // ── Map GGUF tensor names to our architecture tensor names ──
        // GGUF uses names like "blk.0.attn_q.weight" instead of "model.layers.0.self_attn.q_proj.weight"
        let tmap = &arch.tensor_map;

        // GGUF name mapping: try GGUF-style names first, fall back to HF-style
        let gguf_name = |hf_name: &str| -> String {
            // Convert HF-style "model.layers.X.self_attn.q_proj.weight" to GGUF "blk.X.attn_q.weight"
            let s = hf_name
                .replace("model.layers.", "blk.")
                .replace("model.embed_tokens.weight", "token_embd.weight")
                .replace("model.norm.weight", "output_norm.weight")
                .replace("lm_head.weight", "output.weight")
                .replace(".self_attn.q_proj.weight", ".attn_q.weight")
                .replace(".self_attn.k_proj.weight", ".attn_k.weight")
                .replace(".self_attn.v_proj.weight", ".attn_v.weight")
                .replace(".self_attn.o_proj.weight", ".attn_output.weight")
                .replace(".mlp.gate_proj.weight", ".ffn_gate.weight")
                .replace(".mlp.up_proj.weight", ".ffn_up.weight")
                .replace(".mlp.down_proj.weight", ".ffn_down.weight")
                .replace(".input_layernorm.weight", ".attn_norm.weight")
                .replace(".post_attention_layernorm.weight", ".ffn_norm.weight")
                .replace(".self_attn.q_proj.bias", ".attn_q.bias")
                .replace(".self_attn.k_proj.bias", ".attn_k.bias")
                .replace(".self_attn.v_proj.bias", ".attn_v.bias");
            s
        };

        // Try to find tensor by HF name (converted to GGUF), or direct GGUF name
        let resolve_name = |hf_name: &str| -> String {
            let gn = gguf_name(hf_name);
            if gguf.tensors.contains_key(&gn) {
                gn
            } else if gguf.tensors.contains_key(hf_name) {
                hf_name.to_string()
            } else {
                gn // will fail with clear error
            }
        };

        // ── Embedding ──
        let embed_name = resolve_name(&tmap.embed);
        let embed_info = gguf.tensors.get(&embed_name)
            .ok_or_else(|| ModelError::Model(format!("embed tensor '{}' não encontrado", embed_name)))?;
        log::info!("  → embed_tokens: {} (shape={:?}, dtype={:?})", embed_name, embed_info.shape, embed_info.dtype);
        let embed = gpu_bf16_from_gguf(&embed_name)?;

        // ── Camadas ──
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 { log::info!("  → layers {}/{} (GGUF native Q4K/Q6K)", i, cfg.num_layers); }

            let wq_name = resolve_name(&tmap.layer_tensor(i, &tmap.attn_q));
            let wk_name = resolve_name(&tmap.layer_tensor(i, &tmap.attn_k));
            let wv_name = resolve_name(&tmap.layer_tensor(i, &tmap.attn_v));
            let wo_name = resolve_name(&tmap.layer_tensor(i, &tmap.attn_o));
            let wgate_name = resolve_name(&tmap.layer_tensor(i, &tmap.mlp_gate));
            let wup_name = resolve_name(&tmap.layer_tensor(i, &tmap.mlp_up));
            let wdown_name = resolve_name(&tmap.layer_tensor(i, &tmap.mlp_down));
            let in_norm_name = resolve_name(&tmap.layer_tensor(i, &tmap.input_norm));
            let pa_norm_name = resolve_name(&tmap.layer_tensor(i, &tmap.post_attn_norm));

            if i == 0 {
                // Log ALL tensor names and shapes from first layer for debugging
                for (label, name) in [
                    ("wq", &wq_name), ("wk", &wk_name), ("wv", &wv_name), ("wo", &wo_name),
                    ("wgate", &wgate_name), ("wup", &wup_name), ("wdown", &wdown_name),
                    ("in_norm", &in_norm_name), ("pa_norm", &pa_norm_name),
                ] {
                    if let Some(info) = gguf.tensors.get(name) {
                        let rev_shape: Vec<usize> = info.shape.iter().rev().cloned().collect();
                        log::info!("  → L0 {:8}: {} → {:?} (gguf {:?}) {:?}", label, name, rev_shape, info.shape, info.dtype);
                    } else {
                        log::warn!("  → L0 {:8}: {} → NOT FOUND!", label, name);
                    }
                }
            }

            // Optional biases
            let qb_name = tmap.layer_tensor_opt(i, &tmap.attn_q_bias).map(|n| resolve_name(&n));
            let kb_name = tmap.layer_tensor_opt(i, &tmap.attn_k_bias).map(|n| resolve_name(&n));
            let vb_name = tmap.layer_tensor_opt(i, &tmap.attn_v_bias).map(|n| resolve_name(&n));

            layers.push(TransformerLayer {
                wq:    gpu_weight_from_gguf(&wq_name)?,
                wk:    gpu_weight_from_gguf(&wk_name)?,
                wv:    gpu_weight_from_gguf(&wv_name)?,
                wo:    gpu_weight_from_gguf(&wo_name)?,
                wgate: gpu_weight_from_gguf(&wgate_name)?,
                wup:   gpu_weight_from_gguf(&wup_name)?,
                wdown: gpu_weight_from_gguf(&wdown_name)?,
                q_bias: match &qb_name {
                    Some(name) if gguf.tensors.contains_key(name) => Some(gpu_f32_from_gguf(name)?),
                    _ => None,
                },
                k_bias: match &kb_name {
                    Some(name) if gguf.tensors.contains_key(name) => Some(gpu_f32_from_gguf(name)?),
                    _ => None,
                },
                v_bias: match &vb_name {
                    Some(name) if gguf.tensors.contains_key(name) => Some(gpu_f32_from_gguf(name)?),
                    _ => None,
                },
                q_norm_w: None, // Qwen2.5 GGUF doesn't have separate QK norms
                k_norm_w: None,
                input_norm_w: gpu_f32_from_gguf(&in_norm_name)?,
                post_attn_norm_w: gpu_f32_from_gguf(&pa_norm_name)?,
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
            });
        }

        let norm_name = resolve_name(&tmap.final_norm);
        let norm_w = gpu_f32_from_gguf(&norm_name)?;
        let norm_b = None; // RMSNorm doesn't use bias

        // lm_head (may be tied with embeddings)
        let lm_head_name = resolve_name(&tmap.lm_head);
        // lm_head: use native quant if separate, native quant from embed if tied
        let lm_head = if gguf.tensors.contains_key(&lm_head_name) {
            gpu_weight_from_gguf(&lm_head_name)?
        } else {
            // Tied weights: load embed tensor in native quant format (Q4K/Q6K)
            // instead of dequantizing to BF16 (saves ~350MB VRAM and uses fast GEMV)
            log::info!("  → lm_head: usando embed_tokens (tied weights, native quant)");
            gpu_weight_from_gguf(&embed_name)?
        };

        // ── Verify GPU data integrity ──
        // Download first few values from GPU and compare with CPU dequant
        if std::env::var("IARA_DEBUG_GGUF").map(|v| v == "1").unwrap_or(false) {
            log::info!("🔍 GGUF debug: verifying GPU data integrity...");

            // Check embed tensor (BF16)
            let embed_gpu = backend.download_bf16_as_f32(&embed);
            log::info!("  embed GPU first5: {:?}", &embed_gpu[..5.min(embed_gpu.len())]);

            // Check layer 0 wq type
            match &layers[0].wq {
                LayerWeight::Bf16(t) => {
                    let wq_gpu = backend.download_bf16_as_f32(t);
                    log::info!("  L0.wq BF16 GPU first5: {:?}", &wq_gpu[..5.min(wq_gpu.len())]);
                }
                LayerWeight::Q4K { n, k, .. } => log::info!("  L0.wq Q4K native [{}×{}]", n, k),
                LayerWeight::Q6K { n, k, .. } => log::info!("  L0.wq Q6K native [{}×{}]", n, k),
                LayerWeight::Q4(_) => log::info!("  L0.wq Q4 (IARA quantized)"),
                LayerWeight::Ternary { n, k, scale, .. } => log::info!("  L0.wq Ternary [{}×{}] scale={:.4}", n, k, scale),
                LayerWeight::W8 { n, k, .. } => log::info!("  L0.wq W8 INT8 [{}×{}]", n, k),
                LayerWeight::Svd { n, k, rank, .. } => log::info!("  L0.wq SVD [{}×{}] rank={}", n, k, rank),
            }

            // Check layer 0 input_norm (F32)
            let norm_gpu = backend.download_f32(&layers[0].input_norm_w);
            log::info!("  L0.norm GPU first5: {:?}", &norm_gpu[..5.min(norm_gpu.len())]);

            // Check layer 0 q_bias (F32)
            if let Some(ref qb) = layers[0].q_bias {
                let qb_gpu = backend.download_f32(qb);
                log::info!("  L0.q_bias GPU first5: {:?}", &qb_gpu[..5.min(qb_gpu.len())]);
            } else {
                log::warn!("  L0.q_bias: NONE!");
            }
        }

        // ── VRAM accounting ──
        let bf16_bytes = |t: &AnyTensor| t.numel() * 2;
        let f32_bytes = |t: &AnyTensor| t.numel() * 4;
        let mut vram_bytes = embed.numel() * 2 + lm_head.vram_bytes();
        for l in &layers {
            vram_bytes += l.wq.vram_bytes() + l.wk.vram_bytes()
                        + l.wv.vram_bytes() + l.wo.vram_bytes()
                        + l.wgate.vram_bytes() + l.wup.vram_bytes()
                        + l.wdown.vram_bytes();
            vram_bytes += f32_bytes(&l.input_norm_w) + f32_bytes(&l.post_attn_norm_w);
            vram_bytes += l.kv_k.data.numel() * 4 + l.kv_v.data.numel() * 4;
        }

        let scratch = ForwardScratchpad::new(&cfg, &backend);
        let layer_router = OnlineLayerRouter::new(&backend, cfg.num_layers, cfg.hidden_size);

        log::info!("✅ Modelo GGUF [{}] carregado — VRAM: {:.2} GB (native Q4K/Q6K GEMV)", arch.family, vram_bytes as f64 / 1e9);

        Ok(Self {
            embed,
            layers,
            norm_w,
            norm_b,
            lm_head,
            cfg,
            arch,
            vram_bytes,
            backend,
            scratch,
            cfc: None,
            gravity: None,
            quantized: false,
            kuramoto_router: None,
            sparse_scratch: None,
            #[cfg(feature = "experimental")]
            stimulus_tx: None,
            telemetry_enabled: false,
            layer_router,
            graph_replay_mode: std::env::var("IARA_GRAPH").map(|v| v == "2").unwrap_or(false),
        })
    }

    /// Carrega modelo com quantização INT4 on-the-fly.
    /// Pesos são quantizados de BF16→Q4 durante o carregamento.
    /// Reduz VRAM em ~4x (ex: 16GB → 4GB para Qwen3-8B).
    pub fn load_quantized(
        model_dir: &Path,
        cfg: ModelCfg,
        arch: ModelArchitecture,
        backend: Arc<dyn ComputeBackend>,
        enable_cfc: bool,
    ) -> ModelResult<Self> {
        use crate::lnn::cfc::{CfcConfig, CfcLayer};
        use crate::lnn::gravity::{GravityConfig, GravitySystem};

        log::info!(
            "⏳ Carregando modelo [{}] com quantização Q4: {} camadas, hidden={}, vocab={}",
            arch.family, cfg.num_layers, cfg.hidden_size, cfg.vocab_size
        );

        // ── Tentar carregar cache pré-quantizado ──────────────────────────────
        let group_size = 32usize;
        let use_cache = if let Ok(Some((cached_tensors, cached_shapes, cached_group_size))) = load_quantized_cache(model_dir) {
            if cached_group_size == group_size {
                log::info!("⚡ Cache quantizado encontrado e válido, usando...");
                Some((cached_tensors, cached_shapes))
            } else {
                log::warn!(
                    "⚠️  Cache com group_size incompatível: {} vs esperado {}, re-quantizando...",
                    cached_group_size, group_size
                );
                None
            }
        } else {
            None
        };

        // ── Mmaps ────────────────────────────────────────────────────────────
        // Sempre carregamos mmaps: pesos não-quantizados (norms, lm_head, biases,
        // embeddings) precisam de safetensors independente do cache Q4.
        // O cache Q4 otimiza a quantização CPU (BF16→INT4), não a leitura de pesos.
        let mut shards: HashMap<String, memmap2::Mmap> = HashMap::new();
        let tensor_shard_map: Option<HashMap<String, String>>;

        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let weight_map = index["weight_map"].as_object()
                .ok_or_else(|| ModelError::Model("weight_map inválido".into()))?;
            let mut tsm: HashMap<String, String> = HashMap::new();
            for (k, v) in weight_map {
                if let Some(s) = v.as_str() { tsm.insert(k.clone(), s.to_string()); }
            }
            for shard_name in tsm.values() {
                if shards.contains_key(shard_name) { continue; }
                let file = std::fs::File::open(model_dir.join(shard_name))?;
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                shards.insert(shard_name.clone(), mmap);
            }
            tensor_shard_map = Some(tsm);
            log::info!("  → multi-shard ({} shards)", shards.len());
        } else {
            let single = model_dir.join("model.safetensors");
            if !single.exists() {
                return Err(ModelError::Model(
                    "Nenhum safetensors encontrado".into()
                ));
            }
            let file = std::fs::File::open(&single)?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            log::info!("  → single-shard ({:.2} GB)", mmap.len() as f64 / 1e9);
            shards.insert("model.safetensors".to_string(), mmap);
            tensor_shard_map = None;
        }

        // ── Helpers ──────────────────────────────────────────────────────────
        let find_mmap = |name: &str| -> Option<&memmap2::Mmap> {
            match &tensor_shard_map {
                Some(tsm) => tsm.get(name).and_then(|s| shards.get(s)),
                None => shards.get("model.safetensors"),
            }
        };
        let tensor_exists = |name: &str| -> bool {
            match &tensor_shard_map {
                Some(tsm) => tsm.contains_key(name),
                None => shards.get("model.safetensors")
                    .map(|m| shard_has_tensor(m, name))
                    .unwrap_or(false),
            }
        };

        let load_f32 = |name: &str| -> ModelResult<(Vec<f32>, Vec<usize>)> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            read_bf16_f32(mmap, name)
        };

        // ── Helpers para quantização ─────────────────────────────────────────
        // Armazena tensores quantizados para salvar cache depois
        let mut quantized_cache: HashMap<String, QuantizationCpuResult> = HashMap::new();
        let mut quantized_shapes: HashMap<String, [usize; 2]> = HashMap::new();

        let mut quantize_weight = |name: &str| -> ModelResult<LayerWeight> {
            // Se temos cache, usar dele
            if let Some((cached_tensors, cached_shapes)) = &use_cache {
                if let Some(cpu_result) = cached_tensors.get(name) {
                    if let Some(shape) = cached_shapes.get(name) {
                        // Upload direto do cache
                        let quantized = upload_quantized(cpu_result, &[shape[0], shape[1]], group_size, &backend)
                            .map_err(|e| ModelError::Model(format!("Upload quantizado falhou: {}", e)))?;
                        return Ok(LayerWeight::Q4(quantized));
                    }
                }
            }

            // Caso contrário, quantizar do zero
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            let (raw_bytes, shape) = shard_tensor_bytes(mmap, name)?;

            // Casting zero-copy: &[u8] → &[u16] (BF16 bits)
            let bits = unsafe {
                std::slice::from_raw_parts(
                    raw_bytes.as_ptr() as *const u16,
                    raw_bytes.len() / 2
                )
            };

            // Quantizar BF16 → INT4 na CPU
            let cpu_result = quantize_bf16_to_q4(bits, &shape, group_size)
                .map_err(|e| ModelError::Model(format!("Quantização falhou: {}", e)))?;

            // Armazenar para cache se não estamos usando cache
            if use_cache.is_none() {
                quantized_cache.insert(name.to_string(), cpu_result.clone());
                quantized_shapes.insert(name.to_string(), [shape[0], shape[1]]);
            }

            // Upload para GPU
            let quantized = upload_quantized(&cpu_result, &shape, group_size, &backend)
                .map_err(|e| ModelError::Model(format!("Upload quantizado falhou: {}", e)))?;

            Ok(LayerWeight::Q4(quantized))
        };

        // Embedding e lm_head NUNCA são quantizados (muito sensíveis)
        let gpu_bf16 = |name: &str| -> ModelResult<AnyTensor> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            let (raw_bytes, shape) = shard_tensor_bytes(mmap, name)?;
            let bits = unsafe {
                std::slice::from_raw_parts(
                    raw_bytes.as_ptr() as *const u16,
                    raw_bytes.len() / 2
                )
            };
            Ok(backend.upload_bf16(bits, &shape))
        };

        let gpu_f32 = |data: &[f32]| -> AnyTensor {
            backend.upload_f32(data, &[data.len()])
        };

        // ── Nomes de tensores (via architecture descriptor) ─────────────────
        let tmap = &arch.tensor_map;

        // ── Embedding (BF16) ─────────────────────────────────────────────────
        log::info!("  → embed_tokens (BF16)");
        let embed = gpu_bf16(&tmap.embed)?;

        // ── Camadas (Q4 quantizadas) ─────────────────────────────────────────
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 { log::info!("  → layers {}/{} (Q4)", i, cfg.num_layers); }

            let (in_w, _) = load_f32(&tmap.layer_tensor(i, &tmap.input_norm))?;
            let (pa_w, _) = load_f32(&tmap.layer_tensor(i, &tmap.post_attn_norm))?;

            let qbn = tmap.layer_tensor_opt(i, &tmap.attn_q_bias);
            let kbn = tmap.layer_tensor_opt(i, &tmap.attn_k_bias);
            let vbn = tmap.layer_tensor_opt(i, &tmap.attn_v_bias);
            let qnn = tmap.layer_tensor_opt(i, &tmap.q_norm);
            let knn = tmap.layer_tensor_opt(i, &tmap.k_norm);

            layers.push(TransformerLayer {
                wq:    quantize_weight(&tmap.layer_tensor(i, &tmap.attn_q))?,
                wk:    quantize_weight(&tmap.layer_tensor(i, &tmap.attn_k))?,
                wv:    quantize_weight(&tmap.layer_tensor(i, &tmap.attn_v))?,
                wo:    quantize_weight(&tmap.layer_tensor(i, &tmap.attn_o))?,
                wgate: quantize_weight(&tmap.layer_tensor(i, &tmap.mlp_gate))?,
                wup:   quantize_weight(&tmap.layer_tensor(i, &tmap.mlp_up))?,
                wdown: quantize_weight(&tmap.layer_tensor(i, &tmap.mlp_down))?,
                q_bias: match &qbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                k_bias: match &kbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                v_bias: match &vbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                q_norm_w: match &qnn {
                    Some(name) if tensor_exists(name) => {
                        log::debug!("  → q_norm layer {}", i);
                        Some(gpu_f32(&load_f32(name)?.0))
                    },
                    _ => None,
                },
                k_norm_w: match &knn {
                    Some(name) if tensor_exists(name) => {
                        log::debug!("  → k_norm layer {}", i);
                        Some(gpu_f32(&load_f32(name)?.0))
                    },
                    _ => None,
                },
                input_norm_w:     gpu_f32(&in_w),
                post_attn_norm_w: gpu_f32(&pa_w),
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
            });
        }

        // Log KV-cache mode
        match kv_cache::KvCacheMode::detect() {
            kv_cache::KvCacheMode::F32 => log::info!("  → KV-cache: F32 (precisão total)"),
            kv_cache::KvCacheMode::Int8 => log::info!("  → KV-cache: INT8 (4x menos VRAM)"),
            kv_cache::KvCacheMode::Wht => log::info!("  → KV-cache: WHT+INT8 (PolarQuant, 4x menos VRAM + outlier redistribution)"),
        }

        // Log QK norm detection
        if layers.first().map_or(false, |l| l.q_norm_w.is_some()) {
            log::info!("  → QK Normalization detectada: q_norm + k_norm em todas as camadas");
        }

        let (nw, _) = load_f32(&tmap.final_norm)?;
        let norm_w = gpu_f32(&nw);

        // Norm bias (para LayerNorm)
        let norm_b = match &tmap.final_norm_bias {
            Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
            _ => None,
        };

        // lm_head em BF16 (NUNCA quantizar — muito sensível, corrompe distribuição de logits)
        let lm_head = LayerWeight::Bf16(if !arch.tie_word_embeddings && tensor_exists(&tmap.lm_head) {
            gpu_bf16(&tmap.lm_head)?
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            gpu_bf16(&tmap.embed)?
        });

        // ── Contabilidade de VRAM (com quantização) ──────────────────────────
        let bf16_bytes = |t: &AnyTensor| t.numel() * 2;
        let f32_bytes = |t: &AnyTensor| t.numel() * 4;

        let lm_head_bytes = lm_head.vram_bytes();
        let mut vram_bytes = bf16_bytes(&embed) + lm_head_bytes;
        for l in &layers {
            // Pesos quantizados (Q4): ~1 byte por 2 valores + scales/zeros
            let q4_weight_bytes = |qt: &QuantizedTensor| {
                let [rows, cols] = qt.shape;
                // packed: rows × cols/2 bytes
                // scales: rows × (cols/group_size) × 4 bytes
                // zeros: rows × (cols/group_size) × 4 bytes
                let packed_bytes = rows * cols / 2;
                let n_groups = cols / qt.group_size;
                let scales_bytes = rows * n_groups * 4;
                let zeros_bytes = rows * n_groups * 4;
                packed_bytes + scales_bytes + zeros_bytes
            };

            vram_bytes += l.wq.vram_bytes() + l.wk.vram_bytes() + l.wv.vram_bytes()
                        + l.wo.vram_bytes() + l.wgate.vram_bytes()
                        + l.wup.vram_bytes() + l.wdown.vram_bytes();

            vram_bytes += f32_bytes(&l.input_norm_w) + f32_bytes(&l.post_attn_norm_w);
            vram_bytes += l.kv_k.data.numel() * 4 + l.kv_v.data.numel() * 4;
        }

        // ── CfC e Gravity (opcional) ─────────────────────────────────────────
        let cfc = if enable_cfc {
            log::info!("  → ativando CfC correction layer");
            let cfc_cfg = CfcConfig::new(cfg.hidden_size, 8); // cfc_dim = hidden/8
            vram_bytes += CfcLayer::new(&backend, cfc_cfg.clone()).vram_bytes();
            Some(CfcLayer::new(&backend, cfc_cfg))
        } else {
            None
        };

        let gravity = if enable_cfc {
            log::info!("  → ativando gravity scoring");
            Some(GravitySystem::new(
                &backend,
                cfg.max_seq_len,
                cfg.num_kv_heads,
                cfg.head_dim,
                GravityConfig::default(),
            ))
        } else {
            None
        };

        // ── Salvar cache de quantização se não estava usando cache ──────────────
        if use_cache.is_none() && !quantized_cache.is_empty() {
            if let Err(e) = save_quantized_cache(model_dir, &quantized_cache, &quantized_shapes, group_size) {
                log::warn!("⚠️  Falha ao salvar cache quantizado: {}", e);
            }
        }

        log::info!(
            "✅ Modelo [{}] carregado (Q4) — memória estimada: {:.2} GB (4x redução vs BF16)",
            arch.family, vram_bytes as f64 / 1e9
        );

        let scratch = ForwardScratchpad::new(&cfg, &backend);

        // Initialize calibration-based sparse MLP if enabled via IARA_ROUTING=1
        let (kuramoto_router, sparse_scratch) = if std::env::var("IARA_ROUTING").unwrap_or_default() == "1" {
            let active_ratio = std::env::var("IARA_ROUTING_RATIO")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.25);

            log::info!("🔥 Sparse MLP enabled (active_ratio={:.0}%), will calibrate during first prefill...", active_ratio * 100.0);
            let ss = SparseMlpScratch::new(&cfg, &backend, active_ratio);

            (None::<crate::inference::KuramotoRouter>, Some(ss))
        } else {
            (None, None)
        };

        let layer_router = OnlineLayerRouter::new(&backend, cfg.num_layers, cfg.hidden_size);

        Ok(Self {
            embed,
            layers,
            norm_w,
            norm_b,
            lm_head,
            cfg,
            arch,
            vram_bytes,
            backend,
            scratch,
            cfc,
            gravity,
            quantized: true,
            kuramoto_router,
            sparse_scratch,
            #[cfg(feature = "experimental")]
            stimulus_tx: None,
            telemetry_enabled: false,
            layer_router,
            graph_replay_mode: std::env::var("IARA_GRAPH").map(|v| v == "2").unwrap_or(false),
        })
    }

    /// Carrega modelo com quantização ternária {-1, 0, +1} (BitNet 1.58).
    /// Pesos quantizados para 2 bits/weight — 16× menor que BF16.
    /// GEMV usa apenas add/sub, zero multiplicações.
    /// Layers 7-20 skipped (BI≈0) para decode — 2× speedup.
    pub fn load_ternary(
        model_dir: &Path,
        cfg: ModelCfg,
        arch: ModelArchitecture,
        backend: Arc<dyn ComputeBackend>,
    ) -> ModelResult<Self> {
        use crate::storage::bitnet::quantize_to_ternary;

        log::info!(
            "⏳ Carregando modelo [{}] com quantização TERNARY: {} camadas, hidden={}, vocab={}",
            arch.family, cfg.num_layers, cfg.hidden_size, cfg.vocab_size
        );

        // ── Mmaps ────────────────────────────────────────────────────────────
        let mut shards: HashMap<String, memmap2::Mmap> = HashMap::new();
        let tensor_shard_map: Option<HashMap<String, String>>;

        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let weight_map = index["weight_map"].as_object()
                .ok_or_else(|| ModelError::Model("weight_map inválido".into()))?;
            let mut tsm: HashMap<String, String> = HashMap::new();
            for (k, v) in weight_map {
                if let Some(s) = v.as_str() { tsm.insert(k.clone(), s.to_string()); }
            }
            for shard_name in tsm.values() {
                if shards.contains_key(shard_name) { continue; }
                let file = std::fs::File::open(model_dir.join(shard_name))?;
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                shards.insert(shard_name.clone(), mmap);
            }
            tensor_shard_map = Some(tsm);
        } else {
            let single = model_dir.join("model.safetensors");
            if !single.exists() {
                return Err(ModelError::Model("Nenhum safetensors encontrado".into()));
            }
            let file = std::fs::File::open(&single)?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            log::info!("  → single-shard ({:.2} GB)", mmap.len() as f64 / 1e9);
            shards.insert("model.safetensors".to_string(), mmap);
            tensor_shard_map = None;
        }

        // ── Helpers ──────────────────────────────────────────────────────────
        let find_mmap = |name: &str| -> Option<&memmap2::Mmap> {
            match &tensor_shard_map {
                Some(tsm) => tsm.get(name).and_then(|s| shards.get(s)),
                None => shards.get("model.safetensors"),
            }
        };
        let tensor_exists = |name: &str| -> bool {
            match &tensor_shard_map {
                Some(tsm) => tsm.contains_key(name),
                None => shards.get("model.safetensors")
                    .map(|m| shard_has_tensor(m, name))
                    .unwrap_or(false),
            }
        };

        let load_f32 = |name: &str| -> ModelResult<(Vec<f32>, Vec<usize>)> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            read_bf16_f32(mmap, name)
        };

        // Ternary quantize: BF16 → {-1,0,+1} packed 2-bit
        let ternary_weight = |name: &str| -> ModelResult<LayerWeight> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            let (raw_bytes, shape) = shard_tensor_bytes(mmap, name)?;
            let bits = unsafe {
                std::slice::from_raw_parts(raw_bytes.as_ptr() as *const u16, raw_bytes.len() / 2)
            };
            let tw = quantize_to_ternary(bits, [shape[0], shape[1]]);
            let data = backend.upload_ternary(&tw.packed, &[tw.shape[0], (tw.shape[1] + 3) / 4]);
            Ok(LayerWeight::Ternary {
                data,
                n: tw.shape[0],
                k: tw.shape[1],
                scale: tw.scale,
            })
        };

        let gpu_bf16 = |name: &str| -> ModelResult<AnyTensor> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            let (raw_bytes, shape) = shard_tensor_bytes(mmap, name)?;
            let bits = unsafe {
                std::slice::from_raw_parts(raw_bytes.as_ptr() as *const u16, raw_bytes.len() / 2)
            };
            Ok(backend.upload_bf16(bits, &shape))
        };

        let gpu_f32 = |data: &[f32]| -> AnyTensor {
            backend.upload_f32(data, &[data.len()])
        };

        let tmap = &arch.tensor_map;

        // ── Embedding (BF16 — too sensitive for ternary) ─────────────────────
        log::info!("  → embed_tokens (BF16)");
        let embed = gpu_bf16(&tmap.embed)?;

        // ── Hybrid precision: BF16 for critical layers, Ternary for rest ────
        //
        // IARA_HYBRID=1           → enable hybrid mode
        // IARA_BF16_LAYERS=0,1,2  → explicit list of BF16 layers
        // IARA_HYBRID_BI=0.30     → auto: layers with BI >= threshold stay BF16
        //
        // Default BI scores from Qwen2.5-1.5B calibration:
        //   L0=0.98, L1=0.78, L2=0.36, L3=0.23, L5=0.27, L6-25≈0.17, L26=0.22, L27=0.20
        // With threshold 0.30: L0, L1, L2 stay BF16 (~180MB extra vs full ternary)

        let hybrid_enabled = std::env::var("IARA_HYBRID")
            .map(|v| v == "1").unwrap_or(false);

        let bf16_layers: std::collections::HashSet<usize> = if hybrid_enabled {
            if let Ok(list) = std::env::var("IARA_BF16_LAYERS") {
                // Explicit: "0,1,2,27"
                let set: std::collections::HashSet<usize> = list.split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                log::info!("🔬 Hybrid mode: BF16 layers (explicit) = {:?}", set);
                set
            } else {
                // Auto from BI threshold (default 0.30)
                let bi_thresh: f32 = std::env::var("IARA_HYBRID_BI")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(0.30);
                // Calibrated BI scores for Qwen2.5-1.5B
                let calibrated_bi: [f32; 28] = [
                    0.98, 0.78, 0.36, 0.23, 0.19, 0.27, 0.18, 0.17,
                    0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17,
                    0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17,
                    0.17, 0.17, 0.22, 0.20,
                ];
                let set: std::collections::HashSet<usize> = (0..cfg.num_layers)
                    .filter(|&i| i < calibrated_bi.len() && calibrated_bi[i] >= bi_thresh)
                    .collect();
                log::info!("🔬 Hybrid mode: BF16 layers (BI >= {:.2}) = {:?}", bi_thresh, set);
                set
            }
        } else {
            std::collections::HashSet::new()
        };

        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut total_sparsity = 0.0f64;
        let mut total_weights = 0usize;
        let mut bf16_count = 0usize;
        let mut ternary_count = 0usize;

        for i in 0..cfg.num_layers {
            let use_bf16 = bf16_layers.contains(&i);
            let tag = if use_bf16 { "BF16" } else { "Ternary" };
            if i % 8 == 0 || use_bf16 {
                log::info!("  → layer {}/{} ({})", i, cfg.num_layers, tag);
            }
            if use_bf16 { bf16_count += 1; } else { ternary_count += 1; }

            let (in_w, _) = load_f32(&tmap.layer_tensor(i, &tmap.input_norm))?;
            let (pa_w, _) = load_f32(&tmap.layer_tensor(i, &tmap.post_attn_norm))?;

            let qbn = tmap.layer_tensor_opt(i, &tmap.attn_q_bias);
            let kbn = tmap.layer_tensor_opt(i, &tmap.attn_k_bias);
            let vbn = tmap.layer_tensor_opt(i, &tmap.attn_v_bias);
            let qnn = tmap.layer_tensor_opt(i, &tmap.q_norm);
            let knn = tmap.layer_tensor_opt(i, &tmap.k_norm);

            // Per-layer weight loader: BF16 for critical, Ternary for rest
            let load_weight = |name: &str| -> ModelResult<LayerWeight> {
                if use_bf16 {
                    Ok(LayerWeight::Bf16(gpu_bf16(name)?))
                } else {
                    ternary_weight(name)
                }
            };

            layers.push(TransformerLayer {
                wq:    load_weight(&tmap.layer_tensor(i, &tmap.attn_q))?,
                wk:    load_weight(&tmap.layer_tensor(i, &tmap.attn_k))?,
                wv:    load_weight(&tmap.layer_tensor(i, &tmap.attn_v))?,
                wo:    load_weight(&tmap.layer_tensor(i, &tmap.attn_o))?,
                wgate: load_weight(&tmap.layer_tensor(i, &tmap.mlp_gate))?,
                wup:   load_weight(&tmap.layer_tensor(i, &tmap.mlp_up))?,
                wdown: load_weight(&tmap.layer_tensor(i, &tmap.mlp_down))?,
                q_bias: match &qbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                k_bias: match &kbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                v_bias: match &vbn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                q_norm_w: match &qnn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                k_norm_w: match &knn {
                    Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
                    _ => None,
                },
                input_norm_w: gpu_f32(&in_w),
                post_attn_norm_w: gpu_f32(&pa_w),
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
            });
        }

        let (nw, _) = load_f32(&tmap.final_norm)?;
        let norm_w = gpu_f32(&nw);
        let norm_b = match &tmap.final_norm_bias {
            Some(name) if tensor_exists(name) => Some(gpu_f32(&load_f32(name)?.0)),
            _ => None,
        };

        // lm_head: ternary por padrão (3ms→1ms), desativável via IARA_TERNARY_LM_HEAD=0
        let ternary_lm_head = std::env::var("IARA_TERNARY_LM_HEAD")
            .map(|v| v != "0")
            .unwrap_or(true);

        let lm_head_name = if !arch.tie_word_embeddings && tensor_exists(&tmap.lm_head) {
            tmap.lm_head.as_str()
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            tmap.embed.as_str()
        };

        let lm_head = if ternary_lm_head {
            let w = ternary_weight(lm_head_name)?;
            log::info!("  → lm_head: Ternary 2-bit [{}×{}] (IARA_TERNARY_LM_HEAD=1)", cfg.vocab_size, cfg.hidden_size);
            w
        } else {
            log::info!("  → lm_head: BF16 [{}×{}] (IARA_TERNARY_LM_HEAD=0)", cfg.vocab_size, cfg.hidden_size);
            LayerWeight::Bf16(gpu_bf16(lm_head_name)?)
        };

        let mut vram_bytes = embed.numel() * 2 + lm_head.vram_bytes();
        for l in &layers {
            vram_bytes += l.wq.vram_bytes() + l.wk.vram_bytes() + l.wv.vram_bytes()
                + l.wo.vram_bytes() + l.wgate.vram_bytes()
                + l.wup.vram_bytes() + l.wdown.vram_bytes();
            vram_bytes += l.input_norm_w.numel() * 4 + l.post_attn_norm_w.numel() * 4;
            vram_bytes += l.kv_k.data.numel() * 4 + l.kv_v.data.numel() * 4;
        }

        if hybrid_enabled {
            log::info!(
                "✅ Modelo [{}] carregado (Hybrid: {} BF16 + {} Ternary layers) — VRAM: {:.2} GB",
                arch.family, bf16_count, ternary_count, vram_bytes as f64 / 1e9
            );
        } else {
            log::info!(
                "✅ Modelo [{}] carregado (Ternary 2-bit) — VRAM: {:.2} GB (16× redução vs BF16)",
                arch.family, vram_bytes as f64 / 1e9
            );
        }

        let scratch = ForwardScratchpad::new(&cfg, &backend);
        let layer_router = OnlineLayerRouter::new(&backend, cfg.num_layers, cfg.hidden_size);

        Ok(Self {
            embed,
            layers,
            norm_w,
            norm_b,
            lm_head,
            cfg,
            arch,
            vram_bytes,
            backend,
            scratch,
            cfc: None,
            gravity: None,
            quantized: true,
            kuramoto_router: None,
            sparse_scratch: None,
            #[cfg(feature = "experimental")]
            stimulus_tx: None,
            telemetry_enabled: false,
            layer_router,
            graph_replay_mode: std::env::var("IARA_GRAPH").map(|v| v == "2").unwrap_or(false),
        })
    }

    /// Carrega modelo do formato .iara v2 (com suporte a multi-quant: BF16, FP8, Q4, Ternary).
    ///
    /// O arquivo .iara v2 contém metadados + dados de tensores com mmap zero-copy.
    /// Suporta tensores em diferentes dtypes dependendo da receita de quantização.
    pub fn load_from_iara(
        iara_path: &Path,
        cfg: ModelCfg,
        arch: ModelArchitecture,
        backend: Arc<dyn ComputeBackend>,
    ) -> ModelResult<Self> {
        use crate::storage::iara_format::{IaraV2Reader, IaraDtype};

        let reader = IaraV2Reader::open(iara_path)
            .map_err(|e| ModelError::Model(format!("Falha ao abrir arquivo .iara: {}", e)))?;

        log::info!(
            "⏳ Carregando modelo [{}] de .iara v2: {} camadas, hidden={}, vocab={}",
            arch.family, cfg.num_layers, cfg.hidden_size, cfg.vocab_size
        );

        let tmap = &arch.tensor_map;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;

        // ── Helpers para conversão de tipos ──────────────────────────────────
        let bf16_from_bytes = |raw: &[u8]| -> ModelResult<AnyTensor> {
            let bits = unsafe {
                std::slice::from_raw_parts(
                    raw.as_ptr() as *const u16,
                    raw.len() / 2
                )
            };
            // Shape será preenchida pelo chamador via tensor_meta
            Ok(backend.upload_bf16(bits, &[bits.len()]))
        };

        let f32_from_f32_bytes = |raw: &[u8]| -> ModelResult<Vec<f32>> {
            Ok(raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        // Dtype-aware decoder: converts any .iara tensor data to Vec<f32>
        let decode_to_f32 = |name: &str| -> ModelResult<Vec<f32>> {
            use crate::storage::iara_format::IaraDtype;
            let meta = reader.tensor_meta(name)
                .map_err(|e| ModelError::Model(format!("Tensor '{}' meta: {}", name, e)))?;
            let data = reader.tensor_data(name)
                .map_err(|e| ModelError::Model(format!("Tensor '{}' data: {}", name, e)))?;
            match meta.dtype {
                IaraDtype::Bf16 => {
                    // BF16: 2 bytes per element → f32
                    let bits = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2)
                    };
                    Ok(bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect())
                }
                IaraDtype::Fp8E4m3 => {
                    // FP8: 1 byte per element, may have per-row scales
                    let scales = reader.tensor_scales_f32(name).ok();
                    let total = meta.shape.iter().product::<usize>();
                    let cols = if meta.shape.len() >= 2 { meta.shape[1] } else { total };
                    Ok(data.iter().enumerate().map(|(idx, &byte)| {
                        let row = if cols > 0 { idx / cols } else { 0 };
                        let scale = scales.as_ref()
                            .and_then(|s| s.get(row).copied())
                            .unwrap_or(1.0);
                        fp8_e4m3_to_f32(byte) * scale
                    }).collect())
                }
                _ => {
                    // Assume F32 raw bytes
                    f32_from_f32_bytes(data)
                }
            }
        };

        // ── Embedding ────────────────────────────────────────────────────────
        log::info!("  → embed_tokens");
        let embed_meta = reader.tensor_meta(&tmap.embed)
            .map_err(|e| ModelError::Model(format!("Embedding não encontrado: {}", e)))?;
        let embed_data = reader.tensor_data(&tmap.embed)
            .map_err(|e| ModelError::Model(format!("Dados do embedding não acessíveis: {}", e)))?;

        let embed = match embed_meta.dtype {
            IaraDtype::Bf16 => bf16_from_bytes(embed_data)?,
            _ => return Err(ModelError::Model(
                format!("Dtype não suportado para embedding: {:?}", embed_meta.dtype))),
        };

        // ── Camadas ──────────────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 {
                log::info!("  → layers {}/{}", i, cfg.num_layers);
            }

            // ── Norms (dtype-aware: may be BF16 in .iara) ──────────────────
            let in_norm_name = &tmap.layer_tensor(i, &tmap.input_norm);
            let in_w = decode_to_f32(in_norm_name)?;

            let pa_norm_name = &tmap.layer_tensor(i, &tmap.post_attn_norm);
            let pa_w = decode_to_f32(pa_norm_name)?;

            let gpu_f32 = |data: &[f32]| -> AnyTensor {
                backend.upload_f32(data, &[data.len()])
            };

            // ── Pesos de atenção e MLP (podem ser BF16, FP8, Q4) ────────────
            let load_weight = |name: &str| -> ModelResult<LayerWeight> {
                let meta = reader.tensor_meta(name)
                    .map_err(|e| ModelError::Model(format!("Tensor {} não encontrado: {}", name, e)))?;
                let data = reader.tensor_data(name)
                    .map_err(|e| ModelError::Model(format!("Dados tensor {} não acessíveis: {}", name, e)))?;

                match meta.dtype {
                    IaraDtype::Bf16 => {
                        let bits = unsafe {
                            std::slice::from_raw_parts(
                                data.as_ptr() as *const u16,
                                data.len() / 2
                            )
                        };
                        let tensor = backend.upload_bf16(bits, &meta.shape);
                        Ok(LayerWeight::Bf16(tensor))
                    }
                    IaraDtype::Fp8E4m3 => {
                        let scales = reader.tensor_scales_f32(name)
                            .map_err(|e| ModelError::Model(format!("Escalas FP8 layer {}: {}", i, e)))?;

                        let fp8_bytes = data;
                        let total_elements: usize = meta.shape.iter().product();
                        let cols = if meta.shape.len() >= 2 { meta.shape[1] } else { total_elements };
                        let rows = if cols > 0 { total_elements / cols } else { 1 };

                        if fp8_bytes.len() != total_elements {
                            log::warn!("FP8 tensor '{}': data_length={} vs shape_elements={} (delta={})",
                                name, fp8_bytes.len(), total_elements,
                                fp8_bytes.len() as i64 - total_elements as i64);
                        }

                        // Decode FP8→F32 using per-row scales, then F32→BF16
                        // CRITICAL: must produce EXACTLY total_elements values for GPU shape match
                        let mut bf16_bits = vec![0u16; total_elements];
                        for row in 0..rows {
                            let scale = if row < scales.len() { scales[row] } else { 1.0 };
                            for col in 0..cols {
                                let idx = row * cols + col;
                                if idx < fp8_bytes.len() {
                                    let f32_val = fp8_e4m3_to_f32(fp8_bytes[idx]) * scale;
                                    bf16_bits[idx] = (f32_val.to_bits() >> 16) as u16;
                                }
                                // else: stays 0 (zero-padded) — safe for GPU
                            }
                        }

                        log::info!("✅ FP8 tensor '{}' decoded: {}×{} = {} elements, {} scales",
                            name, rows, cols, total_elements, scales.len());
                        let tensor = backend.upload_bf16(&bf16_bits, &meta.shape);
                        Ok(LayerWeight::Bf16(tensor))
                    }
                    IaraDtype::Q4 => {
                        // Q4: dados + scales + zeros
                        let scales_bytes = reader.tensor_scales(name)
                            .map_err(|e| ModelError::Model(format!("Escalas Q4 layer {}: {}", i, e)))?;
                        let zeros_bytes = reader.tensor_zeros(name)
                            .map_err(|e| ModelError::Model(format!("Zeros Q4 layer {}: {}", i, e)))?;

                        let group_size = meta.group_size;
                        let [rows, cols] = [meta.shape[0], meta.shape[1]];
                        let n_groups = cols / group_size;

                        // Converter scales e zeros de bytes para f32
                        let scales_f32: Vec<f32> = scales_bytes.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let zeros_f32: Vec<f32> = zeros_bytes.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();

                        // Upload Q4 via backend
                        let packed = backend.upload_q4(data, &[rows, cols / 2]);
                        let scales = backend.upload_f32(&scales_f32, &[rows, n_groups]);
                        let zeros = backend.upload_f32(&zeros_f32, &[rows, n_groups]);

                        let quantized = crate::storage::quantized::QuantizedTensor {
                            packed,
                            scales,
                            zeros,
                            shape: [rows, cols],
                            group_size,
                        };
                        Ok(LayerWeight::Q4(quantized))
                    }
                    IaraDtype::Ternary => {
                        log::warn!("⚠️  Ternary tensor '{}' — suporte não implementado, pulando", name);
                        // Por enquanto, fall back para BF16
                        let bits = unsafe {
                            std::slice::from_raw_parts(
                                data.as_ptr() as *const u16,
                                (data.len() / 2).min(meta.shape.iter().product::<usize>())
                            )
                        };
                        let tensor = backend.upload_bf16(bits, &meta.shape);
                        Ok(LayerWeight::Bf16(tensor))
                    }
                    _ => Err(ModelError::Model(format!("Dtype não suportado: {:?}", meta.dtype))),
                }
            };

            let qbn = tmap.layer_tensor_opt(i, &tmap.attn_q_bias);
            let kbn = tmap.layer_tensor_opt(i, &tmap.attn_k_bias);
            let vbn = tmap.layer_tensor_opt(i, &tmap.attn_v_bias);
            let qnn = tmap.layer_tensor_opt(i, &tmap.q_norm);
            let knn = tmap.layer_tensor_opt(i, &tmap.k_norm);

            let get_optional_f32 = |name_opt: &Option<String>| -> ModelResult<Option<AnyTensor>> {
                match name_opt {
                    Some(name) => {
                        if reader.tensor_meta(name).is_err() {
                            return Ok(None);
                        }
                        let f32_data = decode_to_f32(name)?;
                        Ok(Some(gpu_f32(&f32_data)))
                    }
                    None => Ok(None),
                }
            };

            layers.push(TransformerLayer {
                wq:    load_weight(&tmap.layer_tensor(i, &tmap.attn_q))?,
                wk:    load_weight(&tmap.layer_tensor(i, &tmap.attn_k))?,
                wv:    load_weight(&tmap.layer_tensor(i, &tmap.attn_v))?,
                wo:    load_weight(&tmap.layer_tensor(i, &tmap.attn_o))?,
                wgate: load_weight(&tmap.layer_tensor(i, &tmap.mlp_gate))?,
                wup:   load_weight(&tmap.layer_tensor(i, &tmap.mlp_up))?,
                wdown: load_weight(&tmap.layer_tensor(i, &tmap.mlp_down))?,
                q_bias: get_optional_f32(&qbn)?,
                k_bias: get_optional_f32(&kbn)?,
                v_bias: get_optional_f32(&vbn)?,
                q_norm_w: get_optional_f32(&qnn)?,
                k_norm_w: get_optional_f32(&knn)?,
                input_norm_w: gpu_f32(&in_w),
                post_attn_norm_w: gpu_f32(&pa_w),
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim, cfg.num_kv_heads, cfg.head_dim),
            });
        }

        // Log QK norm detection
        if layers.first().map_or(false, |l| l.q_norm_w.is_some()) {
            log::info!("  → QK Normalization detectada: q_norm + k_norm em todas as camadas");
        }

        // ── Final norm (dtype-aware) ──────────────────────────────────────
        let nw = decode_to_f32(&tmap.final_norm)?;
        let norm_w = backend.upload_f32(&nw, &[nw.len()]);

        let norm_b = match &tmap.final_norm_bias {
            Some(name) => {
                if reader.tensor_meta(name).is_ok() {
                    let f32_data = decode_to_f32(name)?;
                    Some(backend.upload_f32(&f32_data, &[f32_data.len()]))
                } else {
                    None
                }
            }
            None => None,
        };

        // ── LM head ──────────────────────────────────────────────────────────
        let lm_head = LayerWeight::Bf16(if !arch.tie_word_embeddings && reader.tensor_meta(&tmap.lm_head).is_ok() {
            let meta = reader.tensor_meta(&tmap.lm_head)
                .map_err(|e| ModelError::Model(format!("LM head não encontrado: {}", e)))?;
            let data = reader.tensor_data(&tmap.lm_head)
                .map_err(|e| ModelError::Model(format!("Dados LM head não acessíveis: {}", e)))?;

            match meta.dtype {
                IaraDtype::Bf16 => {
                    let bits = unsafe {
                        std::slice::from_raw_parts(
                            data.as_ptr() as *const u16,
                            data.len() / 2
                        )
                    };
                    backend.upload_bf16(bits, &meta.shape)
                }
                _ => {
                    log::warn!("⚠️  LM head com dtype não-BF16, usando embed_tokens (tied)");
                    let embed_data = reader.tensor_data(&tmap.embed)
                        .map_err(|e| ModelError::Model(format!("Embedding fallback: {}", e)))?;
                    bf16_from_bytes(embed_data)?
                }
            }
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            let embed_data = reader.tensor_data(&tmap.embed)
                .map_err(|e| ModelError::Model(format!("Embedding fallback: {}", e)))?;
            bf16_from_bytes(embed_data)?
        });

        // ── Contabilidade de VRAM ────────────────────────────────────────────
        let bf16_bytes = |t: &AnyTensor| t.numel() * 2;
        let f32_bytes = |t: &AnyTensor| t.numel() * 4;
        let layer_weight_bytes = |w: &LayerWeight| w.vram_bytes();

        let mut vram_bytes = bf16_bytes(&embed) + layer_weight_bytes(&lm_head);
        for l in &layers {
            vram_bytes += layer_weight_bytes(&l.wq) + layer_weight_bytes(&l.wk) + layer_weight_bytes(&l.wv)
                        + layer_weight_bytes(&l.wo) + layer_weight_bytes(&l.wgate)
                        + layer_weight_bytes(&l.wup) + layer_weight_bytes(&l.wdown);
            vram_bytes += f32_bytes(&l.input_norm_w) + f32_bytes(&l.post_attn_norm_w);
            vram_bytes += l.kv_k.data.numel() * 4 + l.kv_v.data.numel() * 4;
        }

        log::info!("✅ Modelo [{}] carregado de .iara v2 — memória estimada: {:.2} GB", arch.family, vram_bytes as f64 / 1e9);

        let scratch = ForwardScratchpad::new(&cfg, &backend);

        // Initialize calibration-based sparse MLP if enabled via IARA_ROUTING=1
        let (kuramoto_router, sparse_scratch) = if std::env::var("IARA_ROUTING").unwrap_or_default() == "1" {
            let active_ratio = std::env::var("IARA_ROUTING_RATIO")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.25);

            let routing_mode = std::env::var("IARA_ROUTING_MODE").unwrap_or_default();
            let mut ss = SparseMlpScratch::new(&cfg, &backend, active_ratio);

            if routing_mode == "sequential" {
                log::info!("🔧 Sequential routing mode (debug): {:.0}% of neurons", active_ratio * 100.0);
                ss.set_sequential_indices(cfg.num_layers, &backend);
            } else {
                log::info!("🔥 Sparse MLP enabled (active_ratio={:.0}%), will calibrate during first tokens...", active_ratio * 100.0);
            }

            (None::<crate::inference::KuramotoRouter>, Some(ss))
        } else {
            (None, None)
        };

        let layer_router = OnlineLayerRouter::new(&backend, cfg.num_layers, cfg.hidden_size);

        Ok(Self {
            embed,
            layers,
            norm_w,
            norm_b,
            lm_head,
            cfg,
            arch,
            vram_bytes,
            backend,
            scratch,
            cfc: None,
            gravity: None,
            quantized: false,
            kuramoto_router,
            sparse_scratch,
            #[cfg(feature = "experimental")]
            stimulus_tx: None,
            telemetry_enabled: false,
            layer_router,
            graph_replay_mode: std::env::var("IARA_GRAPH").map(|v| v == "2").unwrap_or(false),
        })
    }

    pub fn reset_cache(&mut self) {
        for l in &mut self.layers {
            l.kv_k.reset();
            l.kv_v.reset();
        }
        // Reset CfC state (se habilitado)
        if let Some(cfc) = &mut self.cfc {
            cfc.reset(&self.backend);
        }
        // Reset gravity history (se habilitado)
        if let Some(gravity) = &mut self.gravity {
            gravity.reset();
        }
        // Drop any captured HIP graph — pointers are stale after KV reset.
        self.backend.invalidate_graph();
    }

    /// Forward pass via ComputeBackend with optional early exit via phase coherence.
    /// GPU→CPU syncs: 1 mandatory (logits) + up to 2 at boundary layers (coherence check).
    /// Telemetry downloads are ZERO when `self.telemetry_enabled == false` (default).
    /// Returns: (logits, layer_traces) onde layer_traces contém telemetria por camada.
    ///
    /// Parâmetros:
    /// - `token_ids`: token sequence to process
    /// - `layer_budget`: how many layers to use (Fast/Medium/Slow), None = use all
    pub fn forward(&mut self, token_ids: &[u32], layer_budget: Option<LayerBudget>) -> ModelResult<(Vec<f32>, Vec<LayerTrace>)> {
        self.forward_inner(token_ids, layer_budget, false, false)
    }

    fn forward_inner(&mut self, token_ids: &[u32], layer_budget: Option<LayerBudget>, skip_logits_download: bool, skip_embed: bool) -> ModelResult<(Vec<f32>, Vec<LayerTrace>)> {
        let _fwd_t0 = std::time::Instant::now();
        // IARA_FUSED=1 explicitly enables super_fused kernels (~2% slower
        // than the unfused path on RX 6750 XT — fused kernels under-use
        // GPU parallelism). Default = unfused for max throughput.
        // Individual flags: IARA_FUSE_QKV=1, IARA_FUSE_MLP=1, IARA_FUSE_ATTN=1, IARA_FUSE_ROPE=1, IARA_FUSE_Q6KF32=1
        let unfused = {
            use std::sync::OnceLock;
            static U: OnceLock<bool> = OnceLock::new();
            *U.get_or_init(|| {
                // Default: unfused (best perf empirically). IARA_FUSED=1 forces fused.
                if std::env::var("IARA_FUSED").map(|v| v == "1").unwrap_or(false) { false }
                else if std::env::var("IARA_UNFUSED").map(|v| v == "0").unwrap_or(false) { false }
                else { true }
            })
        };
        // QKV super-fusion: empirically a win on Qwen2.5 1.5B Q4K (77 tok/s
        // vs 75 unfused). MLP super-fusion is a loss on the same model
        // (71 tok/s vs 75 unfused) — kept as opt-in only.
        let force_fuse_qkv = std::env::var("IARA_FUSE_QKV").map(|v| v != "0").unwrap_or(true);
        let force_fuse_mlp = std::env::var("IARA_FUSE_MLP").is_ok();
        // fused attention + rope/kv-copy fusion default ON even in unfused
        // mode — these are pure dependent ops (no extra blocks lost) and
        // each saves 2-3 launches per layer.
        let force_fuse_attn = std::env::var("IARA_FUSE_ATTN").map(|v| v != "0").unwrap_or(true);
        let force_fuse_rope = std::env::var("IARA_FUSE_ROPE").map(|v| v != "0").unwrap_or(true);
        let force_fuse_q6kf32 = std::env::var("IARA_FUSE_Q6KF32").is_ok();
        // Q8 dp4a wave64 for MLP & projections: ~40% faster than F32 super-fused.
        // Disable with IARA_NO_Q8_MLP=1 to revert to F32 super-fused path.
        let force_q8_mlp = !std::env::var("IARA_NO_Q8_MLP").is_ok();
        #[cfg(feature = "rocm")]
        hip_ffi::safe::reset_launch_count();
        let n    = token_ids.len();
        let h    = self.cfg.hidden_size;
        let nh   = self.cfg.num_heads;
        let nk   = self.cfg.num_kv_heads;
        let hd   = self.cfg.head_dim;
        let qd   = nh * hd;   // q_dim
        let kd   = nk * hd;   // kv_dim
        let it   = self.cfg.intermediate_size;
        let eps  = self.cfg.rms_eps;
        let theta = self.cfg.rope_theta;
        let scale = 1.0_f32 / (hd as f32).sqrt();
        let vocab = self.cfg.vocab_size;

        // Embedding lookup: BF16 embed → F32 x [n, h]
        // skip_embed=true means caller already did this (e.g. to keep it outside graph capture)
        if !skip_embed {
            self.backend.embed_lookup(&EmbedParams {
                token_ids,
                vocab: &self.embed,
                out: &self.scratch.x,
                n_tokens: n,
                hidden_size: h,
            })?;
        }

        // Debug: check embed lookup output
        if !skip_embed && std::env::var("IARA_DEBUG_GGUF").map(|v| v == "1").unwrap_or(false) {
            static EMBED_DEBUG_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !EMBED_DEBUG_DONE.swap(true, std::sync::atomic::Ordering::SeqCst) {
                let x_data = self.backend.download_f32(&self.scratch.x);
                log::info!("🔍 After embed_lookup: tokens={:?}", &token_ids[..token_ids.len().min(5)]);
                log::info!("  x[0..8] = {:?}", &x_data[..8.min(x_data.len())]);
                let x_norm: f32 = x_data[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                log::info!("  x row0 L2 norm = {:.6}", x_norm);
            }
        }

        // Initialize telemetry vector
        let mut layer_traces: Vec<LayerTrace> = Vec::new();

        let norm_type = self.arch.norm_type;
        let activation = self.arch.activation;
        let n_layers = self.layers.len();

        // ── Early exit thresholds and layer boundaries ──────────────────────
        const EXIT_THRESHOLD_FAST: f32 = 0.85;
        const EXIT_THRESHOLD_MEDIUM: f32 = 0.75;
        let fast_layer_count = (n_layers + 3) / 4;         // first N/4 layers
        let medium_layer_count = (3 * n_layers + 3) / 4;   // first 3N/4 layers

        // Determine final layer to process based on budget
        let final_layer = match layer_budget {
            Some(LayerBudget::Fast) => fast_layer_count.saturating_sub(1),
            Some(LayerBudget::Medium) => medium_layer_count.saturating_sub(1),
            _ => n_layers - 1,  // Slow or None: use all layers
        };

        // ── Per-section GPU profiling (tokens 3-5, with device_sync per section) ──
        use std::sync::atomic::{AtomicU64, AtomicBool, Ordering as AO};
        static PROF_TOKEN: AtomicU64 = AtomicU64::new(0);
        static PROF_NORM: AtomicU64 = AtomicU64::new(0);
        static PROF_QKV: AtomicU64 = AtomicU64::new(0);
        static PROF_ROPE_KV: AtomicU64 = AtomicU64::new(0);
        static PROF_ATTN: AtomicU64 = AtomicU64::new(0);
        static PROF_WO: AtomicU64 = AtomicU64::new(0);
        static PROF_POST_NORM: AtomicU64 = AtomicU64::new(0);
        static PROF_MLP: AtomicU64 = AtomicU64::new(0);
        static PROF_RESIDUAL: AtomicU64 = AtomicU64::new(0);
        static PROF_FINAL: AtomicU64 = AtomicU64::new(0);  // final norm + lm_head
        static PROF_DONE: AtomicBool = AtomicBool::new(false);
        let tok_num = PROF_TOKEN.fetch_add(1, AO::Relaxed);
        let do_section_prof = std::env::var("IARA_SECTION_PROF").map(|v| v == "1").unwrap_or(false);
        // Also keep old profile for backwards compat
        let do_profile = false;
        let forward_t0: Option<std::time::Instant> = None;

        // ── BI-based layer skip for ternary decode ─────────────────────────
        // Bitmask-based skip: each bit in `keep_mask` (u32) = 1 means KEEP that layer.
        // Supports preset modes and custom keep-lists for non-contiguous skipping.
        // Controlled by IARA_TERNARY_SKIP env var (default: enabled for ternary).
        let ternary_skip_enabled = matches!(&self.layers[0].wq, LayerWeight::Ternary { .. })
            && std::env::var("IARA_TERNARY_SKIP").unwrap_or_else(|_| "1".into()) != "0";

        // Build keep_mask: bit i = 1 → execute layer i; bit i = 0 → skip layer i
        let keep_mask: u32 = if let Ok(hex) = std::env::var("IARA_SKIP_MASK") {
            // Raw hex bitmask of layers to SKIP (1=skip). Invert to get keep_mask.
            let skip = u32::from_str_radix(hex.trim_start_matches("0x").trim_start_matches("0X"), 16)
                .unwrap_or(0);
            !skip
        } else if let Ok(layers_str) = std::env::var("IARA_KEEP_LAYERS") {
            // Comma-separated list of layer indices to keep, e.g. "0,1,2,3,14,27"
            let mut mask = 0u32;
            for s in layers_str.split(',') {
                if let Ok(idx) = s.trim().parse::<u32>() {
                    if idx < 32 { mask |= 1 << idx; }
                }
            }
            mask
        } else {
            // Preset modes via IARA_SKIP_MODE (or fallback to IARA_SKIP_START/END)
            let mode = std::env::var("IARA_SKIP_MODE").unwrap_or_default();
            match mode.as_str() {
                // aggressive: keep 0-3, 14, 25-27 → 8 layers executed, 20 skipped
                "aggressive" => {
                    (0b1111)                  // layers 0-3
                    | (1u32 << 14)            // layer 14 (middle anchor)
                    | (0b111 << 25)           // layers 25-27
                }
                // ultra: keep 0-2, 14, 27 → 5 layers executed, 23 skipped
                "ultra" => {
                    (0b111)                   // layers 0-2
                    | (1u32 << 14)            // layer 14 (middle anchor)
                    | (1u32 << 27)            // layer 27 (final)
                }
                // moderate: keep 0-6, 14, 21-27 → 14 layers executed (original behavior)
                "moderate" => {
                    (0b1111111)               // layers 0-6
                    | (1u32 << 14)            // layer 14
                    | (0b1111111 << 21)       // layers 21-27
                }
                // Default: fall back to IARA_SKIP_START/IARA_SKIP_END for compatibility
                _ => {
                    let skip_start = std::env::var("IARA_SKIP_START").ok()
                        .and_then(|s| s.parse::<u32>().ok()).unwrap_or(7);
                    let skip_end = std::env::var("IARA_SKIP_END").ok()
                        .and_then(|s| s.parse::<u32>().ok()).unwrap_or(20);
                    let mut mask = 0xFFFF_FFFFu32; // keep all
                    for i in skip_start..=skip_end {
                        if i < 32 { mask &= !(1 << i); }
                    }
                    mask
                }
            }
        };

        // Log skip configuration once (first decode token)
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if ternary_skip_enabled && !LOGGED.swap(true, Ordering::Relaxed) {
                let kept: u32 = keep_mask.count_ones();
                let skipped = n_layers as u32 - kept.min(n_layers as u32);
                let kept_list: Vec<usize> = (0..n_layers)
                    .filter(|&i| keep_mask & (1 << i) != 0)
                    .collect();
                log::info!(
                    "Ternary layer skip: executing {}/{} layers (skipping {}). Keep: {:?}",
                    kept, n_layers, skipped, kept_list
                );
            }
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Check if we should early exit based on budget
            if layer_idx > final_layer {
                log::debug!(
                    "Early exit at layer {} (budget={:?}): skipping remaining layers",
                    layer_idx, layer_budget
                );
                break;
            }

            // ── Layer skip decision ──────────────────────────────────────────
            // Priority: OnlineLayerRouter (dynamic) > static bitmask (IARA_SKIP_MODE)
            let (router_skip, router_measure) = if n == 1 {
                self.layer_router.should_skip(layer_idx)
            } else {
                (false, false)
            };

            if router_skip {
                continue;  // dynamic skip: router says this layer has low BI
            }
            if !self.layer_router.enabled && ternary_skip_enabled && n == 1 && (keep_mask & (1u32 << layer_idx)) == 0 {
                continue;  // static skip: bitmask says skip (fallback when router is off)
            }

            // Snapshot x BEFORE layer if router wants to measure BI
            if router_measure {
                self.backend.copy_f32_d2d(&self.scratch.x, &self.layer_router.x_snapshot, h).ok();
            }

            let _prof_l0 = false;
            let _layer_t0: Option<std::time::Instant> = None;

            macro_rules! prof_sync {
                ($label:expr) => {};
            }

            // Section profiling: sync GPU and accumulate per-section time
            macro_rules! section_time {
                ($accum:ident, $t0:ident) => {
                    if do_section_prof {
                        self.backend.device_sync();
                        let _elapsed = $t0.elapsed().as_micros() as u64;
                        $accum.fetch_add(_elapsed, AO::Relaxed);
                        $t0 = std::time::Instant::now();
                    }
                };
            }
            let mut _sec_t0 = std::time::Instant::now();

            // ── Input Normalization (RMSNorm ou LayerNorm conforme arquitetura) ──
            // Super-fused: RMSNorm + Q/K/V projection in ONE kernel (decode only, saves norm+q8+qkv launches)
            static SUPER_FUSED_QKV_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let super_fused_qkv = if n == 1 && (!unfused || force_fuse_qkv) && matches!(norm_type, NormType::RMSNorm) {
                match (&layer.wq, &layer.wk, &layer.wv) {
                    // GGML Q4K format (from GGUF native loading)
                    (LayerWeight::Q4K { data: dq, .. }, LayerWeight::Q4K { data: dk, .. }, LayerWeight::Q4K { data: dv, .. }) => {
                        let r = self.backend.fused_rms_qkv_q4k_f32(&compute_core::FusedRmsQkvQ4KF32Params {
                            x: &self.scratch.x, norm_w: &layer.input_norm_w,
                            wq: dq, wk: dk, wv: dv,
                            out_q: &self.scratch.q, out_k: &self.scratch.k, out_v: &self.scratch.v,
                            bias_q: layer.q_bias.as_ref(), bias_k: layer.k_bias.as_ref(), bias_v: layer.v_bias.as_ref(),
                            nq: qd, nk: kd, nv: kd, k: h, eps, x_offset: 0,
                        });
                        if let Err(ref e) = r {
                            static ERR_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                            if !ERR_LOG.swap(true, AO::Relaxed) {
                                log::warn!("SUPER FUSED QKV Q4K FAILED: {:?}", e);
                            }
                        }
                        r.is_ok()
                    }
                    // IARA native Q4 format (from safetensors + IARA_QUANTIZE)
                    (LayerWeight::Q4(qtq), LayerWeight::Q4(qtk), LayerWeight::Q4(qtv)) => {
                        let r = self.backend.fused_rms_qkv_q4(&compute_core::FusedRmsQkvQ4Params {
                            x: &self.scratch.x, norm_w: &layer.input_norm_w,
                            wq_packed: &qtq.packed, wq_scales: &qtq.scales, wq_zeros: &qtq.zeros,
                            wk_packed: &qtk.packed, wk_scales: &qtk.scales, wk_zeros: &qtk.zeros,
                            wv_packed: &qtv.packed, wv_scales: &qtv.scales, wv_zeros: &qtv.zeros,
                            out_q: &self.scratch.q, out_k: &self.scratch.k, out_v: &self.scratch.v,
                            bias_q: layer.q_bias.as_ref(), bias_k: layer.k_bias.as_ref(), bias_v: layer.v_bias.as_ref(),
                            nq: qd, nk: kd, nv: kd, k: h,
                            group_size: qtq.group_size, eps,
                        });
                        if let Err(ref e) = r {
                            static ERR_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                            if !ERR_LOG.swap(true, AO::Relaxed) {
                                log::warn!("SUPER FUSED QKV Q4 FAILED: {:?}", e);
                            }
                        }
                        r.is_ok()
                    }
                    _ => false,
                }
            } else { false };
            if super_fused_qkv && !SUPER_FUSED_QKV_LOG.swap(true, AO::Relaxed) {
                log::warn!("SUPER FUSED RMS+QKV: active (nq={}, nk={}, k={})", qd, kd, h);
            }
            // Try fused RMSNorm+Quantize (1 launch vs 2) for decode — skipped if super_fused_qkv
            // Check if mixed Q4K+Q4K+Q6K path applies (do this BEFORE norm to skip unnecessary norm)
            let is_mixed_q4k_q6k = n == 1 && (!unfused || force_fuse_qkv) && matches!(norm_type, NormType::RMSNorm)
                && matches!((&layer.wq, &layer.wk, &layer.wv),
                    (LayerWeight::Q4K { .. }, LayerWeight::Q4K { .. }, LayerWeight::Q6K { .. }));

            let qkv_q8_pre = if super_fused_qkv || is_mixed_q4k_q6k { false } else if n == 1 && matches!(norm_type, NormType::RMSNorm) {
                if let Some(q8) = self.scratch.q8_scratch.as_ref() {
                    if self.backend.fused_rms_norm_quantize(&compute_core::FusedRmsNormQuantizeParams {
                        input: &self.scratch.x, weight: &layer.input_norm_w,
                        out_f32: &self.scratch.normed, out_q8: q8,
                        hidden_size: h, eps,
                    }).is_ok() {
                        static FUSED_NQ_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !FUSED_NQ_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("FUSED NORM+Q8: active (h={})", h);
                        }
                        true
                    } else {
                        Self::apply_norm(&self.backend, norm_type, &self.scratch.x, &layer.input_norm_w, None, &self.scratch.normed, n, h, eps)?;
                        Self::quantize_input_q8(&self.backend, &self.scratch.normed, 0, q8, h)?;
                        true
                    }
                } else {
                    Self::apply_norm(&self.backend, norm_type, &self.scratch.x, &layer.input_norm_w, None, &self.scratch.normed, n, h, eps)?;
                    false
                }
            } else {
                Self::apply_norm(&self.backend, norm_type, &self.scratch.x, &layer.input_norm_w, None, &self.scratch.normed, n, h, eps)?;
                if n == 1 {
                    if let Some(q8) = self.scratch.q8_scratch.as_ref() {
                        Self::quantize_input_q8(&self.backend, &self.scratch.normed, 0, q8, h)?;
                        true
                    } else { false }
                } else { false }
            };
            prof_sync!("norm+q8");
            section_time!(PROF_NORM, _sec_t0);

            // Try fused QKV kernel (1 launch vs 3) for decode with Q4K weights
            static FUSED_QKV_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let fused_qkv_ok = if n == 1 && (!unfused || force_fuse_qkv) && qkv_q8_pre {
                match (&layer.wq, &layer.wk, &layer.wv) {
                    (LayerWeight::Q4K { data: dq, .. }, LayerWeight::Q4K { data: dk, .. }, LayerWeight::Q4K { data: dv, .. }) => {
                        self.backend.fused_qkv(&compute_core::FusedQkvParams {
                            wq_data: dq, wk_data: dk, wv_data: dv,
                            x_q8: self.scratch.q8_scratch.as_ref().unwrap(),
                            out_q: &self.scratch.q, out_k: &self.scratch.k, out_v: &self.scratch.v,
                            bias_q: layer.q_bias.as_ref(),
                            bias_k: layer.k_bias.as_ref(),
                            bias_v: layer.v_bias.as_ref(),
                            nq: qd, nk: kd, nv: kd, k: h,
                        }).is_ok()
                    }
                    _ => false,
                }
            } else { false };
            // Mixed Q4K+Q4K+Q6K fused path: wv=Q6K breaks the above — use dedicated kernel
            // This is the Qwen2.5-1.5B GGUF layout: saves ~4 launches per layer vs fallback
            let mixed_fused_qkv_ok = if n == 1 && (!unfused || force_fuse_qkv) && !super_fused_qkv && !fused_qkv_ok {
                match (&layer.wq, &layer.wk, &layer.wv) {
                    (LayerWeight::Q4K { data: dq, .. }, LayerWeight::Q4K { data: dk, .. }, LayerWeight::Q6K { data: dv, .. }) => {
                        static MIXED_QKV_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        let r = self.backend.fused_rms_qkv_q4k_q6k_f32(&compute_core::FusedRmsQkvQ4KQ6KF32Params {
                            x: &self.scratch.x, norm_w: &layer.input_norm_w,
                            wq: dq, wk: dk, wv: dv,
                            out_q: &self.scratch.q, out_k: &self.scratch.k, out_v: &self.scratch.v,
                            bias_q: layer.q_bias.as_ref(), bias_k: layer.k_bias.as_ref(), bias_v: layer.v_bias.as_ref(),
                            nq: qd, nk: kd, nv: kd, k: h, eps,
                        });
                        if r.is_ok() && !MIXED_QKV_LOG.swap(true, AO::Relaxed) {
                            log::warn!("FUSED QKV Q4K+Q4K+Q6K: active (nq={}, nk={}, k={}) — 5→1 launches", qd, kd, h);
                        }
                        r.is_ok()
                    }
                    _ => false,
                }
            } else { false };

            if super_fused_qkv || fused_qkv_ok || mixed_fused_qkv_ok {
                if !FUSED_QKV_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    if !mixed_fused_qkv_ok {
                        log::warn!("FUSED QKV: active (nq={}, nk={}, nv={}, k={})", qd, kd, kd, h);
                    }
                }
                // mixed_fused_qkv already wrote normed+q+k+v; skip norm fallback below
                if mixed_fused_qkv_ok {
                    // norm was done inside the fused kernel — mark normed as computed
                    // (no extra norm needed, outputs already in scratch.q/k/v)
                }
            } else {
                Self::project(&self.backend, &layer.wq, &self.scratch.normed, &self.scratch.q, n, qd, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), qkv_q8_pre)?;
                prof_sync!("gemv_q");
                Self::project(&self.backend, &layer.wk, &self.scratch.normed, &self.scratch.k, n, kd, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), qkv_q8_pre)?;
                Self::project(&self.backend, &layer.wv, &self.scratch.normed, &self.scratch.v, n, kd, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), qkv_q8_pre)?;
            }
            prof_sync!("gemv_kv");

            // ── DEBUG Q4: dump first layer's projections ──
            if layer_idx == 0 && std::env::var("IARA_DEBUG_Q4").map(|v| v == "1").unwrap_or(false) {
                static DEBUG_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !DEBUG_DONE.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    let normed_data = self.backend.download_f32(&self.scratch.normed);
                    let q_data = self.backend.download_f32(&self.scratch.q);
                    let k_data = self.backend.download_f32(&self.scratch.k);
                    log::info!("🔍 DEBUG Q4 layer 0 (n={}, qd={}, kd={}, h={}):", n, qd, kd, h);
                    log::info!("  normed[0..8] = {:?}", &normed_data[..8.min(normed_data.len())]);
                    log::info!("  q[0..8]      = {:?}", &q_data[..8.min(q_data.len())]);
                    log::info!("  k[0..8]      = {:?}", &k_data[..8.min(k_data.len())]);
                    let q_nans = q_data.iter().filter(|x| x.is_nan()).count();
                    let q_infs = q_data.iter().filter(|x| x.is_infinite()).count();
                    let q_zeros = q_data.iter().filter(|x| **x == 0.0).count();
                    let q_max = q_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let q_min = q_data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let q_mean = q_data.iter().sum::<f32>() / q_data.len() as f32;
                    log::info!("  q stats: min={:.6}, max={:.6}, mean={:.6}, nans={}, infs={}, zeros={}/{}", q_min, q_max, q_mean, q_nans, q_infs, q_zeros, q_data.len());

                    // Manual dot product check: compute q[0] = dot(normed[0:h], wq_row0[0:h]) on CPU
                    let wq_f32 = match &layer.wq {
                        LayerWeight::Bf16(t) => self.backend.download_bf16_as_f32(t),
                        _ => vec![],
                    };
                    if !wq_f32.is_empty() {
                        let manual_q0: f32 = (0..h).map(|i| normed_data[i] * wq_f32[i]).sum();
                        let manual_q1: f32 = (0..h).map(|i| normed_data[i] * wq_f32[h + i]).sum();
                        log::info!("  Manual CPU: q[0]={:.6} (GPU={:.6}), q[1]={:.6} (GPU={:.6})",
                            manual_q0, q_data[0], manual_q1, q_data[1]);
                        log::info!("  wq shape: {} rows × {} cols (numel={})",
                            wq_f32.len() / h, h, wq_f32.len());
                        log::info!("  wq row0[0..5]: {:?}", &wq_f32[..5]);
                        log::info!("  wq row1[0..5]: {:?}", &wq_f32[h..h+5]);
                    }
                }
            }

            prof_sync!("qkv_done");
            section_time!(PROF_QKV, _sec_t0);
            // Biases: fused into fused_qkv/super_fused/mixed_fused kernel when active, otherwise separate launches
            let any_fused_qkv = fused_qkv_ok || super_fused_qkv || mixed_fused_qkv_ok;
            if !any_fused_qkv {
                if let Some(ref bias) = layer.q_bias {
                    self.backend.add_bias_inplace(&AddBiasParams {
                        tensor: &self.scratch.q,
                        bias,
                        n_tokens: n,
                        dim: qd,
                    })?;
                }
                if let Some(ref bias) = layer.k_bias {
                    self.backend.add_bias_inplace(&AddBiasParams {
                        tensor: &self.scratch.k,
                        bias,
                        n_tokens: n,
                        dim: kd,
                    })?;
                }
                if let Some(ref bias) = layer.v_bias {
                    self.backend.add_bias_inplace(&AddBiasParams {
                        tensor: &self.scratch.v,
                        bias,
                        n_tokens: n,
                        dim: kd,
                    })?;
                }
            }

            // ── QK Normalization (Qwen3) ─────────────────────────────────────
            // T8 TODO: Fused Norm+Bias+RoPE kernel integration point
            // Currently: Apply RMSNorm, then bias, then RoPE as separate operations.
            // Future: When available, fuse into single kernel for Q and K:
            //   fused_norm_bias_rope_q() then fused_norm_bias_rope_k()
            // This would reduce VRAM round-trips: norm(Q) + bias(Q) + rope(Q) → 1 kernel
            // Precondition: Only applicable when QK-norm is present (Qwen3 models).
            // For now: Keep sequential operations; they are correct and not a bottleneck.
            //
            // Aplica RMSNorm per-head em Q e K antes do RoPE.
            // Q: [n, qd] → tratado como [n * nh, hd] → RMSNorm com peso [hd]
            // K: [n, kd] → tratado como [n * nk, hd] → RMSNorm com peso [hd]
            //
            // In-place: o kernel HIP faz read→reduce→write com __syncthreads,
            // cada thread block processa um "token" (= uma cabeça), logo é safe.
            if let Some(ref qnw) = layer.q_norm_w {
                self.backend.rms_norm(&RmsNormParams {
                    input:  &self.scratch.q,
                    weight: qnw,
                    output: &self.scratch.q, // in-place
                    n_tokens: n * nh,        // cada "token" = 1 head
                    hidden_size: hd,
                    eps,
                })?;
            }
            if let Some(ref knw) = layer.k_norm_w {
                self.backend.rms_norm(&RmsNormParams {
                    input:  &self.scratch.k,
                    weight: knw,
                    output: &self.scratch.k, // in-place
                    n_tokens: n * nk,        // cada "token" = 1 kv head
                    hidden_size: hd,
                    eps,
                })?;
            }

            prof_sync!("bias_done");
            // ── RoPE + KV cache append ────────────────────────────────────────
            let pos_start = layer.kv_k.current_len;

            // Try fused RoPE + KV copy (1 launch vs 4) for F32 KV decode
            let fused_rope_ok = if n == 1 && (!unfused || force_fuse_rope) && matches!(layer.kv_k.mode, kv_cache::KvCacheMode::F32) {
                let params = compute_core::FusedRopeKvCopyParams {
                    q: &self.scratch.q, k: &self.scratch.k, v: &self.scratch.v,
                    kv_k_cache: &layer.kv_k.data, kv_v_cache: &layer.kv_v.data,
                    n_heads: nh, n_kv_heads: nk, head_dim: hd,
                    pos_start, theta, kv_dim: kd, max_seq: self.cfg.max_seq_len,
                };
                let res = if self.graph_replay_mode {
                    self.backend.fused_rope_kv_copy_dyn(&params)
                } else {
                    self.backend.fused_rope_kv_copy(&params)
                };
                match res {
                    Ok(()) => {
                        static FUSED_ROPE_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !FUSED_ROPE_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("FUSED ROPE+KV: active (nh={}, nk={}, hd={})", nh, nk, hd);
                        }
                        true
                    }
                    Err(e) => {
                        static FUSED_ROPE_FAIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !FUSED_ROPE_FAIL.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("FUSED ROPE+KV FAILED: {}", e);
                        }
                        false
                    }
                }
            } else { false };

            if !fused_rope_ok {
                self.backend.rope_inplace(&RopeParams {
                    tensor: &self.scratch.q,
                    pos_offset: pos_start,
                    n_tokens: n,
                    n_heads: nh,
                    head_dim: hd,
                    theta,
                })?;
                self.backend.rope_inplace(&RopeParams {
                    tensor: &self.scratch.k,
                    pos_offset: pos_start,
                    n_tokens: n,
                    n_heads: nk,
                    head_dim: hd,
                    theta,
                })?;
            }

            // ── DEBUG: dump Q/K after RoPE ──
            if layer_idx == 0 && std::env::var("IARA_DEBUG_FWD").map(|v| v == "1").unwrap_or(false) {
                let q_data = self.backend.download_f32(&self.scratch.q);
                let k_data = self.backend.download_f32(&self.scratch.k);
                log::info!("🔬 L0 q_after_rope[0..8] = {:?}", &q_data[..8.min(q_data.len())]);
                log::info!("🔬 L0 k_after_rope[0..8] = {:?}", &k_data[..8.min(k_data.len())]);
            }

            prof_sync!("rope_done");
            // ── Append K, V ao KV-cache (skip if already done by fused kernel) ──
            if fused_rope_ok {
                // Fused kernel already copied K and V to cache
            } else {
            match layer.kv_k.mode {
                kv_cache::KvCacheMode::F32 => {
                    self.backend.copy_to_kv_cache(&KvCacheParams {
                        cache:        &layer.kv_k.data,
                        new_kv:       &self.scratch.k,
                        insert_pos:   pos_start,
                        n_new_tokens: n,
                        kv_dim:       kd,
                        max_seq:      self.cfg.max_seq_len,
                    })?;
                },
                kv_cache::KvCacheMode::Int8 => {
                    let scales = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but no scales allocated".into()))?;
                    self.backend.kv_cache_int8_append(&compute_core::KvCacheInt8AppendParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.k,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                },
                kv_cache::KvCacheMode::Wht => {
                    let scales = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but no scales allocated".into()))?;
                    self.backend.kv_cache_wht_append(&compute_core::KvCacheWhtAppendParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.k,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                },
            }

            match layer.kv_v.mode {
                kv_cache::KvCacheMode::F32 => {
                    self.backend.copy_to_kv_cache(&KvCacheParams {
                        cache:        &layer.kv_v.data,
                        new_kv:       &self.scratch.v,
                        insert_pos:   pos_start,
                        n_new_tokens: n,
                        kv_dim:       kd,
                        max_seq:      self.cfg.max_seq_len,
                    })?;
                },
                kv_cache::KvCacheMode::Int8 => {
                    let scales = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but no scales allocated".into()))?;
                    self.backend.kv_cache_int8_append(&compute_core::KvCacheInt8AppendParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.v,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                },
                kv_cache::KvCacheMode::Wht => {
                    let scales = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but no scales allocated".into()))?;
                    self.backend.kv_cache_wht_append(&compute_core::KvCacheWhtAppendParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.v,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                },
            }
            } // end else !fused_rope_ok (KV cache append)

            layer.kv_k.current_len += n;
            layer.kv_v.current_len += n;

            let seq = layer.kv_k.current_len;

            // ── Dequantize KV cache if INT8 / WHT mode ────────────────────────
            let (kv_k_use, kv_v_use) = match layer.kv_k.mode {
                kv_cache::KvCacheMode::F32 => {
                    (&layer.kv_k.data, &layer.kv_v.data)
                },
                kv_cache::KvCacheMode::Int8 => {
                    let kv_k_deq = self.scratch.kv_k_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but kv_k_deq not allocated".into()))?;
                    let kv_v_deq = self.scratch.kv_v_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but kv_v_deq not allocated".into()))?;

                    let scales_k = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but kv_k scales not allocated".into()))?;
                    self.backend.kv_cache_int8_read(&compute_core::KvCacheInt8ReadParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales_k,
                        output: kv_k_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    let scales_v = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but kv_v scales not allocated".into()))?;
                    self.backend.kv_cache_int8_read(&compute_core::KvCacheInt8ReadParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales_v,
                        output: kv_v_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    (kv_k_deq, kv_v_deq)
                },
                kv_cache::KvCacheMode::Wht => {
                    let kv_k_deq = self.scratch.kv_k_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but kv_k_deq not allocated".into()))?;
                    let kv_v_deq = self.scratch.kv_v_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but kv_v_deq not allocated".into()))?;

                    let scales_k = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but kv_k scales not allocated".into()))?;
                    self.backend.kv_cache_wht_read(&compute_core::KvCacheWhtReadParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales_k,
                        output: kv_k_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    let scales_v = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but kv_v scales not allocated".into()))?;
                    self.backend.kv_cache_wht_read(&compute_core::KvCacheWhtReadParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales_v,
                        output: kv_v_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    (kv_k_deq, kv_v_deq)
                },
            };

            // ── Attention: Q × K^T → softmax → × V ───────────────────────────
            // T7: Fused decode attention (n=1) or batched attention (n>1)
            //
            // For decode (n=1): Use fused kernel that does QK^T+softmax+AV in one pass.
            //   This eliminates the scores tensor and 3 kernel launches per layer.
            //   Supports GQA natively via kv_head mapping inside the kernel.
            //
            // For prefill (n>1): Use separate GEMM+softmax+GEMM (batched or GQA grouped).

            prof_sync!("kv_done");
            section_time!(PROF_ROPE_KV, _sec_t0);
            let use_fused_decode = n == 1 && seq > 0 && (!unfused || force_fuse_attn);
            let mut fused_ok = false;

            if use_fused_decode {
                // ── FUSED DECODE ATTENTION (n=1) ─────────────────────────────
                log::trace!("layer {layer_idx}: attn FUSED DECODE seq={seq} nh={nh} nk={nk} hd={hd}");
                let attn_params = FusedAttentionDecodeParams {
                    q: &self.scratch.q,
                    k_cache: kv_k_use,
                    v_cache: kv_v_use,
                    out: &self.scratch.attn_out,
                    seq_len: seq,
                    head_dim: hd,
                    n_heads: nh,
                    n_kv_heads: nk,
                    scale,
                };
                let attn_res = if self.graph_replay_mode {
                    self.backend.fused_attention_decode_dyn(&attn_params, self.cfg.max_seq_len)
                } else {
                    self.backend.fused_attention_decode(&attn_params)
                };
                match attn_res {
                    Ok(()) => {
                        static FUSED_ATTN_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !FUSED_ATTN_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("FUSED ATTN DECODE: active (nh={}, nk={}, hd={}, seq={})", nh, nk, hd, seq);
                        }
                        fused_ok = true;
                    }
                    Err(e) => {
                        static FUSED_ATTN_FAIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !FUSED_ATTN_FAIL.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            log::warn!("FUSED ATTN DECODE FAILED: {}", e);
                        }
                        fused_ok = false;
                    }
                }
            }

            // GQA grouped batched attention (for prefill n>1, or fused fallback)
            if !fused_ok && seq > 0 {
                // GQA grouped batched: group Q heads that share the same KV head
                // For Qwen2.5-1.5b: 12 Q heads / 2 KV heads = 6 Q heads per group
                // Instead of 12×3 = 36 sgemm calls, do 2×2 = 4 sgemm_strided_batched + 12 softmax
                // stride_b = 0 broadcasts the same K/V head to all Q heads in the group
                let heads_per_group = nh / nk;
                log::trace!("layer {layer_idx}: attn GQA GROUPED — {nk} groups × {heads_per_group} Q heads/group");

                for kv_h in 0..nk {
                    let q_start = kv_h * heads_per_group;

                    // T7a-GQA: Batched Q×K^T for all Q heads in this KV group
                    // Q heads [q_start..q_start+heads_per_group] all multiply by K[kv_h]
                    log::trace!("layer {layer_idx}: attn QK GQA kv_h={kv_h} q_start={q_start} batch={heads_per_group}");
                    self.backend.sgemm_strided_batched(&SgemmStridedBatchedParams {
                        trans_a: false,
                        trans_b: true,
                        m: n,
                        n: seq,
                        k: hd,
                        alpha: scale,
                        beta: 0.0,
                        a: &self.scratch.q,
                        lda: qd,           // physical stride between token rows in Q
                        stride_a: hd,      // stride between Q heads within a row
                        a_offset: q_start * hd,  // first Q head in this group
                        b: kv_k_use,
                        ldb: kd,           // physical stride between token rows in K
                        stride_b: 0,       // broadcast: same K head for all Q heads
                        b_offset: kv_h * hd,     // this KV group's K head
                        c: &self.scratch.scores,
                        ldc: seq,          // leading dim of each score block [n, seq]
                        stride_c: n * seq, // stride between score blocks per Q head
                        c_offset: q_start * n * seq,  // scores start for this group
                        batch_count: heads_per_group,
                    })?;

                    // T7b-GQA: Softmax all Q heads in this KV group in one launch
                    // Use causal mask during prefill (n > 1) to prevent attending to future tokens
                    if n > 1 {
                        self.backend.softmax_rows_causal(&compute_core::SoftmaxCausalParams {
                            scores: &self.scratch.scores,
                            offset_elems: q_start * n * seq,
                            n_rows: heads_per_group * n,
                            row_len: seq,
                            n_tokens: n,
                            query_offset: pos_start,
                        })?;
                    } else {
                        self.backend.softmax_rows(&SoftmaxParams {
                            scores: &self.scratch.scores,
                            offset_elems: q_start * n * seq,
                            n_rows: heads_per_group * n,
                            row_len: seq,
                        })?;
                    }

                    // T7c-GQA: Batched Attn×V for all Q heads in this KV group
                    log::trace!("layer {layer_idx}: attn AV GQA kv_h={kv_h} q_start={q_start} batch={heads_per_group}");
                    self.backend.sgemm_strided_batched(&SgemmStridedBatchedParams {
                        trans_a: false,
                        trans_b: false,
                        m: n,
                        n: hd,
                        k: seq,
                        alpha: 1.0,
                        beta: 0.0,
                        a: &self.scratch.scores,
                        lda: seq,          // leading dim of score block
                        stride_a: n * seq, // stride between score blocks
                        a_offset: q_start * n * seq,
                        b: kv_v_use,       // use deq buffer when INT8 mode
                        ldb: kd,           // physical stride between token rows in V
                        stride_b: 0,       // broadcast: same V head for all Q heads
                        b_offset: kv_h * hd,     // this KV group's V head
                        c: &self.scratch.attn_out,
                        ldc: qd,           // physical stride between token rows in output
                        stride_c: hd,      // stride between output heads within a row
                        c_offset: q_start * hd,  // output start for this group
                        batch_count: heads_per_group,
                    })?;
                }
            }

            // ── DEBUG: dump attention output before wo ──
            if layer_idx == 0 && std::env::var("IARA_DEBUG_FWD").map(|v| v == "1").unwrap_or(false) {
                let attn_data = self.backend.download_f32(&self.scratch.attn_out);
                log::info!("🔬 L0 attn_out[0..8] = {:?}", &attn_data[..8.min(attn_data.len())]);
                let x_before = self.backend.download_f32(&self.scratch.x);
                log::info!("🔬 L0 x_before_wo[0..8] = {:?}", &x_before[..8.min(x_before.len())]);
            }

            prof_sync!("attn_done");
            section_time!(PROF_ATTN, _sec_t0);
            // ── Output projection: attn_out × wo^T → x (residual) ─────────────
            // x = x + attn_out × wo^T  (beta=1.0 para acumular em x)
            log::trace!("layer {layer_idx}: wo");
            Self::project(&self.backend, &layer.wo, &self.scratch.attn_out, &self.scratch.x, n, h, qd, 0, 1.0, self.scratch.q8_scratch.as_ref(), false)?;

            prof_sync!("wo_done");
            section_time!(PROF_WO, _sec_t0);
            // ── Post-attention Normalization (+ optional fused quantize for MLP) ──
            // Check if super-fused MLP is possible — avoids norm+quantize if weights are Q4K
            let super_fused_mlp_possible = n == 1 && (!unfused || force_fuse_mlp) && !force_q8_mlp && matches!(norm_type, NormType::RMSNorm)
                && matches!((&layer.wgate, &layer.wup), (LayerWeight::Q4K { .. }, LayerWeight::Q4K { .. }));
            let mlp_q8_pre = if super_fused_mlp_possible { false } else if n == 1 && matches!(norm_type, NormType::RMSNorm) {
                if let Some(q8) = self.scratch.q8_scratch.as_ref() {
                    if self.backend.fused_rms_norm_quantize(&compute_core::FusedRmsNormQuantizeParams {
                        input: &self.scratch.x, weight: &layer.post_attn_norm_w,
                        out_f32: &self.scratch.normed, out_q8: q8,
                        hidden_size: h, eps,
                    }).is_ok() {
                        true
                    } else {
                        Self::apply_norm(&self.backend, norm_type, &self.scratch.x, &layer.post_attn_norm_w, None, &self.scratch.normed, n, h, eps)?;
                        Self::quantize_input_q8(&self.backend, &self.scratch.normed, 0, q8, h)?;
                        true
                    }
                } else {
                    Self::apply_norm(&self.backend, norm_type, &self.scratch.x, &layer.post_attn_norm_w, None, &self.scratch.normed, n, h, eps)?;
                    false
                }
            } else {
                Self::apply_norm(&self.backend, norm_type, &self.scratch.x, &layer.post_attn_norm_w, None, &self.scratch.normed, n, h, eps)?;
                false
            };

            section_time!(PROF_POST_NORM, _sec_t0);
            // ── MLP: Dense, Calibration, or Sparse path ─────────────────────────
            //
            // Calibration: during first prefill (n > 1), run dense MLP but intercept
            // gate activations after SiLU*mul to build per-neuron firing statistics.
            // After calibration, use static hot neuron indices for sparse decode.
            //
            // Sparse path: pre-gathered hot neuron weights (from calibration),
            // reused for all decode tokens. Zero per-token overhead.

            // Calibration threshold: number of tokens to calibrate on before switching to sparse.
            // Prefill tokens + first few decode tokens provide good activation statistics.
            const CALIBRATION_MIN_TOKENS: usize = 16;

            // Auto-finalize calibration after enough tokens
            if let Some(ref mut ss) = self.sparse_scratch {
                if !ss.calibrated && ss.calibration_accum.is_some()
                    && ss.calibration_tokens >= CALIBRATION_MIN_TOKENS
                    && layer_idx == 0
                {
                    log::info!("🔄 Finalizing calibration ({} tokens processed)...", ss.calibration_tokens);
                    ss.finalize_calibration(&self.backend);
                }
            }

            let use_sparse = self.sparse_scratch.is_some()
                && n == 1  // sparse path only for single-token decode
                && self.sparse_scratch.as_ref().map_or(false, |ss| ss.calibrated)
                && matches!(&layer.wgate, LayerWeight::Bf16(_));

            let is_calibrating = self.sparse_scratch.is_some()
                && self.sparse_scratch.as_ref().map_or(false, |ss| {
                    ss.calibration_accum.is_some() && !ss.calibrated
                });

            if use_sparse {
                // ── CALIBRATED-SPARSE MLP PATH ──────────────────────────────
                let ss = self.sparse_scratch.as_mut().unwrap();
                let n_active = ss.cached_n_active[layer_idx];

                // One-time gather: use calibrated hot indices to pre-gather weights
                if !ss.weights_valid[layer_idx] && ss.static_hot_indices[layer_idx].is_some() {
                    if let (LayerWeight::Bf16(wg), LayerWeight::Bf16(wu), LayerWeight::Bf16(wd)) =
                        (&layer.wgate, &layer.wup, &layer.wdown)
                    {
                        use compute_core::params::{GatherRowsBf16Params, GatherColsBf16Params};
                        let idx_buf = ss.static_hot_indices[layer_idx].as_ref().unwrap();

                        let g_buf = self.backend.alloc_bf16(&[n_active, h]);
                        let u_buf = self.backend.alloc_bf16(&[n_active, h]);
                        let d_buf = self.backend.alloc_bf16(&[h, n_active]);

                        self.backend.gather_rows_bf16(&GatherRowsBf16Params {
                            src: wg, dst: &g_buf, indices: idx_buf,
                            n_indices: n_active, cols: h,
                        })?;
                        self.backend.gather_rows_bf16(&GatherRowsBf16Params {
                            src: wu, dst: &u_buf, indices: idx_buf,
                            n_indices: n_active, cols: h,
                        })?;
                        self.backend.gather_cols_bf16(&GatherColsBf16Params {
                            src: wd, dst: &d_buf, indices: idx_buf,
                            rows: h, total_cols: it, n_indices: n_active,
                        })?;

                        ss.cached_weights[layer_idx] = Some((g_buf, u_buf, d_buf));
                        ss.weights_valid[layer_idx] = true;

                        if layer_idx == 0 {
                            log::info!(
                                "🔥 Calibrated sparse MLP: gathered {}/{} hot neurons ({:.0}%)",
                                n_active, it, 100.0 * n_active as f32 / it as f32,
                            );
                        }
                    }
                }

                if layer_idx == 0 {
                    ss.token_counter += 1;
                    if ss.token_counter % 64 == 0 {
                        log::info!(
                            "⚡ SPARSE MLP (calibrated) — {}/{} neurons ({:.0}%), token: {}",
                            n_active, it,
                            100.0 * n_active as f32 / it as f32,
                            ss.token_counter
                        );
                    }
                }

                if n_active > 0 && ss.weights_valid[layer_idx] && ss.cached_weights[layer_idx].is_some() {
                    // ── FAST PATH: pre-gathered calibrated weights ──
                    let (ref cg, ref cu, ref cd) = ss.cached_weights[layer_idx].as_ref().unwrap();

                    self.backend.gemm_bf16_f32(&compute_core::params::GemmBf16Params {
                        m: n, n: n_active, k: h,
                        alpha: 1.0, beta: 0.0,
                        a: &self.scratch.normed, a_offset: 0,
                        b: cg, c: &ss.sparse_gate,
                    })?;
                    self.backend.gemm_bf16_f32(&compute_core::params::GemmBf16Params {
                        m: n, n: n_active, k: h,
                        alpha: 1.0, beta: 0.0,
                        a: &self.scratch.normed, a_offset: 0,
                        b: cu, c: &ss.sparse_up,
                    })?;

                    Self::apply_activation(
                        &self.backend, activation,
                        &ss.sparse_gate, &ss.sparse_up, n, n_active,
                    )?;

                    self.backend.gemm_bf16_f32(&compute_core::params::GemmBf16Params {
                        m: n, n: h, k: n_active,
                        alpha: 1.0, beta: 1.0,
                        a: &ss.sparse_gate, a_offset: 0,
                        b: cd, c: &self.scratch.x,
                    })?;

                } else {
                    // Weights not ready yet — fallback to dense for this layer
                    // Pre-quantize normed for gate+up (same input)
                    let gu_q8_pre = if n == 1 {
                        if let Some(q8) = self.scratch.q8_scratch.as_ref() {
                            Self::quantize_input_q8(&self.backend, &self.scratch.normed, 0, q8, h)?;
                            true
                        } else { false }
                    } else { false };
                    Self::project(&self.backend, &layer.wgate, &self.scratch.normed, &self.scratch.gate, n, it, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), gu_q8_pre)?;
                    Self::project(&self.backend, &layer.wup, &self.scratch.normed, &self.scratch.up, n, it, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), gu_q8_pre)?;
                    Self::apply_activation(&self.backend, activation, &self.scratch.gate, &self.scratch.up, n, it)?;
                    Self::project(&self.backend, &layer.wdown, &self.scratch.gate, &self.scratch.x, n, h, it, 0, 1.0, self.scratch.q8_scratch.as_ref(), false)?;
                }
            } else {
                // ── DENSE MLP PATH (also used for calibration) ─────────────────
                log::trace!("layer {layer_idx}: wgate+wup n={n} it={it} h={h}");

                // Super-fused: RMSNorm + gate+up+SiLU in ONE kernel (saves norm+q8+gate_up launches)
                static SUPER_FUSED_MLP_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                let super_fused_mlp = if super_fused_mlp_possible && !is_calibrating {
                    match (&layer.wgate, &layer.wup) {
                        (LayerWeight::Q4K { data: gd, .. }, LayerWeight::Q4K { data: ud, .. }) => {
                            self.backend.fused_rms_gate_up_silu_q4k_f32(&compute_core::FusedRmsGateUpSiluQ4KF32Params {
                                x: &self.scratch.x, norm_w: &layer.post_attn_norm_w,
                                gate: gd, up: ud, out: &self.scratch.gate,
                                n: it, k: h, eps, x_offset: 0,
                            }).is_ok()
                        }
                        _ => false,
                    }
                } else { false };
                if super_fused_mlp && !SUPER_FUSED_MLP_LOG.swap(true, AO::Relaxed) {
                    log::warn!("SUPER FUSED RMS+MLP: active (n={}, k={})", it, h);
                }

                // Pre-quantize normed for gate+up (reuse from fused norm+q8 if available)
                let gu_q8_pre = if super_fused_mlp { false } else if mlp_q8_pre {
                    true  // already quantized by fused_rms_norm_quantize
                } else if n == 1 {
                    if let Some(q8) = self.scratch.q8_scratch.as_ref() {
                        Self::quantize_input_q8(&self.backend, &self.scratch.normed, 0, q8, h)?;
                        true
                    } else { false }
                } else { false };

                // Try fused gate+up+SiLU kernel (1 launch vs 3) for decode — skipped if super_fused_mlp
                let fused_mlp_ok = if !super_fused_mlp && (!unfused || force_fuse_mlp) && n == 1 && gu_q8_pre && !is_calibrating {
                    match (&layer.wgate, &layer.wup) {
                        (LayerWeight::Q4K { data: gd, .. }, LayerWeight::Q4K { data: ud, .. }) => {
                            self.backend.fused_gate_up_silu(&compute_core::FusedGateUpSiluParams {
                                gate_data: gd, up_data: ud,
                                x_q8: self.scratch.q8_scratch.as_ref().unwrap(),
                                out: &self.scratch.gate, n: it, k: h,
                            }, false).is_ok()
                        }
                        (LayerWeight::Q6K { data: gd, .. }, LayerWeight::Q6K { data: ud, .. }) => {
                            self.backend.fused_gate_up_silu(&compute_core::FusedGateUpSiluParams {
                                gate_data: gd, up_data: ud,
                                x_q8: self.scratch.q8_scratch.as_ref().unwrap(),
                                out: &self.scratch.gate, n: it, k: h,
                            }, true).is_ok()
                        }
                        _ => false,
                    }
                } else { false };

                if fused_mlp_ok {
                    static FUSED_MLP_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !FUSED_MLP_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        log::warn!("FUSED MLP: active (n={}, k={})", it, h);
                    }
                }
                if !super_fused_mlp && !fused_mlp_ok {
                    Self::project(&self.backend, &layer.wgate, &self.scratch.normed, &self.scratch.gate, n, it, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), gu_q8_pre)?;
                    prof_sync!("mlp_gate");
                    Self::project(&self.backend, &layer.wup, &self.scratch.normed, &self.scratch.up, n, it, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), gu_q8_pre)?;
                    prof_sync!("mlp_up");
                    Self::apply_activation(
                        &self.backend, activation,
                        &self.scratch.gate, &self.scratch.up, n, it,
                    )?;
                }
                prof_sync!("mlp_silu");

                // Calibration: intercept gate activations after SiLU*mul
                if is_calibrating {
                    if let Some(ref mut ss) = self.sparse_scratch {
                        ss.accumulate_calibration(
                            layer_idx, &self.scratch.gate, &self.backend, n, it,
                        );
                        if layer_idx == n_layers - 1 {
                            ss.calibration_tokens += n;
                            if ss.calibration_tokens <= 2 || ss.calibration_tokens % 8 == 0 {
                                log::info!(
                                    "📊 Calibration: {} tokens accumulated",
                                    ss.calibration_tokens,
                                );
                            }
                        }
                    }
                }

                // down_proj: x = x + swiglu × wdown^T (beta=1 acumula em x)
                Self::project(&self.backend, &layer.wdown, &self.scratch.gate, &self.scratch.x, n, h, it, 0, 1.0, self.scratch.q8_scratch.as_ref(), false)?;
                prof_sync!("mlp_down");
            }

            prof_sync!("mlp_done");
            section_time!(PROF_MLP, _sec_t0);

            // ── Online BI measurement: cos_sim(x_before, x_after) ──────────
            if router_measure {
                if let Ok(()) = self.backend.cosine_sim_1d(
                    &self.layer_router.x_snapshot,
                    &self.scratch.x,
                    &self.layer_router.cos_sim_buf,
                    h,
                ) {
                    // Download single float (cos_sim result)
                    let sim = self.backend.download_f32(&self.layer_router.cos_sim_buf);
                    if !sim.is_empty() {
                        self.layer_router.update_bi(layer_idx, sim[0]);
                    }
                }
            }

            // ── DEBUG: dump x after full layer ──
            if std::env::var("IARA_DEBUG_FWD").map(|v| v == "1").unwrap_or(false) {
                let x_after = self.backend.download_f32(&self.scratch.x);
                let x_norm: f32 = x_after[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                if layer_idx == 0 {
                    log::info!("🔬 L0 x_after_mlp[0..8] = {:?}", &x_after[..8.min(x_after.len())]);
                }
                log::info!("🔬 L{} x row0 L2 = {:.4}, first4=[{:.4},{:.4},{:.4},{:.4}]",
                    layer_idx, x_norm, x_after[0], x_after[1], x_after[2], x_after[3]);
            }
            log::trace!("layer {layer_idx}: done");

            if let Some(t0) = _layer_t0 {
                self.backend.device_sync();
                let us = t0.elapsed().as_micros();
                log::warn!("PROFILE L{}: {}μs ({:.2}ms)", layer_idx, us, us as f64 / 1000.0);
            }

            // ── Phase coherence check for early exit (decode-only) ────────────────
            // Only check at the 2 boundary layers (fast and medium) to avoid
            // GPU→CPU sync on every layer. This reduces from 28 downloads to at most 2.
            // Gated on telemetry_enabled: each download_f32 is a GPU→CPU sync that
            // stalls the pipeline ~1-2ms. In default mode (telemetry off) we do ZERO
            // mid-forward syncs, keeping maximum throughput for benchmarks.
            let mut early_exit_flag = false;
            let is_boundary = layer_idx + 1 == fast_layer_count || layer_idx + 1 == medium_layer_count;
            if n == 1 && layer_budget.is_none() && is_boundary && self.telemetry_enabled {
                let x_data = self.backend.download_f32(&self.scratch.x);
                let start_idx = (n - 1) * h;
                let end_idx = start_idx + h;
                let token_data = if end_idx <= x_data.len() {
                    &x_data[start_idx..end_idx]
                } else {
                    &x_data[..h.min(x_data.len())]
                };

                let coherence = phase_coherence(token_data);

                // Check phase coherence at layer boundaries
                if layer_idx + 1 == fast_layer_count && coherence > EXIT_THRESHOLD_FAST {
                    log::debug!(
                        "Phase coherence {:.3} > {:.3} at fast layer boundary (layer {}): early exit",
                        coherence, EXIT_THRESHOLD_FAST, layer_idx
                    );
                    early_exit_flag = true;
                } else if layer_idx + 1 == medium_layer_count && coherence > EXIT_THRESHOLD_MEDIUM {
                    log::debug!(
                        "Phase coherence {:.3} > {:.3} at medium layer boundary (layer {}): early exit",
                        coherence, EXIT_THRESHOLD_MEDIUM, layer_idx
                    );
                    early_exit_flag = true;
                }
            }

            // ── Per-layer telemetry (sampled, decode-only) ─────────────────────
            // download_f32 é um sync implícito que para o pipeline GPU.
            // Telemetry is ZERO-COST when disabled (default). Enable via
            // model.telemetry_enabled = true when the dashboard needs live traces.
            // When telemetry_enabled, sample EVERY layer (not just every 12th).
            // The % 12 sampling was for low-overhead background collection;
            // neuroviz explicitly enables telemetry and needs all layers.
            let should_sample = self.telemetry_enabled && n <= 1;

            let trace = if should_sample {
                let x_data = self.backend.download_f32(&self.scratch.x);

                let start_idx = (n - 1) * h;
                let end_idx = start_idx + h;
                let token_data = if end_idx <= x_data.len() {
                    &x_data[start_idx..end_idx]
                } else {
                    &x_data[..h.min(x_data.len())]
                };

                let residual_norm = token_data.iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt();

                let max_activation = token_data.iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max);

                let mean_abs = token_data.iter()
                    .map(|v| v.abs())
                    .sum::<f32>() / h.max(1) as f32;
                let threshold = mean_abs * 2.0;
                let hot: Vec<usize> = token_data.iter()
                    .enumerate()
                    .filter(|(_, v)| v.abs() > threshold)
                    .take(10)
                    .map(|(i, _)| i)
                    .collect();

                // Phase coherence: Kuramoto order parameter from hidden state pairs
                let phase_coherence = iara_observe::phase_coherence_from(
                    &token_data[..h.min(token_data.len())],
                );

                // MLP activation rate: fraction of gate neurons with |act| > threshold
                // Download gate scratch (SiLU(gate)*up) for this layer
                let gate_data = self.backend.download_f32(&self.scratch.gate);
                let n_neurons = gate_data.len().min(it);
                let mlp_activation_rate = if n_neurons > 0 {
                    let alive = gate_data[..n_neurons].iter().filter(|&&v| v.abs() > 1e-6).count();
                    alive as f32 / n_neurons as f32 * 100.0
                } else {
                    0.0
                };

                LayerTrace {
                    layer_idx,
                    attention_energy: residual_norm * 0.6,
                    mlp_energy: residual_norm * 0.4,
                    residual_norm,
                    max_activation,
                    hot_neurons: hot,
                    time_us: 0,
                    phase_coherence,
                    mlp_activation_rate,
                }
            } else {
                LayerTrace {
                    layer_idx,
                    attention_energy: 0.0,
                    mlp_energy: 0.0,
                    residual_norm: 0.0,
                    max_activation: 0.0,
                    hot_neurons: vec![],
                    time_us: 0,
                    phase_coherence: 0.0,
                    mlp_activation_rate: 0.0,
                }
            };

            layer_traces.push(trace.clone());

            // ── iara-observe: real neuron snapshot (sampled, decode-only) ─────
            // Gate: same cadence as LayerTrace (every 12 layers + last).
            // Downloads gate activations (SiLU(gate)*up) from GPU → CPU.
            // ZERO-COST when iara_observe::GLOBAL_COLLECTOR is disabled.
            if should_sample {
                use iara_observe::{GLOBAL_COLLECTOR, NeuronSnapshot, FlowEvent, FlowEventType};

                if GLOBAL_COLLECTOR.is_enabled() {
                // Download gate tensor (contains SiLU(gate)*up post-activation)
                let gate_data = self.backend.download_f32(&self.scratch.gate);
                let n_neurons = gate_data.len().min(it);

                // Sort by absolute activation to find top-32 most active MLP neurons
                let mut indexed: Vec<(usize, f32)> = gate_data[..n_neurons]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v.abs()))
                    .collect();
                indexed.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                let alive = indexed.iter().filter(|(_, v)| *v > 1e-6).count();
                let dead_count = (n_neurons - alive) as u32;
                let mlp_activation_rate = if n_neurons > 0 {
                    alive as f32 / n_neurons as f32 * 100.0
                } else {
                    0.0
                };
                let mlp_gate_sparsity = if n_neurons > 0 {
                    1.0 - alive as f32 / n_neurons as f32
                } else {
                    0.0
                };
                let top_mlp = indexed[..32.min(indexed.len())].to_vec();

                // Use pos_start (KV position before this layer's token) as step proxy.
                // It equals the number of tokens processed so far.
                let step_proxy = layer.kv_k.current_len as u64;

                // Build NeuronSnapshot — residual_norm_post from the trace we just built
                let residual_norm_post = trace.residual_norm;

                // Phase coherence from the same x_data already downloaded for trace.
                // Re-download only if needed; since should_sample is true, x_data
                // was downloaded above. We compute coherence from scratch.x again to
                // avoid lifetime issues with the borrow above.
                let x2 = self.backend.download_f32(&self.scratch.x);
                let start_idx = (n - 1) * h;
                let token_slice = if start_idx + h <= x2.len() {
                    &x2[start_idx..start_idx + h]
                } else {
                    &x2[..h.min(x2.len())]
                };
                let coherence = iara_observe::phase_coherence_from(
                    &token_slice[..h.min(token_slice.len())],
                );

                let mut snap = NeuronSnapshot::new(layer_idx, step_proxy);
                snap.mlp_top_neurons = top_mlp;
                snap.mlp_activation_rate = mlp_activation_rate;
                snap.mlp_dead_neuron_count = dead_count;
                snap.mlp_gate_sparsity = mlp_gate_sparsity;
                snap.residual_norm_post = residual_norm_post;
                snap.phase_coherence = coherence;

                GLOBAL_COLLECTOR.record_neuron(snap);

                // Flow event: layer completed
                GLOBAL_COLLECTOR.record_flow(FlowEvent::now(
                    FlowEventType::LayerCompleted {
                        layer_idx,
                        elapsed_us: trace.time_us as u64,
                    },
                    step_proxy,
                ));
                } // end if GLOBAL_COLLECTOR.is_enabled()
            }

            // ── Send stimulus to orchestrator (experimental) ──────────────────
            #[cfg(feature = "experimental")]
            {
                // Only send every 4th layer to avoid channel pressure.
                // Use the computed residual_norm and max_activation from the trace.
                if let Some(tx) = &self.stimulus_tx {
                    if layer_idx % 4 == 0 {
                        let stim = crate::experimental::orchestrator::LayerStimulus {
                            layer_idx,
                            activation_energy: trace.max_activation,
                            residual_norm: trace.residual_norm,
                        };
                        let _ = tx.try_send(stim);
                    }
                }
            }

            // Early exit if phase coherence threshold exceeded
            if early_exit_flag {
                log::info!("Early exit triggered at layer {}", layer_idx);
                break;
            }
        }

        // ── Gravity + CfC (decode-only: n == 1) ──────────────────────────────
        // Durante o prefill (n > 1) o scratch.scores tem layout [nh, n, seq]
        // — incompatível com o que gravity e CfC esperam ([nh, seq] e [1, h]).
        // Ambas as operações só fazem sentido no decode, token a token.
        if n == 1 {
            // Gravity: scratch.scores = [nh, 1, seq] → gravity usa como [nh, seq]
            if let Some(ref mut grav) = self.gravity {
                let last_layer = self.layers.last().unwrap();
                let seq = last_layer.kv_k.current_len;
                match grav.compute(
                    &self.backend,
                    &self.scratch.scores,
                    &last_layer.kv_v.data,
                    seq,
                    nh,
                ) {
                    Ok(_result) => {
                        log::debug!(
                            "Gravity: mean={:.3}, max={:.3}, threshold={:.3}, high_tokens={}",
                            _result.mean_gravity, _result.max_gravity,
                            _result.effective_threshold, _result.high_gravity_tokens.len(),
                        );
                    }
                    Err(e) => log::warn!("Gravity score falhou: {}", e),
                }
            }

            // CfC: correction_buf é [1, h] — exatamente h floats.
            if let Some(cfc) = &mut self.cfc {
                let correction = cfc.forward(&self.backend, &self.scratch.normed)?;
                // Aplica correção apenas no único token: x[0..h] += correction[0..h]
                self.backend.add_tensors_inplace(
                    &self.scratch.x,
                    correction,
                    h,   // apenas 1 token × hidden_size
                    1.0,
                )?;
                log::debug!("CfC correction aplicada: {} elementos", h);
            }
        }

        // ── Advance layer router token counter ─────────────────────────────
        if n == 1 {
            self.layer_router.advance_token();
        }

        // ── Final Normalization ──────────────────────────────────────────────
        let mut _sec_t0 = std::time::Instant::now();  // re-init for final section
        let _cpu_layers_done = _fwd_t0.elapsed();
        // GPU-SYNC measurement DISABLED — 50 device_sync calls destroy pipeline
        if let Some(t0) = forward_t0 {
            self.backend.device_sync();
            log::warn!("PROF layers_done @ {:?}", t0.elapsed());
        }
        Self::apply_norm(
            &self.backend, norm_type,
            &self.scratch.x, &self.norm_w, self.norm_b.as_ref(),
            &self.scratch.normed, n, h, eps,
        )?;

        // ── lm_head (apenas o último token) ──────────────────────────────────
        // Suporta BF16 (load normal) ou Q4 (load_quantized): dispatch via project()
        // a_offset aponta para o último token: (n-1) * hidden_size
        Self::project(
            &self.backend, &self.lm_head,
            &self.scratch.normed, &self.scratch.logits,
            1, vocab, h, (n - 1) * h, 0.0,
            self.scratch.q8_scratch.as_ref(), false,
        )?;

        // Section prof: final norm + lm_head
        if do_section_prof {
            self.backend.device_sync();
            let _elapsed = _sec_t0.elapsed().as_micros() as u64;
            PROF_FINAL.fetch_add(_elapsed, AO::Relaxed);
        }
        // Report section profiling after 14 tokens (tok_num 0-13)
        if tok_num == 13 && !PROF_DONE.swap(true, AO::Relaxed) {
            let div = 14.0;  // 14 accumulated decode tokens (tok_num 0..=13)
            let n_layers = self.layers.len() as f64;
            log::warn!("=== SECTION PROFILE (avg of 14 tokens, all layers summed, µs per token) ===");
            log::warn!("  NORM+Q8:   {:>7.0}µs  ({:.0}µs/layer)", PROF_NORM.load(AO::Relaxed) as f64 / div, PROF_NORM.load(AO::Relaxed) as f64 / div / n_layers);
            log::warn!("  QKV proj:  {:>7.0}µs  ({:.0}µs/layer)", PROF_QKV.load(AO::Relaxed) as f64 / div, PROF_QKV.load(AO::Relaxed) as f64 / div / n_layers);
            log::warn!("  ROPE+KV:   {:>7.0}µs  ({:.0}µs/layer)", PROF_ROPE_KV.load(AO::Relaxed) as f64 / div, PROF_ROPE_KV.load(AO::Relaxed) as f64 / div / n_layers);
            log::warn!("  ATTN:      {:>7.0}µs  ({:.0}µs/layer)", PROF_ATTN.load(AO::Relaxed) as f64 / div, PROF_ATTN.load(AO::Relaxed) as f64 / div / n_layers);
            log::warn!("  WO proj:   {:>7.0}µs  ({:.0}µs/layer)", PROF_WO.load(AO::Relaxed) as f64 / div, PROF_WO.load(AO::Relaxed) as f64 / div / n_layers);
            log::warn!("  POST NORM: {:>7.0}µs  ({:.0}µs/layer)", PROF_POST_NORM.load(AO::Relaxed) as f64 / div, PROF_POST_NORM.load(AO::Relaxed) as f64 / div / n_layers);
            log::warn!("  MLP:       {:>7.0}µs  ({:.0}µs/layer)", PROF_MLP.load(AO::Relaxed) as f64 / div, PROF_MLP.load(AO::Relaxed) as f64 / div / n_layers);
            log::warn!("  FINAL:     {:>7.0}µs", PROF_FINAL.load(AO::Relaxed) as f64 / div);
            let total = (PROF_NORM.load(AO::Relaxed) + PROF_QKV.load(AO::Relaxed) + PROF_ROPE_KV.load(AO::Relaxed) + PROF_ATTN.load(AO::Relaxed) + PROF_WO.load(AO::Relaxed) + PROF_POST_NORM.load(AO::Relaxed) + PROF_MLP.load(AO::Relaxed) + PROF_FINAL.load(AO::Relaxed)) as f64 / div;
            log::warn!("  TOTAL:     {:>7.0}µs/token", total);
        }

        // ── Download logits (or skip for GPU argmax path) ───────────────────
        if let Some(t0) = forward_t0 {
            self.backend.device_sync();
            log::warn!("PROF lm_head_done @ {:?}", t0.elapsed());
        }
        let logits = if skip_logits_download {
            Vec::new()  // logits stay on device for GPU argmax
        } else {
            self.backend.download_f32(&self.scratch.logits)
        };
        if let Some(t0) = forward_t0 {
            log::warn!("PROF logits_download_done @ {:?}", t0.elapsed());
        }

        // Per-token timing (log every 50 tokens)
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static TOKEN_COUNT: AtomicU64 = AtomicU64::new(0);
            static ACCUM_US: AtomicU64 = AtomicU64::new(0);
            static CPU_DISPATCH_US: AtomicU64 = AtomicU64::new(0);
            let total_us = _fwd_t0.elapsed().as_micros() as u64;
            let dispatch_us = _cpu_layers_done.as_micros() as u64;
            let count = TOKEN_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            ACCUM_US.fetch_add(total_us, Ordering::Relaxed);
            CPU_DISPATCH_US.fetch_add(dispatch_us, Ordering::Relaxed);
            #[cfg(feature = "rocm")]
            let launches = hip_ffi::safe::get_launch_count();
            #[cfg(not(feature = "rocm"))]
            let launches = 0u64;
            if count % 50 == 0 {
                let avg_total = ACCUM_US.load(Ordering::Relaxed) as f64 / count as f64;
                let avg_dispatch = CPU_DISPATCH_US.load(Ordering::Relaxed) as f64 / count as f64;
                log::warn!("PERF [{} tokens] avg: {:.0}µs total, {:.0}µs cpu-dispatch, {:.0}µs gpu-wait, {} launches/token",
                    count, avg_total, avg_dispatch, avg_total - avg_dispatch, launches);
            }
        }

        Ok((logits, layer_traces))
    }

    /// Fast greedy forward: full forward pass + GPU argmax = returns token_id.
    /// Skips downloading 608KB of logits — only downloads 4 bytes (the argmax index).
    pub fn forward_greedy(&mut self, token_ids: &[u32]) -> ModelResult<u32> {
        use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
        static GREEDY_COUNT: AtomicU64 = AtomicU64::new(0);
        static GRAPH_LOGGED: AtomicBool = AtomicBool::new(false);

        let vocab = self.cfg.vocab_size;
        let n = token_ids.len();
        let h = self.cfg.hidden_size;
        let count = GREEDY_COUNT.fetch_add(1, Ordering::Relaxed);

        // HSA boundary sync at entry: drain HIP stream so HSA dispatches see
        // all prior HIP-written state. Active in NOSYNC and FAST modes.
        // Env cached at startup to avoid per-token /proc/self/environ reads.
        #[cfg(feature = "rocm")]
        {
            use std::sync::OnceLock;
            static MODE: OnceLock<bool> = OnceLock::new();
            let active = *MODE.get_or_init(|| {
                std::env::var("IARA_HSA_NOSYNC").map(|v| v == "1").unwrap_or(false)
                    || std::env::var("IARA_HSA_FAST").map(|v| v == "1").unwrap_or(false)
            });
            if active {
                self.backend.device_sync();
            }
        }

        self.backend.embed_lookup(&EmbedParams {
            token_ids,
            vocab: &self.embed,
            out: &self.scratch.x,
            n_tokens: n,
            hidden_size: h,
        })?;

        let greedy_t0 = std::time::Instant::now();
        #[cfg(feature = "rocm")]
        hip_ffi::safe::reset_launch_count();

        // HIP Graph capture for ternary decode:
        // For ternary models with layer skip, the forward path is deterministic
        // (same layers execute every token, same buffer pointers, only scalar
        // args like seq_pos/seq_len change). hipGraphExecUpdate should handle
        // scalar-only changes without re-instantiation.
        //
        // Enable with IARA_GRAPH=1. For decode (n=1), the kernel topology is fixed
        // (same launches every token, only scalar args change). Graph update handles this.
        // Works for both ternary and GGUF Q4K/Q6K models.
        // HIP Graph opt-in. Measured WORSE than direct dispatch on RX 6750 XT (RDNA2)
        // because current impl re-captures every token — update+launch avg 880µs/tok vs
        // 570µs direct dispatch. For win: capture ONCE, replay many; needs rework.
        // Enable with IARA_GRAPH=1 for experimentation.
        let use_graph = n == 1
            && std::env::var("IARA_GRAPH").map(|v| v == "1").unwrap_or(false);
        let use_graph_replay = n == 1 && self.graph_replay_mode;

        if use_graph_replay {
            // Pre-write KV position: kernels read from device buffer; attention adds +1 internally.
            let pre_pos = self.layers[0].kv_k.current_len as i32;
            let _ = self.backend.set_seq_pos(pre_pos);
            // Allow disabling replay shortcut for diagnosis: IARA_GRAPH_REPLAY=0 forces re-capture every token.
            let allow_replay = std::env::var("IARA_GRAPH_REPLAY").map(|v| v != "0").unwrap_or(true);
            if allow_replay && self.backend.try_replay_graph() {
                for layer in &mut self.layers {
                    layer.kv_k.current_len += 1;
                    layer.kv_v.current_len += 1;
                }
                let dispatch_us = greedy_t0.elapsed().as_micros();
                #[cfg(feature = "rocm")]
                let launches_this_token = hip_ffi::safe::get_launch_count();
                #[cfg(not(feature = "rocm"))]
                let launches_this_token = 0u64;
                if count < 3 {
                    log::warn!("HIP Graph REPLAY token {} (skip dispatch, {} launches recorded)", count, launches_this_token);
                }
                let _ = (dispatch_us, launches_this_token);
                return self.backend.gpu_argmax(&self.scratch.logits, vocab)
                    .map(|t| t as u32)
                    .map_err(|e| ModelError::Model(format!("gpu_argmax: {}", e)));
            }
            // First decode token in replay mode: capture full forward_inner.
            self.backend.begin_graph_capture();
        } else if use_graph {
            self.backend.begin_graph_capture();
        }

        // Forward pass WITHOUT embed (already done above)
        let _ = self.forward_inner(token_ids, None, true, true)?;
        let dispatch_us = greedy_t0.elapsed().as_micros();
        #[cfg(feature = "rocm")]
        let launches_this_token = hip_ffi::safe::get_launch_count();
        #[cfg(not(feature = "rocm"))]
        let launches_this_token = 0u64;

        // End capture + launch graph (no-op if use_graph=false)
        if use_graph || use_graph_replay {
            let graph_t0 = std::time::Instant::now();
            let graph_ok = self.backend.end_graph_capture_and_launch();
            let graph_us = graph_t0.elapsed().as_micros();
            if !graph_ok && !GRAPH_LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!("HIP Graph capture/launch FAILED — falling back to direct dispatch");
            } else if graph_ok && count < 3 {
                log::warn!("HIP Graph CAPTURE: {}µs ({} launches collapsed)", graph_us, launches_this_token);
            }
        }

        // FAST mode: drain HSA queue once before reading results.
        #[cfg(feature = "rocm")]
        {
            use std::sync::OnceLock;
            static FAST: OnceLock<bool> = OnceLock::new();
            if *FAST.get_or_init(|| std::env::var("IARA_HSA_FAST").map(|v| v == "1").unwrap_or(false)) {
                self.backend.hsa_barrier_and_wait();
            }
        }

        // GPU argmax runs on same stream (ordered after graph/direct kernels)
        match self.backend.gpu_argmax(&self.scratch.logits, vocab) {
            Ok(token_id) => {
                let total_us = greedy_t0.elapsed().as_micros();
                static GREEDY_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
                static GREEDY_ACCUM_US: AtomicU64 = AtomicU64::new(0);
                static GREEDY_DISPATCH_US: AtomicU64 = AtomicU64::new(0);
                let gc = GREEDY_LOG_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                GREEDY_ACCUM_US.fetch_add(total_us as u64, Ordering::Relaxed);
                GREEDY_DISPATCH_US.fetch_add(dispatch_us as u64, Ordering::Relaxed);
                if gc % 50 == 0 {
                    let avg = GREEDY_ACCUM_US.load(Ordering::Relaxed) as f64 / gc as f64;
                    let avg_d = GREEDY_DISPATCH_US.load(Ordering::Relaxed) as f64 / gc as f64;
                    log::warn!("GREEDY [{gc}tok] avg: {avg:.0}µs total ({avg_d:.0}µs dispatch, {:.0}µs GPU+argmax), {:.1} tok/s, {} launches/token",
                        avg - avg_d, 1_000_000.0 / avg, launches_this_token);
                }
                Ok(token_id as u32)
            },
            Err(_) => {
                let logits = self.backend.download_f32(&self.scratch.logits);
                Ok(logits.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0))
            }
        }
    }

    /// Forward pass with deep per-token trace capturing rich telemetry.
    /// Only operates on single token (n == 1). For sequence, use standard forward().
    ///
    /// Captures per-layer:
    /// - hidden_norm: L2 norm of hidden state
    /// - attn_entropy: Shannon entropy of attention scores
    /// - attn_top_positions: top-5 positions by attention weight
    /// - mlp_activation_rate: % of neurons non-zero after SiLU
    /// - mlp_top_neurons: top-10 activated neurons
    /// - residual_delta: L2 norm of (output - input)
    /// - phase_coherence: Kuramoto order parameter
    ///
    /// NOTE: This requires multiple GPU→CPU downloads and will be ~2-3x slower
    /// than standard forward. Only use when explicitly requested for debugging.
    pub fn forward_deep_trace(&mut self, token_ids: &[u32]) -> ModelResult<crate::inference::DeepTrace> {
        use crate::inference::tracer;

        let token_start = std::time::Instant::now();
        let n = token_ids.len();

        if n != 1 {
            return Err(ModelError::Model(
                "forward_deep_trace only supports single tokens (n=1)".into()
            ));
        }

        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_heads;
        let nk = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let qd = nh * hd;
        let kd = nk * hd;
        let it = self.cfg.intermediate_size;
        let eps = self.cfg.rms_eps;
        let theta = self.cfg.rope_theta;
        let scale = 1.0_f32 / (hd as f32).sqrt();
        let vocab = self.cfg.vocab_size;

        // Embedding lookup
        self.backend.embed_lookup(&EmbedParams {
            token_ids,
            vocab: &self.embed,
            out: &self.scratch.x,
            n_tokens: n,
            hidden_size: h,
        })?;

        let mut layer_traces: Vec<crate::inference::LayerDeepTrace> = Vec::new();
        let norm_type = self.arch.norm_type;
        let activation = self.arch.activation;

        // Store input hidden state for residual delta computation
        let mut prev_hidden = vec![0.0f32; h];

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // ── Download input hidden state (before layer input norm) ──
            let x_data = self.backend.download_f32(&self.scratch.x);
            let hidden_len = h.min(x_data.len());

            // Compute hidden_norm: L2 norm of input
            let hidden_norm = x_data[..hidden_len].iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();

            // Save for residual_delta computation
            let copy_len = h.min(prev_hidden.len());
            prev_hidden[..copy_len].copy_from_slice(&x_data[..copy_len]);

            // ── Input Normalization ──
            Self::apply_norm(
                &self.backend, norm_type,
                &self.scratch.x, &layer.input_norm_w, None,
                &self.scratch.normed, n, h, eps,
            )?;

            // ── Q, K, V projections ──
            Self::project(&self.backend, &layer.wq, &self.scratch.normed, &self.scratch.q, n, qd, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), false)?;
            Self::project(&self.backend, &layer.wk, &self.scratch.normed, &self.scratch.k, n, kd, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), false)?;
            Self::project(&self.backend, &layer.wv, &self.scratch.normed, &self.scratch.v, n, kd, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), false)?;

            // ── Optional biases ──
            if let Some(ref bias) = layer.q_bias {
                self.backend.add_bias_inplace(&AddBiasParams {
                    tensor: &self.scratch.q,
                    bias,
                    n_tokens: n,
                    dim: qd,
                })?;
            }
            if let Some(ref bias) = layer.k_bias {
                self.backend.add_bias_inplace(&AddBiasParams {
                    tensor: &self.scratch.k,
                    bias,
                    n_tokens: n,
                    dim: kd,
                })?;
            }
            if let Some(ref bias) = layer.v_bias {
                self.backend.add_bias_inplace(&AddBiasParams {
                    tensor: &self.scratch.v,
                    bias,
                    n_tokens: n,
                    dim: kd,
                })?;
            }

            // ── QK Normalization (Qwen3) ──
            if let Some(ref qnw) = layer.q_norm_w {
                self.backend.rms_norm(&RmsNormParams {
                    input: &self.scratch.q,
                    weight: qnw,
                    output: &self.scratch.q,
                    n_tokens: n * nh,
                    hidden_size: hd,
                    eps,
                })?;
            }
            if let Some(ref knw) = layer.k_norm_w {
                self.backend.rms_norm(&RmsNormParams {
                    input: &self.scratch.k,
                    weight: knw,
                    output: &self.scratch.k,
                    n_tokens: n * nk,
                    hidden_size: hd,
                    eps,
                })?;
            }

            // ── RoPE ──
            let pos_start = layer.kv_k.current_len;
            self.backend.rope_inplace(&RopeParams {
                tensor: &self.scratch.q,
                pos_offset: pos_start,
                n_tokens: n,
                n_heads: nh,
                head_dim: hd,
                theta,
            })?;
            self.backend.rope_inplace(&RopeParams {
                tensor: &self.scratch.k,
                pos_offset: pos_start,
                n_tokens: n,
                n_heads: nk,
                head_dim: hd,
                theta,
            })?;

            // ── Append K, V to KV-cache ──
            match layer.kv_k.mode {
                kv_cache::KvCacheMode::F32 => {
                    self.backend.copy_to_kv_cache(&KvCacheParams {
                        cache: &layer.kv_k.data,
                        new_kv: &self.scratch.k,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                }
                kv_cache::KvCacheMode::Int8 => {
                    let scales = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but no scales".into()))?;
                    self.backend.kv_cache_int8_append(&compute_core::KvCacheInt8AppendParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.k,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                }
                kv_cache::KvCacheMode::Wht => {
                    let scales = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but no scales".into()))?;
                    self.backend.kv_cache_wht_append(&compute_core::KvCacheWhtAppendParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.k,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                }
            }
            layer.kv_k.current_len += n;

            match layer.kv_v.mode {
                kv_cache::KvCacheMode::F32 => {
                    self.backend.copy_to_kv_cache(&KvCacheParams {
                        cache: &layer.kv_v.data,
                        new_kv: &self.scratch.v,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                }
                kv_cache::KvCacheMode::Int8 => {
                    let scales = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but no scales".into()))?;
                    self.backend.kv_cache_int8_append(&compute_core::KvCacheInt8AppendParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.v,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                }
                kv_cache::KvCacheMode::Wht => {
                    let scales = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but no scales".into()))?;
                    self.backend.kv_cache_wht_append(&compute_core::KvCacheWhtAppendParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales,
                        new_kv: &self.scratch.v,
                        insert_pos: pos_start,
                        n_new_tokens: n,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                        max_seq: self.cfg.max_seq_len,
                    })?;
                }
            }
            layer.kv_v.current_len += n;

            let seq = layer.kv_k.current_len;

            // ── Dequantize KV cache if INT8 / WHT ──
            let (kv_k_use, kv_v_use) = match layer.kv_k.mode {
                kv_cache::KvCacheMode::F32 => {
                    (&layer.kv_k.data, &layer.kv_v.data)
                }
                kv_cache::KvCacheMode::Int8 => {
                    let kv_k_deq = self.scratch.kv_k_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but kv_k_deq not allocated".into()))?;
                    let kv_v_deq = self.scratch.kv_v_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but kv_v_deq not allocated".into()))?;

                    let scales_k = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but scales not allocated".into()))?;
                    self.backend.kv_cache_int8_read(&compute_core::KvCacheInt8ReadParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales_k,
                        output: kv_k_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    let scales_v = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("INT8 mode but scales not allocated".into()))?;
                    self.backend.kv_cache_int8_read(&compute_core::KvCacheInt8ReadParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales_v,
                        output: kv_v_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    (kv_k_deq, kv_v_deq)
                }
                kv_cache::KvCacheMode::Wht => {
                    let kv_k_deq = self.scratch.kv_k_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but kv_k_deq not allocated".into()))?;
                    let kv_v_deq = self.scratch.kv_v_deq.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but kv_v_deq not allocated".into()))?;

                    let scales_k = layer.kv_k.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but scales not allocated".into()))?;
                    self.backend.kv_cache_wht_read(&compute_core::KvCacheWhtReadParams {
                        cache_q: &layer.kv_k.data,
                        cache_scales: scales_k,
                        output: kv_k_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    let scales_v = layer.kv_v.scales.as_ref()
                        .ok_or_else(|| ModelError::Model("WHT mode but scales not allocated".into()))?;
                    self.backend.kv_cache_wht_read(&compute_core::KvCacheWhtReadParams {
                        cache_q: &layer.kv_v.data,
                        cache_scales: scales_v,
                        output: kv_v_deq,
                        seq_len: seq,
                        kv_dim: kd,
                        head_dim: hd,
                        n_kv_heads: nk,
                    })?;

                    (kv_k_deq, kv_v_deq)
                }
            };

            // ── Attention: Q × K^T → softmax → × V ──
            let use_batched = nk == nh;

            if use_batched {
                self.backend.sgemm_strided_batched(&SgemmStridedBatchedParams {
                    trans_a: false,
                    trans_b: true,
                    m: n,
                    n: seq,
                    k: hd,
                    alpha: scale,
                    beta: 0.0,
                    a: &self.scratch.q,
                    lda: hd,
                    stride_a: hd,
                    a_offset: 0,
                    b: kv_k_use,
                    ldb: hd,
                    stride_b: hd,
                    b_offset: 0,
                    c: &self.scratch.scores,
                    ldc: seq,
                    stride_c: n * seq,
                    c_offset: 0,
                    batch_count: nh,
                })?;

                for hi in 0..nh {
                    self.backend.softmax_rows(&SoftmaxParams {
                        scores: &self.scratch.scores,
                        offset_elems: hi * n * seq,
                        n_rows: n,
                        row_len: seq,
                    })?;
                }

                self.backend.sgemm_strided_batched(&SgemmStridedBatchedParams {
                    trans_a: false,
                    trans_b: false,
                    m: n,
                    n: hd,
                    k: seq,
                    alpha: 1.0,
                    beta: 0.0,
                    a: &self.scratch.scores,
                    lda: seq,
                    stride_a: n * seq,
                    a_offset: 0,
                    b: kv_v_use,
                    ldb: hd,
                    stride_b: hd,
                    b_offset: 0,
                    c: &self.scratch.attn_out,
                    ldc: hd,
                    stride_c: hd,
                    c_offset: 0,
                    batch_count: nh,
                })?;
            } else {
                // GQA grouped batched: group Q heads sharing the same KV head
                let heads_per_group = nh / nk;
                log::trace!("layer {layer_idx}: attn GQA GROUPED — {nk} groups × {heads_per_group} Q heads/group");

                for kv_h in 0..nk {
                    let q_start = kv_h * heads_per_group;

                    // Batched Q×K^T for all Q heads in this KV group
                    self.backend.sgemm_strided_batched(&SgemmStridedBatchedParams {
                        trans_a: false,
                        trans_b: true,
                        m: n,
                        n: seq,
                        k: hd,
                        alpha: scale,
                        beta: 0.0,
                        a: &self.scratch.q,
                        lda: qd,
                        stride_a: hd,
                        a_offset: q_start * hd,
                        b: kv_k_use,
                        ldb: kd,
                        stride_b: 0,       // broadcast same K head
                        b_offset: kv_h * hd,
                        c: &self.scratch.scores,
                        ldc: seq,
                        stride_c: n * seq,
                        c_offset: q_start * n * seq,
                        batch_count: heads_per_group,
                    })?;

                    for gi in 0..heads_per_group {
                        let hi = q_start + gi;
                        self.backend.softmax_rows(&SoftmaxParams {
                            scores: &self.scratch.scores,
                            offset_elems: hi * n * seq,
                            n_rows: n,
                            row_len: seq,
                        })?;
                    }

                    // Batched Attn×V for all Q heads in this KV group
                    self.backend.sgemm_strided_batched(&SgemmStridedBatchedParams {
                        trans_a: false,
                        trans_b: false,
                        m: n,
                        n: hd,
                        k: seq,
                        alpha: 1.0,
                        beta: 0.0,
                        a: &self.scratch.scores,
                        lda: seq,
                        stride_a: n * seq,
                        a_offset: q_start * n * seq,
                        b: kv_v_use,
                        ldb: kd,
                        stride_b: 0,       // broadcast same V head
                        b_offset: kv_h * hd,
                        c: &self.scratch.attn_out,
                        ldc: qd,
                        stride_c: hd,
                        c_offset: q_start * hd,
                        batch_count: heads_per_group,
                    })?;
                }
            }

            // ── Download attention scores to compute entropy and top positions ──
            let scores_data = self.backend.download_f32(&self.scratch.scores);
            let attn_entropy: f32;
            let attn_top_positions: Vec<(usize, f32)>;

            if seq > 0 && nh > 0 {
                // Average entropy across all heads
                let mut total_entropy = 0.0f32;
                for hi in 0..nh {
                    let offset = hi * n * seq;
                    if offset + seq <= scores_data.len() {
                        let head_scores = &scores_data[offset..offset + seq];
                        // Scores are already softmaxed
                        total_entropy += tracer::entropy(head_scores);
                    }
                }
                attn_entropy = total_entropy / nh.max(1) as f32;

                // Top positions from first head
                if seq > 0 {
                    let first_head_scores = &scores_data[..seq.min(scores_data.len())];
                    attn_top_positions = tracer::top_k(first_head_scores, 5);
                } else {
                    attn_top_positions = vec![];
                }
            } else {
                attn_entropy = 0.0;
                attn_top_positions = vec![];
            }

            // ── Output projection ──
            Self::project(&self.backend, &layer.wo, &self.scratch.attn_out, &self.scratch.x, n, h, qd, 0, 1.0, self.scratch.q8_scratch.as_ref(), false)?;

            // ── Post-attention Normalization ──
            Self::apply_norm(
                &self.backend, norm_type,
                &self.scratch.x, &layer.post_attn_norm_w, None,
                &self.scratch.normed, n, h, eps,
            )?;

            // ── MLP: gate_proj + up_proj ──
            Self::project(&self.backend, &layer.wgate, &self.scratch.normed, &self.scratch.gate, n, it, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), false)?;
            Self::project(&self.backend, &layer.wup, &self.scratch.normed, &self.scratch.up, n, it, h, 0, 0.0, self.scratch.q8_scratch.as_ref(), false)?;

            // ── Activation: gate = activation(gate) * up ──
            Self::apply_activation(
                &self.backend, activation,
                &self.scratch.gate, &self.scratch.up, n, it,
            )?;

            // ── Download gate to compute MLP activation rate and top neurons ──
            let gate_data = self.backend.download_f32(&self.scratch.gate);
            let mlp_activation_rate: f32;
            let mlp_top_neurons: Vec<(usize, f32)>;

            if it > 0 {
                let non_zero = gate_data.iter()
                    .filter(|v| v.abs() > 1e-10)
                    .count();
                mlp_activation_rate = (non_zero as f32 / it as f32) * 100.0;

                // Top-10 by absolute value
                mlp_top_neurons = tracer::top_k(
                    &gate_data.iter().map(|v| v.abs()).collect::<Vec<_>>(),
                    10
                );
            } else {
                mlp_activation_rate = 0.0;
                mlp_top_neurons = vec![];
            }

            // ── Down projection: residual add ──
            Self::project(&self.backend, &layer.wdown, &self.scratch.gate, &self.scratch.x, n, h, it, 0, 1.0, self.scratch.q8_scratch.as_ref(), false)?;

            // ── Download output hidden state to compute residual_delta and phase_coherence ──
            let x_data = self.backend.download_f32(&self.scratch.x);
            let output_hidden = if h <= x_data.len() {
                &x_data[..h]
            } else {
                &x_data
            };

            // Compute residual_delta: L2 norm of (output - input)
            let residual_delta = prev_hidden.iter().zip(output_hidden)
                .map(|(a, b)| (b - a) * (b - a))
                .sum::<f32>()
                .sqrt();

            // Compute phase_coherence
            let phase_coherence = tracer::phase_coherence(output_hidden);

            layer_traces.push(crate::inference::LayerDeepTrace {
                layer_idx,
                hidden_norm,
                attn_entropy,
                attn_top_positions,
                mlp_activation_rate,
                mlp_top_neurons,
                residual_delta,
                phase_coherence,
                time_us: 0,
            });
        }

        // ── Final Normalization ──
        Self::apply_norm(
            &self.backend, norm_type,
            &self.scratch.x, &self.norm_w, self.norm_b.as_ref(),
            &self.scratch.normed, n, h, eps,
        )?;

        // ── lm_head ──
        Self::project(
            &self.backend, &self.lm_head,
            &self.scratch.normed, &self.scratch.logits,
            1, vocab, h, (n - 1) * h, 0.0,
            self.scratch.q8_scratch.as_ref(), false,
        )?;

        // ── Download logits ──
        let logits = self.backend.download_f32(&self.scratch.logits);

        // ── Compute output trace ──
        let probs = tracer::softmax(&logits);
        let logit_entropy = tracer::entropy(&probs);

        let top_tokens_raw = tracer::top_k(&probs, 10);
        let top_tokens: Vec<(u32, f32)> = top_tokens_raw.into_iter()
            .map(|(idx, prob)| (idx as u32, prob))
            .collect();

        // Find chosen token (highest probability)
        let (chosen_idx, confidence) = if !probs.is_empty() {
            let (idx, &prob) = probs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            (idx as u32, prob)
        } else {
            (0, 0.0)
        };

        // Find rank of chosen token in top-10
        let chosen_token_rank = top_tokens.iter()
            .position(|(id, _)| *id == chosen_idx)
            .map(|i| i as i32)
            .unwrap_or(-1);

        let token_str = crate::tokenizer::token_id_to_text(chosen_idx);

        Ok(crate::inference::DeepTrace {
            token_id: chosen_idx,
            token_str,
            layers: layer_traces,
            output: crate::inference::OutputDeepTrace {
                logit_entropy,
                top_tokens,
                confidence,
                chosen_token_rank,
            },
            time_ms: token_start.elapsed().as_millis(),
        })
    }
}

// ── Leitores safetensors ──────────────────────────────────────────────────────

// A antiga read_bf16_raw foi substituida por zero-copy inline nas closures

fn read_bf16_f32(mmap: &memmap2::Mmap, name: &str) -> ModelResult<(Vec<f32>, Vec<usize>)> {
    let (raw, shape) = shard_tensor_bytes(mmap, name)?;
    let floats = raw.chunks_exact(2).map(|b| {
        bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
    }).collect();
    Ok((floats, shape))
}

fn shard_has_tensor(mmap: &memmap2::Mmap, name: &str) -> bool {
    if mmap.len() < 8 { return false; }
    let hlen = match mmap[0..8].try_into().ok().map(u64::from_le_bytes) {
        Some(n) => n as usize,
        None => return false,
    };
    if mmap.len() < 8 + hlen { return false; }
    let header: HashMap<String, serde_json::Value> = match serde_json::from_slice(&mmap[8..8 + hlen]) {
        Ok(h) => h,
        Err(_) => return false,
    };
    header.contains_key(name)
}

fn shard_tensor_bytes<'a>(mmap: &'a memmap2::Mmap, name: &str) -> ModelResult<(&'a [u8], Vec<usize>)> {
    if mmap.len() < 8 {
        return Err(ModelError::Model("shard inválido".into()));
    }
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&mmap[8..8 + hlen])?;
    let meta = header.get(name)
        .ok_or_else(|| ModelError::Model(format!("tensor '{}' não encontrado", name)))?;
    let shape: Vec<usize> = meta["shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let offs = meta["data_offsets"].as_array().unwrap();
    let s = offs[0].as_u64().unwrap() as usize;
    let e = offs[1].as_u64().unwrap() as usize;
    let base = 8 + hlen;
    Ok((&mmap[base + s..base + e], shape))
}

// ── CPU fallbacks (mantidos para referência e testes) ─────────────────────────
// Usadas apenas em testes unitários — não chamadas durante inferência.

#[allow(dead_code)]
pub(crate) fn apply_rope(
    x: &mut [f32], n_tokens: usize, n_heads: usize, head_dim: usize,
    pos_start: usize, theta: f32,
) {
    let token_stride = n_heads * head_dim;
    let half_dim = head_dim / 2;
    for tok in 0..n_tokens {
        let pos = (pos_start + tok) as f32;
        for h in 0..n_heads {
            let base = tok * token_stride + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0_f32 / theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = pos * freq;
                let (s, c) = angle.sin_cos();
                let x0 = x[base + i];
                let x1 = x[base + half_dim + i];
                x[base + i]            = x0 * c - x1 * s;
                x[base + half_dim + i] = x0 * s + x1 * c;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::iara_format::IaraV2Reader;

    #[test]
    fn test_iara_v2_reader_opens_file() {
        let iara_path = std::path::Path::new("/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b-fp8.iara");

        // Skip test if file doesn't exist (CI environment)
        if !iara_path.exists() {
            println!("Skipping IARA test: file not found");
            return;
        }

        let reader = IaraV2Reader::open(iara_path)
            .expect("Failed to open IARA v2 file");

        // Verify metadata
        assert_eq!(reader.header.metadata.architecture, "qwen2");
        assert_eq!(reader.header.metadata.n_layers, 28);
        assert_eq!(reader.header.metadata.hidden_dim, 1536);
        assert!(reader.header.metadata.vocab_size > 0);

        // Verify tensors exist
        assert!(reader.header.tensors.len() > 0);

        println!("✓ IARA v2 Reader test passed");
        println!("  Architecture: {}", reader.header.metadata.architecture);
        println!("  Layers: {}", reader.header.metadata.n_layers);
        println!("  Hidden dim: {}", reader.header.metadata.hidden_dim);
        println!("  Tensors: {}", reader.header.tensors.len());
    }

    #[test]
    fn test_iara_v2_tensor_meta_access() {
        let iara_path = std::path::Path::new("/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b-fp8.iara");

        if !iara_path.exists() {
            println!("Skipping IARA tensor meta test: file not found");
            return;
        }

        let reader = IaraV2Reader::open(iara_path)
            .expect("Failed to open IARA v2 file");

        // Test accessing a tensor's metadata
        let tensor_names = reader.tensor_names();
        assert!(tensor_names.len() > 0);

        // Try to access first tensor's metadata
        if let Some(first_tensor) = tensor_names.first() {
            let meta = reader.tensor_meta(first_tensor)
                .expect("Failed to get tensor metadata");
            assert!(meta.shape.len() > 0);
            assert!(meta.data_length > 0);
            println!("✓ Tensor metadata access test passed");
            println!("  First tensor: {}", first_tensor);
            println!("  Shape: {:?}", meta.shape);
            println!("  Dtype: {:?}", meta.dtype);
        }
    }

    // ── Phase Coherence Tests ────────────────────────────────────────────────

    #[test]
    fn test_phase_coherence_all_aligned() {
        // All phases aligned (all complex numbers pointing same direction)
        // Should return ~1.0
        let hidden = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coherence = phase_coherence(&hidden);
        assert!(coherence > 0.99, "Expected coherence ~1.0, got {}", coherence);
    }

    #[test]
    fn test_phase_coherence_random() {
        // Random phases (completely chaotic)
        // Should return ~0.0
        let hidden = vec![
            1.0, -1.0, // atan2(-1, 1) ≈ -π/4
            -1.0, 1.0, // atan2(1, -1) ≈ 3π/4
            1.0, 1.0,  // atan2(1, 1) ≈ π/4
            -1.0, -1.0, // atan2(-1, -1) ≈ -3π/4
        ];
        let coherence = phase_coherence(&hidden);
        assert!(coherence < 0.1, "Expected coherence ~0.0 for random phases, got {}", coherence);
    }

    #[test]
    fn test_phase_coherence_partially_aligned() {
        // Half aligned, half not
        // Should return ~0.5
        let hidden = vec![
            1.0, 0.0, // phase 0
            1.0, 0.0, // phase 0
            1.0, 0.0, // phase 0
            0.0, 1.0, // phase π/2
            0.0, 1.0, // phase π/2
            0.0, 1.0, // phase π/2
        ];
        let coherence = phase_coherence(&hidden);
        // Two groups at 90° gives coherence = sqrt(0.5) ≈ 0.707
        assert!(coherence > 0.5 && coherence < 0.85, "Expected coherence ~0.707, got {}", coherence);
    }

    #[test]
    fn test_phase_coherence_empty() {
        let hidden = vec![];
        let coherence = phase_coherence(&hidden);
        assert_eq!(coherence, 0.0);
    }

    #[test]
    fn test_phase_coherence_single_pair() {
        let hidden = vec![1.0, 0.0];
        let coherence = phase_coherence(&hidden);
        assert!(coherence > 0.99);
    }

    #[test]
    fn test_layer_budget_enum() {
        // Test enum values exist and can be compared
        assert_eq!(LayerBudget::Fast, LayerBudget::Fast);
        assert_eq!(LayerBudget::Medium, LayerBudget::Medium);
        assert_eq!(LayerBudget::Slow, LayerBudget::Slow);
        assert_ne!(LayerBudget::Fast, LayerBudget::Medium);
        assert_ne!(LayerBudget::Medium, LayerBudget::Slow);
    }
}
