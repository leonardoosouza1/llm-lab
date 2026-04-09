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
    AddBiasParams, GemvQ4Params, GemvQ4KParams, GemvQ6KParams, FusedAttentionDecodeParams,
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

/// Computes phase coherence (Kuramoto order parameter) from hidden state.
/// Interprets hidden state as complex pairs: h[0]+ih[1], h[2]+ih[3], ...
/// Returns Kuramoto order parameter r = |mean(exp(i*theta))|,
/// where theta = atan2(h[2k+1], h[2k]).
///
/// Returns value in [0.0, 1.0]:
/// - 0.0 = completely chaotic, random phases
/// - 1.0 = total synchronization, all phases aligned
pub fn phase_coherence(hidden: &[f32]) -> f32 {
    if hidden.len() < 2 {
        return 0.0;
    }

    let n = hidden.len() / 2;
    let (mut sum_cos, mut sum_sin) = (0.0f32, 0.0f32);

    for k in 0..n {
        // Complex number: real = hidden[2k], imag = hidden[2k+1]
        let x = hidden[2 * k];
        let y = hidden[2 * k + 1];

        // Normalize to unit circle
        let norm = (x * x + y * y).sqrt().max(1e-8);
        sum_cos += x / norm;
        sum_sin += y / norm;
    }

    // Kuramoto order parameter: r = |mean(exp(i*theta))|
    let mean_cos = sum_cos / n as f32;
    let mean_sin = sum_sin / n as f32;
    (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
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

impl ModelCfg {
    /// Lê parâmetros do config.json. Suporta qualquer modelo causal LM com campos padrão.
    pub fn from_config_json(config_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(config_path)?;
        let v: serde_json::Value = serde_json::from_str(&content)?;

        let arch = v["architectures"].as_array()
            .and_then(|a| a.first())
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        // Verificação básica: deve ser um modelo causal LM (ou ter hidden_size)
        if !arch.contains("CausalLM") && !arch.contains("ForCausalLM")
            && !v["hidden_size"].is_number() && !v["text_config"]["hidden_size"].is_number()
        {
            return Err(format!(
                "Arquitetura '{}' não parece ser um CausalLM. Verifique config.json.",
                arch
            ).into());
        }

        let params = if v["hidden_size"].is_number() { &v }
                     else if v["text_config"]["hidden_size"].is_number() { &v["text_config"] }
                     else { return Err(format!("hidden_size ausente em {}", config_path.display()).into()); };

        let uint = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
            let val = &params[key];
            val.as_u64().or_else(|| val.as_f64().map(|f| f as u64))
               .map(|n| n as usize)
               .ok_or_else(|| format!("'{}' ausente em config.json", key).into())
        };

        let hidden_size = uint("hidden_size")?;
        let num_heads   = uint("num_attention_heads")?;
        let head_dim = params["head_dim"].as_u64()
            .or_else(|| params["head_dim"].as_f64().map(|f| f as u64))
            .map(|n| n as usize)
            .unwrap_or(hidden_size / num_heads);

        Ok(Self {
            hidden_size, num_heads, head_dim,
            num_layers:        uint("num_hidden_layers")?,
            intermediate_size: uint("intermediate_size")?,
            vocab_size:        uint("vocab_size")?,
            num_kv_heads:      uint("num_key_value_heads")?,
            rms_eps:           params["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta:        params["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            // Cap max_seq_len to prevent VRAM bloat.
            // max_position_embeddings for Qwen2.5/3 is often 128K, which would need:
            //   KV cache: layers × 2 × 128K × kv_dim × 4 bytes → 7+ GB for 1.5b model
            //   scores:   n_heads × 128K × 128K × 4 bytes → 824 GB (impossible)
            // A 1024-token context is practical for 12GB GPUs to fit 4B scale models directly.
            max_seq_len: {
                let raw = params["max_position_embeddings"].as_u64()
                    .map(|n| n as usize)
                    .unwrap_or(2048);
                raw.min(1024)
            },
        })
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
            kv_cache::KvCacheMode::Int8 => {
                // Aloca buffers para dequantização INT8 → F32 durante atenção
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
        let q8_scratch = if std::env::var("IARA_GGUF_NATIVE").is_ok() {
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
}

impl CausalLM {
    /// Helper: faz projeção com pesos BF16 ou Q4 quantizados.
    /// Despacha para backend.gemm_bf16_f32 ou backend.gemv_q4 conforme necessário.
    /// Se q8_scratch é Some, usa dp4a path (quantize x → Q8_1, then integer GEMV).
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
        match weight {
            LayerWeight::Bf16(w) => {
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
                if let (Some(q8), true) = (q8_scratch, m == 1) {
                    if !q8_precomputed {
                        backend.quantize_q8(&compute_core::QuantizeQ8Params {
                            x: input, x_offset: a_offset, out: q8, k,
                        })?;
                    }
                    backend.gemv_q4k_q8(&compute_core::GemvQ4KQ8Params {
                        m, n, k, data, x_q8: q8, out: output, beta,
                    })?;
                } else {
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

        // ── Nomes de tensores (via architecture descriptor) ─────────────────
        let tmap = &arch.tensor_map;

        // ── Embedding ────────────────────────────────────────────────────────
        log::info!("  → embed_tokens");
        let embed = gpu(&tmap.embed)?;

        // ── Camadas ──────────────────────────────────────────────────────────
        let kv_dim  = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;

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
                wq:    LayerWeight::Bf16(gpu(&tmap.layer_tensor(i, &tmap.attn_q))?),
                wk:    LayerWeight::Bf16(gpu(&tmap.layer_tensor(i, &tmap.attn_k))?),
                wv:    LayerWeight::Bf16(gpu(&tmap.layer_tensor(i, &tmap.attn_v))?),
                wo:    LayerWeight::Bf16(gpu(&tmap.layer_tensor(i, &tmap.attn_o))?),
                wgate: LayerWeight::Bf16(gpu(&tmap.layer_tensor(i, &tmap.mlp_gate))?),
                wup:   LayerWeight::Bf16(gpu(&tmap.layer_tensor(i, &tmap.mlp_up))?),
                wdown: LayerWeight::Bf16(gpu(&tmap.layer_tensor(i, &tmap.mlp_down))?),
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
        self.backend.embed_lookup(&EmbedParams {
            token_ids,
            vocab: &self.embed,
            out: &self.scratch.x,
            n_tokens: n,
            hidden_size: h,
        })?;

        // Debug: check embed lookup output
        if std::env::var("IARA_DEBUG_GGUF").map(|v| v == "1").unwrap_or(false) {
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

        // ── Per-operation GPU profiling (IARA_PROFILE=1) ──
        let do_profile = std::env::var("IARA_PROFILE").map(|v| v == "1").unwrap_or(false) && n == 1;
        static PROFILE_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_profile = do_profile && !PROFILE_DONE.swap(true, std::sync::atomic::Ordering::SeqCst);
        let forward_t0 = if do_profile { Some(std::time::Instant::now()) } else { None };

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Check if we should early exit based on budget
            if layer_idx > final_layer {
                log::debug!(
                    "Early exit at layer {} (budget={:?}): skipping remaining layers",
                    layer_idx, layer_budget
                );
                break;
            }
            let _prof_l0 = false;  // disabled: per-op profiling stalls pipeline
            let _layer_t0: Option<std::time::Instant> = None;  // disabled: per-layer sync inflates 2×

            macro_rules! prof_sync {
                ($label:expr) => {
                    if _prof_l0 {
                        self.backend.device_sync();
                        log::warn!("PROF-L0 {} @ {:?}", $label, _layer_t0.unwrap().elapsed());
                    }
                };
            }

            // ── Input Normalization (RMSNorm ou LayerNorm conforme arquitetura) ──
            // Try fused RMSNorm+Quantize (1 launch vs 2) for decode
            let qkv_q8_pre = if n == 1 && matches!(norm_type, NormType::RMSNorm) {
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

            // Try fused QKV kernel (1 launch vs 3) for decode with Q4K weights
            static FUSED_QKV_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let fused_qkv_ok = if n == 1 && qkv_q8_pre {
                match (&layer.wq, &layer.wk, &layer.wv) {
                    (LayerWeight::Q4K { data: dq, .. }, LayerWeight::Q4K { data: dk, .. }, LayerWeight::Q4K { data: dv, .. }) => {
                        self.backend.fused_qkv(&compute_core::FusedQkvParams {
                            wq_data: dq, wk_data: dk, wv_data: dv,
                            x_q8: self.scratch.q8_scratch.as_ref().unwrap(),
                            out_q: &self.scratch.q, out_k: &self.scratch.k, out_v: &self.scratch.v,
                            nq: qd, nk: kd, nv: kd, k: h,
                        }).is_ok()
                    }
                    _ => false,
                }
            } else { false };

            if fused_qkv_ok {
                if !FUSED_QKV_LOG.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::warn!("FUSED QKV: active (nq={}, nk={}, nv={}, k={})", qd, kd, kd, h);
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
            // Biases opcionais (Qwen2/2.5)
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
            let fused_rope_ok = if n == 1 && matches!(layer.kv_k.mode, kv_cache::KvCacheMode::F32) {
                self.backend.fused_rope_kv_copy(&compute_core::FusedRopeKvCopyParams {
                    q: &self.scratch.q, k: &self.scratch.k, v: &self.scratch.v,
                    kv_k_cache: &layer.kv_k.data, kv_v_cache: &layer.kv_v.data,
                    n_heads: nh, n_kv_heads: nk, head_dim: hd,
                    pos_start, theta, kv_dim: kd, max_seq: self.cfg.max_seq_len,
                }).is_ok()
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
            }
            } // end else !fused_rope_ok (KV cache append)

            layer.kv_k.current_len += n;
            layer.kv_v.current_len += n;

            let seq = layer.kv_k.current_len;

            // ── Dequantize KV cache if INT8 mode ───────────────────────────────
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
            let use_fused_decode = n == 1 && seq > 0;
            let mut fused_ok = false;

            if use_fused_decode {
                // ── FUSED DECODE ATTENTION (n=1) ─────────────────────────────
                log::trace!("layer {layer_idx}: attn FUSED DECODE seq={seq} nh={nh} nk={nk} hd={hd}");
                match self.backend.fused_attention_decode(&FusedAttentionDecodeParams {
                    q: &self.scratch.q,
                    k_cache: kv_k_use,
                    v_cache: kv_v_use,
                    out: &self.scratch.attn_out,
                    seq_len: seq,
                    head_dim: hd,
                    n_heads: nh,
                    n_kv_heads: nk,
                    scale,
                }) {
                    Ok(()) => {
                        fused_ok = true;
                    }
                    Err(_) => {
                        // Fused not available on this backend — use GQA path
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
                        b: &layer.kv_v.data,
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
            // ── Output projection: attn_out × wo^T → x (residual) ─────────────
            // x = x + attn_out × wo^T  (beta=1.0 para acumular em x)
            log::trace!("layer {layer_idx}: wo");
            Self::project(&self.backend, &layer.wo, &self.scratch.attn_out, &self.scratch.x, n, h, qd, 0, 1.0, self.scratch.q8_scratch.as_ref(), false)?;

            prof_sync!("wo_done");
            // ── Post-attention Normalization (+ optional fused quantize for MLP) ──
            let mlp_q8_pre = if n == 1 && matches!(norm_type, NormType::RMSNorm) {
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

                // Pre-quantize normed for gate+up (reuse from fused norm+q8 if available)
                let gu_q8_pre = if mlp_q8_pre {
                    true  // already quantized by fused_rms_norm_quantize
                } else if n == 1 {
                    if let Some(q8) = self.scratch.q8_scratch.as_ref() {
                        Self::quantize_input_q8(&self.backend, &self.scratch.normed, 0, q8, h)?;
                        true
                    } else { false }
                } else { false };

                // Try fused gate+up+SiLU kernel (1 launch vs 3) for decode
                let fused_mlp_ok = if n == 1 && gu_q8_pre && !is_calibrating {
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
                if !fused_mlp_ok {
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
            let should_sample = self.telemetry_enabled && n <= 1 && (layer_idx % 8 == 0 || layer_idx == n_layers - 1);

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

                LayerTrace {
                    layer_idx,
                    attention_energy: residual_norm * 0.6,
                    mlp_energy: residual_norm * 0.4,
                    residual_norm,
                    max_activation,
                    hot_neurons: hot,
                    time_us: 0,
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
                }
            };

            layer_traces.push(trace.clone());

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

        // ── Final Normalization ──────────────────────────────────────────────
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

        // ── Download logits OR GPU argmax ────────────────────────────────────
        if let Some(t0) = forward_t0 {
            self.backend.device_sync();
            log::warn!("PROF lm_head_done @ {:?}", t0.elapsed());
        }
        let logits = self.backend.download_f32(&self.scratch.logits);
        if let Some(t0) = forward_t0 {
            log::warn!("PROF logits_download_done @ {:?}", t0.elapsed());
        }

        Ok((logits, layer_traces))
    }

    /// Fast greedy forward: runs entire forward pass and returns token_id via GPU argmax.
    /// Skips downloading 608KB of logits — only downloads 4 bytes (the argmax index).
    pub fn forward_greedy(&mut self, token_ids: &[u32]) -> ModelResult<u32> {
        let vocab = self.cfg.vocab_size;

        // Run full forward which computes logits on device AND downloads them
        let (logits, _) = self.forward(token_ids, None)?;

        // GPU argmax on device tensor (logits still in scratch.logits)
        match self.backend.gpu_argmax(&self.scratch.logits, vocab) {
            Ok(token_id) => {
                use std::sync::atomic::{AtomicBool, Ordering};
                static LOGGED: AtomicBool = AtomicBool::new(false);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    log::warn!("GPU ARGMAX: active (vocab={})", vocab);
                }
                Ok(token_id as u32)
            }
            Err(_) => {
                // Fallback: CPU argmax on downloaded logits
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
            }
            layer.kv_v.current_len += n;

            let seq = layer.kv_k.current_len;

            // ── Dequantize KV cache if INT8 ──
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
