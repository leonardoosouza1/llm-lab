// ============================================================================
// model/mod.rs — Pipeline Nativo Qwen: GPU via ComputeBackend
// ============================================================================
// Todas as operações no device (GPU ou CPU-backend):
//   - Pesos: AnyTensor BF16 (metade da memória)
//   - Ativações: AnyTensor F32 (scratchpad pré-alocado, zero-alloc no forward)
//   - KV-cache: GpuKvCache (AnyTensor F32 pré-alocado, append via backend)
//   - Projeções: gemm_bf16_f32 (BF16 pesos × F32 ativações → F32 saída)
//   - Norm/RoPE/SwiGLU/Softmax: kernels via ComputeBackend
//   - Único download_f32(): logits finais [vocab_size] por token gerado
// ============================================================================

mod kv_cache;
pub(crate) use kv_cache::GpuKvCache;
#[allow(unused_imports)]
pub(crate) use kv_cache::BoundedKvCache;

use std::sync::Arc;
use compute_core::{
    ComputeBackend, AnyTensor,
    GemmBf16Params, SgemmParams, EmbedParams, RmsNormParams,
    RopeParams, SoftmaxParams, SiluMulParams, KvCacheParams, AddBiasParams, GemvQ4Params,
};
use half::bf16;
use std::collections::HashMap;
use std::path::Path;
use crate::storage::quantized::{QuantizedTensor, quantize_bf16_to_q4, upload_quantized, QuantizationCpuResult, save_quantized_cache, load_quantized_cache};

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
    /// Lê parâmetros do config.json. Suporta Qwen2, Qwen2.5, Qwen3, Llama, Mistral.
    pub fn from_config_json(config_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(config_path)?;
        let v: serde_json::Value = serde_json::from_str(&content)?;

        let arch = v["architectures"].as_array()
            .and_then(|a| a.first())
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        let supported = ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM",
                         "MistralForCausalLM", "Qwen2_5ForCausalLM"];
        if !supported.iter().any(|s| arch.contains(s.trim_end_matches("ForCausalLM"))
                                  || *s == arch) {
            return Err(format!(
                "Arquitetura '{}' não suportada. Apenas causal LM (Qwen3, Llama, Mistral).",
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
}

// ── Camada Transformer ────────────────────────────────────────────────────────

/// Pesos e estado de uma camada Qwen — tensores no device via AnyTensor ou QuantizedTensor.
pub struct QwenLayer {
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
    // Norm weights no device (pequenos, mas evita round-trip host→device por token)
    pub input_norm_w:    AnyTensor, // F32
    pub post_attn_norm_w: AnyTensor, // F32
    // KV-cache no device (janela deslizante via copy_to_kv_cache)
    pub kv_k: GpuKvCache,
    pub kv_v: GpuKvCache,
}

// ── Modelo completo ───────────────────────────────────────────────────────────

pub struct QwenModel {
    pub embed:   AnyTensor,     // BF16 [vocab, hidden] — sempre (sensível a quantização)
    pub layers:  Vec<QwenLayer>,
    pub norm_w:  AnyTensor,     // F32 [hidden]
    pub lm_head: AnyTensor,     // BF16 [vocab, hidden] — sempre (sensível a quantização)
    pub cfg:     ModelCfg,
    /// Bytes estimados no device: pesos BF16/Q4 + scratchpad + KV-cache.
    pub vram_bytes: usize,
    pub backend: Arc<dyn ComputeBackend>,
    pub scratch: ForwardScratchpad,
    // ────────────────────────────────────────────────────────────────────────
    // Novos campos para CfC e Gravity (quantização + correção adaptativa)
    // ────────────────────────────────────────────────────────────────────────
    /// CfC Liquid Correction Layer (opcional).
    /// Se None, a correção está desabilitada (sem custo extra).
    pub cfc: Option<crate::lnn::cfc::CfcLayer>,
    /// Sistema de Gravidade Semântica (opcional).
    /// Usado para detectar tokens críticos que requerem CfC correction.
    pub gravity: Option<crate::lnn::gravity::GravitySystem>,
    /// Flag: se true, pesos estão em Q4 (quantizados INT4).
    /// Se false, pesos são BF16 (padrão).
    pub quantized: bool,
}

impl QwenModel {
    /// Helper: faz projeção com pesos BF16 ou Q4 quantizados.
    /// Despacha para backend.gemm_bf16_f32 ou backend.gemv_q4 conforme necessário.
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
                    out: output,
                    beta,
                })?;
            }
        }
        Ok(())
    }

    /// Carrega pesos do diretório. Suporta single-shard e multi-shard.
    pub fn load(model_dir: &Path, cfg: ModelCfg, backend: Arc<dyn ComputeBackend>) -> ModelResult<Self> {
        log::info!(
            "⏳ Carregando Qwen: {} camadas, hidden={}, vocab={}",
            cfg.num_layers, cfg.hidden_size, cfg.vocab_size
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

        // ── Embedding ────────────────────────────────────────────────────────
        log::info!("  → embed_tokens");
        let embed = gpu("model.embed_tokens.weight")?;

        // ── Camadas ──────────────────────────────────────────────────────────
        let kv_dim  = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 { log::info!("  → layers {}/{}", i, cfg.num_layers); }
            let p = format!("model.layers.{}", i);

            let (in_w, _)  = load_f32(&format!("{}.input_layernorm.weight", p))?;
            let (pa_w, _)  = load_f32(&format!("{}.post_attention_layernorm.weight", p))?;

            let qbn = format!("{}.self_attn.q_proj.bias", p);
            let kbn = format!("{}.self_attn.k_proj.bias", p);
            let vbn = format!("{}.self_attn.v_proj.bias", p);

            layers.push(QwenLayer {
                wq:    LayerWeight::Bf16(gpu(&format!("{}.self_attn.q_proj.weight", p))?),
                wk:    LayerWeight::Bf16(gpu(&format!("{}.self_attn.k_proj.weight", p))?),
                wv:    LayerWeight::Bf16(gpu(&format!("{}.self_attn.v_proj.weight", p))?),
                wo:    LayerWeight::Bf16(gpu(&format!("{}.self_attn.o_proj.weight", p))?),
                wgate: LayerWeight::Bf16(gpu(&format!("{}.mlp.gate_proj.weight", p))?),
                wup:   LayerWeight::Bf16(gpu(&format!("{}.mlp.up_proj.weight", p))?),
                wdown: LayerWeight::Bf16(gpu(&format!("{}.mlp.down_proj.weight", p))?),
                q_bias: if tensor_exists(&qbn) {
                    Some(gpu_f32(&load_f32(&qbn)?.0))
                } else { None },
                k_bias: if tensor_exists(&kbn) {
                    Some(gpu_f32(&load_f32(&kbn)?.0))
                } else { None },
                v_bias: if tensor_exists(&vbn) {
                    Some(gpu_f32(&load_f32(&vbn)?.0))
                } else { None },
                input_norm_w:    gpu_f32(&in_w),
                post_attn_norm_w: gpu_f32(&pa_w),
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim),
            });
        }

        let (nw, _) = load_f32("model.norm.weight")?;
        let norm_w = gpu_f32(&nw);

        let lm_head = if tensor_exists("lm_head.weight") {
            gpu("lm_head.weight")?
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            gpu("model.embed_tokens.weight")?
        };

        // ── Contabilidade de VRAM/memória ────────────────────────────────────
        // Estimativa: BF16 = 2 bytes/elem, F32 = 4 bytes/elem
        let bf16_bytes = |t: &AnyTensor| t.numel() * 2;
        let f32_bytes  = |t: &AnyTensor| t.numel() * 4;
        let layer_weight_bytes = |w: &LayerWeight| match w {
            LayerWeight::Bf16(t) => bf16_bytes(t),
            LayerWeight::Q4(qt) => {
                let [rows, cols] = qt.shape;
                // packed: rows × cols/2 bytes
                // scales: rows × (cols/group_size) × 4 bytes
                // zeros: rows × (cols/group_size) × 4 bytes
                let packed_bytes = rows * cols / 2;
                let n_groups = cols / qt.group_size;
                let scales_bytes = rows * n_groups * 4;
                let zeros_bytes = rows * n_groups * 4;
                packed_bytes + scales_bytes + zeros_bytes
            }
        };

        let mut vram_bytes = bf16_bytes(&embed) + bf16_bytes(&lm_head);
        for l in &layers {
            vram_bytes += layer_weight_bytes(&l.wq) + layer_weight_bytes(&l.wk) + layer_weight_bytes(&l.wv)
                        + layer_weight_bytes(&l.wo) + layer_weight_bytes(&l.wgate)
                        + layer_weight_bytes(&l.wup) + layer_weight_bytes(&l.wdown);
            vram_bytes += f32_bytes(&l.input_norm_w) + f32_bytes(&l.post_attn_norm_w);
            vram_bytes += l.kv_k.data.numel() * 4 + l.kv_v.data.numel() * 4;
        }

        log::info!("✅ Qwen carregado — memória estimada: {:.2} GB", vram_bytes as f64 / 1e9);

        let scratch = ForwardScratchpad::new(&cfg, &backend);

        Ok(Self {
            embed,
            layers,
            norm_w,
            lm_head,
            cfg,
            vram_bytes,
            backend,
            scratch,
            cfc: None,
            gravity: None,
            quantized: false,
        })
    }

    /// Carrega modelo com quantização INT4 on-the-fly.
    /// Pesos são quantizados de BF16→Q4 durante o carregamento.
    /// Reduz VRAM em ~4x (ex: 16GB → 4GB para Qwen3-8B).
    pub fn load_quantized(
        model_dir: &Path,
        cfg: ModelCfg,
        backend: Arc<dyn ComputeBackend>,
        enable_cfc: bool,
    ) -> ModelResult<Self> {
        use crate::lnn::cfc::{CfcConfig, CfcLayer};
        use crate::lnn::gravity::{GravityConfig, GravitySystem};

        log::info!(
            "⏳ Carregando Qwen com quantização Q4: {} camadas, hidden={}, vocab={}",
            cfg.num_layers, cfg.hidden_size, cfg.vocab_size
        );

        // ── Tentar carregar cache pré-quantizado ──────────────────────────────
        let group_size = 128usize;
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

        // ── Mmaps (idêntico ao load()) ────────────────────────────────────────
        let mut shards: HashMap<String, memmap2::Mmap> = HashMap::new();
        let tensor_shard_map: Option<HashMap<String, String>>;

        // Só carrega mmaps se não estamos usando cache
        let should_load_mmaps = use_cache.is_none();

        if should_load_mmaps {
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
        } else {
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

        // ── Embedding (BF16) ─────────────────────────────────────────────────
        log::info!("  → embed_tokens (BF16)");
        let embed = gpu_bf16("model.embed_tokens.weight")?;

        // ── Camadas (Q4 quantizadas) ─────────────────────────────────────────
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let max_seq = cfg.max_seq_len;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 { log::info!("  → layers {}/{} (Q4)", i, cfg.num_layers); }
            let p = format!("model.layers.{}", i);

            let (in_w, _) = load_f32(&format!("{}.input_layernorm.weight", p))?;
            let (pa_w, _) = load_f32(&format!("{}.post_attention_layernorm.weight", p))?;

            let qbn = format!("{}.self_attn.q_proj.bias", p);
            let kbn = format!("{}.self_attn.k_proj.bias", p);
            let vbn = format!("{}.self_attn.v_proj.bias", p);

            layers.push(QwenLayer {
                wq:    quantize_weight(&format!("{}.self_attn.q_proj.weight", p))?,
                wk:    quantize_weight(&format!("{}.self_attn.k_proj.weight", p))?,
                wv:    quantize_weight(&format!("{}.self_attn.v_proj.weight", p))?,
                wo:    quantize_weight(&format!("{}.self_attn.o_proj.weight", p))?,
                wgate: quantize_weight(&format!("{}.mlp.gate_proj.weight", p))?,
                wup:   quantize_weight(&format!("{}.mlp.up_proj.weight", p))?,
                wdown: quantize_weight(&format!("{}.mlp.down_proj.weight", p))?,
                q_bias: if tensor_exists(&qbn) {
                    Some(gpu_f32(&load_f32(&qbn)?.0))
                } else { None },
                k_bias: if tensor_exists(&kbn) {
                    Some(gpu_f32(&load_f32(&kbn)?.0))
                } else { None },
                v_bias: if tensor_exists(&vbn) {
                    Some(gpu_f32(&load_f32(&vbn)?.0))
                } else { None },
                input_norm_w:     gpu_f32(&in_w),
                post_attn_norm_w: gpu_f32(&pa_w),
                kv_k: GpuKvCache::new(&backend, max_seq, kv_dim),
                kv_v: GpuKvCache::new(&backend, max_seq, kv_dim),
            });
        }

        let (nw, _) = load_f32("model.norm.weight")?;
        let norm_w = gpu_f32(&nw);

        // lm_head em BF16 (não quantizar)
        let lm_head = if tensor_exists("lm_head.weight") {
            gpu_bf16("lm_head.weight")?
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            gpu_bf16("model.embed_tokens.weight")?
        };

        // ── Contabilidade de VRAM (com quantização) ──────────────────────────
        let bf16_bytes = |t: &AnyTensor| t.numel() * 2;
        let f32_bytes = |t: &AnyTensor| t.numel() * 4;

        let mut vram_bytes = bf16_bytes(&embed) + bf16_bytes(&lm_head);
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

            vram_bytes += match &l.wq { LayerWeight::Bf16(t) => bf16_bytes(t), LayerWeight::Q4(qt) => q4_weight_bytes(qt) };
            vram_bytes += match &l.wk { LayerWeight::Bf16(t) => bf16_bytes(t), LayerWeight::Q4(qt) => q4_weight_bytes(qt) };
            vram_bytes += match &l.wv { LayerWeight::Bf16(t) => bf16_bytes(t), LayerWeight::Q4(qt) => q4_weight_bytes(qt) };
            vram_bytes += match &l.wo { LayerWeight::Bf16(t) => bf16_bytes(t), LayerWeight::Q4(qt) => q4_weight_bytes(qt) };
            vram_bytes += match &l.wgate { LayerWeight::Bf16(t) => bf16_bytes(t), LayerWeight::Q4(qt) => q4_weight_bytes(qt) };
            vram_bytes += match &l.wup { LayerWeight::Bf16(t) => bf16_bytes(t), LayerWeight::Q4(qt) => q4_weight_bytes(qt) };
            vram_bytes += match &l.wdown { LayerWeight::Bf16(t) => bf16_bytes(t), LayerWeight::Q4(qt) => q4_weight_bytes(qt) };

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
            "✅ Qwen carregado (Q4) — memória estimada: {:.2} GB (4x redução vs BF16)",
            vram_bytes as f64 / 1e9
        );

        let scratch = ForwardScratchpad::new(&cfg, &backend);

        Ok(Self {
            embed,
            layers,
            norm_w,
            lm_head,
            cfg,
            vram_bytes,
            backend,
            scratch,
            cfc,
            gravity,
            quantized: true,
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

    /// Forward pass via ComputeBackend.
    /// Único download_f32(): logits finais [vocab_size] → CPU para sampling.
    pub fn forward(&mut self, token_ids: &[u32]) -> ModelResult<Vec<f32>> {
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

        for layer in &mut self.layers {
            // ── Input RMSNorm ─────────────────────────────────────────────────
            self.backend.rms_norm(&RmsNormParams {
                input:  &self.scratch.x,
                weight: &layer.input_norm_w,
                output: &self.scratch.normed,
                n_tokens: n,
                hidden_size: h,
                eps,
            })?;

            // ── Q, K, V projections (BF16/Q4 pesos × F32 ativações → F32) ──────
            Self::project(&self.backend, &layer.wq, &self.scratch.normed, &self.scratch.q, n, qd, h, 0, 0.0)?;
            Self::project(&self.backend, &layer.wk, &self.scratch.normed, &self.scratch.k, n, kd, h, 0, 0.0)?;
            Self::project(&self.backend, &layer.wv, &self.scratch.normed, &self.scratch.v, n, kd, h, 0, 0.0)?;

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

            // ── RoPE (in-place) ───────────────────────────────────────────────
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

            // ── Append K, V ao KV-cache ───────────────────────────────────────
            self.backend.copy_to_kv_cache(&KvCacheParams {
                cache:        &layer.kv_k.data,
                new_kv:       &self.scratch.k,
                insert_pos:   pos_start,
                n_new_tokens: n,
                kv_dim:       kd,
                max_seq:      self.cfg.max_seq_len,
            })?;
            layer.kv_k.current_len += n;

            self.backend.copy_to_kv_cache(&KvCacheParams {
                cache:        &layer.kv_v.data,
                new_kv:       &self.scratch.v,
                insert_pos:   pos_start,
                n_new_tokens: n,
                kv_dim:       kd,
                max_seq:      self.cfg.max_seq_len,
            })?;
            layer.kv_v.current_len += n;

            let seq = layer.kv_k.current_len;

            // ── Attention: Q × K^T → softmax → × V ───────────────────────────
            // Processa cabeçote por cabeçote via sgemm com offsets.
            // K e V estão em [seq, kv_dim] com stride kv_dim entre tokens.
            //
            // Para cabeçote hi:
            //   Q_h offset: hi * hd  (stride qd entre tokens)
            //   K_h offset: kv_h * hd (stride kd entre tokens, no cache)
            //   scores offset: hi * n * seq
            //
            // scores_h[n, seq] = scale × Q_h × K_h^T  — row-major
            // C[qi*seq + ki] = scale × Q_h[qi, :] · K_h[ki, :]
            // Kernel row-major: M=n (queries), N=seq (keys), K=hd
            //   A = Q_h[n, hd]:   trans_a=false, a_offset=hi*hd, lda=qd
            //   B = K_h[seq, hd]: trans_b=true,  b_offset=kv_h*hd, ldb=kd
            //   C = scores[n, seq]: c_offset=hi*n*seq, ldc=seq
            for hi in 0..nh {
                let kv_h = hi * nk / nh;

                self.backend.sgemm(&SgemmParams {
                    trans_a: false,
                    trans_b: true,
                    m: n, n: seq, k: hd,
                    alpha: scale, beta: 0.0,
                    a: &self.scratch.q,
                    lda: qd,
                    a_offset: hi * hd,
                    b: &layer.kv_k.data,
                    ldb: kd,
                    b_offset: kv_h * hd,
                    c: &self.scratch.scores,
                    ldc: seq,
                    c_offset: hi * n * seq,
                })?;

                // Softmax row-wise: cada query normaliza sobre seq keys
                self.backend.softmax_rows(&SoftmaxParams {
                    scores: &self.scratch.scores,
                    offset_elems: hi * n * seq,
                    n_rows: n,
                    row_len: seq,
                })?;

                // out_h[n, hd] = attn_weights[n, seq] × V_h[seq, hd]  — row-major
                // C[qi*qd + hi*hd + d] = Σ_ki attn[qi,ki] × V_h[ki, d]
                // A = attn[n, seq]: trans_a=false, a_offset=hi*n*seq, lda=seq
                // B = V_h[seq, hd]: trans_b=false, b_offset=kv_h*hd, ldb=kd
                // C = attn_out[n, qd]: c_offset=hi*hd, ldc=qd
                self.backend.sgemm(&SgemmParams {
                    trans_a: false,
                    trans_b: false,
                    m: n, n: hd, k: seq,
                    alpha: 1.0, beta: 0.0,
                    a: &self.scratch.scores,
                    lda: seq,
                    a_offset: hi * n * seq,
                    b: &layer.kv_v.data,
                    ldb: kd,
                    b_offset: kv_h * hd,
                    c: &self.scratch.attn_out,
                    ldc: qd,
                    c_offset: hi * hd,
                })?;
            }

            // ── Output projection: attn_out × wo^T → x (residual) ─────────────
            // x = x + attn_out × wo^T  (beta=1.0 para acumular em x)
            Self::project(&self.backend, &layer.wo, &self.scratch.attn_out, &self.scratch.x, n, h, qd, 0, 1.0)?;

            // ── Post-attention RMSNorm ────────────────────────────────────────
            self.backend.rms_norm(&RmsNormParams {
                input:  &self.scratch.x,
                weight: &layer.post_attn_norm_w,
                output: &self.scratch.normed,
                n_tokens: n,
                hidden_size: h,
                eps,
            })?;

            // ── MLP: gate_proj, up_proj ───────────────────────────────────────
            Self::project(&self.backend, &layer.wgate, &self.scratch.normed, &self.scratch.gate, n, it, h, 0, 0.0)?;
            Self::project(&self.backend, &layer.wup, &self.scratch.normed, &self.scratch.up, n, it, h, 0, 0.0)?;

            // SwiGLU: gate = silu(gate) * up (in-place em gate)
            self.backend.silu_mul_inplace(&SiluMulParams {
                gate: &self.scratch.gate,
                up:   &self.scratch.up,
                n_tokens: n,
                intermediate_size: it,
            })?;

            // down_proj: x = x + swiglu × wdown^T (beta=1 acumula em x)
            Self::project(&self.backend, &layer.wdown, &self.scratch.gate, &self.scratch.x, n, h, it, 0, 1.0)?;
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

        // ── Final RMSNorm ─────────────────────────────────────────────────────
        self.backend.rms_norm(&RmsNormParams {
            input:  &self.scratch.x,
            weight: &self.norm_w,
            output: &self.scratch.normed,
            n_tokens: n,
            hidden_size: h,
            eps,
        })?;

        // ── lm_head (apenas o último token) ──────────────────────────────────
        // lm_head SEMPRE em BF16 (não quantizável — muito sensível)
        // a_offset aponta para o último token: (n-1) * hidden_size
        self.backend.gemm_bf16_f32(&GemmBf16Params {
            m: 1,
            n: vocab,
            k: h,
            alpha: 1.0,
            beta: 0.0,
            a: &self.scratch.normed,
            a_offset: (n - 1) * h,
            b: &self.lm_head,
            c: &self.scratch.logits,
        })?;

        // ── Download logits: único to_host() de todo o forward ───────────────
        let logits = self.backend.download_f32(&self.scratch.logits);
        Ok(logits)
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
