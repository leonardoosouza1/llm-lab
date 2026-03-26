// ============================================================================
// model.rs — Pipeline Nativo Qwen (sem candle, sem framework)
// ============================================================================
// Carrega pesos safetensors direto em GPU via hip_ffi.
// Forward pass: embedding → N×transformer_block → lm_head
// Cada tensor é um GpuTensor residente na VRAM da RX 6750 XT.
//
// FASE ATUAL: Implementação de referência CPU (f32 puro).
// PRÓXIMA FASE: rocBLAS GEMM para projections + kernels HIP para ops point-wise.
// ============================================================================

use hip_ffi::GpuTensor;
use half::bf16;
use std::collections::HashMap;
use std::path::Path;

// ── Tipos de erro ──────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("HIP error: {0}")]
    Hip(#[from] hip_ffi::HipError),
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
}

impl ModelCfg {
    /// Lê os parâmetros diretamente do config.json do modelo.
    /// Suporta configs planas (Qwen3, Llama) e configs aninhadas em `text_config` (VLMs).
    /// Retorna erro claro se a arquitetura não for um causal LM suportado.
    pub fn from_config_json(config_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(config_path)?;
        let v: serde_json::Value = serde_json::from_str(&content)?;

        // Verifica arquitetura suportada
        let arch = v["architectures"].as_array()
            .and_then(|a| a.first())
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        let supported = ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM",
                         "MistralForCausalLM", "Qwen2_5ForCausalLM"];
        if !supported.iter().any(|s| arch.contains(s.trim_end_matches("ForCausalLM"))
                                  || *s == arch) {
            return Err(format!(
                "Arquitetura '{}' não é suportada. O engine suporta apenas modelos \
                 causal LM (Qwen3, Llama, Mistral). Modelos multimodais (VLM) como \
                 Qwen3_5ForConditionalGeneration não são compatíveis.",
                arch
            ).into());
        }

        // Alguns configs têm parâmetros no nível raiz; VLMs os colocam em text_config.
        // Usamos o nível raiz se hidden_size existir lá, senão tentamos text_config.
        let params = if v["hidden_size"].is_number() {
            &v
        } else if v["text_config"]["hidden_size"].is_number() {
            &v["text_config"]
        } else {
            return Err(format!(
                "config.json de '{}' não contém 'hidden_size' nem 'text_config.hidden_size'",
                config_path.display()
            ).into());
        };

        // Helper: extrai usize de inteiro ou float JSON
        let uint = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
            let val = &params[key];
            val.as_u64()
                .or_else(|| val.as_f64().map(|f| f as u64))
                .map(|n| n as usize)
                .ok_or_else(|| format!("campo '{}' ausente ou inválido em config.json", key).into())
        };

        let hidden_size = uint("hidden_size")?;
        let num_heads   = uint("num_attention_heads")?;

        // head_dim pode não existir em configs antigos → calcula como hidden/heads
        let head_dim = params["head_dim"].as_u64()
            .or_else(|| params["head_dim"].as_f64().map(|f| f as u64))
            .map(|n| n as usize)
            .unwrap_or(hidden_size / num_heads);

        Ok(Self {
            hidden_size,
            num_heads,
            head_dim,
            num_layers:        uint("num_hidden_layers")?,
            intermediate_size: uint("intermediate_size")?,
            vocab_size:        uint("vocab_size")?,
            num_kv_heads:      uint("num_key_value_heads")?,
            rms_eps:           params["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        })
    }

    /// Fallback hardcoded para Qwen3-8B (usado em testes)
    pub fn qwen_8b() -> Self {
        Self {
            hidden_size: 4096, num_layers: 36, intermediate_size: 12288,
            vocab_size: 151936, num_heads: 32, num_kv_heads: 8,
            head_dim: 128, rms_eps: 1e-6,
        }
    }
}

// ── Camada Transformer ────────────────────────────────────────────────────────

/// Pesos de uma camada Qwen (todos em GPU como GpuTensor).
pub struct QwenLayer {
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
    pub wgate: GpuTensor,
    pub wup: GpuTensor,
    pub wdown: GpuTensor,
    // Norm weights ficam em host (vetores pequenos, sem ganho em GPU)
    pub input_norm_w: Vec<f32>,
    pub post_attn_norm_w: Vec<f32>,
    // KV-Cache em host (cresce com o contexto)
    pub kv_k: Vec<f32>,  // [seq, kv_dim]
    pub kv_v: Vec<f32>,  // [seq, kv_dim]
}

// ── Modelo completo ───────────────────────────────────────────────────────────

pub struct QwenModel {
    pub embed: GpuTensor,       // [vocab, hidden]
    pub layers: Vec<QwenLayer>,
    pub norm_w: Vec<f32>,       // [hidden]
    pub lm_head: GpuTensor,     // [vocab, hidden]
    pub cfg: ModelCfg,
}

impl QwenModel {
    /// Carrega pesos do diretório de safetensors.
    /// Converte BF16 → F32 host → envia para VRAM via HIP.
    pub fn load(model_dir: &Path, cfg: ModelCfg) -> ModelResult<Self> {
        log::info!(
            "⏳ Carregando Qwen nativo: {} camadas, hidden={}, vocab={}",
            cfg.num_layers, cfg.hidden_size, cfg.vocab_size
        );

        // Monta mapa de shards
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_file = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_file)?;
        let weight_map = index["weight_map"].as_object()
            .ok_or_else(|| ModelError::Model("weight_map inválido".into()))?;

        // mmap de cada shard
        let mut shards: HashMap<String, memmap2::Mmap> = HashMap::new();
        for shard_name in weight_map.values().filter_map(|v| v.as_str()) {
            if shards.contains_key(shard_name) { continue; }
            let path = model_dir.join(shard_name);
            let file = std::fs::File::open(&path)?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            shards.insert(shard_name.to_string(), mmap);
        }

        // Helper de carregamento
        let load = |name: &str| -> ModelResult<(Vec<f32>, Vec<usize>)> {
            let shard = weight_map[name].as_str()
                .ok_or_else(|| ModelError::Model(format!("'{}' não no index", name)))?;
            read_bf16_f32(shards.get(shard).unwrap(), name)
        };

        let gpu = |name: &str| -> ModelResult<GpuTensor> {
            let (data, shape) = load(name)?;
            Ok(GpuTensor::from_slice(&data, &shape)?)
        };

        // Embedding
        log::info!("  → embed_tokens");
        let embed = gpu("model.embed_tokens.weight")?;

        // Camadas
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 {
                log::info!("  → layers {}/{}", i, cfg.num_layers);
            }
            let p = format!("model.layers.{}", i);
            let (in_w, _)  = load(&format!("{}.input_layernorm.weight", p))?;
            let (pa_w, _)  = load(&format!("{}.post_attention_layernorm.weight", p))?;

            layers.push(QwenLayer {
                wq:    gpu(&format!("{}.self_attn.q_proj.weight", p))?,
                wk:    gpu(&format!("{}.self_attn.k_proj.weight", p))?,
                wv:    gpu(&format!("{}.self_attn.v_proj.weight", p))?,
                wo:    gpu(&format!("{}.self_attn.o_proj.weight", p))?,
                wgate: gpu(&format!("{}.mlp.gate_proj.weight", p))?,
                wup:   gpu(&format!("{}.mlp.up_proj.weight", p))?,
                wdown: gpu(&format!("{}.mlp.down_proj.weight", p))?,
                input_norm_w: in_w,
                post_attn_norm_w: pa_w,
                kv_k: Vec::new(),
                kv_v: Vec::new(),
            });
        }

        let (norm_w, _) = load("model.norm.weight")?;
        let lm_head = gpu("lm_head.weight")?;

        log::info!("✅ Qwen carregado em VRAM (HIP nativo)");
        Ok(Self { embed, layers, norm_w, lm_head, cfg })
    }

    pub fn reset_cache(&mut self) {
        for l in &mut self.layers {
            l.kv_k.clear();
            l.kv_v.clear();
        }
    }

    /// Forward pass — retorna logits [vocab] para o último token.
    /// Implementação de referência em CPU pura f32.
    /// rocBLAS GEMM na próxima fase (substituição direta desta função).
    pub fn forward(&mut self, token_ids: &[u32]) -> ModelResult<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let n = token_ids.len();

        // Embedding lookup
        let embed = self.embed.to_host()?;
        let mut x: Vec<f32> = token_ids.iter().flat_map(|&id| {
            let off = id as usize * h;
            embed[off..off + h].to_vec()
        }).collect();

        // Transformer blocks
        for layer in &mut self.layers {
            let normed = rms_norm(&x, &layer.input_norm_w, self.cfg.rms_eps, n, h);

            let wq = layer.wq.to_host()?;
            let wk = layer.wk.to_host()?;
            let wv = layer.wv.to_host()?;
            let wo = layer.wo.to_host()?;

            let q_dim = layer.wq.shape[0];
            let kv_dim = layer.wk.shape[0];

            let q = gemm(&normed, &wq, n, h, q_dim);
            let k = gemm(&normed, &wk, n, h, kv_dim);
            let v = gemm(&normed, &wv, n, h, kv_dim);

            layer.kv_k.extend_from_slice(&k);
            layer.kv_v.extend_from_slice(&v);
            let seq = layer.kv_k.len() / kv_dim;

            // Multi-head attention
            let head_dim = self.cfg.head_dim;
            let n_heads  = self.cfg.num_heads;
            let n_kv     = self.cfg.num_kv_heads;
            let scale    = (head_dim as f32).sqrt();
            let mut attn = vec![0.0f32; n * q_dim];

            for h_i in 0..n_heads {
                let kv_h = h_i * n_kv / n_heads; // GQA mapping
                let q_off = h_i * head_dim;
                let k_off = kv_h * head_dim;

                for qi in 0..n {
                    let mut scores = vec![0.0f32; seq];
                    for ki in 0..seq {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[qi * q_dim + q_off + d]
                                 * layer.kv_k[ki * kv_dim + k_off + d];
                        }
                        scores[ki] = dot / scale;
                    }
                    softmax_inplace(&mut scores);
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for ki in 0..seq {
                            acc += scores[ki] * layer.kv_v[ki * kv_dim + k_off + d];
                        }
                        attn[qi * q_dim + q_off + d] = acc;
                    }
                }
            }

            let attn_proj = gemm(&attn, &wo, n, q_dim, h);
            for i in 0..n * h { x[i] += attn_proj[i]; }

            // MLP
            let normed2 = rms_norm(&x, &layer.post_attn_norm_w, self.cfg.rms_eps, n, h);
            let wg = layer.wgate.to_host()?;
            let wu = layer.wup.to_host()?;
            let wd = layer.wdown.to_host()?;
            let inter = layer.wgate.shape[0];

            let gate = gemm(&normed2, &wg, n, h, inter);
            let up   = gemm(&normed2, &wu, n, h, inter);
            let swiglu: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
            let mlp = gemm(&swiglu, &wd, n, inter, h);
            for i in 0..n * h { x[i] += mlp[i]; }
        }

        // Final norm
        let x_norm = rms_norm(&x, &self.norm_w, self.cfg.rms_eps, n, h);

        // lm_head — só no último token
        let last = &x_norm[(n - 1) * h..];
        let lm = self.lm_head.to_host()?;
        let vocab = self.cfg.vocab_size;
        let logits = gemm(last, &lm, 1, h, vocab);
        Ok(logits)
    }
}

// ── Primitivas CPU ────────────────────────────────────────────────────────────

fn rms_norm(x: &[f32], w: &[f32], eps: f32, n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let rms = (row.iter().map(|&v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        for j in 0..d { out[i * d + j] = row[j] / rms * w[j]; }
    }
    out
}

/// A[m,k] × B^T[n,k] → C[m,n]  (weights armazenados como [out, in])
fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k { acc += a[i * k + p] * b[j * k + p]; }
            c[i * n + j] = acc;
        }
    }
    c
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in x.iter_mut() { *v /= sum; }
}

#[inline(always)]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

// ── Leitor BF16 safetensors ───────────────────────────────────────────────────

fn read_bf16_f32(mmap: &memmap2::Mmap, name: &str) -> ModelResult<(Vec<f32>, Vec<usize>)> {
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
    let raw = &mmap[base + s..base + e];

    let floats: Vec<f32> = raw.chunks_exact(2).map(|b| {
        bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
    }).collect();

    Ok((floats, shape))
}
