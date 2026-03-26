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

mod kv_cache;
use kv_cache::BoundedKvCache;
use hip_ffi::GpuTensorBF16;
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
    pub rope_theta: f32,
    pub max_seq_len: usize,
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
            rope_theta:        params["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            max_seq_len:       params["max_position_embeddings"].as_u64().map(|n| n as usize).unwrap_or(2048),
        })
    }

    /// Fallback hardcoded para Qwen3-8B (usado em testes)
    pub fn qwen_8b() -> Self {
        Self {
            hidden_size: 4096, num_layers: 36, intermediate_size: 12288,
            vocab_size: 151936, num_heads: 32, num_kv_heads: 8,
            head_dim: 128, rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 2048,
        }
    }
}

// ── Camada Transformer ────────────────────────────────────────────────────────

/// Pesos de uma camada Qwen (GPU como BF16 — metade da VRAM vs F32).
pub struct QwenLayer {
    pub wq: GpuTensorBF16,
    pub wk: GpuTensorBF16,
    pub wv: GpuTensorBF16,
    pub wo: GpuTensorBF16,
    pub wgate: GpuTensorBF16,
    pub wup: GpuTensorBF16,
    pub wdown: GpuTensorBF16,
    // Biases de atenção (Qwen2/2.5 têm bias em Q/K/V; Qwen3 não tem)
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    // Norm weights ficam em host (vetores pequenos, sem ganho em GPU)
    pub input_norm_w: Vec<f32>,
    pub post_attn_norm_w: Vec<f32>,
    // KV-Cache em host (janela deslizante, tamanho fixo = max_seq_len)
    pub kv_k: BoundedKvCache,  // [seq, kv_dim]
    pub kv_v: BoundedKvCache,  // [seq, kv_dim]
}

// ── Modelo completo ───────────────────────────────────────────────────────────

pub struct QwenModel {
    pub embed: GpuTensorBF16,   // [vocab, hidden]
    pub layers: Vec<QwenLayer>,
    pub norm_w: Vec<f32>,       // [hidden]
    pub lm_head: GpuTensorBF16, // [vocab, hidden]
    pub cfg: ModelCfg,
    /// Bytes totais ocupados em VRAM pelos GpuTensorBF16 (2 bytes/elemento).
    /// Calculado uma vez no load(). Não inclui KV-cache (host) nem ativações.
    pub vram_bytes: usize,
}

impl QwenModel {
    /// Carrega pesos do diretório de safetensors.
    /// Suporta single-shard (model.safetensors) e multi-shard (model.safetensors.index.json).
    /// Mantém BF16 cru → envia para VRAM como u16 (metade da VRAM vs F32).
    pub fn load(model_dir: &Path, cfg: ModelCfg) -> ModelResult<Self> {
        log::info!(
            "⏳ Carregando Qwen nativo: {} camadas, hidden={}, vocab={}",
            cfg.num_layers, cfg.hidden_size, cfg.vocab_size
        );

        // Detecta single-shard vs multi-shard e monta mmaps
        let mut shards: HashMap<String, memmap2::Mmap> = HashMap::new();
        // Some(map) = multi-shard: tensor_name → shard_filename
        // None = single-shard: tudo em "model.safetensors"
        let tensor_shard_map: Option<HashMap<String, String>>;

        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_file = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_file)?;
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
                    "Nenhum arquivo safetensors encontrado (sem index.json nem model.safetensors)".into()
                ));
            }
            let file = std::fs::File::open(&single)?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            log::info!("  → single-shard ({:.2} GB)", mmap.len() as f64 / 1e9);
            shards.insert("model.safetensors".to_string(), mmap);
            tensor_shard_map = None;
        }

        // Resolve mmap para um tensor (ou None se tensor não existe no índice)
        let find_mmap = |name: &str| -> Option<&memmap2::Mmap> {
            match &tensor_shard_map {
                Some(tsm) => tsm.get(name).and_then(|s| shards.get(s)),
                None => shards.get("model.safetensors"),
            }
        };

        let tensor_exists = |name: &str| -> bool {
            match &tensor_shard_map {
                Some(tsm) => tsm.contains_key(name),
                // single-shard: tenta parsear o header pra verificar
                None => shards.get("model.safetensors")
                    .map(|m| shard_has_tensor(m, name))
                    .unwrap_or(false),
            }
        };

        // Helper: lê BF16 cru (u16 bits) — sem converter para f32
        let load_raw = |name: &str| -> ModelResult<(Vec<u16>, Vec<usize>)> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            read_bf16_raw(mmap, name)
        };

        // Helper: lê f32 (para vetores de norm/bias — pequenos, ficam na CPU)
        let load_f32 = |name: &str| -> ModelResult<(Vec<f32>, Vec<usize>)> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            read_bf16_f32(mmap, name)
        };

        let gpu = |name: &str| -> ModelResult<GpuTensorBF16> {
            let (bits, shape) = load_raw(name)?;
            Ok(GpuTensorBF16::from_bf16_bits(&bits, &shape)?)
        };

        // Embedding
        log::info!("  → embed_tokens");
        let embed = gpu("model.embed_tokens.weight")?;

        // Camadas
        let kv_dim   = cfg.num_kv_heads * cfg.head_dim;
        let max_seq  = cfg.max_seq_len;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i % 8 == 0 {
                log::info!("  → layers {}/{}", i, cfg.num_layers);
            }
            let p = format!("model.layers.{}", i);
            let (in_w, _)  = load_f32(&format!("{}.input_layernorm.weight", p))?;
            let (pa_w, _)  = load_f32(&format!("{}.post_attention_layernorm.weight", p))?;

            let q_bias_name = format!("{}.self_attn.q_proj.bias", p);
            let k_bias_name = format!("{}.self_attn.k_proj.bias", p);
            let v_bias_name = format!("{}.self_attn.v_proj.bias", p);

            layers.push(QwenLayer {
                wq:    gpu(&format!("{}.self_attn.q_proj.weight", p))?,
                wk:    gpu(&format!("{}.self_attn.k_proj.weight", p))?,
                wv:    gpu(&format!("{}.self_attn.v_proj.weight", p))?,
                wo:    gpu(&format!("{}.self_attn.o_proj.weight", p))?,
                wgate: gpu(&format!("{}.mlp.gate_proj.weight", p))?,
                wup:   gpu(&format!("{}.mlp.up_proj.weight", p))?,
                wdown: gpu(&format!("{}.mlp.down_proj.weight", p))?,
                q_bias: if tensor_exists(&q_bias_name) { Some(load_f32(&q_bias_name)?.0) } else { None },
                k_bias: if tensor_exists(&k_bias_name) { Some(load_f32(&k_bias_name)?.0) } else { None },
                v_bias: if tensor_exists(&v_bias_name) { Some(load_f32(&v_bias_name)?.0) } else { None },
                input_norm_w: in_w,
                post_attn_norm_w: pa_w,
                kv_k: BoundedKvCache::new(max_seq, kv_dim),
                kv_v: BoundedKvCache::new(max_seq, kv_dim),
            });
        }

        let (norm_w, _) = load_f32("model.norm.weight")?;
        // Alguns modelos usam tie_word_embeddings=true → lm_head = embed_tokens
        let lm_head = if tensor_exists("lm_head.weight") {
            gpu("lm_head.weight")?
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            gpu("model.embed_tokens.weight")?
        };

        // Calcula total de bytes em VRAM (GpuTensorBF16 = 2 bytes/elemento)
        let tensor_bytes = |t: &GpuTensorBF16| -> usize {
            t.shape.iter().product::<usize>() * 2
        };
        let mut vram_bytes = tensor_bytes(&embed) + tensor_bytes(&lm_head);
        for l in &layers {
            vram_bytes += tensor_bytes(&l.wq) + tensor_bytes(&l.wk) + tensor_bytes(&l.wv)
                        + tensor_bytes(&l.wo) + tensor_bytes(&l.wgate)
                        + tensor_bytes(&l.wup) + tensor_bytes(&l.wdown);
        }

        log::info!("✅ Qwen carregado — pesos BF16 em VRAM: {:.2} GB", vram_bytes as f64 / 1e9);
        Ok(Self { embed, layers, norm_w, lm_head, cfg, vram_bytes })
    }

    pub fn reset_cache(&mut self) {
        for l in &mut self.layers {
            l.kv_k.reset();
            l.kv_v.reset();
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

            let mut q = gemm(&normed, &wq, n, h, q_dim);
            let mut k = gemm(&normed, &wk, n, h, kv_dim);
            let mut v = gemm(&normed, &wv, n, h, kv_dim);

            // Bias (Qwen2/2.5): adiciona bias a Q, K, V
            if let Some(ref b) = layer.q_bias {
                for tok in 0..n { for d in 0..q_dim { q[tok*q_dim+d] += b[d]; } }
            }
            if let Some(ref b) = layer.k_bias {
                for tok in 0..n { for d in 0..kv_dim { k[tok*kv_dim+d] += b[d]; } }
            }
            if let Some(ref b) = layer.v_bias {
                for tok in 0..n { for d in 0..kv_dim { v[tok*kv_dim+d] += b[d]; } }
            }

            // RoPE: pos_start = tokens já em cache antes desta chamada
            let pos_start = layer.kv_k.seq_len();
            apply_rope(&mut q, n, self.cfg.num_heads,    self.cfg.head_dim, pos_start, self.cfg.rope_theta);
            apply_rope(&mut k, n, self.cfg.num_kv_heads, self.cfg.head_dim, pos_start, self.cfg.rope_theta);

            layer.kv_k.append(&k);
            layer.kv_v.append(&v);
            let seq = layer.kv_k.seq_len();

            // Multi-head attention
            let head_dim = self.cfg.head_dim;
            let n_heads  = self.cfg.num_heads;
            let n_kv     = self.cfg.num_kv_heads;
            let scale    = (head_dim as f32).sqrt();
            let mut attn = vec![0.0f32; n * q_dim];
            let kv_k_s = layer.kv_k.as_slice();
            let kv_v_s = layer.kv_v.as_slice();

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
                                 * kv_k_s[ki * kv_dim + k_off + d];
                        }
                        scores[ki] = dot / scale;
                    }
                    softmax_inplace(&mut scores);
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for ki in 0..seq {
                            acc += scores[ki] * kv_v_s[ki * kv_dim + k_off + d];
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

/// Aplica RoPE (Rotary Position Embedding) in-place em Q ou K.
///
/// Layout esperado: buffer plano de shape [n_tokens, n_heads, head_dim].
/// Usa a convenção "halves" do HuggingFace/Qwen: o par rotacionado é
/// (x[i], x[i + head_dim/2]) para i in 0..half_dim.
///
/// `pos_start` é a posição absoluta do primeiro token no buffer —
/// para decode com KV-cache, isso é o comprimento do cache antes do append.
pub(crate) fn apply_rope(
    x: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    pos_start: usize,
    theta: f32,
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
                let (sin_val, cos_val) = angle.sin_cos();
                let x0 = x[base + i];
                let x1 = x[base + half_dim + i];
                x[base + i]            = x0 * cos_val - x1 * sin_val;
                x[base + half_dim + i] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

// ── Leitores BF16 safetensors ─────────────────────────────────────────────────

/// Lê tensor BF16 como bits crus (u16) — sem conversão para f32.
/// Usado para pesos matriciais enviados à GPU como GpuTensorBF16.
fn read_bf16_raw(mmap: &memmap2::Mmap, name: &str) -> ModelResult<(Vec<u16>, Vec<usize>)> {
    let (raw, shape) = shard_tensor_bytes(mmap, name)?;
    let bits: Vec<u16> = raw.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    Ok((bits, shape))
}

/// Lê tensor BF16 convertendo para f32 no host.
/// Usado para vetores de norm (pequenos, ficam na CPU).
fn read_bf16_f32(mmap: &memmap2::Mmap, name: &str) -> ModelResult<(Vec<f32>, Vec<usize>)> {
    let (raw, shape) = shard_tensor_bytes(mmap, name)?;
    let floats: Vec<f32> = raw.chunks_exact(2).map(|b| {
        bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
    }).collect();
    Ok((floats, shape))
}

/// Verifica se um tensor existe no header de um shard (single-shard mode).
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
