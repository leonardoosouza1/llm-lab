// ============================================================================
// model/mod.rs — Pipeline Nativo Qwen: GPU-only via hipBLAS + kernels HIP
// ============================================================================
// Todas as operações em VRAM:
//   - Pesos: GpuTensorBF16 (u16, metade da VRAM)
//   - Ativações: DeviceBuffer<f32> (scratchpad pré-alocado, zero-alloc no forward)
//   - KV-cache: GpuKvCache (DeviceBuffer<f32> pré-alocado, append via kernel)
//   - Projeções: hipBLAS GEMM (BF16 pesos × F32 ativações → F32 saída)
//   - Norm/RoPE/SwiGLU/Softmax: kernels HIP customizados (llm_kernels.hip)
//   - Único to_host(): logits finais [vocab_size] por token gerado
// ============================================================================

mod kv_cache;
pub(crate) use kv_cache::GpuKvCache;
#[allow(unused_imports)]
pub(crate) use kv_cache::BoundedKvCache;

use hip_ffi::{
    safe::DeviceBuffer,
    GpuTensorBF16,
    BlasHandle,
    GpuKernels,
    gemm_bf16_f32, sgemm_strided,
};
use half::bf16;
use std::collections::HashMap;
use std::path::Path;

// ── Tipos de erro ────────────────────────────────────────────────────────────

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
            // A 4096-token context is practical for interactive inference and fits all models.
            max_seq_len: {
                let raw = params["max_position_embeddings"].as_u64()
                    .map(|n| n as usize)
                    .unwrap_or(2048);
                raw.min(4096)
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

/// Buffers GPU pré-alocados para o forward pass. Zero alocação durante geração.
pub struct ForwardScratchpad {
    pub x:        DeviceBuffer<f32>,  // [max_seq, hidden]
    pub normed:   DeviceBuffer<f32>,  // [max_seq, hidden]
    pub q:        DeviceBuffer<f32>,  // [max_seq, q_dim]
    pub k:        DeviceBuffer<f32>,  // [max_seq, kv_dim]
    pub v:        DeviceBuffer<f32>,  // [max_seq, kv_dim]
    pub attn_out: DeviceBuffer<f32>,  // [max_seq, q_dim]
    pub gate:     DeviceBuffer<f32>,  // [max_seq, inter]
    pub up:       DeviceBuffer<f32>,  // [max_seq, inter]
    pub scores:   DeviceBuffer<f32>,  // [n_heads * max_seq, max_seq] — scores de atenção
    pub logits:   DeviceBuffer<f32>,  // [vocab_size]
    pub tok_ids:  DeviceBuffer<u32>,  // [max_seq] — IDs de token para embed_lookup
}

impl ForwardScratchpad {
    fn new(cfg: &ModelCfg) -> Result<Self, hip_ffi::HipError> {
        let ms = cfg.max_seq_len;
        let h  = cfg.hidden_size;
        let qd = cfg.num_heads * cfg.head_dim;
        let kd = cfg.num_kv_heads * cfg.head_dim;
        let it = cfg.intermediate_size;
        let nh = cfg.num_heads;
        let v  = cfg.vocab_size;
        Ok(Self {
            x:        DeviceBuffer::alloc(ms * h)?,
            normed:   DeviceBuffer::alloc(ms * h)?,
            q:        DeviceBuffer::alloc(ms * qd)?,
            k:        DeviceBuffer::alloc(ms * kd)?,
            v:        DeviceBuffer::alloc(ms * kd)?,
            attn_out: DeviceBuffer::alloc(ms * qd)?,
            gate:     DeviceBuffer::alloc(ms * it)?,
            up:       DeviceBuffer::alloc(ms * it)?,
            scores:   DeviceBuffer::alloc(nh * ms * ms)?,
            logits:   DeviceBuffer::alloc(v)?,
            tok_ids:  DeviceBuffer::alloc(ms)?,
        })
    }
}

// ── Camada Transformer ────────────────────────────────────────────────────────

/// Pesos e estado de uma camada Qwen — tudo em GPU.
pub struct QwenLayer {
    // Pesos matriciais: BF16 em GPU (metade da VRAM vs F32)
    pub wq:    GpuTensorBF16,
    pub wk:    GpuTensorBF16,
    pub wv:    GpuTensorBF16,
    pub wo:    GpuTensorBF16,
    pub wgate: GpuTensorBF16,
    pub wup:   GpuTensorBF16,
    pub wdown: GpuTensorBF16,
    // Biases opcionais (Qwen2/2.5 têm bias em Q/K/V; Qwen3 não tem)
    pub q_bias_gpu: Option<DeviceBuffer<f32>>,
    pub k_bias_gpu: Option<DeviceBuffer<f32>>,
    pub v_bias_gpu: Option<DeviceBuffer<f32>>,
    // Norm weights em GPU (pequenos, mas evita round-trip host→device por token)
    pub input_norm_w_gpu:    DeviceBuffer<f32>,
    pub post_attn_norm_w_gpu: DeviceBuffer<f32>,
    // KV-cache em GPU (janela deslizante via copy_to_kv_cache kernel)
    pub kv_k: GpuKvCache,
    pub kv_v: GpuKvCache,
}

// ── Modelo completo ───────────────────────────────────────────────────────────

pub struct QwenModel {
    pub embed:   GpuTensorBF16,  // [vocab, hidden]
    pub layers:  Vec<QwenLayer>,
    pub norm_w_gpu: DeviceBuffer<f32>,  // [hidden]
    pub lm_head: GpuTensorBF16,  // [vocab, hidden]
    pub cfg:     ModelCfg,
    /// Bytes em VRAM: pesos BF16 (2 bytes/elem) + scratchpad + KV-cache (4 bytes/elem).
    pub vram_bytes: usize,
    // Contexto GPU
    pub blas:    BlasHandle,
    pub kernels: GpuKernels,
    pub scratch: ForwardScratchpad,
}

impl QwenModel {
    /// Carrega pesos do diretório. Suporta single-shard e multi-shard.
    pub fn load(model_dir: &Path, cfg: ModelCfg) -> ModelResult<Self> {
        log::info!(
            "⏳ Carregando Qwen GPU: {} camadas, hidden={}, vocab={}",
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

        let load_raw = |name: &str| -> ModelResult<(Vec<u16>, Vec<usize>)> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            read_bf16_raw(mmap, name)
        };
        let load_f32 = |name: &str| -> ModelResult<(Vec<f32>, Vec<usize>)> {
            let mmap = find_mmap(name)
                .ok_or_else(|| ModelError::Model(format!("'{}' não encontrado", name)))?;
            read_bf16_f32(mmap, name)
        };

        // Envia BF16 cru para GPU
        let gpu = |name: &str| -> ModelResult<GpuTensorBF16> {
            let (bits, shape) = load_raw(name)?;
            Ok(GpuTensorBF16::from_bf16_bits(&bits, &shape)?)
        };
        // Envia vetor f32 para GPU
        let gpu_f32 = |data: &[f32]| -> ModelResult<DeviceBuffer<f32>> {
            Ok(DeviceBuffer::from_slice(data)?)
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
                wq:    gpu(&format!("{}.self_attn.q_proj.weight", p))?,
                wk:    gpu(&format!("{}.self_attn.k_proj.weight", p))?,
                wv:    gpu(&format!("{}.self_attn.v_proj.weight", p))?,
                wo:    gpu(&format!("{}.self_attn.o_proj.weight", p))?,
                wgate: gpu(&format!("{}.mlp.gate_proj.weight", p))?,
                wup:   gpu(&format!("{}.mlp.up_proj.weight", p))?,
                wdown: gpu(&format!("{}.mlp.down_proj.weight", p))?,
                q_bias_gpu: if tensor_exists(&qbn) { Some(gpu_f32(&load_f32(&qbn)?.0)?) } else { None },
                k_bias_gpu: if tensor_exists(&kbn) { Some(gpu_f32(&load_f32(&kbn)?.0)?) } else { None },
                v_bias_gpu: if tensor_exists(&vbn) { Some(gpu_f32(&load_f32(&vbn)?.0)?) } else { None },
                input_norm_w_gpu:    gpu_f32(&in_w)?,
                post_attn_norm_w_gpu: gpu_f32(&pa_w)?,
                kv_k: GpuKvCache::new(max_seq, kv_dim)?,
                kv_v: GpuKvCache::new(max_seq, kv_dim)?,
            });
        }

        let (nw, _) = load_f32("model.norm.weight")?;
        let norm_w_gpu = gpu_f32(&nw)?;

        let lm_head = if tensor_exists("lm_head.weight") {
            gpu("lm_head.weight")?
        } else {
            log::info!("  → lm_head: usando embed_tokens (tied weights)");
            gpu("model.embed_tokens.weight")?
        };

        // ── VRAM contabil ────────────────────────────────────────────────────
        let bf16_bytes = |t: &GpuTensorBF16| t.shape.iter().product::<usize>() * 2;
        let f32_bytes  = |t: &DeviceBuffer<f32>| t.len() * 4;

        let mut vram_bytes = bf16_bytes(&embed) + bf16_bytes(&lm_head);
        for l in &layers {
            vram_bytes += bf16_bytes(&l.wq) + bf16_bytes(&l.wk) + bf16_bytes(&l.wv)
                        + bf16_bytes(&l.wo) + bf16_bytes(&l.wgate)
                        + bf16_bytes(&l.wup) + bf16_bytes(&l.wdown);
            vram_bytes += f32_bytes(&l.input_norm_w_gpu) + f32_bytes(&l.post_attn_norm_w_gpu);
            vram_bytes += l.kv_k.data.len() * 4 + l.kv_v.data.len() * 4;
        }

        log::info!("✅ Qwen carregado — VRAM: {:.2} GB", vram_bytes as f64 / 1e9);

        let blas   = BlasHandle::new()?;
        let kernels = GpuKernels::new()?;
        let scratch = ForwardScratchpad::new(&cfg)?;

        Ok(Self { embed, layers, norm_w_gpu, lm_head, cfg, vram_bytes, blas, kernels, scratch })
    }

    pub fn reset_cache(&mut self) {
        for l in &mut self.layers {
            l.kv_k.reset();
            l.kv_v.reset();
        }
    }

    /// Forward pass GPU-only.
    /// Único to_host(): logits finais [vocab_size] → CPU para sampling.
    pub fn forward(&mut self, token_ids: &[u32]) -> ModelResult<Vec<f32>> {
        let n  = token_ids.len() as i32;
        let h  = self.cfg.hidden_size as i32;
        let nh = self.cfg.num_heads as i32;
        let nk = self.cfg.num_kv_heads as i32;
        let hd = self.cfg.head_dim as i32;
        let qd = nh * hd;   // q_dim
        let kd = nk * hd;   // kv_dim
        let it = self.cfg.intermediate_size as i32;
        let eps = self.cfg.rms_eps;
        let theta = self.cfg.rope_theta;
        let scale = 1.0_f32 / (hd as f32).sqrt();
        let vocab = self.cfg.vocab_size as i32;

        // Upload token IDs para GPU
        unsafe { hip_ffi::safe::hip_h2d(self.scratch.tok_ids.as_mut_ptr(), token_ids)?; }

        // Embedding lookup: BF16 embed → f32 x [n, h]
        unsafe { self.kernels.embed_lookup(
            self.embed.dev_ptr(),
            self.scratch.tok_ids.as_ptr(),
            self.scratch.x.as_mut_ptr(),
            n, h,
        )?; }

        for layer in &mut self.layers {
            // ── Input RMSNorm ─────────────────────────────────────────────────
            unsafe { self.kernels.rms_norm(
                self.scratch.normed.as_mut_ptr(),
                self.scratch.x.as_ptr(),
                layer.input_norm_w_gpu.as_ptr(),
                n, h, eps,
            )?; }

            // ── Q, K, V projections (hipBLAS BF16×F32) ───────────────────────
            gemm_bf16_f32(&self.blas, n, qd, h, 1.0, 0.0,
                self.scratch.normed.as_ptr(), layer.wq.dev_ptr(),
                self.scratch.q.as_mut_ptr())?;

            gemm_bf16_f32(&self.blas, n, kd, h, 1.0, 0.0,
                self.scratch.normed.as_ptr(), layer.wk.dev_ptr(),
                self.scratch.k.as_mut_ptr())?;

            gemm_bf16_f32(&self.blas, n, kd, h, 1.0, 0.0,
                self.scratch.normed.as_ptr(), layer.wv.dev_ptr(),
                self.scratch.v.as_mut_ptr())?;

            // Biases opcionais (Qwen2/2.5)
            if let Some(ref b) = layer.q_bias_gpu {
                unsafe { self.kernels.add_bias_inplace(self.scratch.q.as_mut_ptr(), b.as_ptr(), n, qd)?; }
            }
            if let Some(ref b) = layer.k_bias_gpu {
                unsafe { self.kernels.add_bias_inplace(self.scratch.k.as_mut_ptr(), b.as_ptr(), n, kd)?; }
            }
            if let Some(ref b) = layer.v_bias_gpu {
                unsafe { self.kernels.add_bias_inplace(self.scratch.v.as_mut_ptr(), b.as_ptr(), n, kd)?; }
            }

            // ── RoPE (in-place) ───────────────────────────────────────────────
            let pos_start = layer.kv_k.current_len as i32;
            unsafe {
                self.kernels.rope_inplace(self.scratch.q.as_mut_ptr(), n, nh, hd, pos_start, theta)?;
                self.kernels.rope_inplace(self.scratch.k.as_mut_ptr(), n, nk, hd, pos_start, theta)?;
            }

            // ── Append K, V ao KV-cache GPU ───────────────────────────────────
            unsafe {
                self.kernels.copy_to_kv_cache(
                    layer.kv_k.data.as_mut_ptr(),
                    self.scratch.k.as_ptr(),
                    n, kd, pos_start,
                )?;
                self.kernels.copy_to_kv_cache(
                    layer.kv_v.data.as_mut_ptr(),
                    self.scratch.v.as_ptr(),
                    n, kd, pos_start,
                )?;
            }
            layer.kv_k.current_len += n as usize;
            layer.kv_v.current_len += n as usize;
            let seq = layer.kv_k.current_len as i32;

            // ── Attention: Q × K^T → softmax → × V ───────────────────────────
            // Processa cabeçote por cabeçote via hipBLAS GEMM com stride.
            // K e V estão em [seq, kv_dim] com stride kv_dim entre tokens.
            //
            // Para cabeçote h: Q_h = q[*, h*hd..(h+1)*hd], lda = qd
            //                  K_h = kv_k[*, kv_h*hd..(kv_h+1)*hd], lda = kd
            //
            // scores_h[n, seq] = Q_h[n, hd] × K_h^T[hd, seq]
            // Em col-major: scores^T[seq, n] = K_h[seq, hd]×Q_h^T[hd, n]
            //   sgemm_strided(OP_N=K, OP_T=Q_h, M=seq, N=n, K=hd, lda=kd, ldb=qd, ldc=seq)
            //   → stores in column-major, reads as row-major scores[n, seq] ✓

            let kv_k_ptr = layer.kv_k.data.as_ptr();
            let kv_v_ptr = layer.kv_v.data.as_ptr();

            for hi in 0..(nh as usize) {
                let kv_h = hi * (nk as usize) / (nh as usize);

                let q_h  = unsafe { self.scratch.q.as_ptr().add(hi * hd as usize) };
                let k_h  = unsafe { kv_k_ptr.add(kv_h * hd as usize) };
                let v_h  = unsafe { kv_v_ptr.add(kv_h * hd as usize) };
                // scores_h buffer: offset by head
                let sc_h = unsafe { self.scratch.scores.as_mut_ptr()
                                        .add(hi * (n as usize) * (seq as usize)) };
                let out_h = unsafe { self.scratch.attn_out.as_mut_ptr()
                                         .add(hi * hd as usize) };

                // scores_h[n, seq] = scale × Q_h × K_h^T
                // col-major: [seq, n] = K_h[seq, hd](no T) × Q_h^T[hd, n] (T)
                sgemm_strided(&self.blas,
                    false, true,       // transA=K_h(no T in row-major), transB=Q_h(T)
                    seq, n, hd,        // M=seq, N=n, K=hd
                    scale, 0.0,
                    k_h, kd,           // K_h ptr, lda=kd (stride between K rows)
                    q_h, qd,           // Q_h ptr, ldb=qd (stride between Q rows)
                    sc_h, seq,         // output, ldc=seq
                )?;

                // Softmax over seq dimension (rows of scores_h[n, seq])
                unsafe { self.kernels.softmax_rows(sc_h, n, seq)?; }

                // out_h[n, hd] = scores_h[n, seq] × V_h[seq, hd]
                // col-major: [hd, n] = V_h^T[hd, seq] × scores_h^T[seq, n]
                sgemm_strided(&self.blas,
                    true, false,       // transA=V_h(T), transB=scores(no T)
                    hd, n, seq,        // M=hd, N=n, K=seq
                    1.0, 0.0,
                    v_h, kd,           // V_h ptr, lda=kd
                    sc_h, seq,         // scores ptr, ldb=seq
                    out_h, qd,         // output into attn_out[n, hi*hd..], stride=qd
                )?;
            }

            // ── Output projection: attn_out × wo^T → x (residual) ─────────────
            // x = x + normed_proj (onde normed_proj = attn_out × wo^T)
            // Usa beta=1.0 para acumular direto em x
            gemm_bf16_f32(&self.blas, n, h, qd, 1.0, 1.0,
                self.scratch.attn_out.as_ptr(), layer.wo.dev_ptr(),
                self.scratch.x.as_mut_ptr())?;

            // ── Post-attention RMSNorm ────────────────────────────────────────
            unsafe { self.kernels.rms_norm(
                self.scratch.normed.as_mut_ptr(),
                self.scratch.x.as_ptr(),
                layer.post_attn_norm_w_gpu.as_ptr(),
                n, h, eps,
            )?; }

            // ── MLP: gate_proj, up_proj ──────────────────────────────────────
            gemm_bf16_f32(&self.blas, n, it, h, 1.0, 0.0,
                self.scratch.normed.as_ptr(), layer.wgate.dev_ptr(),
                self.scratch.gate.as_mut_ptr())?;

            gemm_bf16_f32(&self.blas, n, it, h, 1.0, 0.0,
                self.scratch.normed.as_ptr(), layer.wup.dev_ptr(),
                self.scratch.up.as_mut_ptr())?;

            // SwiGLU: gate = silu(gate) * up (in-place em gate)
            unsafe { self.kernels.silu_mul_inplace(
                self.scratch.gate.as_mut_ptr(),
                self.scratch.up.as_ptr(),
                n, it,
            )?; }

            // down_proj: x = x + swiglu × wdown^T (beta=1 acumula em x)
            gemm_bf16_f32(&self.blas, n, h, it, 1.0, 1.0,
                self.scratch.gate.as_ptr(), layer.wdown.dev_ptr(),
                self.scratch.x.as_mut_ptr())?;
        }

        // ── Final RMSNorm ─────────────────────────────────────────────────────
        unsafe { self.kernels.rms_norm(
            self.scratch.normed.as_mut_ptr(),
            self.scratch.x.as_ptr(),
            self.norm_w_gpu.as_ptr(),
            n, h, eps,
        )?; }

        // ── lm_head (apenas o último token) ──────────────────────────────────
        // Envia só o último token: normed[(n-1)*h .. n*h]
        let last_normed = unsafe { self.scratch.normed.as_ptr().add((n as usize - 1) * h as usize) };
        gemm_bf16_f32(&self.blas, 1, vocab, h, 1.0, 0.0,
            last_normed, self.lm_head.dev_ptr(),
            self.scratch.logits.as_mut_ptr())?;

        // ── Download logits: único to_host() de todo o forward ───────────────
        let mut logits = vec![0.0f32; vocab as usize];
        unsafe { hip_ffi::safe::hip_d2h(&mut logits, self.scratch.logits.as_ptr())?; }
        Ok(logits)
    }
}

// ── Leitores safetensors ──────────────────────────────────────────────────────

fn read_bf16_raw(mmap: &memmap2::Mmap, name: &str) -> ModelResult<(Vec<u16>, Vec<usize>)> {
    let (raw, shape) = shard_tensor_bytes(mmap, name)?;
    let bits = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    Ok((bits, shape))
}

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
// Usadas apenas em testes unitários — não chamadas durante inferência GPU.

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
