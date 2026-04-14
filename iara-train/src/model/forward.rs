// ============================================================================
// forward.rs — Full CPU forward pass for Qwen2.5 (training mode)
// ============================================================================
// Pure f32 CPU implementation. No KV cache — runs full sequence at once.
// Matches iara-engine's GPU forward pass logic but computes in f32 on CPU.
//
// Architecture: Qwen2.5-1.5B
//   - 28 transformer layers
//   - hidden_size = 1536
//   - num_heads = 12, num_kv_heads = 2, head_dim = 128
//   - intermediate_size = 8960
//   - vocab_size = 151936
//   - GQA (Grouped Query Attention)
//   - RoPE positional encoding
//   - RMSNorm (pre-norm architecture)
//   - SwiGLU MLP (gate * silu + up, then down projection)
//   - Q/K/V have biases, O/MLP do not
// ============================================================================

use crate::model::weights::ModelWeights;
use std::time::Instant;

/// Result of a forward pass.
pub struct ForwardResult {
    /// Logits [seq_len * vocab_size] — flat row-major.
    pub logits: Vec<f32>,
    /// Final hidden states [seq_len * hidden_size] (for LoRA backward).
    pub hidden: Vec<f32>,
    /// Per-layer hidden states BEFORE each layer (for LoRA grad computation).
    /// Layer i contains the hidden state that was input to layer i's attention.
    /// Length = num_layers, each [seq_len * hidden_size].
    pub layer_inputs: Vec<Vec<f32>>,
    /// Time taken for the forward pass.
    pub elapsed_ms: f32,
}

// ── Math primitives ──────────────────────────────────────────────────────────

/// RMSNorm: out[i] = x[i] / rms(x) * w[i]
/// x and w are [size], operates row-wise on [seq_len, size].
fn rms_norm_inplace(x: &mut [f32], w: &[f32], eps: f32, seq_len: usize, size: usize) {
    for t in 0..seq_len {
        let row = &mut x[t * size..(t + 1) * size];
        let rms = {
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / size as f32;
            (ss + eps).sqrt()
        };
        for (xi, wi) in row.iter_mut().zip(w.iter()) {
            *xi = (*xi / rms) * wi;
        }
    }
}

/// Matrix multiply: C = A @ B
/// A: [m, k], B: [k, n], C: [m, n]
fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Simple loop — for correctness first, not speed
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

/// Matrix multiply with bias add: C = A @ B^T + bias
/// A: [m, k], B: [n, k] (stored as [out, in] — PyTorch convention), bias: [n]
/// C: [m, n]
fn matmul_transb_bias(
    a: &[f32], b: &[f32], bias: Option<&[f32]>,
    c: &mut [f32], m: usize, k: usize, n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                // b[j, l] = b[j * k + l]
                s += a[i * k + l] * b[j * k + l];
            }
            if let Some(bias) = bias {
                s += bias[j];
            }
            c[i * n + j] = s;
        }
    }
}

/// SiLU activation: x * sigmoid(x)
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply RoPE to a [seq_len, n_heads, head_dim] tensor (stored as [seq_len, n_heads * head_dim]).
/// Modifies in-place.
fn apply_rope(
    x: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_offset: usize,
) {
    let half_dim = head_dim / 2;
    for t in 0..seq_len {
        let pos = (t + pos_offset) as f32;
        for h in 0..n_heads {
            let offset = t * (n_heads * head_dim) + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let x0 = x[offset + i];
                let x1 = x[offset + i + half_dim];
                x[offset + i]            = x0 * cos_a - x1 * sin_a;
                x[offset + i + half_dim] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

/// Causal self-attention (GQA).
/// q: [seq_len, n_heads * head_dim]
/// k: [seq_len, n_kv_heads * head_dim]
/// v: [seq_len, n_kv_heads * head_dim]
/// out: [seq_len, n_heads * head_dim]
fn attention(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let n_rep = n_heads / n_kv_heads;  // GQA repeat factor
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let kv_h = h / n_rep;  // which KV head this Q head uses

        // Compute attention scores: [seq_len, seq_len]
        // scores[t, s] = dot(q[t, h], k[s, kv_h]) * scale
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for t in 0..seq_len {
            let q_off = t * (n_heads * head_dim) + h * head_dim;
            for s in 0..=t {  // causal: only attend to s <= t
                let k_off = s * (n_kv_heads * head_dim) + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                scores[t * seq_len + s] = dot * scale;
            }
            // Fill future positions with -inf (already 0 from vec init, set explicitly)
            for s in (t + 1)..seq_len {
                scores[t * seq_len + s] = f32::NEG_INFINITY;
            }
        }

        // Softmax over last dim (per query token)
        for t in 0..seq_len {
            let row = &mut scores[t * seq_len..(t + 1) * seq_len];
            let max_val = row[..=t].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in 0..=t {
                row[s] = (row[s] - max_val).exp();
                sum += row[s];
            }
            for s in 0..=t {
                row[s] /= sum;
            }
            for s in (t + 1)..seq_len {
                row[s] = 0.0;
            }
        }

        // Weighted sum of values
        for t in 0..seq_len {
            let out_off = t * (n_heads * head_dim) + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for s in 0..=t {
                    let v_off = s * (n_kv_heads * head_dim) + kv_h * head_dim;
                    acc += scores[t * seq_len + s] * v[v_off + d];
                }
                out[out_off + d] = acc;
            }
        }
    }
}

// ── Main forward pass ────────────────────────────────────────────────────────

/// Run a full forward pass through the Qwen2.5 model.
///
/// Returns logits [seq_len * vocab_size] and auxiliary data for training.
/// `collect_layer_inputs` must be true to populate `layer_inputs` in the result
/// (needed for LoRA backward pass).
pub fn forward(
    weights: &ModelWeights,
    input_ids: &[u32],
    collect_layer_inputs: bool,
) -> ForwardResult {
    let t0 = Instant::now();

    let cfg = &weights.config;
    let seq_len = input_ids.len();
    let hidden_size = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let inter_size = cfg.intermediate_size;
    let vocab_size = cfg.vocab_size;
    let n_layers = cfg.num_hidden_layers;
    let eps = cfg.rms_norm_eps as f32;
    let rope_theta = cfg.rope_theta as f32;

    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // 1. Embedding lookup: hidden = embed_tokens[input_ids]
    let embed_w = weights.get("model.embed_tokens.weight");
    let mut hidden = vec![0.0f32; seq_len * hidden_size];
    for (t, &tok) in input_ids.iter().enumerate() {
        let src = &embed_w[tok as usize * hidden_size..(tok as usize + 1) * hidden_size];
        hidden[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(src);
    }

    let mut layer_inputs: Vec<Vec<f32>> = Vec::new();

    // Scratchpad buffers
    let mut normed    = vec![0.0f32; seq_len * hidden_size];
    let mut q_buf     = vec![0.0f32; seq_len * q_dim];
    let mut k_buf     = vec![0.0f32; seq_len * kv_dim];
    let mut v_buf     = vec![0.0f32; seq_len * kv_dim];
    let mut attn_out  = vec![0.0f32; seq_len * q_dim];
    let mut o_out     = vec![0.0f32; seq_len * hidden_size];
    let mut gate_buf  = vec![0.0f32; seq_len * inter_size];
    let mut up_buf    = vec![0.0f32; seq_len * inter_size];
    let mut mlp_out   = vec![0.0f32; seq_len * hidden_size];

    // 2. Transformer layers
    for layer_idx in 0..n_layers {
        let p = format!("model.layers.{}", layer_idx);

        // Optionally save the input to this layer for LoRA backward
        if collect_layer_inputs {
            layer_inputs.push(hidden.clone());
        }

        // ── Pre-attention RMSNorm ──────────────────────────────────────────
        let norm1_w = weights.get(&format!("{p}.input_layernorm.weight"));
        normed.copy_from_slice(&hidden);
        rms_norm_inplace(&mut normed, norm1_w, eps, seq_len, hidden_size);

        // ── Q, K, V projections ───────────────────────────────────────────
        let wq = weights.get(&format!("{p}.self_attn.q_proj.weight"));
        let wk = weights.get(&format!("{p}.self_attn.k_proj.weight"));
        let wv = weights.get(&format!("{p}.self_attn.v_proj.weight"));
        let bq = weights.get_opt(&format!("{p}.self_attn.q_proj.bias"));
        let bk = weights.get_opt(&format!("{p}.self_attn.k_proj.bias"));
        let bv = weights.get_opt(&format!("{p}.self_attn.v_proj.bias"));

        matmul_transb_bias(&normed, wq, bq, &mut q_buf, seq_len, hidden_size, q_dim);
        matmul_transb_bias(&normed, wk, bk, &mut k_buf, seq_len, hidden_size, kv_dim);
        matmul_transb_bias(&normed, wv, bv, &mut v_buf, seq_len, hidden_size, kv_dim);

        // ── RoPE ──────────────────────────────────────────────────────────
        apply_rope(&mut q_buf, seq_len, n_heads,    head_dim, rope_theta, 0);
        apply_rope(&mut k_buf, seq_len, n_kv_heads, head_dim, rope_theta, 0);

        // ── Causal attention ──────────────────────────────────────────────
        attention(&q_buf, &k_buf, &v_buf, &mut attn_out,
                  seq_len, n_heads, n_kv_heads, head_dim);

        // ── O projection ──────────────────────────────────────────────────
        let wo = weights.get(&format!("{p}.self_attn.o_proj.weight"));
        matmul_transb_bias(&attn_out, wo, None, &mut o_out, seq_len, q_dim, hidden_size);

        // ── Residual add ──────────────────────────────────────────────────
        for i in 0..hidden.len() {
            hidden[i] += o_out[i];
        }

        // ── Post-attention RMSNorm ─────────────────────────────────────────
        let norm2_w = weights.get(&format!("{p}.post_attention_layernorm.weight"));
        normed.copy_from_slice(&hidden);
        rms_norm_inplace(&mut normed, norm2_w, eps, seq_len, hidden_size);

        // ── MLP: SwiGLU ───────────────────────────────────────────────────
        let wgate = weights.get(&format!("{p}.mlp.gate_proj.weight"));
        let wup   = weights.get(&format!("{p}.mlp.up_proj.weight"));
        let wdown = weights.get(&format!("{p}.mlp.down_proj.weight"));

        matmul_transb_bias(&normed, wgate, None, &mut gate_buf, seq_len, hidden_size, inter_size);
        matmul_transb_bias(&normed, wup,   None, &mut up_buf,   seq_len, hidden_size, inter_size);

        // gate = silu(gate) * up
        for i in 0..gate_buf.len() {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }

        matmul_transb_bias(&gate_buf, wdown, None, &mut mlp_out, seq_len, inter_size, hidden_size);

        // ── Residual add ──────────────────────────────────────────────────
        for i in 0..hidden.len() {
            hidden[i] += mlp_out[i];
        }
    }

    // 3. Final RMSNorm
    let final_norm_w = weights.get("model.norm.weight");
    rms_norm_inplace(&mut hidden, final_norm_w, eps, seq_len, hidden_size);

    // 4. LM head: logits [seq_len, vocab_size]
    // Qwen2.5-1.5B uses weight tying: lm_head.weight = embed_tokens.weight^T
    // lm_head weight is [vocab_size, hidden_size] (same as embed_tokens.weight)
    let lm_head_w = weights.get_opt("lm_head.weight")
        .unwrap_or_else(|| weights.get("model.embed_tokens.weight"));

    let mut logits = vec![0.0f32; seq_len * vocab_size];
    matmul_transb_bias(&hidden, lm_head_w, None, &mut logits, seq_len, hidden_size, vocab_size);

    let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;

    ForwardResult {
        logits,
        hidden,
        layer_inputs,
        elapsed_ms,
    }
}

// ── Top-k sampling helper ────────────────────────────────────────────────────

/// Extract logits for the last token and return top-k (token_id, probability) pairs.
pub fn top_k_last_token(logits: &[f32], vocab_size: usize, k: usize) -> Vec<(u32, f32)> {
    let seq_len = logits.len() / vocab_size;
    let last_row = &logits[(seq_len - 1) * vocab_size..];

    // Softmax
    let max_val = last_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = last_row.iter().map(|&x| (x - max_val).exp()).sum();
    let probs: Vec<f32> = last_row.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();

    // Top-k
    let mut indexed: Vec<(u32, f32)> = probs.iter().enumerate()
        .map(|(i, &p)| (i as u32, p))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}
