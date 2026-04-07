// ============================================================================
// DoubleGatedConv — LFM2.5-style Short Convolution Block
// ============================================================================
// Implements the core compute primitive from Liquid AI's LFM2.5 architecture:
//   1. RMSNorm(x)
//   2. (B, C, h̃) = Linear(normed, w_expand)         expand d → 3d
//   3. y = σ(B) ⊙ h̃                                  gate 1 (input gating)
//   4. z = DepthwiseConv1D_causal(y, w_conv)          short local context
//   5. output = x + (C ⊙ z) @ w_out                   gate 2 + residual
//
// Key properties:
//   - O(L·d·k) per layer (k = kernel size, typically 4-8)
//   - Zero KV-cache: only needs kernel_size tokens of history
//   - Causal: each token only sees past tokens (autoregressive safe)
//   - 62.5% of LFM2.5 layers are Conv, 37.5% are GQA Attention
//
// Also defines:
//   - LivOperator trait: unifies Conv, Attention, SSM under one abstraction
//   - HybridArchitecture: describes Conv/Attn block sequence
// ============================================================================

use std::f32::consts::PI;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a DoubleGatedConv layer.
#[derive(Debug, Clone)]
pub struct DoubleGatedConvConfig {
    pub hidden_dim: usize,
    pub expand_ratio: usize,   // splits: B, C, h̃ → typically 3
    pub kernel_size: usize,    // depthwise conv kernel (4-8 for LFM2.5)
    pub activation: ConvActivation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvActivation {
    SiLU,
    GELU,
    ReLU,
    None,
}

impl DoubleGatedConvConfig {
    /// LFM2.5 default: hidden_dim=2048, expand=3, kernel=4, SiLU
    pub fn default_lfm(hidden_dim: usize) -> Self {
        Self { hidden_dim, expand_ratio: 3, kernel_size: 4, activation: ConvActivation::SiLU }
    }

    pub fn expanded_dim(&self) -> usize { self.hidden_dim * self.expand_ratio }
}

// ── Weights ───────────────────────────────────────────────────────────────────

/// Weights for a single DoubleGatedConv layer.
pub struct DoubleGatedConvWeights {
    /// Expansion projection: [hidden_dim, 3*hidden_dim]  (B, C, h̃)
    pub w_expand: Vec<f32>,
    pub b_expand: Option<Vec<f32>>,
    /// Depthwise conv kernel: [hidden_dim, kernel_size]
    pub w_conv: Vec<f32>,
    /// Output projection: [hidden_dim, hidden_dim]
    pub w_out: Vec<f32>,
    pub b_out: Option<Vec<f32>>,
    /// RMSNorm weight: [hidden_dim]
    pub norm_weight: Vec<f32>,
    pub config: DoubleGatedConvConfig,
}

impl DoubleGatedConvWeights {
    /// Deterministic pseudo-random init (Xavier scale, no rand dependency).
    pub fn new_random(config: DoubleGatedConvConfig) -> Self {
        let h = config.hidden_dim;
        let exp = config.expanded_dim();
        let k = config.kernel_size;

        let s_exp = (2.0 / (h + exp) as f32).sqrt();
        let s_conv = (2.0 / (h + k) as f32).sqrt();
        let s_out = (2.0 / (2 * h) as f32).sqrt();

        Self {
            w_expand: (0..h * exp).map(|i| (i as f32 * 0.1 + 0.3).sin() * s_exp).collect(),
            b_expand: None,
            w_conv:   (0..h * k).map(|i| (i as f32 * 0.07 + 0.5).cos() * s_conv).collect(),
            w_out:    (0..h * h).map(|i| (i as f32 * 0.03 + 0.9).sin() * s_out).collect(),
            b_out:    None,
            norm_weight: vec![1.0; h],
            config,
        }
    }
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

/// Per-layer activation statistics (mirrors LayerTrace from forward.rs).
#[derive(Debug, Clone, Default)]
pub struct ConvLayerTrace {
    pub layer_idx: usize,
    pub gate1_energy: f32,   // mean(y²)  — input gate activation
    pub gate2_energy: f32,   // mean((C⊙z)²)  — output gate activation
    pub conv_energy: f32,    // mean(z²) — convolution output
    pub time_us: u128,
}

// ── Primitive Ops (CPU, f32) ──────────────────────────────────────────────────

/// RMSNorm: out[t,i] = (x[t,i] / rms(x[t,:])) · weight[i]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    let n_tokens = x.len() / dim;
    let mut out = vec![0.0; x.len()];
    for t in 0..n_tokens {
        let row = &x[t * dim..(t + 1) * dim];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
        let inv = 1.0 / rms;
        for i in 0..dim {
            out[t * dim + i] = row[i] * inv * weight[i];
        }
    }
    out
}

/// Linear: x [n_tokens, in_dim] @ w [in_dim, out_dim] + bias → [n_tokens, out_dim]
fn linear(
    x: &[f32], w: &[f32], bias: Option<&[f32]>,
    n_tokens: usize, in_dim: usize, out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; n_tokens * out_dim];
    for t in 0..n_tokens {
        for j in 0..out_dim {
            let mut s = 0.0f32;
            for i in 0..in_dim {
                s += x[t * in_dim + i] * w[i * out_dim + j];
            }
            if let Some(b) = bias { s += b[j]; }
            out[t * out_dim + j] = s;
        }
    }
    out
}

/// Depthwise causal 1D convolution per channel.
/// x: [n_tokens, channels], kernel: [channels, kernel_size]
/// Causal: token t only sees tokens [t-k+1 .. t].
fn depthwise_conv1d_causal(
    x: &[f32], kernel: &[f32],
    n_tokens: usize, channels: usize, kernel_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; n_tokens * channels];
    for t in 0..n_tokens {
        for c in 0..channels {
            let mut s = 0.0f32;
            for k in 0..kernel_size {
                let src_t = t as isize - k as isize;
                if src_t >= 0 {
                    s += x[src_t as usize * channels + c] * kernel[c * kernel_size + k];
                }
            }
            out[t * channels + c] = s;
        }
    }
    out
}

fn elementwise_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

/// SiLU (Swish): x · σ(x)
pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// GELU (approx): x · Φ(x)
pub fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn apply_activation(x: &mut [f32], act: ConvActivation) {
    match act {
        ConvActivation::SiLU => x.iter_mut().for_each(|v| *v = silu(*v)),
        ConvActivation::GELU => x.iter_mut().for_each(|v| *v = gelu(*v)),
        ConvActivation::ReLU => x.iter_mut().for_each(|v| *v = v.max(0.0)),
        ConvActivation::None => {}
    }
}

// ── DoubleGatedConv Forward ───────────────────────────────────────────────────

/// Forward pass of a single DoubleGatedConv layer (LFM2.5 core block).
///
/// ```text
/// x ─── RMSNorm → expand → split(B,C,h̃) → σ(B)⊙h̃ → Conv1D → C⊙z → project ─┐
/// │                                                                              │
/// └──────────────────────────────────────────── + ◄──────────────────────────────┘
/// ```
pub fn double_gated_conv_forward(
    x: &[f32],
    weights: &DoubleGatedConvWeights,
) -> (Vec<f32>, ConvLayerTrace) {
    let start = std::time::Instant::now();
    let h = weights.config.hidden_dim;
    let n_tokens = x.len() / h;
    let k = weights.config.kernel_size;

    // 1. RMSNorm
    let normed = rms_norm(x, &weights.norm_weight, 1e-6);

    // 2. Expand: normed → [n_tokens, 3h]
    let expanded = linear(
        &normed, &weights.w_expand, weights.b_expand.as_deref(),
        n_tokens, h, 3 * h,
    );

    // 3. Split into B (gate1), C (gate2), h̃ (value)
    let mut b_gate = vec![0.0; n_tokens * h];
    let mut c_gate = vec![0.0; n_tokens * h];
    let mut h_tilde = vec![0.0; n_tokens * h];
    for t in 0..n_tokens {
        let row = &expanded[t * 3 * h..(t + 1) * 3 * h];
        b_gate[t * h..(t + 1) * h].copy_from_slice(&row[..h]);
        c_gate[t * h..(t + 1) * h].copy_from_slice(&row[h..2 * h]);
        h_tilde[t * h..(t + 1) * h].copy_from_slice(&row[2 * h..]);
    }

    // 4. Gate 1: y = σ(B) ⊙ h̃
    apply_activation(&mut b_gate, weights.config.activation);
    let y = elementwise_mul(&b_gate, &h_tilde);
    let gate1_energy = y.iter().map(|v| v * v).sum::<f32>() / y.len().max(1) as f32;

    // 5. Causal depthwise conv1d
    let z = depthwise_conv1d_causal(&y, &weights.w_conv, n_tokens, h, k);
    let conv_energy = z.iter().map(|v| v * v).sum::<f32>() / z.len().max(1) as f32;

    // 6. Gate 2: gated = C ⊙ z, then project
    let gated = elementwise_mul(&c_gate, &z);
    let gate2_energy = gated.iter().map(|v| v * v).sum::<f32>() / gated.len().max(1) as f32;

    let projected = linear(
        &gated, &weights.w_out, weights.b_out.as_deref(),
        n_tokens, h, h,
    );

    // 7. Residual: output = x + projected
    let output: Vec<f32> = x.iter().zip(projected.iter()).map(|(&a, &b)| a + b).collect();

    let trace = ConvLayerTrace {
        layer_idx: 0,
        gate1_energy,
        gate2_energy,
        conv_energy,
        time_us: start.elapsed().as_micros(),
    };
    (output, trace)
}

// ── HybridArchitecture ────────────────────────────────────────────────────────

/// Which block type a layer uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockType {
    /// DoubleGatedConv: local context, O(Lk), zero KV-cache.
    Conv,
    /// GQA Attention: global context, O(L²), KV-cache per layer.
    Attention,
}

/// Describes the full block sequence for a hybrid model.
///
/// ```text
/// LFM2.5-1.2B (16 layers):  C C A C C A C C C A C C A C C A
/// IARA-hybrid (36 layers):  C C C A C C A C C C A C C A C C C A C C A C C C A C C A C C C A C C A C A
/// ```
#[derive(Debug, Clone)]
pub struct HybridArchitecture {
    pub name: String,
    pub blocks: Vec<BlockType>,
    pub hidden_dim: usize,
    pub conv_kernel_size: usize,
    pub n_kv_heads: usize,
    pub n_q_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
}

impl HybridArchitecture {
    /// LFM2.5-style: ~37.5% attention, rest conv, interleaved.
    pub fn lfm_style(n_layers: usize, hidden_dim: usize) -> Self {
        let n_attn = (n_layers as f32 * 0.375).round() as usize;
        let spacing = n_layers as f32 / (n_attn + 1) as f32;
        let mut blocks = vec![BlockType::Conv; n_layers];
        for i in 0..n_attn {
            let pos = ((i + 1) as f32 * spacing) as usize;
            if pos < n_layers { blocks[pos] = BlockType::Attention; }
        }
        Self {
            name: "lfm-hybrid".into(), blocks, hidden_dim,
            conv_kernel_size: 4, n_kv_heads: 8, n_q_heads: 32,
            head_dim: hidden_dim / 32, intermediate_dim: hidden_dim * 4,
        }
    }

    /// Pure transformer (Qwen3/Llama baseline).
    pub fn transformer(n_layers: usize, hidden_dim: usize) -> Self {
        Self {
            name: "transformer".into(),
            blocks: vec![BlockType::Attention; n_layers],
            hidden_dim, conv_kernel_size: 0, n_kv_heads: 8, n_q_heads: 32,
            head_dim: hidden_dim / 32, intermediate_dim: hidden_dim * 4,
        }
    }

    /// All conv, zero attention (extreme edge: no global context).
    pub fn conv_only(n_layers: usize, hidden_dim: usize) -> Self {
        Self {
            name: "conv-only".into(),
            blocks: vec![BlockType::Conv; n_layers],
            hidden_dim, conv_kernel_size: 8, n_kv_heads: 0, n_q_heads: 0,
            head_dim: 0, intermediate_dim: hidden_dim * 4,
        }
    }

    pub fn n_conv_layers(&self) -> usize {
        self.blocks.iter().filter(|&&b| b == BlockType::Conv).count()
    }

    pub fn n_attn_layers(&self) -> usize {
        self.blocks.iter().filter(|&&b| b == BlockType::Attention).count()
    }

    /// KV-cache bytes per sequence position (BF16, K+V).
    pub fn kv_cache_bytes_per_token(&self) -> usize {
        self.n_attn_layers() * 2 * self.n_kv_heads * self.head_dim * 2
    }

    /// Fraction of KV-cache eliminated vs full transformer (0.0 .. 1.0).
    pub fn kv_cache_reduction(&self) -> f32 {
        let total = self.blocks.len();
        if total == 0 { return 0.0; }
        1.0 - (self.n_attn_layers() as f32 / total as f32)
    }

    /// Estimated FLOPs per token → (conv_total, attn_total).
    pub fn estimate_flops_per_token(&self, seq_len: usize) -> (f64, f64) {
        let h = self.hidden_dim as f64;
        let k = self.conv_kernel_size as f64;
        let heads = self.n_q_heads as f64;
        let hd = self.head_dim as f64;
        let inter = self.intermediate_dim as f64;
        let s = seq_len as f64;

        let conv_per = 2.0 * h * 3.0 * h   // expand
            + h * k                           // depthwise conv
            + 2.0 * h * h                    // output projection
            + 2.0 * h * inter;              // MLP

        let attn_per = 2.0 * h * (heads * hd * 3.0)  // QKV
            + 2.0 * heads * hd * s                     // Q·K scores
            + 2.0 * heads * hd * s                     // attn × V
            + 2.0 * h * h                              // output proj
            + 2.0 * h * inter;                        // MLP

        (conv_per * self.n_conv_layers() as f64,
         attn_per * self.n_attn_layers() as f64)
    }
}

// ── LIV Abstraction (Linear Input-Varying) ────────────────────────────────────

/// LIV: y = T(x) · x, where T(x) is input-dependent.
///
/// Unifies under one trait:
///   - Attention: T(x) = softmax(Q(x)·K(x)^T / √d)
///   - GatedConv: T(x) = diag(σ(Wg·x)) · Conv_k
///   - SSM:       T(x) = A(x)  (state-dependent transition)
///   - Linear RNN: T(x) = diag(forget_gate(x)) · W_h
pub trait LivOperator: Send + Sync {
    fn name(&self) -> &str;
    /// Apply the LIV transform: x → y.
    fn forward(&self, x: &[f32], n_tokens: usize, hidden_dim: usize) -> Vec<f32>;
    /// Bytes of state per sequence position (KV-cache, RNN state, conv history).
    fn memory_per_token(&self) -> usize;
    /// Can this be used autoregressively?
    fn is_causal(&self) -> bool;
}

/// LIV-Conv: DoubleGatedConv wrapped as a LivOperator.
pub struct LivConv {
    pub weights: DoubleGatedConvWeights,
}

impl LivOperator for LivConv {
    fn name(&self) -> &str { "LIV-Conv" }

    fn forward(&self, x: &[f32], _n_tokens: usize, _hidden_dim: usize) -> Vec<f32> {
        double_gated_conv_forward(x, &self.weights).0
    }

    fn memory_per_token(&self) -> usize {
        // Only needs (kernel_size - 1) tokens of history × hidden_dim × f32
        (self.weights.config.kernel_size - 1) * self.weights.config.hidden_dim * 4
    }

    fn is_causal(&self) -> bool { true }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_unit() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let out = rms_norm(&x, &w, 1e-6);
        let rms = (30.0f32 / 4.0).sqrt();
        for i in 0..4 {
            assert!((out[i] - x[i] / rms).abs() < 1e-5);
        }
    }

    #[test]
    fn test_depthwise_conv1d_identity() {
        // kernel = [1, 0, 0, 0] → identity
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens, 3 channels
        let kernel = vec![1.0, 0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0, 0.0,
                          1.0, 0.0, 0.0, 0.0];
        let out = depthwise_conv1d_causal(&x, &kernel, 2, 3, 4);
        for i in 0..6 { assert!((out[i] - x[i]).abs() < 1e-6); }
    }

    #[test]
    fn test_depthwise_conv1d_causal_sum() {
        // kernel = [1, 1] → sum(current, previous)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens, 2 ch
        let kernel = vec![1.0, 1.0, 1.0, 1.0];
        let out = depthwise_conv1d_causal(&x, &kernel, 3, 2, 2);
        assert!((out[0] - 1.0).abs() < 1e-6);  // t=0 ch0: 1×1 (no prev)
        assert!((out[1] - 2.0).abs() < 1e-6);  // t=0 ch1
        assert!((out[2] - 4.0).abs() < 1e-6);  // t=1 ch0: 3+1
        assert!((out[3] - 6.0).abs() < 1e-6);  // t=1 ch1: 4+2
        assert!((out[4] - 8.0).abs() < 1e-6);  // t=2 ch0: 5+3
        assert!((out[5] - 10.0).abs() < 1e-6); // t=2 ch1: 6+4
    }

    #[test]
    fn test_conv_forward_shape() {
        let cfg = DoubleGatedConvConfig::default_lfm(64);
        let w = DoubleGatedConvWeights::new_random(cfg);
        let x = vec![0.1; 4 * 64]; // 4 tokens
        let (out, tr) = double_gated_conv_forward(&x, &w);
        assert_eq!(out.len(), 4 * 64);
        assert!(tr.gate1_energy >= 0.0);
        assert!(tr.time_us > 0);
    }

    #[test]
    fn test_conv_forward_residual_adds() {
        let cfg = DoubleGatedConvConfig::default_lfm(64);
        let w = DoubleGatedConvWeights::new_random(cfg);
        // Use varied input so activations aren't all identical
        let x: Vec<f32> = (0..3 * 64).map(|i| (i as f32 * 0.13).sin()).collect();
        let (out, _) = double_gated_conv_forward(&x, &w);
        // At least some elements should differ from input (residual adds projection)
        let diffs: usize = out.iter().zip(x.iter())
            .filter(|(&o, &i)| (o - i).abs() > 1e-6)
            .count();
        assert!(diffs > 0, "output should differ from input somewhere");
    }

    #[test]
    fn test_hybrid_lfm_block_distribution() {
        let arch = HybridArchitecture::lfm_style(16, 2048);
        assert_eq!(arch.blocks.len(), 16);
        assert_eq!(arch.n_attn_layers(), 6);
        assert_eq!(arch.n_conv_layers(), 10);
        assert!(arch.kv_cache_reduction() > 0.5);
    }

    #[test]
    fn test_hybrid_kv_savings() {
        let tf = HybridArchitecture::transformer(36, 4096);
        let hy = HybridArchitecture::lfm_style(36, 4096);
        let co = HybridArchitecture::conv_only(36, 4096);
        assert_eq!(tf.kv_cache_reduction(), 0.0);
        assert!(hy.kv_cache_reduction() > 0.5);
        assert_eq!(co.kv_cache_reduction(), 1.0);
    }

    #[test]
    fn test_conv_cheaper_than_attention_at_long_seq() {
        let arch = HybridArchitecture::lfm_style(16, 2048);
        let (cf, af) = arch.estimate_flops_per_token(1024);
        let conv_per = cf / arch.n_conv_layers() as f64;
        let attn_per = af / arch.n_attn_layers() as f64;
        assert!(conv_per < attn_per, "conv={:.0} attn={:.0}", conv_per, attn_per);
    }

    #[test]
    fn test_liv_conv_trait() {
        let cfg = DoubleGatedConvConfig::default_lfm(32);
        let w = DoubleGatedConvWeights::new_random(cfg);
        let liv = LivConv { weights: w };
        assert!(liv.is_causal());
        assert!(liv.memory_per_token() > 0);
        let x = vec![0.5; 2 * 32];
        let out = liv.forward(&x, 2, 32);
        assert_eq!(out.len(), 2 * 32);
    }

    #[test]
    fn test_silu_values() {
        assert!((silu(0.0)).abs() < 1e-6);
        assert!(silu(1.0) > 0.7);
        assert!(silu(-10.0).abs() < 0.001);
    }

    #[test]
    fn test_elementwise_mul() {
        assert_eq!(elementwise_mul(&[1.0, 2.0], &[3.0, 4.0]), vec![3.0, 8.0]);
    }
}
