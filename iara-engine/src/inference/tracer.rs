// ============================================================================
// inference/tracer.rs — Deep inference tracer for per-token analysis
// ============================================================================
// Captures full path a token takes through the model during forward pass.
// Instruments every layer with attention entropy, MLP activation rates,
// phase coherence, and other rich telemetry.
//
// Zero overhead when trace_deep=false (conditional compilation + feature gates)
// ============================================================================

use serde::{Deserialize, Serialize};

// ── Layer-level deep trace ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDeepTrace {
    pub layer_idx: usize,

    /// L2 norm of hidden state entering the layer (after input norm)
    pub hidden_norm: f32,

    /// Shannon entropy of attention scores after softmax.
    /// H = -sum(p_i * log(p_i)) where p_i are attention weights.
    /// Range [0.0, log(seq_len)]:
    ///   0.0 = completely focused attention (one token gets all weight)
    ///   log(seq_len) = perfectly diffuse attention (uniform)
    pub attn_entropy: f32,

    /// Top-5 positions that received highest attention weights.
    /// Each element is (position_index, attention_weight).
    pub attn_top_positions: Vec<(usize, f32)>,

    /// Percentage of MLP neurons that are non-zero after SiLU gating.
    /// Reveals sparsity: high % = dense activation, low % = sparse.
    pub mlp_activation_rate: f32,

    /// Top-10 most activated MLP neurons by absolute value.
    /// Each element is (neuron_index, activation_value).
    pub mlp_top_neurons: Vec<(usize, f32)>,

    /// L2 norm of (layer_output - layer_input), measuring residual delta.
    /// Shows how much this layer changed the hidden state.
    pub residual_delta: f32,

    /// Kuramoto order parameter of hidden state.
    /// Interprets hidden state as complex pairs: h[0]+ih[1], h[2]+ih[3], ...
    /// Range [0.0, 1.0]:
    ///   0.0 = completely chaotic, random phases
    ///   1.0 = total synchronization, phases aligned
    pub phase_coherence: f32,

    /// Microseconds spent in this layer (estimated)
    pub time_us: u128,
}

// ── Final output deep trace ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputDeepTrace {
    /// Shannon entropy of final logit distribution.
    /// H = -sum(p_i * log(p_i)) where p_i = softmax(logits).
    /// Range [0.0, log(vocab_size)]:
    ///   0.0 = completely certain (one token has all probability)
    ///   log(vocab_size) = completely uncertain (uniform)
    pub logit_entropy: f32,

    /// Top-10 token candidates with their probabilities.
    /// Each element is (token_id, probability).
    pub top_tokens: Vec<(u32, f32)>,

    /// Probability of the chosen/generated token.
    pub confidence: f32,

    /// Index of the chosen token in the top-10 (or -1 if not in top-10)
    pub chosen_token_rank: i32,
}

// ── Complete per-token trace ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepTrace {
    /// Token that was generated
    pub token_id: u32,
    pub token_str: String,

    /// Per-layer traces
    pub layers: Vec<LayerDeepTrace>,

    /// Final logit and probability analysis
    pub output: OutputDeepTrace,

    /// Total forward pass time in milliseconds
    pub time_ms: u128,
}

// ── Helper functions for computing metrics ────────────────────────────────

/// Compute Shannon entropy of a probability distribution
pub fn entropy(probs: &[f32]) -> f32 {
    probs.iter()
        .filter(|&p| *p > 1e-10)
        .map(|p| -p * p.ln())
        .sum()
}

/// Get top-K elements from a slice, returning indices and values
pub fn top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = data.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    // Partial sort to find top K
    let n = k.min(indexed.len());
    if n > 0 {
        indexed.select_nth_unstable_by(n - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result: Vec<_> = indexed[..n].to_vec();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    } else {
        vec![]
    }
}

/// Compute softmax of logits (numerically stable)
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter()
        .map(|&x| (x - max).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(1e-10)).collect()
}

/// Compute phase coherence (Kuramoto order parameter) from hidden state
/// Interprets hidden state as complex pairs: h[0]+ih[1], h[2]+ih[3], ...
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy() {
        // Uniform distribution: maximum entropy
        let uniform = vec![0.5, 0.5];
        let e = entropy(&uniform);
        assert!((e - 2.0_f32.ln()).abs() < 0.01, "Uniform entropy should be ln(2)");

        // Certain distribution: zero entropy
        let certain = vec![1.0, 0.0];
        let e = entropy(&certain);
        assert!(e < 0.01, "Certain distribution should have ~0 entropy");
    }

    #[test]
    fn test_top_k() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let top2 = top_k(&data, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], (4, 5.0));
        assert_eq!(top2[1], (2, 4.0));
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Probabilities should sum to ~1");
    }

    #[test]
    fn test_phase_coherence() {
        // Perfectly aligned phases: all vectors point in same direction
        let aligned = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coh = phase_coherence(&aligned);
        assert!((coh - 1.0).abs() < 0.01, "Aligned phases should give coherence ~1");

        // Random phases: should be low
        let random = vec![0.5, 0.5, -0.3, 0.7, 0.1, -0.9];
        let coh = phase_coherence(&random);
        assert!(coh < 0.8, "Random phases should give lower coherence");
    }
}
