// ============================================================================
// inference/sampling.rs — Token sampling: temperature, top-k, top-p (nucleus)
// ============================================================================
// Otimizado para zero-allocation no hot path:
//   - SamplingBuffer reutiliza Vec entre tokens (evita 3× alloc/dealloc por token)
//   - Greedy shortcut: argmax sem softmax
//   - Top-K via partial sort O(n) — select_nth_unstable
//   - Top-P via cumsum em buffer pré-ordenado
// ============================================================================

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Hiperparâmetros de amostragem passados em cada requisição de geração.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperatura dos logits. 0.0 ou < 0.001 → greedy argmax.
    pub temperature: f32,
    /// Top-K: mantém apenas os K tokens de maior probabilidade. 0 = desabilitado.
    pub top_k: usize,
    /// Top-P (nucleus): mantém o menor conjunto com cumsum >= top_p. 1.0 = desabilitado.
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, top_p: 0.9 }
    }
}

/// Reusable buffers for sampling — avoids allocation per token.
/// Create once, reuse across all tokens in a generation.
pub struct SamplingBuffer {
    probs: Vec<f32>,
    indexed: Vec<(usize, f32)>,
    order: Vec<usize>,
}

impl SamplingBuffer {
    pub fn new() -> Self {
        Self {
            probs: Vec::new(),
            indexed: Vec::new(),
            order: Vec::new(),
        }
    }

    /// Pre-allocate buffers for a known vocab size.
    pub fn with_vocab_size(vocab_size: usize) -> Self {
        Self {
            probs: Vec::with_capacity(vocab_size),
            indexed: Vec::with_capacity(vocab_size),
            order: Vec::with_capacity(vocab_size),
        }
    }
}

/// Sample the next token using pre-allocated buffers (zero-alloc hot path).
pub fn sample_token_buffered(logits: &[f32], params: &SamplingParams, buf: &mut SamplingBuffer) -> u32 {
    // ── 1. Shortcut greedy ────────────────────────────────────────────────
    if params.temperature < 0.001 {
        return argmax(logits);
    }

    let vocab = logits.len();

    // ── 2. Temperature scale into reused buffer ──────────────────────────
    buf.probs.clear();
    let inv_temp = 1.0 / params.temperature;
    buf.probs.extend(logits.iter().map(|&l| l * inv_temp));

    // ── 3. Softmax ────────────────────────────────────────────────────────
    softmax_inplace(&mut buf.probs);

    // ── 4. Top-K filter ───────────────────────────────────────────────────
    if params.top_k > 0 && params.top_k < vocab {
        buf.indexed.clear();
        buf.indexed.extend(buf.probs.iter().enumerate().map(|(i, &p)| (i, p)));
        // select_nth_unstable: O(n) — coloca o K-ésimo maior na posição top_k-1
        buf.indexed.select_nth_unstable_by(params.top_k - 1, |a, b| b.1.total_cmp(&a.1));
        let kth_prob = buf.indexed[params.top_k - 1].1;
        for p in buf.probs.iter_mut() {
            if *p < kth_prob { *p = 0.0; }
        }
    }

    // ── 5. Top-P (nucleus) filter ─────────────────────────────────────────
    if params.top_p < 1.0 {
        buf.order.clear();
        buf.order.extend(0..vocab);
        buf.order.sort_unstable_by(|&a, &b| buf.probs[b].total_cmp(&buf.probs[a]));
        let mut cumulative = 0.0f32;
        let mut cutoff = false;
        for &idx in &buf.order {
            if cutoff { buf.probs[idx] = 0.0; continue; }
            cumulative += buf.probs[idx];
            if cumulative >= params.top_p { cutoff = true; }
        }
    }

    // ── 6. Renormalize ────────────────────────────────────────────────────
    let sum: f32 = buf.probs.iter().sum();
    if sum <= 0.0 {
        return argmax(logits); // degenerate: fallback greedy
    }
    for p in buf.probs.iter_mut() { *p /= sum; }

    // ── 7. Multinomial sampling ───────────────────────────────────────────
    match WeightedIndex::new(&buf.probs) {
        Ok(dist) => dist.sample(&mut thread_rng()) as u32,
        Err(_) => argmax(logits),
    }
}

/// Legacy API — allocates per call. Use sample_token_buffered for hot path.
pub fn sample_token(logits: &[f32], params: &SamplingParams) -> u32 {
    let mut buf = SamplingBuffer::new();
    sample_token_buffered(logits, params, &mut buf)
}

#[inline]
fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[inline]
fn softmax_inplace(x: &mut [f32]) {
    // Substitui NaN por -inf para não contaminar o resultado
    for v in x.iter_mut() { if v.is_nan() { *v = f32::NEG_INFINITY; } }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    if sum > 0.0 { for v in x.iter_mut() { *v /= sum; } }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        let params = SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
        assert_eq!(sample_token(&logits, &params), 2);
    }

    #[test]
    fn greedy_returns_argmax_buffered() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        let params = SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
        let mut buf = SamplingBuffer::new();
        assert_eq!(sample_token_buffered(&logits, &params, &mut buf), 2);
    }

    #[test]
    fn top_k_1_deterministic() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        let params = SamplingParams { temperature: 1.0, top_k: 1, top_p: 1.0 };
        let mut buf = SamplingBuffer::new();
        for _ in 0..20 {
            assert_eq!(sample_token_buffered(&logits, &params, &mut buf), 2);
        }
    }

    #[test]
    fn dominant_logit_always_selected() {
        let mut logits = vec![0.0f32; 50];
        logits[7] = 100.0;
        let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        let mut buf = SamplingBuffer::new();
        assert_eq!(sample_token_buffered(&logits, &params, &mut buf), 7);
    }

    #[test]
    fn degenerate_no_panic() {
        let logits = vec![-1e10f32; 100];
        let params = SamplingParams { temperature: 0.001, top_k: 0, top_p: 1.0 };
        let mut buf = SamplingBuffer::new();
        let tok = sample_token_buffered(&logits, &params, &mut buf);
        assert!(tok < 100);
    }

    #[test]
    fn buffer_reuse_consistent() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        let params = SamplingParams { temperature: 1.0, top_k: 2, top_p: 0.95 };
        let mut buf = SamplingBuffer::with_vocab_size(4);
        // Call multiple times — buffer is reused
        for _ in 0..10 {
            let tok = sample_token_buffered(&logits, &params, &mut buf);
            assert!(tok < 4);
        }
    }
}
