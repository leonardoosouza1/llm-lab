// ============================================================================
// qwen/sampling.rs — Amostragem de tokens: temperatura, top-k, top-p (núcleo)
// ============================================================================
// Fluxo:
//   1. temperature < 0.001 → argmax (greedy)
//   2. Escala logits por 1/T
//   3. Softmax → distribuição de probabilidade
//   4. Top-K: zera tudo abaixo do K-ésimo maior
//   5. Top-P: zera tokens após cumsum >= top_p
//   6. Renormaliza
//   7. Amostra via WeightedIndex (rand)
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

/// Amostra o próximo token a partir de logits brutos.
pub fn sample_token(logits: &[f32], params: &SamplingParams) -> u32 {
    // ── 1. Shortcut greedy ────────────────────────────────────────────────
    if params.temperature < 0.001 {
        return argmax(logits);
    }

    // ── 2. Escala por temperatura ─────────────────────────────────────────
    let inv_temp = 1.0 / params.temperature;
    let mut probs: Vec<f32> = logits.iter().map(|&l| l * inv_temp).collect();

    // ── 3. Softmax ────────────────────────────────────────────────────────
    softmax_inplace(&mut probs);

    // ── 4. Filtro Top-K ───────────────────────────────────────────────────
    if params.top_k > 0 && params.top_k < probs.len() {
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        // select_nth_unstable: O(n) — coloca o K-ésimo maior na posição top_k-1
        indexed.select_nth_unstable_by(params.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        let kth_prob = indexed[params.top_k - 1].1;
        for p in probs.iter_mut() {
            if *p < kth_prob { *p = 0.0; }
        }
    }

    // ── 5. Filtro Top-P (nucleus) ─────────────────────────────────────────
    if params.top_p < 1.0 {
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cumulative = 0.0f32;
        let mut cutoff = false;
        for &idx in &order {
            if cutoff { probs[idx] = 0.0; continue; }
            cumulative += probs[idx];
            if cumulative >= params.top_p { cutoff = true; }
        }
    }

    // ── 6. Renormaliza ────────────────────────────────────────────────────
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return argmax(logits); // caso degenerado: volta ao greedy
    }
    for p in probs.iter_mut() { *p /= sum; }

    // ── 7. Amostragem multinomial ─────────────────────────────────────────
    match WeightedIndex::new(&probs) {
        Ok(dist) => dist.sample(&mut thread_rng()) as u32,
        Err(_) => argmax(logits),
    }
}

#[inline]
fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[inline]
fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in x.iter_mut() { *v /= sum; }
}

// ── Testes ────────────────────────────────────────────────────────────────────

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
    fn top_k_1_deterministic() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        let params = SamplingParams { temperature: 1.0, top_k: 1, top_p: 1.0 };
        for _ in 0..20 {
            assert_eq!(sample_token(&logits, &params), 2);
        }
    }

    #[test]
    fn dominant_logit_always_selected() {
        let mut logits = vec![0.0f32; 50];
        logits[7] = 100.0;
        let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        assert_eq!(sample_token(&logits, &params), 7);
    }

    #[test]
    fn degenerate_no_panic() {
        let logits = vec![-1e10f32; 100];
        let params = SamplingParams { temperature: 0.001, top_k: 0, top_p: 1.0 };
        let tok = sample_token(&logits, &params);
        assert!(tok < 100);
    }
}
