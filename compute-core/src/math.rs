// ============================================================================
// math.rs — Shared math primitives for LLM inference and training
// ============================================================================
// Consolidated from iara-engine's tracer.rs and sampling.rs.
// All functions are pure, allocation-minimal, and numerically stable.
// ============================================================================

/// Compute stable softmax, returning a new Vec.
///
/// Uses max-subtraction trick to prevent overflow.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter()
        .map(|&x| (x - max).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(1e-10)).collect()
}

/// In-place softmax with NaN sanitization.
///
/// Replaces NaN values with -inf before computing, preventing contamination.
pub fn softmax_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if v.is_nan() { *v = f32::NEG_INFINITY; }
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() { *v /= sum; }
    }
}

/// Compute Shannon entropy of a probability distribution.
///
/// H = -sum(p_i * ln(p_i)) for p_i > 0.
/// Returns 0.0 for empty slices.
pub fn entropy(probs: &[f32]) -> f32 {
    probs.iter()
        .filter(|&p| *p > 1e-10)
        .map(|p| -p * p.ln())
        .sum()
}

/// Get top-K elements from a slice, returning (index, value) pairs sorted descending.
///
/// Uses partial sort (select_nth_unstable) for O(n) average performance.
pub fn top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = data.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

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

/// Compute phase coherence (Kuramoto order parameter) from a hidden state.
///
/// Interprets the hidden state as complex pairs: h[0]+ih[1], h[2]+ih[3], ...
/// Returns a value in [0.0, 1.0]:
/// - 0.0 = completely chaotic, random phases
/// - 1.0 = total synchronization, all phases aligned
pub fn phase_coherence(hidden: &[f32]) -> f32 {
    if hidden.len() < 2 {
        return 0.0;
    }

    let n = hidden.len() / 2;
    let (mut sum_cos, mut sum_sin) = (0.0f32, 0.0f32);

    for k in 0..n {
        let x = hidden[2 * k];
        let y = hidden[2 * k + 1];

        let norm = (x * x + y * y).sqrt().max(1e-8);
        sum_cos += x / norm;
        sum_sin += y / norm;
    }

    let mean_cos = sum_cos / n as f32;
    let mean_sin = sum_sin / n as f32;
    (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_inplace_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_inplace_nan_handling() {
        let mut x = vec![1.0, f32::NAN, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        assert!(!x.iter().any(|v| v.is_nan()));
    }

    #[test]
    fn test_entropy_uniform() {
        let uniform = vec![0.5, 0.5];
        let e = entropy(&uniform);
        assert!((e - 2.0_f32.ln()).abs() < 0.01);
    }

    #[test]
    fn test_entropy_certain() {
        let certain = vec![1.0, 0.0];
        let e = entropy(&certain);
        assert!(e < 0.01);
    }

    #[test]
    fn test_top_k_basic() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let top2 = top_k(&data, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], (4, 5.0));
        assert_eq!(top2[1], (2, 4.0));
    }

    #[test]
    fn test_phase_coherence_aligned() {
        let aligned = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coh = phase_coherence(&aligned);
        assert!((coh - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_phase_coherence_random() {
        let random = vec![0.5, 0.5, -0.3, 0.7, 0.1, -0.9];
        let coh = phase_coherence(&random);
        assert!(coh < 0.8);
    }

    #[test]
    fn test_phase_coherence_empty() {
        assert_eq!(phase_coherence(&[]), 0.0);
        assert_eq!(phase_coherence(&[1.0]), 0.0);
    }
}
