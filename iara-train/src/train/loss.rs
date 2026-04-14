//! Loss functions for causal language model training.
//!
//! Native implementations (no burn dependency).
//! These operate on raw f32 slices and will be replaced with GPU kernels.

/// Cross-entropy loss for next-token prediction.
///
/// `logits`: flat slice of shape [seq_len * vocab_size]
/// `targets`: token IDs of shape [seq_len]
/// `vocab_size`: vocabulary size
///
/// Returns (scalar loss, gradient of loss w.r.t. logits [seq_len * vocab_size]).
/// Gradient: grad[t, v] = softmax(logits[t])[v] - one_hot(targets[t])[v]
pub fn cross_entropy_loss(
    logits: &[f32],
    targets: &[u32],
    vocab_size: usize,
) -> (f32, Vec<f32>) {
    let seq_len = targets.len();
    assert_eq!(logits.len(), seq_len * vocab_size,
        "logits length {} != seq_len {} * vocab_size {}", logits.len(), seq_len, vocab_size);

    let mut total_loss = 0.0f64;
    let mut grad = vec![0.0f32; seq_len * vocab_size];

    for t in 0..seq_len {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];

        // Numerically stable softmax
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f64> = row.iter().map(|&x| ((x - max_val) as f64).exp()).collect();
        let sum_exp: f64 = exp_vals.iter().sum();

        // Loss = -log(softmax[target])
        let target_idx = targets[t] as usize;
        let log_prob = (exp_vals[target_idx] / sum_exp).ln();
        total_loss -= log_prob;

        // Gradient = softmax - one_hot(target)
        let grad_row = &mut grad[offset..offset + vocab_size];
        for v in 0..vocab_size {
            let sm = (exp_vals[v] / sum_exp) as f32;
            grad_row[v] = sm - if v == target_idx { 1.0 } else { 0.0 };
        }
    }

    let loss = (total_loss / seq_len as f64) as f32;

    // Scale gradients by 1/seq_len (matches loss normalization)
    let inv_seq = 1.0 / seq_len as f32;
    for g in &mut grad {
        *g *= inv_seq;
    }

    (loss, grad)
}

/// Convenience function that returns only the loss (for evaluation).
pub fn cross_entropy_loss_scalar(
    logits: &[f32],
    targets: &[u32],
    vocab_size: usize,
) -> f32 {
    cross_entropy_loss(logits, targets, vocab_size).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_confident() {
        // Logits strongly favor correct token
        let logits = vec![
            10.0, 0.0, 0.0, 0.0,  // token 0 is correct
            0.0, 10.0, 0.0, 0.0,  // token 1 is correct
        ];
        let targets = vec![0, 1];
        let (loss, grad) = cross_entropy_loss(&logits, &targets, 4);
        assert!(loss < 0.01, "expected near-zero loss, got {loss}");
        // Gradient at correct token should be near 0 (softmax ~1 - 1 = 0)
        assert!(grad[0].abs() < 0.01, "grad at correct token: {}", grad[0]);
        assert!(grad[5].abs() < 0.01, "grad at correct token: {}", grad[5]);
    }

    #[test]
    fn test_cross_entropy_wrong() {
        // Logits point away from correct token
        let logits = vec![
            0.0, 10.0, 0.0, 0.0,  // token 0 is correct but logit favors 1
            10.0, 0.0, 0.0, 0.0,  // token 1 is correct but logit favors 0
        ];
        let targets = vec![0, 1];
        let (loss, _grad) = cross_entropy_loss(&logits, &targets, 4);
        assert!(loss > 5.0, "expected high loss, got {loss}");
    }

    #[test]
    fn test_gradient_sum_zero() {
        // Gradients should sum to 0 per position (softmax - one_hot sums to 0)
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let targets = vec![2u32];
        let (_loss, grad) = cross_entropy_loss(&logits, &targets, 4);
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-6, "grad should sum to 0, got {sum}");
    }

    #[test]
    fn test_uniform_logits_loss() {
        // With uniform logits over vocab_size=4, loss = ln(4) ≈ 1.386
        let vocab_size = 4;
        let logits = vec![0.0f32; vocab_size];
        let targets = vec![0u32];
        let (loss, _) = cross_entropy_loss(&logits, &targets, vocab_size);
        let expected = (vocab_size as f32).ln();
        assert!((loss - expected).abs() < 1e-5,
            "expected loss={expected}, got {loss}");
    }
}
