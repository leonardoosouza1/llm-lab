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
/// Returns scalar loss value.
pub fn cross_entropy_loss(
    logits: &[f32],
    targets: &[u32],
    vocab_size: usize,
) -> f32 {
    // CPU reference implementation
    let seq_len = targets.len();
    assert_eq!(logits.len(), seq_len * vocab_size);

    let mut total_loss = 0.0f64;
    for t in 0..seq_len {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];

        // log-sum-exp for numerical stability
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = row.iter().map(|&x| ((x - max_val) as f64).exp()).sum();
        let log_sum_exp = max_val as f64 + sum_exp.ln();

        let target_logit = row[targets[t] as usize] as f64;
        total_loss += log_sum_exp - target_logit;
    }

    (total_loss / seq_len as f64) as f32
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
        let loss = cross_entropy_loss(&logits, &targets, 4);
        assert!(loss < 0.01, "expected near-zero loss, got {loss}");
    }

    #[test]
    fn test_cross_entropy_wrong() {
        // Logits point away from correct token
        let logits = vec![
            0.0, 10.0, 0.0, 0.0,  // token 0 is correct but logit favors 1
            10.0, 0.0, 0.0, 0.0,  // token 1 is correct but logit favors 0
        ];
        let targets = vec![0, 1];
        let loss = cross_entropy_loss(&logits, &targets, 4);
        assert!(loss > 5.0, "expected high loss, got {loss}");
    }
}
