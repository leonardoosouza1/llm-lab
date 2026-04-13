//! Loss functions for causal language model training.
//!
//! Provides three loss components:
//! 1. **Cross-entropy** — standard next-token prediction loss
//! 2. **KL-divergence** — for knowledge distillation from a teacher model
//! 3. **Combined loss** — weighted blend of CE + KL for distillation training

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Int, Tensor};

// ---------------------------------------------------------------------------
// 1. Cross-entropy loss for next-token prediction
// ---------------------------------------------------------------------------

/// Cross-entropy loss for causal language modeling.
///
/// Accepts 3-D logits `[batch, seq_len, vocab_size]` and 2-D targets
/// `[batch, seq_len]`, reshapes internally to use standard 2-D CE, and
/// returns a scalar loss.
pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,    // [batch, seq_len, vocab_size]
    targets: Tensor<B, 2, Int>, // [batch, seq_len]
) -> Tensor<B, 1> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let n_tokens = batch_size * seq_len;

    // Reshape to [n_tokens, vocab_size] and [n_tokens]
    let logits_2d = logits.reshape([n_tokens, vocab_size]);
    let targets_1d = targets.reshape([n_tokens]);

    // log_softmax over vocab dimension, then gather the target indices
    let log_probs = log_softmax(logits_2d, 1); // [n_tokens, vocab_size]
    let target_log_probs = log_probs.gather(1, targets_1d.reshape([n_tokens, 1])); // [n_tokens, 1]

    // Mean negative log-likelihood
    target_log_probs.mean().neg()
}

/// Cross-entropy loss with an optional pad token ID to ignore.
///
/// Tokens matching `pad_id` do not contribute to the loss.
pub fn cross_entropy_loss_with_padding<B: Backend>(
    logits: Tensor<B, 3>,       // [batch, seq_len, vocab_size]
    targets: Tensor<B, 2, Int>, // [batch, seq_len]
    pad_id: i64,
) -> Tensor<B, 1> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let n_tokens = batch_size * seq_len;

    let logits_2d = logits.reshape([n_tokens, vocab_size]);
    let targets_1d = targets.reshape([n_tokens]);

    let log_probs = log_softmax(logits_2d, 1);
    let target_log_probs = log_probs
        .gather(1, targets_1d.clone().reshape([n_tokens, 1]))
        .reshape([n_tokens]);

    // Mask: 1.0 where target != pad_id, 0.0 where it is pad
    let mask = targets_1d.equal_elem(pad_id).bool_not().float();
    let count = mask.clone().sum();

    // Masked mean
    let masked_loss = (target_log_probs * mask).sum().neg();

    // Avoid division by zero when the entire batch is padding
    masked_loss / count.clamp_min(1.0)
}

// ---------------------------------------------------------------------------
// 2. KL-divergence for knowledge distillation
// ---------------------------------------------------------------------------

/// KL-divergence loss for knowledge distillation.
///
/// Computes KL(teacher || student) with temperature scaling:
///
/// ```text
/// p_teacher = softmax(teacher_logits / T)
/// q_student = log_softmax(student_logits / T)
/// loss = sum(p_teacher * (log(p_teacher) - q_student)) * T^2
/// ```
///
/// The T^2 factor ensures gradients remain on the same scale regardless of
/// the temperature value.
pub fn kl_divergence_loss<B: Backend>(
    student_logits: Tensor<B, 3>,  // [batch, seq_len, vocab_size]
    teacher_logits: Tensor<B, 3>,  // [batch, seq_len, vocab_size]
    temperature: f64,
) -> Tensor<B, 1> {
    let [batch_size, seq_len, vocab_size] = student_logits.dims();
    let n_tokens = batch_size * seq_len;

    // Scale by temperature
    let student_scaled = student_logits.reshape([n_tokens, vocab_size]).div_scalar(temperature);
    let teacher_scaled = teacher_logits.reshape([n_tokens, vocab_size]).div_scalar(temperature);

    // Teacher probabilities and student log-probabilities
    let p_teacher = softmax(teacher_scaled.clone(), 1); // [n_tokens, vocab_size]
    let log_q_student = log_softmax(student_scaled, 1);  // [n_tokens, vocab_size]

    // KL = sum(p * (log(p) - log(q)))
    // Using log-domain for teacher for numerical stability
    let log_p_teacher = log_softmax(teacher_scaled, 1);
    let kl_per_token = (p_teacher * (log_p_teacher - log_q_student)).sum_dim(1); // [n_tokens, 1]

    // Mean over all tokens, scaled by T^2
    let t_sq = temperature * temperature;
    kl_per_token.mean().mul_scalar(t_sq)
}

// ---------------------------------------------------------------------------
// 3. Combined distillation loss
// ---------------------------------------------------------------------------

/// Combined loss for knowledge distillation training.
///
/// ```text
/// loss = alpha * CE(student_logits, targets) + (1 - alpha) * KL(teacher, student, T)
/// ```
///
/// - `alpha = 1.0` => pure supervised (no distillation)
/// - `alpha = 0.0` => pure distillation (no ground truth)
/// - Typical values: `alpha = 0.5`, `temperature = 2.0`
pub fn distillation_loss<B: Backend>(
    student_logits: Tensor<B, 3>,  // [batch, seq_len, vocab_size]
    teacher_logits: Tensor<B, 3>,  // [batch, seq_len, vocab_size]
    targets: Tensor<B, 2, Int>,    // [batch, seq_len]
    alpha: f64,
    temperature: f64,
) -> Tensor<B, 1> {
    let ce = cross_entropy_loss(student_logits.clone(), targets);
    let kl = kl_divergence_loss(student_logits, teacher_logits, temperature);

    ce.mul_scalar(alpha) + kl.mul_scalar(1.0 - alpha)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn test_cross_entropy_basic() {
        let device = <B as Backend>::Device::default();

        // batch=1, seq_len=2, vocab_size=4
        // Logits strongly favor the correct tokens
        let logits = Tensor::<B, 3>::from_data(
            TensorData::from([[[10.0_f32, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]]]),
            &device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::from([[0_i64, 1]]),
            &device,
        );

        let loss = cross_entropy_loss(logits, targets);
        let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

        // With logit=10 for the correct class, CE should be very small
        assert!(loss_val < 0.01, "expected near-zero loss, got {loss_val}");
    }

    #[test]
    fn test_cross_entropy_high_loss() {
        let device = <B as Backend>::Device::default();

        // Logits point AWAY from the correct token
        let logits = Tensor::<B, 3>::from_data(
            TensorData::from([[[0.0_f32, 10.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]]),
            &device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::from([[0_i64, 1]]),
            &device,
        );

        let loss = cross_entropy_loss(logits, targets);
        let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

        // Should be high (around 10)
        assert!(loss_val > 5.0, "expected high loss, got {loss_val}");
    }

    #[test]
    fn test_kl_divergence_identical() {
        let device = <B as Backend>::Device::default();

        // When student == teacher, KL should be ~0
        let logits = Tensor::<B, 3>::from_data(
            TensorData::from([[[1.0_f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]),
            &device,
        );
        let kl = kl_divergence_loss(logits.clone(), logits, 2.0);
        let kl_val = kl.into_data().as_slice::<f32>().unwrap()[0];

        assert!(
            kl_val.abs() < 1e-5,
            "KL of identical distributions should be ~0, got {kl_val}"
        );
    }

    #[test]
    fn test_distillation_loss_alpha_one() {
        let device = <B as Backend>::Device::default();

        let student = Tensor::<B, 3>::from_data(
            TensorData::from([[[10.0_f32, 0.0, 0.0, 0.0]]]),
            &device,
        );
        let teacher = Tensor::<B, 3>::from_data(
            TensorData::from([[[0.0_f32, 10.0, 0.0, 0.0]]]),
            &device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::from([[0_i64]]),
            &device,
        );

        // alpha=1 => pure CE, teacher contribution is zero
        let loss_combined = distillation_loss(
            student.clone(),
            teacher,
            targets.clone(),
            1.0,
            2.0,
        );
        let loss_ce = cross_entropy_loss(student, targets);

        let combined_val = loss_combined.into_data().as_slice::<f32>().unwrap()[0];
        let ce_val = loss_ce.into_data().as_slice::<f32>().unwrap()[0];

        assert!(
            (combined_val - ce_val).abs() < 1e-5,
            "alpha=1 should give pure CE: combined={combined_val}, ce={ce_val}"
        );
    }
}
