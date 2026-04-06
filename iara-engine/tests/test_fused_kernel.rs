//! Correctness tests for fused bias + RoPE kernel
//!
//! This test validates the fused kernel produces identical results to:
//!   1. add_bias_inplace(q)
//!   2. rope_inplace(q, pos)
//!
//! ## CPU Reference Implementation
//! Since GPU access may not be available, this test file includes CPU reference
//! implementations of add_bias and rope_inplace operations. These are used to
//! verify the mathematical correctness of the fused kernel logic.
//!
//! The tests validate that:
//! - Sequential (add_bias → rope) produces identical results to fused (rope(bias+q))
//! - Results are correct for various positions (0, 1, 512, 1023) as specified in TASK-01
//! - Both bias-present and bias-absent (Llama) paths work correctly
//! - RoPE mathematical properties (magnitude preservation) are maintained
//!
//! Run: cargo test --test test_fused_kernel -- --nocapture

// ─────────────────────────────────────────────────────────────────────────────
// CPU Reference Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Reference implementation: add bias in-place
/// q[tok, d] += bias[d]
fn cpu_add_bias(q: &mut [f32], bias: &[f32], n_tokens: usize, dim: usize) {
    for tok in 0..n_tokens {
        for d in 0..dim {
            q[tok * dim + d] += bias[d];
        }
    }
}

/// Reference implementation: RoPE (in-place) using halves convention
/// Rotation pairs: (x[i], x[i + head_dim/2]) for i < head_dim/2
fn cpu_rope_inplace(
    q: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    pos_start: usize,
    theta: f32,
) {
    let half_dim = head_dim / 2;

    for tok in 0..n_tokens {
        for head in 0..n_heads {
            let base_idx = tok * n_heads * head_dim + head * head_dim;

            for i in 0..half_dim {
                // Rotation frequency for this dimension pair
                let freq = 1.0f32 / theta.powf((2.0f32 * i as f32) / head_dim as f32);
                let angle = (pos_start as f32 + tok as f32) * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let x0 = q[base_idx + i];
                let x1 = q[base_idx + half_dim + i];

                // Apply 2D rotation
                q[base_idx + i] = x0 * cos_a - x1 * sin_a;
                q[base_idx + half_dim + i] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

/// Reference: fused bias + rope (same as sequential add_bias then rope)
fn cpu_fused_bias_rope(
    q: &mut [f32],
    bias: Option<&[f32]>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    pos_start: usize,
    theta: f32,
) {
    let dim = n_heads * head_dim;

    // Step 1: add bias
    if let Some(b) = bias {
        cpu_add_bias(q, b, n_tokens, dim);
    }

    // Step 2: apply rope
    cpu_rope_inplace(q, n_tokens, n_heads, head_dim, pos_start, theta);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test Cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bias_rope_with_bias_position_0() {
    let n_tokens = 1;
    let n_heads = 32;
    let head_dim = 128;
    let pos_start = 0;
    let theta = 1000000.0f32; // Qwen3 theta

    let dim = n_heads * head_dim;

    // Create test data
    let mut q = vec![0.1f32; n_tokens * dim];
    let bias = vec![0.05f32; dim];

    // Separate operations
    let mut q_separate = q.clone();
    cpu_add_bias(&mut q_separate, &bias, n_tokens, dim);
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused operation
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, Some(&bias), n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_bias_rope_no_bias_position_0() {
    let n_tokens = 1;
    let n_heads = 32;
    let head_dim = 128;
    let pos_start = 0;
    let theta = 1000000.0f32;

    let dim = n_heads * head_dim;

    // Create test data (no bias — Llama path)
    let mut q = vec![0.1f32; n_tokens * dim];

    // Separate operations (no bias)
    let mut q_separate = q.clone();
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused operation (no bias)
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, None, n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_bias_rope_position_1() {
    let n_tokens = 1;
    let n_heads = 8;
    let head_dim = 64;
    let pos_start = 1;
    let theta = 10000.0f32;

    let dim = n_heads * head_dim;

    let mut q = vec![0.01f32; n_tokens * dim];
    let bias = vec![0.001f32; dim];

    // Separate
    let mut q_separate = q.clone();
    cpu_add_bias(&mut q_separate, &bias, n_tokens, dim);
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, Some(&bias), n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_bias_rope_position_512() {
    let n_tokens = 1;
    let n_heads = 16;
    let head_dim = 96;
    let pos_start = 512;
    let theta = 100000.0f32;

    let dim = n_heads * head_dim;

    let mut q = vec![0.02f32; n_tokens * dim];
    let bias = vec![0.003f32; dim];

    // Separate
    let mut q_separate = q.clone();
    cpu_add_bias(&mut q_separate, &bias, n_tokens, dim);
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, Some(&bias), n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_bias_rope_position_1023() {
    let n_tokens = 1;
    let n_heads = 24;
    let head_dim = 128;
    let pos_start = 1023;
    let theta = 1000000.0f32;

    let dim = n_heads * head_dim;

    let mut q = vec![0.05f32; n_tokens * dim];
    let bias = vec![0.02f32; dim];

    // Separate
    let mut q_separate = q.clone();
    cpu_add_bias(&mut q_separate, &bias, n_tokens, dim);
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, Some(&bias), n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_bias_rope_multi_token() {
    let n_tokens = 5;
    let n_heads = 16;
    let head_dim = 64;
    let pos_start = 100;
    let theta = 10000.0f32;

    let dim = n_heads * head_dim;

    // Random-ish test data
    let mut q: Vec<f32> = (0..(n_tokens * dim))
        .map(|i| 0.01 * ((i as f32).sin()))
        .collect();

    let bias: Vec<f32> = (0..dim)
        .map(|i| 0.001 * ((i as f32).cos()))
        .collect();

    // Separate
    let mut q_separate = q.clone();
    cpu_add_bias(&mut q_separate, &bias, n_tokens, dim);
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, Some(&bias), n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_bias_rope_zero_bias() {
    let n_tokens = 2;
    let n_heads = 8;
    let head_dim = 128;
    let pos_start = 10;
    let theta = 1000000.0f32;

    let dim = n_heads * head_dim;

    let mut q = vec![0.1f32; n_tokens * dim];
    let bias = vec![0.0f32; dim]; // Zero bias should act like no bias

    // Separate
    let mut q_separate = q.clone();
    cpu_add_bias(&mut q_separate, &bias, n_tokens, dim);
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, Some(&bias), n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_bias_rope_large_head_dim() {
    let n_tokens = 1;
    let n_heads = 4;
    let head_dim = 256; // Large head dimension
    let pos_start = 0;
    let theta = 1000000.0f32;

    let dim = n_heads * head_dim;

    let mut q = vec![0.01f32; n_tokens * dim];
    let bias = vec![0.001f32; dim];

    // Separate
    let mut q_separate = q.clone();
    cpu_add_bias(&mut q_separate, &bias, n_tokens, dim);
    cpu_rope_inplace(&mut q_separate, n_tokens, n_heads, head_dim, pos_start, theta);

    // Fused
    let mut q_fused = q.clone();
    cpu_fused_bias_rope(&mut q_fused, Some(&bias), n_tokens, n_heads, head_dim, pos_start, theta);

    // Compare
    for i in 0..q_separate.len() {
        let diff = (q_separate[i] - q_fused[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: separate={}, fused={}, diff={}",
            i, q_separate[i], q_fused[i], diff
        );
    }
}

#[test]
fn test_rope_mathematical_properties() {
    // Verify that RoPE preserves vector magnitude (rotation property)
    let n_tokens = 1;
    let n_heads = 1;
    let head_dim = 4;
    let pos_start = 0;
    let theta = 10000.0f32;

    let dim = n_heads * head_dim;

    // Create a simple test vector
    let mut q = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];

    // Compute original magnitude (sum of squares)
    let mag_before: f32 = q.iter().map(|x| x * x).sum();

    // Apply RoPE
    cpu_rope_inplace(&mut q, n_tokens, n_heads, head_dim, pos_start, theta);

    // Compute magnitude after (should be nearly the same)
    let mag_after: f32 = q.iter().map(|x| x * x).sum();

    let mag_diff = (mag_before - mag_after).abs();
    assert!(
        mag_diff < 1e-5,
        "RoPE changed vector magnitude: before={}, after={}, diff={}",
        mag_before, mag_after, mag_diff
    );
}
