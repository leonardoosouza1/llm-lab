//! DNA V4 compression + polish pipelines.
//!
//! Two entry points:
//! - `compress_tensor`: Thin SVD factorization W [rows, cols] → A [rows, rank], B [rank, cols]
//! - `polish_svd`: LoRA-style gradient descent on A, B to close RMSE gap against teacher
//!
//! Compression is deterministic (no training). Polish is optional but recommended
//! for ranks < ~min(rows, cols) / 4 where truncation error becomes visible.

use half::bf16;
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// BF16 tensor residing on host. Shape: [rows, cols] row-major.
pub struct HostBf16Matrix {
    pub data: Vec<u16>,
    pub rows: usize,
    pub cols: usize,
}

impl HostBf16Matrix {
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let bf16_data: Vec<u16> = data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        Self { data: bf16_data, rows, cols }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
    }

    /// Frobenius RMSE between this matrix and another of the same shape.
    pub fn rmse(&self, other: &Self) -> f32 {
        assert_eq!(self.data.len(), other.data.len());
        let n = self.data.len() as f32;
        let sum: f32 = self.data.iter().zip(&other.data)
            .map(|(a, b)| {
                let d = bf16::from_bits(*a).to_f32() - bf16::from_bits(*b).to_f32();
                d * d
            })
            .sum();
        (sum / n).sqrt()
    }
}

/// Output of Thin SVD: W ≈ A · B.
pub struct SvdFactors {
    pub a_bf16: Vec<u16>, // [rows, rank] row-major
    pub b_bf16: Vec<u16>, // [rank, cols] row-major
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    /// Frobenius RMSE of (W - A·B) per element (sanity metric for polish target).
    pub residual_rmse: f32,
}

/// Thin SVD factorization of a BF16 weight matrix.
///
/// Uses nalgebra's SVD on the full F32 matrix. For production models this is slow
/// and memory-heavy; consider randomized SVD (Halko) for matrices > ~4096².
/// Absorbs singular values into A: A = U·Σ, B = V^T.
pub fn compress_tensor(w: &HostBf16Matrix, rank: usize) -> SvdFactors {
    let (rows, cols) = (w.rows, w.cols);
    assert!(rank > 0 && rank <= rows.min(cols), "rank out of range");

    let f32_data = w.to_f32();
    // nalgebra expects column-major. Our data is row-major, so transpose during build.
    let m = DMatrix::<f32>::from_fn(rows, cols, |i, j| f32_data[i * cols + j]);
    let svd = m.clone().svd(true, true);
    let u = svd.u.expect("SVD U");
    let s = svd.singular_values;
    let vt = svd.v_t.expect("SVD V_t");

    let mut a = DMatrix::<f32>::zeros(rows, rank);
    let mut b = DMatrix::<f32>::zeros(rank, cols);
    for k in 0..rank {
        let sigma = s[k];
        for i in 0..rows { a[(i, k)] = u[(i, k)] * sigma; }
        for j in 0..cols { b[(k, j)] = vt[(k, j)]; }
    }

    // Residual (F32 domain, before BF16 quantization).
    let approx = &a * &b;
    let residual = &m - &approx;
    let residual_rmse = (residual.norm_squared() / (rows * cols) as f32).sqrt();

    // Row-major BF16 dump.
    let mut a_bf16 = Vec::with_capacity(rows * rank);
    for i in 0..rows {
        for k in 0..rank {
            a_bf16.push(bf16::from_f32(a[(i, k)]).to_bits());
        }
    }
    let mut b_bf16 = Vec::with_capacity(rank * cols);
    for k in 0..rank {
        for j in 0..cols {
            b_bf16.push(bf16::from_f32(b[(k, j)]).to_bits());
        }
    }

    SvdFactors { a_bf16, b_bf16, rows, cols, rank, residual_rmse }
}

/// Randomized SVD (Halko-Martinsson-Tropp 2009) — O(m·n·(r+p)) instead of O(m·n·min(m,n)).
///
/// For rank ≪ min(rows, cols) this is orders of magnitude faster than full SVD.
/// Quality: with `oversample=10` and `power_iters=2` the approximation is within
/// a factor ~(1+ε) of the best rank-r in spectral norm — indistinguishable from
/// truncated full SVD for typical LLM weight spectra.
///
/// Returns factors absorbing Σ into A (A = U·Σ, B = Vᵀ).
pub fn compress_tensor_randomized(
    w: &HostBf16Matrix,
    rank: usize,
    oversample: usize,
    power_iters: usize,
    seed: u64,
) -> SvdFactors {
    let (rows, cols) = (w.rows, w.cols);
    assert!(rank > 0 && rank <= rows.min(cols), "rank out of range");
    let k = (rank + oversample).min(rows.min(cols));

    let f32_data = w.to_f32();
    let a_mat = DMatrix::<f32>::from_fn(rows, cols, |i, j| f32_data[i * cols + j]);
    drop(f32_data);

    // Ω ~ N(0,1) [cols, k]
    let mut rng = StdRng::seed_from_u64(seed);
    let omega = DMatrix::<f32>::from_fn(cols, k, |_, _| {
        // Box-Muller via rand's f32 uniform — good enough for subspace seeding
        let u1: f32 = rng.gen_range(1e-9..1.0);
        let u2: f32 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    });

    // Y = A · Ω  [rows, k]
    let mut y = &a_mat * &omega;

    // Power iterations: Y ← A · (Aᵀ · Y). Re-orthogonalize between iters to fight roundoff.
    let at = a_mat.transpose();
    for _ in 0..power_iters {
        let z = &at * &y;            // [cols, k]
        y = &a_mat * &z;             // [rows, k]
        // Orthonormalize (thin QR) to stabilize
        let qr = y.clone().qr();
        y = qr.q();
    }

    // Q, _ = QR(Y). Q [rows, k].
    let qr = y.qr();
    let q = qr.q();

    // B = Qᵀ · A  [k, cols]
    let b_small = q.transpose() * &a_mat;

    // SVD of the small B [k × cols] — cheap since k ≪ rows.
    let svd = b_small.clone().svd(true, true);
    let u_tilde = svd.u.expect("rsvd U");
    let s = svd.singular_values;
    let vt = svd.v_t.expect("rsvd Vt");

    // U = Q · U_tilde  [rows, k]
    let u_full = &q * &u_tilde;

    // Truncate to rank r, absorb Σ into A.
    let mut a_out = DMatrix::<f32>::zeros(rows, rank);
    let mut b_out = DMatrix::<f32>::zeros(rank, cols);
    for kk in 0..rank {
        let sigma = s[kk];
        for i in 0..rows { a_out[(i, kk)] = u_full[(i, kk)] * sigma; }
        for j in 0..cols { b_out[(kk, j)] = vt[(kk, j)]; }
    }

    // Residual in F32 domain
    let approx = &a_out * &b_out;
    let residual = &a_mat - &approx;
    let residual_rmse = (residual.norm_squared() / (rows * cols) as f32).sqrt();

    // BF16 dump (row-major)
    let mut a_bf16 = Vec::with_capacity(rows * rank);
    for i in 0..rows {
        for kk in 0..rank {
            a_bf16.push(bf16::from_f32(a_out[(i, kk)]).to_bits());
        }
    }
    let mut b_bf16 = Vec::with_capacity(rank * cols);
    for kk in 0..rank {
        for j in 0..cols {
            b_bf16.push(bf16::from_f32(b_out[(kk, j)]).to_bits());
        }
    }

    SvdFactors { a_bf16, b_bf16, rows, cols, rank, residual_rmse }
}

/// Hybrid rank selector: global default with per-name overrides.
///
/// Patterns match by suffix (`*.q_proj.weight`) or exact name.
/// Precedence: exact > suffix > default.
pub struct RankPolicy {
    pub default_rank: usize,
    pub exact: std::collections::HashMap<String, usize>,
    pub suffix: Vec<(String, usize)>, // pattern like "*.q_proj.weight"
}

impl RankPolicy {
    pub fn new(default_rank: usize) -> Self {
        Self {
            default_rank,
            exact: std::collections::HashMap::new(),
            suffix: Vec::new(),
        }
    }

    pub fn with_exact(mut self, name: impl Into<String>, rank: usize) -> Self {
        self.exact.insert(name.into(), rank);
        self
    }

    pub fn with_suffix(mut self, pattern: impl Into<String>, rank: usize) -> Self {
        self.suffix.push((pattern.into(), rank));
        self
    }

    pub fn rank_for(&self, name: &str) -> usize {
        if let Some(&r) = self.exact.get(name) { return r; }
        for (pat, r) in &self.suffix {
            let stripped = pat.strip_prefix('*').unwrap_or(pat);
            if name.ends_with(stripped) { return *r; }
        }
        self.default_rank
    }
}

/// Default policy: embed/lm_head and attention projections get higher rank.
/// Tuned for Qwen-style models; adjust per-architecture as needed.
pub fn default_rank_policy(default_rank: usize) -> RankPolicy {
    RankPolicy::new(default_rank)
        .with_suffix(".embed_tokens.weight", default_rank.max(256))
        .with_suffix("lm_head.weight", default_rank.max(256))
        .with_suffix(".q_proj.weight", default_rank.max(128))
        .with_suffix(".k_proj.weight", default_rank.max(128))
        .with_suffix(".v_proj.weight", default_rank.max(128))
}

/// Tensor-level polish: LoRA-style SGD on (A, B) with W_target as frozen ground truth.
///
/// Loss: ½·||A·B - W_target||²_F  per-element MSE.
/// Gradients:
///   dL/dA = (A·B - W) · Bᵀ
///   dL/dB = Aᵀ · (A·B - W)
///
/// Returns the final RMSE. In-place update of factors.
pub fn polish_svd(
    factors: &mut SvdFactors,
    w_target: &HostBf16Matrix,
    steps: usize,
    lr: f32,
) -> f32 {
    assert_eq!(factors.rows, w_target.rows);
    assert_eq!(factors.cols, w_target.cols);

    let (rows, cols, rank) = (factors.rows, factors.cols, factors.rank);

    // Move to F32 for stable gradient descent.
    let w = DMatrix::<f32>::from_fn(rows, cols, |i, j| {
        bf16::from_bits(w_target.data[i * cols + j]).to_f32()
    });
    let mut a = DMatrix::<f32>::from_fn(rows, rank, |i, k| {
        bf16::from_bits(factors.a_bf16[i * rank + k]).to_f32()
    });
    let mut b = DMatrix::<f32>::from_fn(rank, cols, |k, j| {
        bf16::from_bits(factors.b_bf16[k * cols + j]).to_f32()
    });

    let n = (rows * cols) as f32;
    let mut last_rmse = f32::INFINITY;

    // Normalize gradient step by element count — keeps lr scale-invariant across shapes.
    let inv_n = 1.0 / n;

    for step in 0..steps {
        let diff = &a * &b - &w;           // [rows, cols]
        last_rmse = (diff.norm_squared() * inv_n).sqrt();

        let grad_a = &diff * b.transpose() * inv_n; // [rows, rank]
        let grad_b = a.transpose() * &diff * inv_n; // [rank, cols]

        a -= grad_a * lr;
        b -= grad_b * lr;

        if step % 10 == 0 {
            log::debug!("polish step {step}: rmse={last_rmse:.6e}");
        }
    }

    // Write back BF16.
    for i in 0..rows {
        for k in 0..rank {
            factors.a_bf16[i * rank + k] = bf16::from_f32(a[(i, k)]).to_bits();
        }
    }
    for k in 0..rank {
        for j in 0..cols {
            factors.b_bf16[k * cols + j] = bf16::from_f32(b[(k, j)]).to_bits();
        }
    }
    factors.residual_rmse = last_rmse;
    last_rmse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_full_rank_near_zero_residual() {
        // 8x8 random matrix, rank 8 = full rank. Residual should be tiny (just BF16 noise).
        let rows = 8;
        let cols = 8;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let w = HostBf16Matrix::from_f32(&data, rows, cols);
        let factors = compress_tensor(&w, 8);
        assert!(factors.residual_rmse < 1e-3, "residual too large: {}", factors.residual_rmse);
    }

    #[test]
    fn compress_low_rank_recovers_structure() {
        // Build a known rank-2 matrix: W = u·vᵀ + u2·v2ᵀ
        let rows = 6;
        let cols = 10;
        let u = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = u[i] * v[j];
            }
        }
        let w = HostBf16Matrix::from_f32(&data, rows, cols);
        let factors = compress_tensor(&w, 2);
        assert!(factors.residual_rmse < 1e-2, "rank-2 approx not clean: {}", factors.residual_rmse);
    }

    #[test]
    fn polish_reduces_rmse_from_noisy_init() {
        // Start from SVD-optimal factors then perturb A — polish must recover RMSE.
        let rows = 16;
        let cols = 16;
        let data: Vec<f32> = (0..rows * cols).map(|i| ((i * 17 + 3) % 31) as f32 * 0.1).collect();
        let w = HostBf16Matrix::from_f32(&data, rows, cols);
        let mut factors = compress_tensor(&w, 8);
        // Perturb A with +10% noise to move away from local min.
        for bits in factors.a_bf16.iter_mut() {
            let v = bf16::from_bits(*bits).to_f32() * 1.1;
            *bits = bf16::from_f32(v).to_bits();
        }
        let perturbed_rmse = {
            let w_approx = HostBf16Matrix::from_f32(&{
                let a = DMatrix::<f32>::from_fn(rows, 8, |i, k| {
                    bf16::from_bits(factors.a_bf16[i * 8 + k]).to_f32()
                });
                let b = DMatrix::<f32>::from_fn(8, cols, |k, j| {
                    bf16::from_bits(factors.b_bf16[k * cols + j]).to_f32()
                });
                let m = &a * &b;
                (0..rows * cols).map(|i| m[(i / cols, i % cols)]).collect::<Vec<f32>>()
            }, rows, cols);
            w.rmse(&w_approx)
        };
        let after = polish_svd(&mut factors, &w, 1000, 1.0);
        assert!(
            after < perturbed_rmse,
            "polish failed to recover: perturbed={} final={}",
            perturbed_rmse, after
        );
    }

    #[test]
    fn randomized_svd_matches_full_on_low_rank() {
        // Known rank-3 matrix (outer products). Randomized SVD with rank=3 should recover near-exact.
        let rows = 64;
        let cols = 128;
        let mut data = vec![0.0f32; rows * cols];
        for r in 0..3 {
            let u: Vec<f32> = (0..rows).map(|i| ((i + r * 7) % 13) as f32 * 0.1).collect();
            let v: Vec<f32> = (0..cols).map(|j| ((j + r * 11) % 17) as f32 * 0.1).collect();
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] += u[i] * v[j];
                }
            }
        }
        let w = HostBf16Matrix::from_f32(&data, rows, cols);
        let rsvd = compress_tensor_randomized(&w, 3, 10, 2, 42);
        assert!(rsvd.residual_rmse < 5e-2, "rsvd residual too large: {}", rsvd.residual_rmse);
    }

    #[test]
    fn randomized_svd_faster_path_produces_valid_factors() {
        // Arbitrary 256×512 matrix, rank=16. Just verify it runs and residual is finite.
        let rows = 256;
        let cols = 512;
        let data: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 + 11) % 97) as f32 * 0.01).collect();
        let w = HostBf16Matrix::from_f32(&data, rows, cols);
        let factors = compress_tensor_randomized(&w, 16, 10, 2, 7);
        assert_eq!(factors.a_bf16.len(), rows * 16);
        assert_eq!(factors.b_bf16.len(), 16 * cols);
        assert!(factors.residual_rmse.is_finite());
    }

    #[test]
    fn rank_policy_precedence() {
        let p = RankPolicy::new(64)
            .with_suffix(".q_proj.weight", 128)
            .with_exact("model.embed_tokens.weight", 256);
        assert_eq!(p.rank_for("model.embed_tokens.weight"), 256);
        assert_eq!(p.rank_for("model.layers.0.self_attn.q_proj.weight"), 128);
        assert_eq!(p.rank_for("model.layers.0.mlp.gate_proj.weight"), 64);
    }
}
