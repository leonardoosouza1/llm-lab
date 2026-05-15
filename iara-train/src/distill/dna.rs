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
///
/// Calibrated via SV-spectrum analysis (scripts/dna/sv_spectrum.py):
/// - Qwen2.5-1.5B V/K_proj had only 256 SVs total (lucky narrow shape) → rank 256
///   = 100% lossless. Qwen2.5 just worked at rank 256.
/// - Qwen3-4B V/K_proj has 1024 SVs (8 KV heads × 128) → rank 256 captures only
///   50% energy → 42% L2 drift in V → garbage logits.
/// - Solution: rank multipliers per tensor type. Caller controls baseline via
///   --rank; each suffix overrides upward when needed.
pub fn default_rank_policy(default_rank: usize) -> RankPolicy {
    RankPolicy::new(default_rank)
        .with_suffix(".embed_tokens.weight", default_rank.max(256))
        .with_suffix("lm_head.weight", default_rank.max(256))
        .with_suffix(".q_proj.weight", default_rank.max(512))
        .with_suffix(".k_proj.weight", default_rank.max(512))
        .with_suffix(".v_proj.weight", default_rank.max(512))
        .with_suffix(".o_proj.weight", default_rank.max(512))
        .with_suffix(".gate_proj.weight", default_rank.max(512))
        .with_suffix(".up_proj.weight", default_rank.max(512))
        .with_suffix(".down_proj.weight", default_rank.max(512))
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

/// GPU-accelerated polish: same semantics as `polish_svd` but all GEMMs run on
/// the provided `ComputeBackend` via `sgemm`. Does NOT depend on hipSOLVER —
/// only sgemm + `add_tensors_inplace` + `copy_f32_d2d` (the last with a
/// CPU-reupload fallback for backends that don't override it).
///
/// Loss: ½·‖A·B − W‖²_F. Gradients:
///   dL/dA = (A·B − W)·Bᵀ · (1/n)
///   dL/dB = Aᵀ·(A·B − W)  · (1/n)
///
/// Returns the final RMSE (pre-last-update) and writes back BF16 factors.
pub fn polish_svd_gpu(
    backend: &dyn compute_core::ComputeBackend,
    factors: &mut SvdFactors,
    w_target: &HostBf16Matrix,
    steps: usize,
    lr: f32,
) -> f32 {
    use compute_core::SgemmParams;

    assert_eq!(factors.rows, w_target.rows);
    assert_eq!(factors.cols, w_target.cols);
    let (rows, cols, rank) = (factors.rows, factors.cols, factors.rank);

    // Host-side F32 conversions (single shot).
    let w_host: Vec<f32> = w_target.to_f32();
    let mut a_host: Vec<f32> = factors.a_bf16.iter()
        .map(|&b| bf16::from_bits(b).to_f32()).collect();
    let mut b_host: Vec<f32> = factors.b_bf16.iter()
        .map(|&b| bf16::from_bits(b).to_f32()).collect();

    // Device buffers.
    let w_dev = backend.upload_f32(&w_host, &[rows, cols]);
    let a_dev = backend.upload_f32(&a_host, &[rows, rank]);
    let b_dev = backend.upload_f32(&b_host, &[rank, cols]);
    let diff_dev = backend.alloc_f32(&[rows, cols]);
    let grad_a_dev = backend.alloc_f32(&[rows, rank]);
    let grad_b_dev = backend.alloc_f32(&[rank, cols]);

    let n = (rows * cols) as f32;
    let inv_n = 1.0 / n;
    let mut last_rmse = f32::INFINITY;

    // why: this routine needs device-to-device copy to prefill diff_dev with W
    //      each step (for the sgemm beta=-1 trick). The trait default returns
    //      NotImplemented, so we probe once and bail early if the backend lacks
    //      it — the CPU `polish_svd` already covers that case.
    // TODO: drop this requirement once a device axpy/fill kernel is on the trait.
    if backend.copy_f32_d2d(&w_dev, &diff_dev, rows * cols).is_err() {
        log::warn!("polish_svd_gpu: backend lacks copy_f32_d2d; falling back to CPU polish");
        // Discard device state; delegate to CPU path.
        drop((w_dev, a_dev, b_dev, diff_dev, grad_a_dev, grad_b_dev));
        return polish_svd(factors, w_target, steps, lr);
    }

    for step in 0..steps {
        // Prefill diff with W so sgemm below yields A·B − W.
        backend
            .copy_f32_d2d(&w_dev, &diff_dev, rows * cols)
            .expect("copy_f32_d2d after probe succeeded should not fail");

        // diff = A·B − W.  Row-major C[rows,cols] = A[rows,rank] · B[rank,cols].
        backend.sgemm(&SgemmParams {
            trans_a: false, trans_b: false,
            m: rows, n: cols, k: rank,
            alpha: 1.0, beta: -1.0,
            a: &a_dev, lda: rank, a_offset: 0,
            b: &b_dev, ldb: cols, b_offset: 0,
            c: &diff_dev, ldc: cols, c_offset: 0,
        }).expect("sgemm A·B - W");

        // why: measure RMSE before the weight update (matches CPU polish_svd).
        //      One device→host transfer per step; dominated by sgemm cost for
        //      reasonable shapes.
        let diff_host = backend.download_f32(&diff_dev);
        let ss: f64 = diff_host.iter().map(|&v| (v as f64) * (v as f64)).sum();
        last_rmse = ((ss as f32) * inv_n).sqrt();

        // gradA [rows, rank] = diff[rows,cols] · B[rank,cols]ᵀ · inv_n.
        // Row-major with trans_b=true: lda=cols, ldb=cols (stride of B rows
        // before transpose), ldc=rank.
        backend.sgemm(&SgemmParams {
            trans_a: false, trans_b: true,
            m: rows, n: rank, k: cols,
            alpha: inv_n, beta: 0.0,
            a: &diff_dev, lda: cols, a_offset: 0,
            b: &b_dev, ldb: cols, b_offset: 0,
            c: &grad_a_dev, ldc: rank, c_offset: 0,
        }).expect("sgemm gradA");

        // gradB [rank, cols] = A[rows,rank]ᵀ · diff[rows,cols] · inv_n.
        // Row-major with trans_a=true: lda=rank (pre-transpose stride of A).
        backend.sgemm(&SgemmParams {
            trans_a: true, trans_b: false,
            m: rank, n: cols, k: rows,
            alpha: inv_n, beta: 0.0,
            a: &a_dev, lda: rank, a_offset: 0,
            b: &diff_dev, ldb: cols, b_offset: 0,
            c: &grad_b_dev, ldc: cols, c_offset: 0,
        }).expect("sgemm gradB");

        // A -= lr · gradA ;  B -= lr · gradB. axpy via add_tensors_inplace.
        backend
            .add_tensors_inplace(&a_dev, &grad_a_dev, rows * rank, -lr)
            .expect("axpy A");
        backend
            .add_tensors_inplace(&b_dev, &grad_b_dev, rank * cols, -lr)
            .expect("axpy B");

        if step % 10 == 0 {
            log::debug!("polish_gpu step {step}: rmse={last_rmse:.6e}");
        }
    }

    // Download A, B → BF16 write-back.
    a_host = backend.download_f32(&a_dev);
    b_host = backend.download_f32(&b_dev);
    for i in 0..rows {
        for k in 0..rank {
            factors.a_bf16[i * rank + k] = bf16::from_f32(a_host[i * rank + k]).to_bits();
        }
    }
    for k in 0..rank {
        for j in 0..cols {
            factors.b_bf16[k * cols + j] = bf16::from_f32(b_host[k * cols + j]).to_bits();
        }
    }
    factors.residual_rmse = last_rmse;
    last_rmse
}

/// Full-GPU randomized SVD (Halko-Martinsson-Tropp). Uses hipSOLVER QR + SVD
/// for the orthogonalization and small-SVD steps; hipBLAS sgemm for the
/// two big matmuls. W is uploaded once as F32; factors come back host-side BF16.
///
/// Requires a backend that implements `qr_f32` and `svd_f32`. Caller should
/// probe and fall back to `compress_tensor_randomized` (CPU) if unavailable.
///
/// Residual RMSE is estimated via Eckart-Young: ‖W−AB‖²_F ≈ Σ_{i≥rank} σᵢ².
/// This approximates the true residual (rsvd singular values aren't exact).
pub fn compress_tensor_rsvd_gpu(
    backend: &dyn compute_core::ComputeBackend,
    w: &HostBf16Matrix,
    rank: usize,
    oversample: usize,
    seed: u64,
) -> Result<SvdFactors, compute_core::BackendError> {
    compress_tensor_rsvd_gpu_raw(backend, &w.data, w.rows, w.cols, rank, oversample, seed)
}

/// Zero-copy variant: takes BF16 bits as a raw slice (e.g., reinterpreted from
/// a safetensors mmap). Avoids the `HostBf16Matrix` host-side allocation.
pub fn compress_tensor_rsvd_gpu_raw(
    backend: &dyn compute_core::ComputeBackend,
    w_bf16: &[u16],
    rows: usize,
    cols: usize,
    rank: usize,
    oversample: usize,
    seed: u64,
) -> Result<SvdFactors, compute_core::BackendError> {
    use compute_core::SgemmParams;

    assert_eq!(w_bf16.len(), rows * cols, "w_bf16 len mismatch");
    assert!(rank > 0 && rank <= rows.min(cols), "rank out of range");
    let k = (rank + oversample).min(rows.min(cols));

    // 1. Upload W as BF16 (half the bytes of F32). Widen to F32 on device,
    //    dropping the BF16 buffer afterwards. No host-side bf16→f32 pass.
    let w_bf16_dev = backend.upload_bf16(w_bf16, &[rows, cols]);
    let w_dev = backend.alloc_f32(&[rows, cols]);
    backend.widen_bf16_f32(&w_bf16_dev, &w_dev, rows * cols)?;
    drop(w_bf16_dev);

    // 2. Generate Ω ~ N(0,1) row-major [cols, k], upload.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega_host = vec![0.0f32; cols * k];
    for v in omega_host.iter_mut() {
        let u1: f32 = rng.gen_range(1e-9..1.0);
        let u2: f32 = rng.gen();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    let omega_dev = backend.upload_f32(&omega_host, &[cols, k]);
    drop(omega_host);

    // 3. y_trans bytes row-major [k, rows] ≡ col-major [rows, k] = Y = W·Ω.
    //    Computed as Yᵀ_rm = Ωᵀ·Wᵀ.
    let y_trans_dev = backend.alloc_f32(&[k, rows]);
    backend.sgemm(&SgemmParams {
        trans_a: true, trans_b: true,
        m: k, n: rows, k: cols,
        alpha: 1.0, beta: 0.0,
        a: &omega_dev, lda: k, a_offset: 0,
        b: &w_dev,     ldb: cols, b_offset: 0,
        c: &y_trans_dev, ldc: rows, c_offset: 0,
    })?;

    // 4. QR of Y col-major [rows, k]. q_dev bytes = col-major Q [rows, k] = row-major [k, rows] = Qᵀ.
    let q_dev = backend.alloc_f32(&[rows, k]);
    backend.qr_f32(&y_trans_dev, &q_dev, rows, k)?;

    // 5. B_trans bytes row-major [cols, k] ≡ col-major [k, cols] = B = Qᵀ·W.
    //    Computed as Bᵀ_rm = Wᵀ·Q.
    let b_trans_dev = backend.alloc_f32(&[cols, k]);
    backend.sgemm(&SgemmParams {
        trans_a: true, trans_b: true,
        m: cols, n: k, k: rows,
        alpha: 1.0, beta: 0.0,
        a: &w_dev, lda: cols, a_offset: 0,
        b: &q_dev, ldb: rows, b_offset: 0,
        c: &b_trans_dev, ldc: k, c_offset: 0,
    })?;

    // 6. SVD of B col-major [k, cols] → U [k,k], S [k], Vt [k, cols] (all col-major).
    let u_dev  = backend.alloc_f32(&[k, k]);
    let s_dev  = backend.alloc_f32(&[k]);
    let vt_dev = backend.alloc_f32(&[k, cols]);
    backend.svd_f32(&b_trans_dev, &u_dev, &s_dev, &vt_dev, k, cols)?;

    // 7. Download S, U (small), Vt (medium).
    let s_host  = backend.download_f32(&s_dev);   // len = k
    let u_host  = backend.download_f32(&u_dev);   // col-major [k, k]
    let vt_host = backend.download_f32(&vt_dev);  // col-major [k, cols]

    // 8. Build U_scaled col-major [k, rank] absorbing Σ into U. Host-side, tiny.
    let mut u_scaled = vec![0.0f32; k * rank];
    for j in 0..rank {
        let sigma = s_host[j];
        for i in 0..k {
            u_scaled[j * k + i] = u_host[j * k + i] * sigma;
        }
    }
    // Re-upload as row-major [rank, k] (same bytes as col-major [k, rank]).
    let u_scaled_dev = backend.upload_f32(&u_scaled, &[rank, k]);
    drop(u_scaled);

    // 9. A_factor row-major [rows, rank] = Q · U_scaled.
    //    Q bytes = row-major [k, rows] → trans_a=true yields Q [rows, k].
    //    U_scaled bytes = row-major [rank, k] → trans_b=true yields [k, rank].
    let a_factor_dev = backend.alloc_f32(&[rows, rank]);
    backend.sgemm(&SgemmParams {
        trans_a: true, trans_b: true,
        m: rows, n: rank, k,
        alpha: 1.0, beta: 0.0,
        a: &q_dev, lda: rows, a_offset: 0,
        b: &u_scaled_dev, ldb: k, b_offset: 0,
        c: &a_factor_dev, ldc: rank, c_offset: 0,
    })?;

    // 10. Download A (row-major) and synthesize B row-major [rank, cols] from Vt col-major.
    let a_factor = backend.download_f32(&a_factor_dev);
    let mut b_factor = vec![0.0f32; rank * cols];
    for i in 0..rank {
        for j in 0..cols {
            b_factor[i * cols + j] = vt_host[j * k + i];
        }
    }

    // 11. Quantize to BF16.
    let a_bf16: Vec<u16> = a_factor.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
    let b_bf16: Vec<u16> = b_factor.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();

    // 12. Residual estimate (Eckart-Young lower bound).
    let residual_sq: f32 = s_host.iter().skip(rank).map(|&s| s * s).sum();
    let residual_rmse = (residual_sq / (rows * cols) as f32).sqrt();

    Ok(SvdFactors { a_bf16, b_bf16, rows, cols, rank, residual_rmse })
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

    #[cfg(feature = "rocm")]
    #[test]
    fn polish_gpu_reduces_rmse() {
        let backend = match rocm_backend::RocmBackend::probe() {
            Ok(b) => b,
            Err(_) => return, // no GPU available at test time — skip
        };

        // 64×64 matrix, compress to rank 8, perturb A, polish on GPU.
        let rows = 64;
        let cols = 64;
        let rank = 8;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
            .collect();
        let w = HostBf16Matrix::from_f32(&data, rows, cols);
        let mut factors = compress_tensor(&w, rank);

        // Perturb A with +10% noise.
        for bits in factors.a_bf16.iter_mut() {
            let v = bf16::from_bits(*bits).to_f32() * 1.1;
            *bits = bf16::from_f32(v).to_bits();
        }

        // Measure RMSE of perturbed approximation.
        let perturbed_rmse = {
            let a = DMatrix::<f32>::from_fn(rows, rank, |i, k| {
                bf16::from_bits(factors.a_bf16[i * rank + k]).to_f32()
            });
            let b = DMatrix::<f32>::from_fn(rank, cols, |k, j| {
                bf16::from_bits(factors.b_bf16[k * cols + j]).to_f32()
            });
            let m = &a * &b;
            let recon_data: Vec<f32> = (0..rows * cols)
                .map(|i| m[(i / cols, i % cols)])
                .collect();
            let w_approx = HostBf16Matrix::from_f32(&recon_data, rows, cols);
            w.rmse(&w_approx)
        };

        let after = polish_svd_gpu(&backend, &mut factors, &w, 50, 1.0);
        assert!(
            after < perturbed_rmse,
            "GPU polish failed to recover: perturbed={} final={}",
            perturbed_rmse, after
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rsvd_gpu_matches_cpu_on_low_rank() {
        let backend = match rocm_backend::RocmBackend::probe() {
            Ok(b) => b,
            Err(_) => return,
        };
        let rows = 128;
        let cols = 256;
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
        let gpu = compress_tensor_rsvd_gpu(&backend, &w, 3, 10, 42)
            .expect("rsvd_gpu");
        // Reconstruct on CPU, compare to ground truth.
        let mut recon = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for jj in 0..cols {
                let mut s = 0.0f32;
                for kk in 0..3 {
                    let a = bf16::from_bits(gpu.a_bf16[i * 3 + kk]).to_f32();
                    let b = bf16::from_bits(gpu.b_bf16[kk * cols + jj]).to_f32();
                    s += a * b;
                }
                recon[i * cols + jj] = s;
            }
        }
        let sse: f32 = recon.iter().zip(&data).map(|(a, b)| (a - b).powi(2)).sum();
        let rmse = (sse / (rows * cols) as f32).sqrt();
        assert!(rmse < 5e-2, "gpu rsvd recon rmse too large: {}", rmse);
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
