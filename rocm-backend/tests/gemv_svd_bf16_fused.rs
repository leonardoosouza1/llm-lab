// Numerical equivalence test: fused gemv_svd_bf16 kernel vs the default 2-gemm path.
//
// Runs only when an AMD GPU is detectable via `RocmBackend::probe()`. If probe
// fails (no /dev/kfd, no driver), the test prints a notice and returns Ok.
//
// Shapes chosen per task requirements: rows=256, cols=512, rank=64.
// Rank cap is 1024, (cols+rank) cap is 12288 — well under both.

use compute_core::{
    params::{GemvSvdBf16Params},
    ComputeBackend,
};
use rocm_backend::RocmBackend;

fn f32_to_bf16_bits(v: f32) -> u16 {
    // Round-to-nearest-even truncation to upper 16 bits
    let bits = v.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

#[test]
fn fused_matches_default_2gemm() {
    let backend = match RocmBackend::probe() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[skip] RocmBackend unavailable: {e}. Build-only verification.");
            return;
        }
    };

    let rows: usize = 256;
    let cols: usize = 512;
    let rank: usize = 64;

    // Deterministic pseudo-random fill (no extra deps)
    let rand_f32 = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = ((s >> 33) as u32) as f32 / (u32::MAX as f32);
            (x - 0.5) * 0.1  // small scale to avoid bf16 overflow accumulations
        }).collect()
    };

    let a_f32 = rand_f32(1, rows * rank);
    let b_f32 = rand_f32(2, rank * cols);
    let x_f32 = rand_f32(3, cols);

    // BF16 quantize (kernel reads bf16, so both paths see same bits)
    let a_bf16: Vec<u16> = a_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let b_bf16: Vec<u16> = b_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    let a_t = backend.upload_bf16(&a_bf16, &[rows, rank]);
    let b_t = backend.upload_bf16(&b_bf16, &[rank, cols]);
    let x_t = backend.upload_f32(&x_f32, &[cols]);
    let scratch_t = backend.alloc_f32(&[rank]);

    let out_fused = backend.alloc_f32(&[rows]);
    let out_ref   = backend.alloc_f32(&[rows]);

    // ── Fused path ────────────────────────────────────────────────────────
    backend.gemv_svd_bf16(&GemvSvdBf16Params {
        m: 1,
        rows, cols, rank,
        a: &a_t, b: &b_t, x: &x_t,
        x_offset: 0,
        scratch: &scratch_t,
        out: &out_fused,
        beta: 0.0,
    }).expect("fused gemv_svd_bf16");
    backend.device_sync();
    let y_fused = backend.download_f32(&out_fused);

    // ── Reference path: manually call 2 gemms (mirrors default trait impl) ─
    use compute_core::params::GemmBf16Params;
    backend.gemm_bf16_f32(&GemmBf16Params {
        m: 1, n: rank, k: cols,
        alpha: 1.0, beta: 0.0,
        a: &x_t, a_offset: 0,
        b: &b_t, c: &scratch_t,
    }).expect("gemm step 1");
    backend.gemm_bf16_f32(&GemmBf16Params {
        m: 1, n: rows, k: rank,
        alpha: 1.0, beta: 0.0,
        a: &scratch_t, a_offset: 0,
        b: &a_t, c: &out_ref,
    }).expect("gemm step 2");
    backend.device_sync();
    let y_ref = backend.download_f32(&out_ref);

    // ── Numerical equivalence within 1e-3 rel error ───────────────────────
    let mut max_rel = 0.0f32;
    let mut max_abs = 0.0f32;
    for i in 0..rows {
        let a = y_fused[i];
        let b = y_ref[i];
        let abs = (a - b).abs();
        let rel = abs / (b.abs().max(1e-6));
        if rel > max_rel { max_rel = rel; }
        if abs > max_abs { max_abs = abs; }
    }
    eprintln!("gemv_svd_bf16_fused: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}");
    assert!(max_rel < 1e-3 || max_abs < 1e-4,
        "fused vs default mismatch: max_rel={max_rel}, max_abs={max_abs}");
}
