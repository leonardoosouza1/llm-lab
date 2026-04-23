// hipSOLVER GPU integration: QR and thin SVD sanity checks.
//
// Skips gracefully when a ROCm-capable GPU is not present (probe() fails).

use compute_core::ComputeBackend;
use rocm_backend::RocmBackend;

/// QᵀQ ≈ I for the Q produced by qr_f32 on a tall, skinny matrix.
#[test]
fn qr_produces_orthonormal_q() {
    let backend = match RocmBackend::probe() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[skip] RocmBackend unavailable: {e}");
            return;
        }
    };

    // 4×3 matrix. In column-major this is stored as 3 columns of 4 rows each.
    // Columns chosen to be linearly independent.
    //   col0 = [1, 2, 0, 1]
    //   col1 = [0, 1, 1, 2]
    //   col2 = [1, 0, 2, 1]
    let rows = 4usize;
    let cols = 3usize;
    let a_col_major: Vec<f32> = vec![
        1.0, 2.0, 0.0, 1.0,   // col 0
        0.0, 1.0, 1.0, 2.0,   // col 1
        1.0, 0.0, 2.0, 1.0,   // col 2
    ];

    let a_t = backend.upload_f32(&a_col_major, &[rows * cols]);
    let q_t = backend.alloc_f32(&[rows * cols]);

    backend.qr_f32(&a_t, &q_t, rows, cols).expect("qr_f32");
    backend.device_sync();

    let q = backend.download_f32(&q_t);
    assert_eq!(q.len(), rows * cols);

    // Column-major Q is rows × cols; QᵀQ should be the cols × cols identity.
    // (Q has orthonormal columns.)
    for i in 0..cols {
        for j in 0..cols {
            let mut dot = 0.0f32;
            for r in 0..rows {
                dot += q[i * rows + r] * q[j * rows + r];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-4,
                "QᵀQ[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

/// SVD of diag(3, 2, 1) must return singular values [3, 2, 1].
#[test]
fn svd_of_diagonal_matrix() {
    let backend = match RocmBackend::probe() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[skip] RocmBackend unavailable: {e}");
            return;
        }
    };

    let n = 3usize;
    // diag(3, 2, 1) is layout-invariant (same in row-major and column-major).
    let a_col_major: Vec<f32> = vec![
        3.0, 0.0, 0.0,   // col 0
        0.0, 2.0, 0.0,   // col 1
        0.0, 0.0, 1.0,   // col 2
    ];

    let a_t  = backend.upload_f32(&a_col_major, &[n * n]);
    let u_t  = backend.alloc_f32(&[n * n]);
    let s_t  = backend.alloc_f32(&[n]);
    let vt_t = backend.alloc_f32(&[n * n]);

    backend.svd_f32(&a_t, &u_t, &s_t, &vt_t, n, n).expect("svd_f32");
    backend.device_sync();

    let s = backend.download_f32(&s_t);
    eprintln!("singular values: {:?}", s);

    // hipSOLVER returns S in non-increasing order.
    let expected = [3.0f32, 2.0, 1.0];
    for (i, &want) in expected.iter().enumerate() {
        let got = s[i];
        assert!(
            (got - want).abs() < 1e-4,
            "singular value {i}: got {got}, want {want}"
        );
    }
}
