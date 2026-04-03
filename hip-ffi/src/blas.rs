//! Safe hipBLAS wrapper for GPU matrix multiplication.
//! Supports BF16 weights × F32 activations → F32 output (mixed precision GEMM).

use crate::safe::HipError;

mod raw {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/hipblas_bindings.rs"));
}

use raw::*;

fn blas_check(status: hipblasStatus_t) -> Result<(), HipError> {
    if status == hipblasStatus_t_HIPBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HipError::Hip { code: status, msg: format!("hipBLAS error {}", status) })
    }
}

/// RAII handle for a hipBLAS context.
pub struct BlasHandle {
    handle: hipblasHandle_t,
}

unsafe impl Send for BlasHandle {}
unsafe impl Sync for BlasHandle {}

impl BlasHandle {
    pub fn new() -> Result<Self, HipError> {
        let mut handle = std::ptr::null_mut();
        unsafe { blas_check(hipblasCreate(&mut handle))? };
        Ok(Self { handle })
    }
}

impl Drop for BlasHandle {
    fn drop(&mut self) {
        unsafe { hipblasDestroy(self.handle); }
    }
}

/// GEMM: BF16 weights × F32 activations → F32 output.
///
/// Computes: C[m, n] = alpha × A[m, k] × B^T[n, k] + beta × C[m, n]
///
/// - A (activations): F32, shape [m, k], row-major
/// - B (weights): BF16, shape [n, k], row-major (stored as [out, in])
/// - C (output): F32, shape [m, n], row-major
///
/// hipBLAS is column-major, so we swap A/B and use OP_T for the activation matrix
/// to achieve the equivalent row-major semantics.
pub fn gemm_bf16_f32(
    handle: &BlasHandle,
    m: i32,   // output rows (n_tokens)
    n: i32,   // output cols (out_dim)
    k: i32,   // inner dim (hidden)
    alpha: f32,
    beta: f32,
    a_bf16: *const u16,   // activations [m, k] (BF16 bits)
    b_bf16: *const u16,   // weights [n, k] (BF16 bits)
    c_f32: *mut f32,      // output [m, n]
) -> Result<(), HipError> {
    // Column-major interpretation:
    // We want C_col[n, m] = B_col[n, k] × A_col^T[k, m]
    // Which gives row-major: C[m, n] = A[m, k] × B^T[n, k]
    unsafe {
        blas_check(hipblasGemmEx_v2(
            handle.handle,
            hipblasOperation_t_HIPBLAS_OP_T,  // weights: transposed in col-major
            hipblasOperation_t_HIPBLAS_OP_N,  // activations: non-transposed
            n, m, k,
            &alpha as *const f32 as *const std::ffi::c_void,
            b_bf16 as *const std::ffi::c_void,
            hipDataType_HIP_R_16BF,
            k,
            a_bf16 as *const std::ffi::c_void,
            hipDataType_HIP_R_16BF,
            k,
            &beta as *const f32 as *const std::ffi::c_void,
            c_f32 as *mut std::ffi::c_void,
            hipDataType_HIP_R_32F,
            n,
            hipblasComputeType_t_HIPBLAS_COMPUTE_32F,
            hipblasGemmAlgo_t_HIPBLAS_GEMM_DEFAULT,
        ))
    }
}

/// General strided F32 GEMM with custom leading dimensions.
///
/// Computes (row-major): C[m, n] = alpha × op(A) × op(B) + beta × C
/// - trans_a=false: A is [m, k], lda ≥ k
/// - trans_a=true:  A is [k, m], lda ≥ m  (A gets transposed)
/// - trans_b=false: B is [k, n], ldb ≥ n
/// - trans_b=true:  B is [n, k], ldb ≥ k  (B gets transposed)
///
/// hipBLAS is column-major, so we feed the matrices swapped.
pub fn sgemm_strided(
    handle: &BlasHandle,
    trans_a: bool,
    trans_b: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    c: *mut f32,
    ldc: i32,
) -> Result<(), HipError> {
    // Row-major C[m,n] = op(A) × op(B)  →  col-major: C^T = op(B)^T × op(A)^T
    // We swap A↔B in the BLAS call. The transpose flags map DIRECTLY:
    //   row-major trans_b=true  →  BLAS ta = OP_T  (transpose the B data)
    //   row-major trans_a=true  →  BLAS tb = OP_T  (transpose the A data)
    let ta = if trans_b { hipblasOperation_t_HIPBLAS_OP_T } else { hipblasOperation_t_HIPBLAS_OP_N };
    let tb = if trans_a { hipblasOperation_t_HIPBLAS_OP_T } else { hipblasOperation_t_HIPBLAS_OP_N };
    unsafe {
        blas_check(hipblasGemmEx_v2(
            handle.handle,
            ta, tb,
            n, m, k,
            &alpha as *const f32 as *const std::ffi::c_void,
            b as *const std::ffi::c_void, hipDataType_HIP_R_32F, ldb,
            a as *const std::ffi::c_void, hipDataType_HIP_R_32F, lda,
            &beta as *const f32 as *const std::ffi::c_void,
            c as *mut std::ffi::c_void, hipDataType_HIP_R_32F, ldc,
            hipblasComputeType_t_HIPBLAS_COMPUTE_32F,
            hipblasGemmAlgo_t_HIPBLAS_GEMM_DEFAULT,
        ))
    }
}

/// Strided batched F32 GEMM: perform m×n×k GEMMs with fixed strides.
///
/// Computes (row-major): C[b, m, n] = alpha × op(A[b, m, k]) × op(B[b, k, n]) + beta × C[b, m, n]
/// where b is the batch index (0 to batch_count-1).
///
/// - stride_a: offset (in elements) between A[b] and A[b+1]
/// - stride_b: offset (in elements) between B[b] and B[b+1]
/// - stride_c: offset (in elements) between C[b] and C[b+1]
/// - trans_a=false: A is [m, k], lda ≥ k
/// - trans_a=true:  A is [k, m], lda ≥ m  (A gets transposed)
/// - trans_b=false: B is [k, n], ldb ≥ n
/// - trans_b=true:  B is [n, k], ldb ≥ k  (B gets transposed)
///
/// Used for multi-head attention: all 32 attention heads in a single BLAS call.
pub fn sgemm_strided_batched(
    handle: &BlasHandle,
    trans_a: bool,
    trans_b: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
    a: *const f32,
    lda: i32,
    stride_a: i64,
    b: *const f32,
    ldb: i32,
    stride_b: i64,
    c: *mut f32,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) -> Result<(), HipError> {
    // Row-major C[b, m, n] = op(A[b]) × op(B[b])  →  col-major: C^T = op(B)^T × op(A)^T
    // We swap A↔B in the BLAS call. The transpose flags map DIRECTLY:
    //   row-major trans_b=true  →  BLAS ta = OP_T  (transpose the B data)
    //   row-major trans_a=true  →  BLAS tb = OP_T  (transpose the A data)
    let ta = if trans_b { hipblasOperation_t_HIPBLAS_OP_T } else { hipblasOperation_t_HIPBLAS_OP_N };
    let tb = if trans_a { hipblasOperation_t_HIPBLAS_OP_T } else { hipblasOperation_t_HIPBLAS_OP_N };
    unsafe {
        blas_check(hipblasSgemmStridedBatched(
            handle.handle,
            ta, tb,
            n, m, k,
            &alpha as *const f32,
            b, ldb, stride_b,
            a, lda, stride_a,
            &beta as *const f32,
            c, ldc, stride_c,
            batch_count,
        ))
    }
}

/// Pure F32 GEMM for small ops (biases, scratchpad).
/// C[m, n] = alpha × A[m, k] × B^T[n, k] + beta × C[m, n]
pub fn sgemm(
    handle: &BlasHandle,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) -> Result<(), HipError> {
    unsafe {
        blas_check(hipblasGemmEx_v2(
            handle.handle,
            hipblasOperation_t_HIPBLAS_OP_T,
            hipblasOperation_t_HIPBLAS_OP_N,
            n, m, k,
            &alpha as *const f32 as *const std::ffi::c_void,
            b as *const std::ffi::c_void, hipDataType_HIP_R_32F, k,
            a as *const std::ffi::c_void, hipDataType_HIP_R_32F, k,
            &beta as *const f32 as *const std::ffi::c_void,
            c as *mut std::ffi::c_void, hipDataType_HIP_R_32F, n,
            hipblasComputeType_t_HIPBLAS_COMPUTE_32F,
            hipblasGemmAlgo_t_HIPBLAS_GEMM_DEFAULT,
        ))
    }
}
