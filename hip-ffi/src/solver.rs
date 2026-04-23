//! Safe hipSOLVER wrapper for dense linear algebra on GPU (QR, thin SVD).
//!
//! hipSOLVER (like LAPACK/cuSOLVER) expects **column-major** matrices. Callers
//! working from row-major tensors must transpose before/after, or reinterpret
//! dims (an m×n row-major matrix is the transpose of an n×m column-major one).

use crate::raw::{
    hipsolverDnCreate, hipsolverDnDestroy, hipsolverDnSetStream,
    hipsolverDnSgeqrf, hipsolverDnSgeqrf_bufferSize,
    hipsolverDnSgesvd, hipsolverDnSgesvd_bufferSize,
    hipsolverDnSorgqr, hipsolverDnSorgqr_bufferSize,
    hipsolverHandle_t, hipsolverStatus_t,
    hipsolverStatus_t_HIPSOLVER_STATUS_SUCCESS,
    hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToDevice,
};
use crate::safe::{hip_check, hip_malloc, HipError};

fn solver_check(status: hipsolverStatus_t) -> Result<(), HipError> {
    if status == hipsolverStatus_t_HIPSOLVER_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HipError::Hip { code: status, msg: format!("hipSOLVER error {}", status) })
    }
}

/// RAII handle for a hipSOLVER dense context.
pub struct SolverHandle {
    handle: hipsolverHandle_t,
}

unsafe impl Send for SolverHandle {}
unsafe impl Sync for SolverHandle {}

impl SolverHandle {
    pub fn new() -> Result<Self, HipError> {
        let mut handle: hipsolverHandle_t = std::ptr::null_mut();
        unsafe { solver_check(hipsolverDnCreate(&mut handle))? };
        Ok(Self { handle })
    }

    /// Route subsequent solver operations through `stream`.
    pub fn set_stream(&self, stream: crate::raw::hipStream_t) -> Result<(), HipError> {
        unsafe { solver_check(hipsolverDnSetStream(self.handle, stream as _)) }
    }

    pub fn raw(&self) -> hipsolverHandle_t {
        self.handle
    }
}

impl Drop for SolverHandle {
    fn drop(&mut self) {
        unsafe { hipsolverDnDestroy(self.handle); }
    }
}

/// QR factorisation for a dense F32 matrix: A = Q · R, materialising Q explicitly.
///
/// **Layout**: hipSOLVER is column-major. `a_f32_device` is a device pointer to
/// an `m × n` column-major matrix (lda = m). On return, `q_out` holds the
/// orthonormal Q (column-major, `m × n`, with only the first `min(m, n)` columns
/// meaningful; for the overdetermined case `m ≥ n` this is all of Q).
///
/// `A` is overwritten in-place during the computation. Caller must provide
/// `q_out` with capacity for `m * n` floats on device; A is copied into it
/// before `orgqr` so the factorisation reuses the copy.
pub fn qr_f32(
    handle: &SolverHandle,
    m: i32,
    n: i32,
    a_f32_device: *mut f32,
    q_out: *mut f32,
) -> Result<(), HipError> {
    if m < 0 || n < 0 {
        return Err(HipError::Hip { code: 0, msg: "qr_f32: negative dims".into() });
    }
    let k = m.min(n);
    let total = (m as usize) * (n as usize);

    unsafe {
        hip_check(hipMemcpy(
            q_out as *mut std::ffi::c_void,
            a_f32_device as *const std::ffi::c_void,
            total * std::mem::size_of::<f32>(),
            hipMemcpyKind_hipMemcpyDeviceToDevice,
        ))?;

        let mut lwork: i32 = 0;
        solver_check(hipsolverDnSgeqrf_bufferSize(
            handle.handle, m, n, q_out, m, &mut lwork,
        ))?;

        let work = hip_malloc::<f32>(lwork.max(1) as usize)?;
        let tau  = hip_malloc::<f32>(k.max(1) as usize)?;
        let dev_info = hip_malloc::<i32>(1)?;

        let r_geqrf = solver_check(hipsolverDnSgeqrf(
            handle.handle, m, n, q_out, m, tau, work, lwork, dev_info,
        ));

        if let Err(e) = r_geqrf {
            let _ = crate::safe::hip_free(work);
            let _ = crate::safe::hip_free(tau);
            let _ = crate::safe::hip_free(dev_info);
            return Err(e);
        }

        // ORGQR needs its own workspace; reuse the allocator pattern for clarity.
        let mut lwork2: i32 = 0;
        let r_bs2 = solver_check(hipsolverDnSorgqr_bufferSize(
            handle.handle, m, k, k, q_out, m, tau, &mut lwork2,
        ));
        if let Err(e) = r_bs2 {
            let _ = crate::safe::hip_free(work);
            let _ = crate::safe::hip_free(tau);
            let _ = crate::safe::hip_free(dev_info);
            return Err(e);
        }

        // If the new workspace is larger, reallocate.
        let work2 = if lwork2 > lwork {
            let _ = crate::safe::hip_free(work);
            hip_malloc::<f32>(lwork2.max(1) as usize)?
        } else {
            work
        };

        let r_orgqr = solver_check(hipsolverDnSorgqr(
            handle.handle, m, k, k, q_out, m, tau, work2, lwork2, dev_info,
        ));

        let _ = crate::safe::hip_free(work2);
        let _ = crate::safe::hip_free(tau);
        let _ = crate::safe::hip_free(dev_info);

        r_orgqr
    }
}

/// Thin SVD for a dense F32 matrix: A = U · diag(S) · Vᵀ.
///
/// **Layout**: hipSOLVER is column-major. `a_f32_device` is an `m × n` column-major
/// matrix (lda = m). Uses jobu='S', jobv='S' (thin form):
/// - `u_out`  : column-major `m × min(m, n)`  (ldu = m)
/// - `s_out`  : `min(m, n)` singular values, non-increasing
/// - `vt_out` : column-major `min(m, n) × n`  (ldv = min(m, n))
///
/// `A` is destroyed in-place.
pub fn svd_f32(
    handle: &SolverHandle,
    m: i32,
    n: i32,
    a_f32_device: *mut f32,
    u_out: *mut f32,
    s_out: *mut f32,
    vt_out: *mut f32,
) -> Result<(), HipError> {
    if m < 0 || n < 0 {
        return Err(HipError::Hip { code: 0, msg: "svd_f32: negative dims".into() });
    }
    let k = m.min(n);

    unsafe {
        let mut lwork: i32 = 0;
        solver_check(hipsolverDnSgesvd_bufferSize(handle.handle, m, n, &mut lwork))?;

        let work = hip_malloc::<f32>(lwork.max(1) as usize)?;
        // rwork is used for complex variants; pass null for real SVD.
        let dev_info = hip_malloc::<i32>(1)?;

        let status = solver_check(hipsolverDnSgesvd(
            handle.handle,
            b'S' as std::os::raw::c_schar,
            b'S' as std::os::raw::c_schar,
            m, n,
            a_f32_device, m,
            s_out,
            u_out, m,
            vt_out, k,
            work, lwork,
            std::ptr::null_mut(),
            dev_info,
        ));

        let _ = crate::safe::hip_free(work);
        let _ = crate::safe::hip_free(dev_info);

        status
    }
}
