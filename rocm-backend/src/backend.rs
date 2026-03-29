use std::sync::Arc;

use compute_core::{
    AnyTensor, BackendError, ComputeBackend, DType,
    params::{
        AddBiasParams, EmbedParams, GemmBf16Params, KvCacheParams,
        RmsNormParams, RopeParams, SgemmParams, SiluMulParams, SoftmaxParams,
    },
};

use hip_ffi::safe::DeviceBuffer;

// ── Type alias for clarity ────────────────────────────────────────────────────

/// BF16 bits are stored as u16 on the device.
type DeviceBufferU16 = DeviceBuffer<u16>;

// ── Downcast helpers ──────────────────────────────────────────────────────────

/// Extract a read-only F32 device pointer from an `AnyTensor`.
///
/// # Panics
/// Panics if the inner buffer is not a `DeviceBuffer<f32>`.
fn f32_ptr(t: &AnyTensor) -> *const f32 {
    t.downcast_ref::<DeviceBuffer<f32>>()
        .expect("RocmBackend: expected F32 DeviceBuffer")
        .as_ptr()
}

/// Extract a mutable F32 device pointer from an `AnyTensor`.
///
/// The underlying allocation is GPU memory; taking `*mut` from `*const` is safe
/// because no aliasing with host memory is possible through this pointer.
fn f32_mut_ptr(t: &AnyTensor) -> *mut f32 {
    f32_ptr(t) as *mut f32
}

/// Extract a read-only u16 (BF16 bits) device pointer from an `AnyTensor`.
///
/// # Panics
/// Panics if the inner buffer is not a `DeviceBuffer<u16>`.
fn u16_ptr(t: &AnyTensor) -> *const u16 {
    t.downcast_ref::<DeviceBufferU16>()
        .expect("RocmBackend: expected BF16/u16 DeviceBuffer")
        .as_ptr()
}

// ── RocmBackend ───────────────────────────────────────────────────────────────

pub struct RocmBackend {
    blas:    hip_ffi::BlasHandle,
    kernels: hip_ffi::GpuKernels,
}

// SAFETY: `BlasHandle` and `GpuKernels` both declare `unsafe impl Send + Sync`
// in hip-ffi.  We propagate that guarantee here.
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Initialise the ROCm backend on device 0.
    ///
    /// Returns `Err(BackendError::NotAvailable)` if HIP cannot be initialised,
    /// hipBLAS context creation fails, or the kernel module fails to load.
    pub fn probe() -> Result<Self, BackendError> {
        hip_ffi::hip_set_device(0)
            .map_err(|e| BackendError::NotAvailable(format!("HIP: {e}")))?;

        let blas = hip_ffi::BlasHandle::new()
            .map_err(|e| BackendError::NotAvailable(format!("hipBLAS: {e}")))?;

        let kernels = hip_ffi::GpuKernels::new()
            .map_err(|e| BackendError::NotAvailable(format!("GpuKernels: {e}")))?;

        Ok(Self { blas, kernels })
    }
}

// ── ComputeBackend impl ───────────────────────────────────────────────────────

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str {
        "ROCm/HIP gfx1032"
    }

    // ── Memory management ────────────────────────────────────────────────────

    fn alloc_f32(&self, shape: &[usize]) -> AnyTensor {
        let n = shape.iter().product();
        let buf = DeviceBuffer::<f32>::alloc(n).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::F32)
    }

    fn alloc_bf16(&self, shape: &[usize]) -> AnyTensor {
        let n = shape.iter().product();
        let buf = DeviceBufferU16::alloc(n).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::BF16)
    }

    fn upload_f32(&self, data: &[f32], shape: &[usize]) -> AnyTensor {
        let buf = DeviceBuffer::<f32>::from_slice(data).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::F32)
    }

    fn upload_bf16(&self, data: &[u16], shape: &[usize]) -> AnyTensor {
        // `DeviceBuffer::from_slice` calls `hip_h2d` generically over T = u16.
        let buf = DeviceBufferU16::from_slice(data).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::BF16)
    }

    fn download_f32(&self, t: &AnyTensor) -> Vec<f32> {
        let buf = t.downcast_ref::<DeviceBuffer<f32>>().unwrap();
        buf.to_host().unwrap()
    }

    // ── GEMM ─────────────────────────────────────────────────────────────────

    fn gemm_bf16_f32(&self, p: &GemmBf16Params) -> Result<(), BackendError> {
        let a_ptr = unsafe { f32_ptr(p.a).add(p.a_offset) };
        let b_ptr = u16_ptr(p.b);
        let c_ptr = f32_mut_ptr(p.c);

        hip_ffi::gemm_bf16_f32(
            &self.blas,
            p.m as i32,
            p.n as i32,
            p.k as i32,
            p.alpha,
            p.beta,
            a_ptr,
            b_ptr,
            c_ptr,
        )
        .map_err(|e| BackendError::Gemm(e.to_string()))
    }

    fn sgemm(&self, p: &SgemmParams) -> Result<(), BackendError> {
        let a_ptr = unsafe { f32_ptr(p.a).add(p.a_offset) };
        let b_ptr = unsafe { f32_ptr(p.b).add(p.b_offset) };
        let c_ptr = unsafe { f32_mut_ptr(p.c).add(p.c_offset) };

        hip_ffi::sgemm_strided(
            &self.blas,
            p.trans_a,
            p.trans_b,
            p.m as i32,
            p.n as i32,
            p.k as i32,
            p.alpha,
            p.beta,
            a_ptr,
            p.lda as i32,
            b_ptr,
            p.ldb as i32,
            c_ptr,
            p.ldc as i32,
        )
        .map_err(|e| BackendError::Gemm(e.to_string()))
    }

    // ── Custom kernels ────────────────────────────────────────────────────────

    fn embed_lookup(&self, p: &EmbedParams) -> Result<(), BackendError> {
        // Upload token IDs (host slice) to a temporary GPU buffer.
        let tok_buf = DeviceBuffer::<u32>::from_slice(p.token_ids)
            .map_err(|e| BackendError::Transfer(e.to_string()))?;

        unsafe {
            self.kernels.embed_lookup(
                u16_ptr(p.vocab),
                tok_buf.as_ptr(),
                f32_mut_ptr(p.out),
                p.n_tokens as i32,
                p.hidden_size as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn rms_norm(&self, p: &RmsNormParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.rms_norm(
                f32_mut_ptr(p.output),
                f32_ptr(p.input),
                f32_ptr(p.weight),
                p.n_tokens as i32,
                p.hidden_size as i32,
                p.eps,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn rope_inplace(&self, p: &RopeParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.rope_inplace(
                f32_mut_ptr(p.tensor),
                p.n_tokens as i32,
                p.n_heads as i32,
                p.head_dim as i32,
                p.pos_offset as i32,
                p.theta,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn softmax_rows(&self, p: &SoftmaxParams) -> Result<(), BackendError> {
        let scores_ptr = unsafe { f32_mut_ptr(p.scores).add(p.offset_elems) };
        unsafe {
            self.kernels.softmax_rows(
                scores_ptr,
                p.n_rows as i32,
                p.row_len as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn silu_mul_inplace(&self, p: &SiluMulParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.silu_mul_inplace(
                f32_mut_ptr(p.gate),
                f32_ptr(p.up),
                p.n_tokens as i32,
                p.intermediate_size as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn copy_to_kv_cache(&self, p: &KvCacheParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.copy_to_kv_cache(
                f32_mut_ptr(p.cache),
                f32_ptr(p.new_kv),
                p.n_new_tokens as i32,
                p.kv_dim as i32,
                p.insert_pos as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn add_bias_inplace(&self, p: &AddBiasParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.add_bias_inplace(
                f32_mut_ptr(p.tensor),
                f32_ptr(p.bias),
                p.n_tokens as i32,
                p.dim as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    // ── Sync and info ─────────────────────────────────────────────────────────

    fn sync(&self) -> Result<(), BackendError> {
        hip_ffi::safe::hip_device_sync()
            .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn mem_info(&self) -> Option<(usize, usize)> {
        hip_ffi::safe::hip_mem_info().ok()
    }
}
