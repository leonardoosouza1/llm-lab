use std::sync::Arc;

use compute_core::{
    params::{
        AddBiasParams, CfcForwardParams, EmbedParams, FusedNormBiasRopeParams, GeluMulParams, GemmBf16Params, GemvQ4Params, GemvQ4KParams, GemvQ6KParams,
        GravityScoreParams, KvCacheParams, KvCacheQuantizedParams, LayerNormParams, RmsNormParams,
        RopeParams, SgemmParams, SgemmStridedBatchedParams, SiluMulParams, SoftmaxParams, SoftmaxCausalParams,
        QuantizeQ8Params, GemvQ4KQ8Params, GemvQ6KQ8Params, GemvQ4KF32Params, GemvQ6KF32Params, GemvBf16F32Params,
        FusedRmsQkvQ4KF32Params, FusedRmsGateUpSiluQ4KF32Params, FusedRmsQkvQ4KQ6KF32Params,
        FusedRmsQkvQ4Params, FusedRmsGateUpSiluQ4Params,
        KvCacheInt8AppendParams, KvCacheInt8ReadParams,
        KvCacheWhtAppendParams, KvCacheWhtReadParams,
        GatherRowsBf16Params, GatherColsBf16Params,
        KuramotoCoherenceParams, TopkSelectParams, FusedAttentionDecodeParams, FusedGateUpSiluParams, FusedQkvParams, FusedRmsNormQuantizeParams, FusedRopeKvCopyParams,
        GemvTernaryF32Params, GemvW8A16Params, GemvSvdBf16Params,
    },
    AnyTensor, BackendError, ComputeBackend, DType,
};

use hip_ffi::safe::DeviceBuffer;

// ── Type alias for clarity ────────────────────────────────────────────────────

/// BF16 bits are stored as u16 on the device.
type DeviceBufferU16 = DeviceBuffer<u16>;

/// Q4 packed data is stored as u8 on the device.
type DeviceBufferU8 = DeviceBuffer<u8>;

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

/// Extract a read-only u8 (Q4 packed) device pointer from an `AnyTensor`.
///
/// # Panics
/// Panics if the inner buffer is not a `DeviceBuffer<u8>`.
fn u8_ptr(t: &AnyTensor) -> *const u8 {
    t.downcast_ref::<DeviceBuffer<u8>>()
        .expect("RocmBackend: expected Q4/u8 DeviceBuffer")
        .as_ptr()
}

/// Extract a read-only i8 device pointer from an `AnyTensor` (stored as u8 buffer).
/// Used for W8A16 INT8 weights: data is uploaded as u8 bytes, cast to i8 for the kernel.
fn i8_ptr(t: &AnyTensor) -> *const i8 {
    u8_ptr(t) as *const i8
}

/// Extract a read-only i32 device pointer from an `AnyTensor`.
fn i32_ptr(t: &AnyTensor) -> *const i32 {
    t.downcast_ref::<DeviceBuffer<i32>>()
        .expect("RocmBackend: expected I32 DeviceBuffer")
        .as_ptr()
}

fn u16_mut_ptr(t: &AnyTensor) -> *mut u16 {
    u16_ptr(t) as *mut u16
}

/// Extract a mutable u8 (Q4 packed) device pointer from an `AnyTensor`.
fn u8_mut_ptr(t: &AnyTensor) -> *mut u8 {
    u8_ptr(t) as *mut u8
}

// ── RocmBackend ───────────────────────────────────────────────────────────────

pub struct RocmBackend {
    blas: hip_ffi::BlasHandle,
    // Lazy-init: only constructed when QR/SVD are first requested (avoids
    // paying hipSOLVER init cost for inference-only workloads).
    solver: std::sync::OnceLock<hip_ffi::SolverHandle>,
    kernels: hip_ffi::GpuKernels,
    tok_buf: std::sync::Mutex<hip_ffi::safe::DeviceBuffer<u32>>,
    act_bf16: std::sync::Mutex<hip_ffi::safe::DeviceBuffer<u16>>,
    argmax_buf: hip_ffi::safe::DeviceBuffer<i32>,
    /// Device-resident scalar (i32, 1 elem) for HIP graph replay: holds seq_pos / current_len.
    /// Updated via hipMemcpyAsync per token so kernels read latest value during graph replay.
    pos_dev: hip_ffi::safe::DeviceBuffer<i32>,
    // HIP stream + graph capture state
    stream: hip_ffi::raw::hipStream_t,
    graph_exec: std::sync::Mutex<Option<hip_ffi::raw::hipGraphExec_t>>,
    capturing: std::sync::atomic::AtomicBool,
    /// Use Wave64 GEMV kernels (IARA_WAVE64=1) for better RDNA2 bandwidth utilization
    use_wave64: bool,
    use_mmvq_llama: bool,
    /// Use Flash Attention 2 kernel (IARA_FLASH_ATTN=1) instead of original fused decode
    use_flash_attn: bool,
    /// Use V2 kernels (IARA_V2=1): wave32 + dp4a + multi-row + no LDS
    use_v2: bool,
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
        hip_ffi::hip_set_device(0).map_err(|e| BackendError::NotAvailable(format!("HIP: {e}")))?;

        // Create a dedicated stream for all operations (required for graph capture)
        let stream = hip_ffi::hip_stream_create()
            .map_err(|e| BackendError::NotAvailable(format!("HIP stream: {e}")))?;

        let blas = hip_ffi::BlasHandle::new()
            .map_err(|e| BackendError::NotAvailable(format!("hipBLAS: {e}")))?;

        // Route hipBLAS through our stream
        blas.set_stream(stream)
            .map_err(|e| BackendError::NotAvailable(format!("hipBLAS set_stream: {e}")))?;

        let mut kernels = hip_ffi::GpuKernels::new()
            .map_err(|e| BackendError::NotAvailable(format!("GpuKernels: {e}")))?;

        // Route kernel launches through our stream
        kernels.set_stream(stream);

        // Optional HSA AQL fast path (IARA_HSA=1).
        if std::env::var("IARA_HSA").map(|v| v == "1").unwrap_or(false) {
            if let Err(e) = kernels.enable_hsa() {
                eprintln!("⚠️  HSA dispatcher failed to init: {}", e);
            }
        }

        let tok_buf = hip_ffi::safe::DeviceBuffer::<u32>::alloc(4096)
            .map_err(|e| BackendError::NotAvailable(format!("HIP alloc: {e}")))?;

        let act_bf16 = hip_ffi::safe::DeviceBuffer::<u16>::alloc(4096 * 8192)
            .map_err(|e| BackendError::NotAvailable(format!("HIP alloc act_bf16: {e}")))?;

        let argmax_buf = hip_ffi::safe::DeviceBuffer::<i32>::alloc(1)
            .map_err(|e| BackendError::NotAvailable(format!("HIP alloc argmax: {e}")))?;

        let pos_dev = hip_ffi::safe::DeviceBuffer::<i32>::alloc(1)
            .map_err(|e| BackendError::NotAvailable(format!("HIP alloc pos_dev: {e}")))?;

        // MMVQ (llama.cpp-ported Q4K wave32 dp4a kernel) — fastest Q4K GEMV
        // empirically. Default ON; opt-out via IARA_MMVQ=0.
        let use_mmvq_llama = std::env::var("IARA_MMVQ").map(|v| v != "0").unwrap_or(true);
        let use_v2 = !use_mmvq_llama && std::env::var("IARA_V2").map(|v| v != "0").unwrap_or(true);
        let use_wave64 = !use_mmvq_llama && !use_v2 && std::env::var("IARA_WAVE64").map(|v| v != "0").unwrap_or(true);
        let use_flash_attn = std::env::var("IARA_FLASH_ATTN").map(|v| v != "0").unwrap_or(true);
        eprintln!("ROCm backend: using dedicated HIP stream + graph capture support{}{}{}",
            if use_v2 { " + V2 GEMV (wave32/dp4a/no-LDS)" } else if use_wave64 { " + Wave64 GEMV" } else { "" },
            if use_flash_attn { " + Flash Attention 2" } else { "" },
            if use_v2 { "" } else { "" });

        Ok(Self {
            blas,
            solver: std::sync::OnceLock::new(),
            kernels,
            tok_buf: std::sync::Mutex::new(tok_buf),
            act_bf16: std::sync::Mutex::new(act_bf16),
            argmax_buf,
            pos_dev,
            stream,
            graph_exec: std::sync::Mutex::new(None),
            capturing: std::sync::atomic::AtomicBool::new(false),
            use_wave64,
            use_flash_attn,
            use_v2,
            use_mmvq_llama,
        })
    }

    /// Lazily create (and cache) a hipSOLVER handle bound to the backend stream.
    fn solver_handle(&self) -> Result<&hip_ffi::SolverHandle, BackendError> {
        if let Some(h) = self.solver.get() {
            return Ok(h);
        }
        let h = hip_ffi::SolverHandle::new()
            .map_err(|e| BackendError::NotAvailable(format!("hipSOLVER: {e}")))?;
        h.set_stream(self.stream)
            .map_err(|e| BackendError::NotAvailable(format!("hipSOLVER set_stream: {e}")))?;
        // Tolerate a race: if another thread set it, discard ours.
        let _ = self.solver.set(h);
        Ok(self.solver.get().unwrap())
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
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected,
            "upload_bf16: data.len()={} != shape.product()={} shape={:?}",
            data.len(), expected, shape);
        // `DeviceBuffer::from_slice` calls `hip_h2d` generically over T = u16.
        let buf = DeviceBufferU16::from_slice(data).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::BF16)
    }

    fn upload_q4(&self, data: &[u8], shape: &[usize]) -> AnyTensor {
        // Upload Q4 packed data (2 valores por byte) como u8 no device.
        let buf = DeviceBufferU8::from_slice(data).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::Q4)
    }

    fn upload_ternary(&self, packed: &[u8], shape: &[usize]) -> AnyTensor {
        // Upload ternary packed data (4 weights per byte, 2 bits each) as raw u8 on device.
        let buf = DeviceBufferU8::from_slice(packed).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::Q4) // reuse Q4 tag for u8 storage
    }

    fn upload_i32(&self, data: &[i32], shape: &[usize]) -> AnyTensor {
        let buf = DeviceBuffer::<i32>::from_slice(data).unwrap();
        // Reuse F32 dtype tag — the kernel treats it as i32 via raw pointer
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::F32)
    }

    fn alloc_q4(&self, shape: &[usize]) -> AnyTensor {
        // Aloca buffer u8 não inicializado para dados Q4 comprimidos.
        let n = shape.iter().product();
        let buf = DeviceBufferU8::alloc(n).unwrap();
        AnyTensor::new(Arc::new(buf), shape.to_vec(), DType::Q4)
    }

    fn download_f32(&self, t: &AnyTensor) -> Vec<f32> {
        let buf = t.downcast_ref::<DeviceBuffer<f32>>().unwrap();
        buf.to_host().unwrap()
    }

    fn device_sync(&self) {
        unsafe { hip_ffi::hip_stream_sync(self.stream).ok(); }
    }

    fn download_bf16_as_f32(&self, t: &AnyTensor) -> Vec<f32> {
        let buf = t.downcast_ref::<DeviceBufferU16>()
            .expect("download_bf16_as_f32: expected BF16/u16 DeviceBuffer");
        let u16_host = buf.to_host().unwrap();
        u16_host.iter().map(|&bits| f32::from_bits((bits as u32) << 16)).collect()
    }

    // ── GEMM ─────────────────────────────────────────────────────────────────

    fn gemm_bf16_f32(&self, p: &GemmBf16Params) -> Result<(), BackendError> {
        let a_ptr = unsafe { f32_ptr(p.a).add(p.a_offset) };
        let b_ptr = u16_ptr(p.b);
        let c_ptr = f32_mut_ptr(p.c);

        let mut act_guard = self.act_bf16.lock().unwrap();
        let expected_elems = p.m * p.k;
        if expected_elems > act_guard.len() {
            return Err(BackendError::Gemm(format!(
                "act_bf16 overflow: {expected_elems} elems"
            )));
        }

        unsafe {
            self.kernels
                .f32_to_bf16(a_ptr, act_guard.as_mut_ptr(), expected_elems as i32)
                .map_err(|e| BackendError::Gemm(format!("Cast: {e}")))?;

            // No sync needed: f32_to_bf16 and hipblasGemmEx both run on the
            // default stream (null), so they are already serialized by the HIP
            // runtime.  The old explicit sync was flushing the entire GPU pipeline
            // ~140 times per token, killing throughput.
        }

        let r = hip_ffi::gemm_bf16_f32(
            &self.blas,
            p.m as i32,
            p.n as i32,
            p.k as i32,
            p.alpha,
            p.beta,
            act_guard.as_ptr(),
            b_ptr,
            c_ptr,
        )
        .map_err(|e| BackendError::Gemm(e.to_string()));
        // HSA drain disabled — caused regression in baseline
        r
    }

    fn sgemm(&self, p: &SgemmParams) -> Result<(), BackendError> {
        let a_ptr = unsafe { f32_ptr(p.a).add(p.a_offset) };
        let b_ptr = unsafe { f32_ptr(p.b).add(p.b_offset) };
        let c_ptr = unsafe { f32_mut_ptr(p.c).add(p.c_offset) };

        let r = hip_ffi::sgemm_strided(
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
        .map_err(|e| BackendError::Gemm(e.to_string()));
        // HSA drain disabled — caused regression in baseline
        r
    }

    fn sgemm_strided_batched(&self, p: &SgemmStridedBatchedParams) -> Result<(), BackendError> {
        let a_ptr = unsafe { f32_ptr(p.a).add(p.a_offset) };
        let b_ptr = unsafe { f32_ptr(p.b).add(p.b_offset) };
        let c_ptr = unsafe { f32_mut_ptr(p.c).add(p.c_offset) };

        let r = hip_ffi::sgemm_strided_batched(
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
            p.stride_a as i64,
            b_ptr,
            p.ldb as i32,
            p.stride_b as i64,
            c_ptr,
            p.ldc as i32,
            p.stride_c as i64,
            p.batch_count as i32,
        )
        .map_err(|e| BackendError::Gemm(e.to_string()));
        // HSA drain disabled — caused regression in baseline
        r
    }

    // ── Custom kernels ────────────────────────────────────────────────────────

    fn embed_lookup(&self, p: &EmbedParams) -> Result<(), BackendError> {
        // Obter lock do buffer pré-alocado
        let mut buf_guard = self.tok_buf.lock().unwrap();

        if p.token_ids.len() > buf_guard.len() {
            return Err(BackendError::Transfer(
                "Muitos tokens para tok_buf pre-alocado".into(),
            ));
        }

        // Upload token IDs (host slice) to the pre-allocated GPU buffer via our stream.
        // Drain HIP stream right after so the upcoming HSA-side embed kernel sees
        // the token bytes (HIP stream and HSA queue are independent).
        unsafe {
            hip_ffi::safe::hip_h2d_async(buf_guard.as_mut_ptr(), p.token_ids, self.stream)
                .map_err(|e| BackendError::Transfer(e.to_string()))?;
            if self.kernels.hsa.is_some() {
                hip_ffi::raw::hipStreamSynchronize(self.stream);
            }
        }

        unsafe {
            self.kernels.embed_lookup(
                u16_ptr(p.vocab),
                buf_guard.as_ptr(),
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

    fn layer_norm(&self, _p: &LayerNormParams) -> Result<(), BackendError> {
        Err(BackendError::Kernel(
            "LayerNorm not yet implemented — use RMSNorm models (Qwen, Llama, Mistral)".into(),
        ))
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
            self.kernels
                .softmax_rows(scores_ptr, p.n_rows as i32, p.row_len as i32)
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn softmax_rows_causal(&self, p: &SoftmaxCausalParams) -> Result<(), BackendError> {
        let scores_ptr = unsafe { f32_mut_ptr(p.scores).add(p.offset_elems) };
        unsafe {
            self.kernels.softmax_rows_causal(
                scores_ptr,
                p.n_rows as i32,
                p.row_len as i32,
                p.n_tokens as i32,
                p.query_offset as i32,
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

    fn gelu_mul_inplace(&self, _p: &GeluMulParams) -> Result<(), BackendError> {
        Err(BackendError::Kernel(
            "GELU activation not yet implemented — use SiLU models (Qwen, Llama, Mistral)".into(),
        ))
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

    fn gemv_q4(&self, p: &GemvQ4Params) -> Result<(), BackendError> {
        // GEMV dequantizado: pesos INT4 comprimido × ativações F32 → F32
        // x_offset permite selecionar token específico (ex: lm_head usa último token)
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        unsafe {
            self.kernels.dequant_gemv_q4(
                u8_ptr(p.packed),
                f32_ptr(p.scales),
                f32_ptr(p.zeros),
                x_ptr,
                f32_mut_ptr(p.out),
                p.m as i32,
                p.n as i32,
                p.k as i32,
                p.group_size as i32,
                p.beta,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_q4k(&self, p: &GemvQ4KParams) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        // RDNA2 decode optimisation: use 1-warp-per-row kernel for m≤4 (decode).
        // Eliminates cross-warp LDS reduction; all 1536 blocks fit in 28 CUs simultaneously.
        let result = if p.m <= 4 {
            unsafe {
                self.kernels.dequant_gemv_q4k_w32(
                    u8_ptr(p.data), x_ptr, f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            }
        } else {
            unsafe {
                self.kernels.dequant_gemv_q4k(
                    u8_ptr(p.data), x_ptr, f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            }
        };
        result.map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_q6k(&self, p: &GemvQ6KParams) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        unsafe {
            self.kernels.dequant_gemv_q6k(
                u8_ptr(p.data),
                x_ptr,
                f32_mut_ptr(p.out),
                p.m as i32,
                p.n as i32,
                p.k as i32,
                p.beta,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn quantize_q8(&self, p: &QuantizeQ8Params) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        unsafe {
            self.kernels.quantize_f32_q8_1(x_ptr, u8_mut_ptr(p.out), p.k as i32)
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_q4k_q8(&self, p: &GemvQ4KQ8Params) -> Result<(), BackendError> {
        unsafe {
            if self.use_mmvq_llama {
                self.kernels.mmvq_q4k_llama(
                    u8_ptr(p.data), u8_ptr(p.x_q8), f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            } else if self.use_v2 {
                self.kernels.gemv_q4k_q8_v2(
                    u8_ptr(p.data), u8_ptr(p.x_q8), f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            } else if self.use_wave64 {
                self.kernels.gemv_q4k_q8_w64(
                    u8_ptr(p.data), u8_ptr(p.x_q8), f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            } else {
                self.kernels.gemv_q4k_q8_multi(
                    u8_ptr(p.data), u8_ptr(p.x_q8), f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            }
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_q6k_q8(&self, p: &GemvQ6KQ8Params) -> Result<(), BackendError> {
        unsafe {
            if self.use_v2 {
                self.kernels.gemv_q6k_q8_v2(
                    u8_ptr(p.data), u8_ptr(p.x_q8), f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            } else if self.use_wave64 {
                self.kernels.gemv_q6k_q8_w64(
                    u8_ptr(p.data), u8_ptr(p.x_q8), f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            } else {
                self.kernels.gemv_q6k_q8(
                    u8_ptr(p.data), u8_ptr(p.x_q8), f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            }
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_q4k_f32(&self, p: &GemvQ4KF32Params) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        // RDNA2 decode optimisation: 1 warp/row, zero LDS.
        // Eliminates shared-memory limit (36KB for K=8960) that caps occupancy to
        // 1 block/CU with the old kernel. New: all N rows run in parallel.
        let result = if p.m <= 4 {
            unsafe {
                self.kernels.gemv_q4k_f32_w32(
                    u8_ptr(p.data), x_ptr, f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            }
        } else {
            unsafe {
                self.kernels.gemv_q4k_f32_multi(
                    u8_ptr(p.data), x_ptr, f32_mut_ptr(p.out),
                    p.m as i32, p.n as i32, p.k as i32, p.beta,
                )
            }
        };
        result.map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_q6k_f32(&self, p: &GemvQ6KF32Params) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        unsafe {
            self.kernels.gemv_q6k_f32_w32(
                u8_ptr(p.data), x_ptr, f32_mut_ptr(p.out),
                p.m as i32, p.n as i32, p.k as i32, p.beta,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_bf16_f32(&self, p: &GemvBf16F32Params) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        log::debug!("gemv_bf16_f32: wave64={}, v2={}, n={}, k={}", self.use_wave64, self.use_v2, p.n, p.k);
        let result = if self.use_wave64 && !self.use_v2 {
            unsafe {
                self.kernels.gemv_bf16_f32_w64(
                    u16_ptr(p.w), x_ptr, f32_mut_ptr(p.out),
                    p.n as i32, p.k as i32,
                )
            }
        } else {
            unsafe {
                self.kernels.gemv_bf16_f32_multi(
                    u16_ptr(p.w), x_ptr, f32_mut_ptr(p.out),
                    p.n as i32, p.k as i32,
                )
            }
        };
        result.map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_svd_bf16(&self, p: &GemvSvdBf16Params) -> Result<(), BackendError> {
        // LDS budget on RDNA2: 64 KB per block. Kernel uses (cols + rank) * 4 bytes.
        // Cap rank ≤ 1024 and (cols + rank) ≤ 12288 (48 KB) to leave headroom for
        // warp_sums[8] (32 B) + compiler spill. Fall back to default 2-gemm path otherwise.
        const RANK_CAP: usize = 1024;
        const LDS_FLOATS_CAP: usize = 12288;
        // Fused kernel only handles decode (m=1). Fall back to 2-gemm chain
        // for prefill (m>1) or when LDS budget is exceeded.
        if p.m != 1 || p.rank > RANK_CAP || (p.cols + p.rank) > LDS_FLOATS_CAP {
            self.gemm_bf16_f32(&GemmBf16Params {
                m: p.m, n: p.rank, k: p.cols,
                alpha: 1.0, beta: 0.0,
                a: p.x, a_offset: p.x_offset,
                b: p.b, c: p.scratch,
            })?;
            return self.gemm_bf16_f32(&GemmBf16Params {
                m: p.m, n: p.rows, k: p.rank,
                alpha: 1.0, beta: p.beta,
                a: p.scratch, a_offset: 0,
                b: p.a, c: p.out,
            });
        }

        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        unsafe {
            self.kernels.gemv_svd_bf16_fused(
                u16_ptr(p.a),
                u16_ptr(p.b),
                x_ptr,
                f32_mut_ptr(p.out),
                p.rows as i32,
                p.cols as i32,
                p.rank as i32,
                p.beta,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_w8a16(&self, p: &GemvW8A16Params) -> Result<(), BackendError> {
        // W8A32 GEMV: INT8 weights × F32 activations → F32 (wave64 decode path)
        // Supports beta=0 (overwrite) and beta=1 (residual accumulate).
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        unsafe {
            self.kernels.gemv_i8_f32_w64(
                i8_ptr(p.w),
                f32_ptr(p.scales),
                x_ptr,
                f32_mut_ptr(p.out),
                p.n as i32,
                p.k as i32,
                p.beta,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gemv_ternary_f32(&self, p: &GemvTernaryF32Params) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        // Dispatch to the optimal variant:
        //   decode (m=1) or small K (≤1536) → w32: zero LDS, 32 threads/row, max occupancy
        //   prefill (m>1) and large K (>1536) → smem: x cached in LDS for 8-warp reuse
        unsafe {
            self.kernels.gemv_ternary_f32(
                u8_ptr(p.packed), x_ptr, f32_mut_ptr(p.out),
                p.m as i32, p.n as i32, p.k as i32,
                p.scale, p.beta,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_rms_qkv_q4k_f32(&self, p: &FusedRmsQkvQ4KF32Params) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        let bias_q = p.bias_q.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        let bias_k = p.bias_k.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        let bias_v = p.bias_v.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        unsafe {
            self.kernels.fused_rms_qkv_q4k_f32(
                x_ptr,
                f32_ptr(p.norm_w),
                u8_ptr(p.wq), u8_ptr(p.wk), u8_ptr(p.wv),
                f32_mut_ptr(p.out_q), f32_mut_ptr(p.out_k), f32_mut_ptr(p.out_v),
                bias_q, bias_k, bias_v,
                p.nq as i32, p.nk as i32, p.nv as i32, p.k as i32, p.eps,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_rms_qkv_q4k_q6k_f32(&self, p: &FusedRmsQkvQ4KQ6KF32Params) -> Result<(), BackendError> {
        let bias_q = p.bias_q.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        let bias_k = p.bias_k.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        let bias_v = p.bias_v.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        unsafe {
            self.kernels.fused_rms_qkv_q4k_q6k_f32(
                f32_ptr(p.x),
                f32_ptr(p.norm_w),
                u8_ptr(p.wq), u8_ptr(p.wk), u8_ptr(p.wv),
                f32_mut_ptr(p.out_q), f32_mut_ptr(p.out_k), f32_mut_ptr(p.out_v),
                bias_q, bias_k, bias_v,
                p.nq as i32, p.nk as i32, p.nv as i32, p.k as i32, p.eps,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_rms_gate_up_silu_q4k_f32(&self, p: &FusedRmsGateUpSiluQ4KF32Params) -> Result<(), BackendError> {
        let x_ptr = unsafe { f32_ptr(p.x).add(p.x_offset) };
        unsafe {
            self.kernels.fused_rms_gate_up_silu_q4k_f32(
                x_ptr,
                f32_ptr(p.norm_w),
                u8_ptr(p.gate), u8_ptr(p.up),
                f32_mut_ptr(p.out),
                p.n as i32, p.k as i32, p.eps,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_rms_qkv_q4(&self, p: &FusedRmsQkvQ4Params) -> Result<(), BackendError> {
        let bias_q = p.bias_q.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        let bias_k = p.bias_k.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        let bias_v = p.bias_v.map(|b| f32_ptr(b)).unwrap_or(std::ptr::null());
        unsafe {
            self.kernels.fused_rms_qkv_q4_f32(
                f32_ptr(p.x), f32_ptr(p.norm_w),
                u8_ptr(p.wq_packed), f32_ptr(p.wq_scales), f32_ptr(p.wq_zeros),
                u8_ptr(p.wk_packed), f32_ptr(p.wk_scales), f32_ptr(p.wk_zeros),
                u8_ptr(p.wv_packed), f32_ptr(p.wv_scales), f32_ptr(p.wv_zeros),
                f32_mut_ptr(p.out_q), f32_mut_ptr(p.out_k), f32_mut_ptr(p.out_v),
                bias_q, bias_k, bias_v,
                p.nq as i32, p.nk as i32, p.nv as i32, p.k as i32,
                p.group_size as i32, p.eps,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_rms_gate_up_silu_q4(&self, p: &FusedRmsGateUpSiluQ4Params) -> Result<(), BackendError> {
        unsafe {
            self.kernels.fused_rms_gate_up_silu_q4_f32(
                f32_ptr(p.x), f32_ptr(p.norm_w),
                u8_ptr(p.gate_packed), f32_ptr(p.gate_scales), f32_ptr(p.gate_zeros),
                u8_ptr(p.up_packed),   f32_ptr(p.up_scales),   f32_ptr(p.up_zeros),
                f32_mut_ptr(p.out),
                p.n as i32, p.k as i32, p.group_size as i32, p.eps,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gravity_score(&self, p: &GravityScoreParams) -> Result<(), BackendError> {
        // Calcula importância semântica: gravity(i) = média(attn_score(i) * ||v(i)||₂)
        // Passo 1: calcula normas L2 dos value vectors do cache V
        // value_norms kernel: (v_cache, norms_out, seq_len, n_kv_heads, head_dim)
        unsafe {
            self.kernels.value_norms(
                f32_ptr(p.v_cache),           // [seq_len, n_kv_heads * head_dim] — input
                f32_mut_ptr(p.value_norms),   // [n_kv_heads, seq_len] — output
                p.seq_len as i32,
                p.n_kv_heads as i32,
                p.head_dim as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(format!("value_norms: {e}")))?;

        // Depois calcula gravity score como produto atenção × normas, média sobre heads
        unsafe {
            self.kernels.gravity_score(
                f32_ptr(p.attn_weights),
                f32_ptr(p.value_norms),
                f32_mut_ptr(p.gravity_out),
                p.n_heads as i32,
                p.seq_len as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn cfc_forward(&self, p: &CfcForwardParams) -> Result<(), BackendError> {
        // CfC Closed-form Continuous liquid correction layer
        // Fluxo:
        //   1. f_proj = x @ wf^T + h_prev @ wf2^T + bf   (SGEMM)
        //   2. g_proj = x @ wg^T + h_prev @ wg2^T + bg   (SGEMM)
        //   3. cfc_gate_update kernel (sigmoid/tanh gating)
        //   4. correction = h_new @ wout^T + bout         (SGEMM)

        // Aloca buffers temporários para f_proj, g_proj (pequeno: cfc_dim)
        let f_proj = self.alloc_f32(&[p.cfc_dim]);
        let g_proj = self.alloc_f32(&[p.cfc_dim]);

        // Passo 1: f_proj = x @ wf^T + h_prev @ wf2^T + bf
        // f_proj = zeros; x @ wf^T → f_proj; h_prev @ wf2^T → f_proj (acumula)
        {
            // x @ wf^T: [n_tokens, hidden] @ [cfc_dim, hidden]^T → [n_tokens, cfc_dim]
            // Mas queremos apenas o último token projétado. Para simplificar,
            // fazemos GEMM completo (projeta todos os tokens) e usamos o último.
            // Alternativamente, usamos apenas a última linha de x.
            // Vamos fazer versão simples: última linha de x = x[(n_tokens-1) * hidden : ]

            let x_last_offset = (p.n_tokens - 1) * p.hidden;

            // f_proj = x_last @ wf^T + bf
            let sg_f1 = SgemmParams {
                trans_a: false,
                trans_b: true,
                m: 1,
                n: p.cfc_dim,
                k: p.hidden,
                alpha: 1.0,
                beta: 0.0,
                a: p.x,
                lda: p.hidden,
                a_offset: x_last_offset,
                b: p.wf,
                ldb: p.hidden,
                b_offset: 0,
                c: &f_proj,
                ldc: p.cfc_dim,
                c_offset: 0,
            };
            self.sgemm(&sg_f1)?;

            // f_proj += h_prev @ wf2^T + bf (acumula)
            let sg_f2 = SgemmParams {
                trans_a: false,
                trans_b: true,
                m: 1,
                n: p.cfc_dim,
                k: p.cfc_dim,
                alpha: 1.0,
                beta: 1.0, // acumula
                a: p.h_prev,
                lda: p.cfc_dim,
                a_offset: 0,
                b: p.wf2,
                ldb: p.cfc_dim,
                b_offset: 0,
                c: &f_proj,
                ldc: p.cfc_dim,
                c_offset: 0,
            };
            self.sgemm(&sg_f2)?;

            // f_proj += bf (add_bias_inplace)
            let ab = AddBiasParams {
                tensor: &f_proj,
                bias: p.bf,
                n_tokens: 1,
                dim: p.cfc_dim,
            };
            self.add_bias_inplace(&ab)?;
        }

        // Passo 2: g_proj = x_last @ wg^T + h_prev @ wg2^T + bg
        {
            let x_last_offset = (p.n_tokens - 1) * p.hidden;

            let sg_g1 = SgemmParams {
                trans_a: false,
                trans_b: true,
                m: 1,
                n: p.cfc_dim,
                k: p.hidden,
                alpha: 1.0,
                beta: 0.0,
                a: p.x,
                lda: p.hidden,
                a_offset: x_last_offset,
                b: p.wg,
                ldb: p.hidden,
                b_offset: 0,
                c: &g_proj,
                ldc: p.cfc_dim,
                c_offset: 0,
            };
            self.sgemm(&sg_g1)?;

            let sg_g2 = SgemmParams {
                trans_a: false,
                trans_b: true,
                m: 1,
                n: p.cfc_dim,
                k: p.cfc_dim,
                alpha: 1.0,
                beta: 1.0,
                a: p.h_prev,
                lda: p.cfc_dim,
                a_offset: 0,
                b: p.wg2,
                ldb: p.cfc_dim,
                b_offset: 0,
                c: &g_proj,
                ldc: p.cfc_dim,
                c_offset: 0,
            };
            self.sgemm(&sg_g2)?;

            let ab = AddBiasParams {
                tensor: &g_proj,
                bias: p.bg,
                n_tokens: 1,
                dim: p.cfc_dim,
            };
            self.add_bias_inplace(&ab)?;
        }

        // Passo 3: h_new = sigmoid(f) * h_prev + (1 - sigmoid(f)) * tanh(g)
        unsafe {
            self.kernels.cfc_gate_update(
                f32_ptr(&f_proj),
                f32_ptr(&g_proj),
                f32_ptr(p.h_prev),
                f32_mut_ptr(p.h_out),
                p.cfc_dim as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(format!("cfc_gate_update: {e}")))?;

        // Passo 4: correction = h_new @ wout^T + bout
        // correction é [n_tokens, hidden]: projeta h_new de volta pro espaço hidden
        let sg_out = SgemmParams {
            trans_a: false,
            trans_b: true,
            m: p.n_tokens,
            n: p.hidden,
            k: p.cfc_dim,
            alpha: 1.0,
            beta: 0.0,
            a: p.h_out,
            lda: p.cfc_dim,
            a_offset: 0,
            b: p.wout,
            ldb: p.cfc_dim,
            b_offset: 0,
            c: p.correction,
            ldc: p.hidden,
            c_offset: 0,
        };
        self.sgemm(&sg_out)?;

        // correction += bout (broadcast aos tokens)
        let ab = AddBiasParams {
            tensor: p.correction,
            bias: p.bout,
            n_tokens: p.n_tokens,
            dim: p.hidden,
        };
        self.add_bias_inplace(&ab)?;

        Ok(())
    }

    fn kv_cache_quantized_append(&self, p: &KvCacheQuantizedParams) -> Result<(), BackendError> {
        // Append KV cache com quantização on-the-fly para INT4
        unsafe {
            self.kernels.kv_cache_quantize_append(
                u8_mut_ptr(p.cache),
                f32_mut_ptr(p.scales),
                f32_mut_ptr(p.zeros),
                f32_ptr(p.new_kv),
                p.insert_pos as i32,
                p.n_new_tokens as i32,
                p.kv_dim as i32,
                p.max_seq as i32,
                p.group_size as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn add_tensors_inplace(&self, a: &AnyTensor, b: &AnyTensor, n_elems: usize, alpha: f32) -> Result<(), BackendError> {
        unsafe {
            self.kernels.add_tensors_inplace(
                f32_mut_ptr(a),
                f32_ptr(b),
                n_elems as i32,
                alpha,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_norm_bias_rope(&self, p: &FusedNormBiasRopeParams) -> Result<(), BackendError> {
        let has_bias = if p.bias.is_some() { 1i32 } else { 0i32 };
        let bias_ptr = if let Some(bias) = p.bias {
            f32_ptr(bias)
        } else {
            std::ptr::null()
        };

        unsafe {
            self.kernels.fused_norm_bias_rope(
                f32_mut_ptr(p.output),
                f32_ptr(p.input),
                f32_ptr(p.norm_weight),
                bias_ptr,
                p.n_tokens as i32,
                p.hidden_size as i32,
                p.n_heads as i32,
                p.pos_offset as i32,
                p.theta,
                p.eps,
                has_bias,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    // ── GPU-side sampling ──────────────────────────────────────────────────────

    fn gpu_argmax(&self, logits: &AnyTensor, ncols: usize) -> Result<i32, BackendError> {
        let logits_ptr = f32_ptr(logits);
        let dst_ptr = self.argmax_buf.as_ptr() as *mut i32;
        unsafe {
            self.kernels.argmax_f32(
                logits_ptr,
                dst_ptr,
                std::ptr::null_mut(),  // no max value needed
                ncols as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))?;

        // Download just 4 bytes instead of 608KB
        let result = self.argmax_buf.to_host()
            .map_err(|e| BackendError::Kernel(e.to_string()))?;
        Ok(result[0])
    }

    // ── Online BI measurement (adaptive layer skip) ────────────────────────────

    fn cosine_sim_1d(&self, a: &AnyTensor, b: &AnyTensor, out: &AnyTensor, dim: usize) -> Result<(), BackendError> {
        unsafe {
            self.kernels.cosine_sim_1d(
                f32_ptr(a),
                f32_ptr(b),
                f32_mut_ptr(out),
                dim as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn copy_f32_d2d(&self, src: &AnyTensor, dst: &AnyTensor, n: usize) -> Result<(), BackendError> {
        unsafe {
            self.kernels.copy_f32(
                f32_ptr(src),
                f32_mut_ptr(dst),
                n as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn widen_bf16_f32(&self, src_bf16: &AnyTensor, dst_f32: &AnyTensor, n: usize) -> Result<(), BackendError> {
        unsafe {
            self.kernels.widen_bf16_f32(
                u16_ptr(src_bf16),
                f32_mut_ptr(dst_f32),
                n as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    // ── HIP Graph capture (trait override) ────────────────────────────────────

    /// Capture-once-replay: if exec already instantiated, just launch it.
    /// Returns true if replay succeeded (skip dispatch); false if caller must dispatch+capture.
    fn try_replay_graph(&self) -> bool {
        let exec_guard = self.graph_exec.lock().unwrap();
        if let Some(exec) = exec_guard.as_ref() {
            unsafe {
                hip_ffi::hip_graph_launch(*exec, self.stream).is_ok()
            }
        } else {
            false
        }
    }

    fn invalidate_graph(&self) {
        let mut exec_guard = self.graph_exec.lock().unwrap();
        if let Some(exec) = exec_guard.take() {
            unsafe { let _ = hip_ffi::hip_graph_exec_destroy(exec); }
        }
    }

    fn hsa_barrier_and_wait(&self) -> bool {
        if let Some(hsa) = self.kernels.hsa.as_ref() {
            let _ = hsa.submit_barrier_and_wait();
            // Belt-and-suspenders: also drain via HIP runtime (covers any
            // residual HIP-stream activity AND forces a system-wide GPU
            // synchronization that the HSA barrier alone may not provide).
            unsafe { hip_ffi::raw::hipDeviceSynchronize(); }
            true
        } else {
            false
        }
    }

    fn begin_graph_capture(&self) -> bool {
        unsafe {
            match hip_ffi::hip_stream_begin_capture(self.stream, hip_ffi::CAPTURE_MODE_RELAXED) {
                Ok(()) => {
                    self.capturing.store(true, std::sync::atomic::Ordering::Relaxed);
                    true
                }
                Err(e) => {
                    eprintln!("graph capture begin failed: {}", e);
                    false
                }
            }
        }
    }

    fn end_graph_capture_and_launch(&self) -> bool {
        if !self.capturing.swap(false, std::sync::atomic::Ordering::Relaxed) {
            return false;
        }
        unsafe {
            let t0 = std::time::Instant::now();
            let graph = match hip_ffi::hip_stream_end_capture(self.stream) {
                Ok(g) => g,
                Err(e) => { eprintln!("graph capture end failed: {}", e); return false; }
            };
            let capture_us = t0.elapsed().as_micros();
            let mut exec_guard = self.graph_exec.lock().unwrap();
            let (exec, method) = if let Some(existing) = exec_guard.as_ref() {
                match hip_ffi::hip_graph_exec_update(*existing, graph) {
                    Ok(true) => (*existing, "update"),
                    Ok(false) => {
                        let _ = hip_ffi::hip_graph_exec_destroy(*existing);
                        match hip_ffi::hip_graph_instantiate(graph) {
                            Ok(e) => (e, "re-instantiate(update-rejected)"),
                            Err(err) => {
                                eprintln!("graph instantiate failed: {}", err);
                                let _ = hip_ffi::hip_graph_destroy(graph);
                                *exec_guard = None;
                                return false;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("graph exec update error: {}", e);
                        let _ = hip_ffi::hip_graph_exec_destroy(*existing);
                        match hip_ffi::hip_graph_instantiate(graph) {
                            Ok(e) => (e, "re-instantiate(update-error)"),
                            Err(err) => {
                                eprintln!("graph instantiate failed: {}", err);
                                let _ = hip_ffi::hip_graph_destroy(graph);
                                *exec_guard = None;
                                return false;
                            }
                        }
                    }
                }
            } else {
                match hip_ffi::hip_graph_instantiate(graph) {
                    Ok(e) => (e, "first-instantiate"),
                    Err(err) => {
                        eprintln!("graph instantiate failed: {}", err);
                        let _ = hip_ffi::hip_graph_destroy(graph);
                        return false;
                    }
                }
            };
            let prep_us = t0.elapsed().as_micros();
            *exec_guard = Some(exec);
            let launch_ok = hip_ffi::hip_graph_launch(exec, self.stream).is_ok();
            let total_us = t0.elapsed().as_micros();
            let _ = hip_ffi::hip_graph_destroy(graph);

            // Log graph overhead periodically
            use std::sync::atomic::{AtomicU64, Ordering};
            static GRAPH_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
            static GRAPH_UPDATE_OK: AtomicU64 = AtomicU64::new(0);
            static GRAPH_REINSTANTIATE: AtomicU64 = AtomicU64::new(0);
            static GRAPH_TOTAL_US: AtomicU64 = AtomicU64::new(0);
            let gc = GRAPH_CALL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            GRAPH_TOTAL_US.fetch_add(total_us as u64, Ordering::Relaxed);
            if method == "update" {
                GRAPH_UPDATE_OK.fetch_add(1, Ordering::Relaxed);
            } else if method.starts_with("re-instantiate") {
                GRAPH_REINSTANTIATE.fetch_add(1, Ordering::Relaxed);
            }
            if gc <= 3 || gc % 50 == 0 {
                let avg = GRAPH_TOTAL_US.load(Ordering::Relaxed) as f64 / gc as f64;
                let updates = GRAPH_UPDATE_OK.load(Ordering::Relaxed);
                let reins = GRAPH_REINSTANTIATE.load(Ordering::Relaxed);
                eprintln!("GRAPH [#{gc}] {method}: capture={capture_us}µs prep={prep_us}µs launch={total_us}µs | avg={avg:.0}µs | updates={updates} reins={reins}");
            }

            launch_ok
        }
    }

    // ── Dense linear algebra (hipSOLVER) ──────────────────────────────────────

    fn qr_f32(&self, a: &AnyTensor, q_out: &AnyTensor, rows: usize, cols: usize) -> Result<(), BackendError> {
        let handle = self.solver_handle()?;
        hip_ffi::qr_f32(
            handle,
            rows as i32,
            cols as i32,
            f32_mut_ptr(a),
            f32_mut_ptr(q_out),
        )
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn svd_f32(
        &self,
        a: &AnyTensor,
        u: &AnyTensor,
        s: &AnyTensor,
        vt: &AnyTensor,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        let handle = self.solver_handle()?;
        hip_ffi::svd_f32(
            handle,
            rows as i32,
            cols as i32,
            f32_mut_ptr(a),
            f32_mut_ptr(u),
            f32_mut_ptr(s),
            f32_mut_ptr(vt),
        )
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    // ── Sync and info ─────────────────────────────────────────────────────────

    fn sync(&self) -> Result<(), BackendError> {
        hip_ffi::safe::hip_device_sync().map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn mem_info(&self) -> Option<(usize, usize)> {
        hip_ffi::safe::hip_mem_info().ok()
    }

    fn kv_cache_int8_append(&self, p: &KvCacheInt8AppendParams) -> Result<(), BackendError> {
        // Quantize F32 K/V to INT8 and append to cache
        unsafe {
            self.kernels.kv_cache_int8_quantize_append(
                u8_mut_ptr(p.cache_q) as *mut i8,
                f32_mut_ptr(p.cache_scales),
                f32_ptr(p.new_kv),
                p.insert_pos as i32,
                p.n_new_tokens as i32,
                p.kv_dim as i32,
                p.head_dim as i32,
                p.n_kv_heads as i32,
                p.max_seq as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn kv_cache_int8_read(&self, p: &KvCacheInt8ReadParams) -> Result<(), BackendError> {
        // Dequantize INT8 K/V cache to F32 for attention
        unsafe {
            self.kernels.kv_cache_int8_dequantize(
                u8_ptr(p.cache_q) as *const i8,
                f32_ptr(p.cache_scales),
                f32_mut_ptr(p.output),
                p.seq_len as i32,
                p.kv_dim as i32,
                p.head_dim as i32,
                p.n_kv_heads as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn kv_cache_wht_append(&self, p: &KvCacheWhtAppendParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.kv_cache_wht_quant_append(
                u8_mut_ptr(p.cache_q) as *mut i8,
                f32_mut_ptr(p.cache_scales),
                f32_ptr(p.new_kv),
                p.insert_pos as i32,
                p.n_new_tokens as i32,
                p.kv_dim as i32,
                p.head_dim as i32,
                p.n_kv_heads as i32,
                p.max_seq as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn kv_cache_wht_read(&self, p: &KvCacheWhtReadParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.kv_cache_wht_dequant(
                f32_mut_ptr(p.output),
                u8_ptr(p.cache_q) as *const i8,
                f32_ptr(p.cache_scales),
                p.seq_len as i32,
                p.kv_dim as i32,
                p.head_dim as i32,
                p.n_kv_heads as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_rope_kv_copy(&self, p: &FusedRopeKvCopyParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.fused_rope_kv_copy(
                f32_mut_ptr(p.q), f32_mut_ptr(p.k), f32_ptr(p.v),
                f32_mut_ptr(p.kv_k_cache), f32_mut_ptr(p.kv_v_cache),
                p.n_heads as i32, p.n_kv_heads as i32, p.head_dim as i32,
                p.pos_start as i32, p.theta,
                p.kv_dim as i32, p.max_seq as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_rms_norm_quantize(&self, p: &FusedRmsNormQuantizeParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.fused_rms_norm_quantize(
                f32_ptr(p.input), f32_ptr(p.weight),
                f32_mut_ptr(p.out_f32), u8_mut_ptr(p.out_q8),
                p.hidden_size as i32, p.eps,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_qkv(&self, p: &FusedQkvParams) -> Result<(), BackendError> {
        unsafe {
            let bq = p.bias_q.map(|t| f32_ptr(t)).unwrap_or(std::ptr::null());
            let bk = p.bias_k.map(|t| f32_ptr(t)).unwrap_or(std::ptr::null());
            let bv = p.bias_v.map(|t| f32_ptr(t)).unwrap_or(std::ptr::null());
            if self.use_wave64 && !self.use_v2 {
                self.kernels.fused_qkv_q4k_w64(
                    u8_ptr(p.wq_data), u8_ptr(p.wk_data), u8_ptr(p.wv_data),
                    u8_ptr(p.x_q8),
                    f32_mut_ptr(p.out_q), f32_mut_ptr(p.out_k), f32_mut_ptr(p.out_v),
                    bq, bk, bv,
                    p.nq as i32, p.nk as i32, p.nv as i32, p.k as i32,
                )
            } else {
                // V2 and non-wave64 both use wave32 fused QKV
                self.kernels.fused_qkv_q4k(
                    u8_ptr(p.wq_data), u8_ptr(p.wk_data), u8_ptr(p.wv_data),
                    u8_ptr(p.x_q8),
                    f32_mut_ptr(p.out_q), f32_mut_ptr(p.out_k), f32_mut_ptr(p.out_v),
                    bq, bk, bv,
                    p.nq as i32, p.nk as i32, p.nv as i32, p.k as i32,
                )
            }
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_gate_up_silu(&self, p: &FusedGateUpSiluParams, is_q6k: bool) -> Result<(), BackendError> {
        unsafe {
            let gate_ptr = u8_ptr(p.gate_data);
            let up_ptr = u8_ptr(p.up_data);
            let q8_ptr = u8_ptr(p.x_q8);
            let out_ptr = f32_mut_ptr(p.out);
            if is_q6k {
                self.kernels.fused_gate_up_silu_q6k(
                    gate_ptr, up_ptr, q8_ptr, out_ptr,
                    p.n as i32, p.k as i32,
                )
            } else if self.use_v2 {
                self.kernels.fused_gate_up_silu_q4k_v2(
                    gate_ptr, up_ptr, q8_ptr, out_ptr,
                    p.n as i32, p.k as i32,
                )
            } else if self.use_wave64 {
                self.kernels.fused_gate_up_silu_q4k_w64(
                    gate_ptr, up_ptr, q8_ptr, out_ptr,
                    p.n as i32, p.k as i32,
                )
            } else {
                self.kernels.fused_gate_up_silu_q4k_multi(
                    gate_ptr, up_ptr, q8_ptr, out_ptr,
                    p.n as i32, p.k as i32,
                )
            }
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn set_seq_pos(&self, value: i32) -> Result<(), BackendError> {
        // Sync H2D: 4 bytes, negligible latency. Async with stack-borne &[value]
        // is unsafe because hipMemcpyAsync may read src after function returns.
        let buf = [value];
        unsafe {
            hip_ffi::safe::hip_h2d(self.pos_dev.as_ptr() as *mut i32, &buf)
        }
        .map_err(|e| BackendError::Kernel(format!("set_seq_pos h2d: {e}")))
    }

    fn fused_rope_kv_copy_dyn(&self, p: &FusedRopeKvCopyParams) -> Result<(), BackendError> {
        let kv_dim = p.n_kv_heads * p.head_dim;
        unsafe {
            self.kernels.fused_rope_kv_copy_dyn(
                f32_mut_ptr(p.q),
                f32_mut_ptr(p.k),
                f32_ptr(p.v),
                f32_mut_ptr(p.kv_k_cache),
                f32_mut_ptr(p.kv_v_cache),
                p.n_heads as i32,
                p.n_kv_heads as i32,
                p.head_dim as i32,
                self.pos_dev.as_ptr(),
                p.theta,
                kv_dim as i32,
                p.max_seq as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn copy_to_kv_cache_dyn(&self, p: &KvCacheParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.copy_to_kv_cache_dyn(
                f32_mut_ptr(p.cache),
                f32_ptr(p.new_kv),
                p.n_new_tokens as i32,
                p.kv_dim as i32,
                self.pos_dev.as_ptr(),
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_attention_decode_dyn(&self, p: &FusedAttentionDecodeParams, max_seq_for_smem: usize) -> Result<(), BackendError> {
        unsafe {
            self.kernels.fused_attention_decode_dyn(
                f32_ptr(p.q),
                f32_ptr(p.k_cache),
                f32_ptr(p.v_cache),
                f32_mut_ptr(p.out),
                self.pos_dev.as_ptr(),
                p.head_dim as i32,
                p.n_heads as i32,
                p.n_kv_heads as i32,
                p.scale,
                max_seq_for_smem as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn fused_attention_decode(&self, p: &FusedAttentionDecodeParams) -> Result<(), BackendError> {
        unsafe {
            if self.use_flash_attn {
                self.kernels.flash_attention_decode_v2(
                    f32_ptr(p.q),
                    f32_ptr(p.k_cache),
                    f32_ptr(p.v_cache),
                    f32_mut_ptr(p.out),
                    p.seq_len as i32,
                    p.head_dim as i32,
                    p.n_heads as i32,
                    p.n_kv_heads as i32,
                    p.scale,
                )
            } else {
                self.kernels.fused_attention_decode(
                    f32_ptr(p.q),
                    f32_ptr(p.k_cache),
                    f32_ptr(p.v_cache),
                    f32_mut_ptr(p.out),
                    p.seq_len as i32,
                    p.head_dim as i32,
                    p.n_heads as i32,
                    p.n_kv_heads as i32,
                    p.scale,
                )
            }
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    // ── Sparse MLP gather kernels ────────────────────────────────────────────

    fn gather_rows_bf16(&self, p: &GatherRowsBf16Params) -> Result<(), BackendError> {
        unsafe {
            self.kernels.gather_rows_bf16(
                u16_ptr(p.src),
                u16_mut_ptr(p.dst),
                i32_ptr(p.indices),
                p.n_indices as i32,
                p.cols as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn gather_cols_bf16(&self, p: &GatherColsBf16Params) -> Result<(), BackendError> {
        unsafe {
            self.kernels.gather_cols_bf16(
                u16_ptr(p.src),
                u16_mut_ptr(p.dst),
                i32_ptr(p.indices),
                p.rows as i32,
                p.total_cols as i32,
                p.n_indices as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn kuramoto_coherence(&self, p: &KuramotoCoherenceParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.kuramoto_coherence(
                f32_ptr(p.hidden),
                f32_ptr(p.tuning_forks),
                f32_mut_ptr(p.coherences),
                p.n_neurons as i32,
                p.hidden_dim as i32,
                p.phase_dim as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }

    fn topk_select(&self, p: &TopkSelectParams) -> Result<(), BackendError> {
        unsafe {
            self.kernels.topk_select(
                f32_ptr(p.coherences),
                i32_ptr(p.indices) as *mut i32,
                p.n_neurons as i32,
                p.top_k as i32,
            )
        }
        .map_err(|e| BackendError::Kernel(e.to_string()))
    }
}
