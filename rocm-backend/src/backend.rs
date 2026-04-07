use std::sync::Arc;

use compute_core::{
    params::{
        AddBiasParams, CfcForwardParams, EmbedParams, FusedNormBiasRopeParams, GeluMulParams, GemmBf16Params, GemvQ4Params,
        GravityScoreParams, KvCacheParams, KvCacheQuantizedParams, LayerNormParams, RmsNormParams,
        RopeParams, SgemmParams, SgemmStridedBatchedParams, SiluMulParams, SoftmaxParams,
        KvCacheInt8AppendParams, KvCacheInt8ReadParams,
        GatherRowsBf16Params, GatherColsBf16Params,
        KuramotoCoherenceParams, TopkSelectParams,
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
    kernels: hip_ffi::GpuKernels,
    tok_buf: std::sync::Mutex<hip_ffi::safe::DeviceBuffer<u32>>,
    act_bf16: std::sync::Mutex<hip_ffi::safe::DeviceBuffer<u16>>,
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

        let blas = hip_ffi::BlasHandle::new()
            .map_err(|e| BackendError::NotAvailable(format!("hipBLAS: {e}")))?;

        let kernels = hip_ffi::GpuKernels::new()
            .map_err(|e| BackendError::NotAvailable(format!("GpuKernels: {e}")))?;

        // Pre-allocate a buffer for up to 4096 tokens (max_seq_len) to avoid
        // VRAM fragmentation (memory leak footprint) during the generation loop.
        let tok_buf = hip_ffi::safe::DeviceBuffer::<u32>::alloc(4096)
            .map_err(|e| BackendError::NotAvailable(format!("HIP alloc: {e}")))?;

        // Pre-allocate BF16 activations buffer for GEMM (max: 4096 seq x 8192 hidden) = 33.5M elems (~67 MB)
        let act_bf16 = hip_ffi::safe::DeviceBuffer::<u16>::alloc(4096 * 8192)
            .map_err(|e| BackendError::NotAvailable(format!("HIP alloc act_bf16: {e}")))?;

        Ok(Self {
            blas,
            kernels,
            tok_buf: std::sync::Mutex::new(tok_buf),
            act_bf16: std::sync::Mutex::new(act_bf16),
        })
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

        hip_ffi::gemm_bf16_f32(
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

    fn sgemm_strided_batched(&self, p: &SgemmStridedBatchedParams) -> Result<(), BackendError> {
        let a_ptr = unsafe { f32_ptr(p.a).add(p.a_offset) };
        let b_ptr = unsafe { f32_ptr(p.b).add(p.b_offset) };
        let c_ptr = unsafe { f32_mut_ptr(p.c).add(p.c_offset) };

        hip_ffi::sgemm_strided_batched(
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
        .map_err(|e| BackendError::Gemm(e.to_string()))
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

        // Upload token IDs (host slice) to the pre-allocated GPU buffer.
        unsafe {
            hip_ffi::safe::hip_h2d(buf_guard.as_mut_ptr(), p.token_ids)
                .map_err(|e| BackendError::Transfer(e.to_string()))?;
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
        unsafe {
            self.kernels.dequant_gemv_q4(
                u8_ptr(p.packed),
                f32_ptr(p.scales),
                f32_ptr(p.zeros),
                f32_ptr(p.x),
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
