use crate::{AnyTensor, BackendError, params::*};
use crate::params::{KvCacheInt8AppendParams, KvCacheInt8ReadParams, KvCacheWhtAppendParams, KvCacheWhtReadParams};

/// Trait principal de backend de compute.
/// Toda implementação (ROCm, OpenCL, CPU) implementa este trait.
/// iara-engine usa apenas este trait — nunca importa HIP ou OpenCL diretamente.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Nome legível do backend (ex: "ROCm/HIP gfx1032", "OpenCL Intel Iris Plus")
    fn name(&self) -> &str;

    // ── Gerenciamento de memória ──────────────────────────────────────────────

    /// Aloca buffer F32 não inicializado no device.
    fn alloc_f32(&self, shape: &[usize]) -> AnyTensor;

    /// Aloca buffer BF16 (u16) não inicializado no device.
    fn alloc_bf16(&self, shape: &[usize]) -> AnyTensor;

    /// Upload de slice F32 host → device.
    fn upload_f32(&self, data: &[f32], shape: &[usize]) -> AnyTensor;

    /// Upload de slice BF16 (u16 bits) host → device.
    fn upload_bf16(&self, data: &[u16], shape: &[usize]) -> AnyTensor;

    /// Upload de dados INT4 comprimido (2 valores por byte) host → device.
    fn upload_q4(&self, data: &[u8], shape: &[usize]) -> AnyTensor;

    /// Upload de slice I32 host → device (para índices de gather/scatter).
    fn upload_i32(&self, data: &[i32], shape: &[usize]) -> AnyTensor {
        // Default: cast to f32 and upload (CPU backends without native i32 support)
        let _ = data;
        let _ = shape;
        panic!("upload_i32 not implemented for this backend");
    }

    /// Aloca buffer Q4 não inicializado no device.
    fn alloc_q4(&self, shape: &[usize]) -> AnyTensor;

    /// Upload raw bytes to device (for GGML quantized blocks: Q4_K, Q6_K, etc.).
    /// Shape is informational only — the actual buffer size is data.len() bytes.
    fn upload_raw(&self, data: &[u8], shape: &[usize]) -> AnyTensor {
        // Default: delegate to upload_q4 which stores as u8
        self.upload_q4(data, shape)
    }

    /// Download de tensor F32 device → Vec<f32> host. Sincroniza o device.
    fn download_f32(&self, t: &AnyTensor) -> Vec<f32>;

    /// Synchronize device (wait for all pending GPU operations to complete).
    /// Default implementation is a no-op (for CPU backends).
    fn device_sync(&self) {}

    /// Download de tensor BF16 device → Vec<f32> host, convertendo BF16→F32.
    /// Default: tenta download_f32 (para backends que armazenam tudo como F32).
    fn download_bf16_as_f32(&self, t: &AnyTensor) -> Vec<f32> {
        self.download_f32(t)
    }

    // ── GEMM ─────────────────────────────────────────────────────────────────

    fn gemm_bf16_f32(&self, p: &GemmBf16Params) -> Result<(), BackendError>;
    fn sgemm(&self, p: &SgemmParams) -> Result<(), BackendError>;
    fn sgemm_strided_batched(&self, p: &SgemmStridedBatchedParams) -> Result<(), BackendError>;

    // ── Kernels customizados ──────────────────────────────────────────────────

    fn embed_lookup         (&self, p: &EmbedParams)              -> Result<(), BackendError>;
    fn rms_norm             (&self, p: &RmsNormParams)            -> Result<(), BackendError>;
    fn layer_norm           (&self, p: &LayerNormParams)          -> Result<(), BackendError>;
    fn rope_inplace         (&self, p: &RopeParams)               -> Result<(), BackendError>;
    fn softmax_rows         (&self, p: &SoftmaxParams)            -> Result<(), BackendError>;
    fn softmax_rows_causal  (&self, p: &SoftmaxCausalParams)     -> Result<(), BackendError>;
    fn silu_mul_inplace     (&self, p: &SiluMulParams)            -> Result<(), BackendError>;
    fn gelu_mul_inplace     (&self, p: &GeluMulParams)            -> Result<(), BackendError>;
    fn copy_to_kv_cache     (&self, p: &KvCacheParams)            -> Result<(), BackendError>;
    fn add_bias_inplace     (&self, p: &AddBiasParams)            -> Result<(), BackendError>;
    fn gemv_q4              (&self, p: &GemvQ4Params)             -> Result<(), BackendError>;

    /// Native GGML Q4_K GEMV: operates directly on raw Q4_K blocks on device.
    /// Dequantizes on-the-fly in F32, no BF16 intermediate.
    fn gemv_q4k             (&self, _p: &GemvQ4KParams)           -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_q4k".into()))
    }

    /// Native GGML Q6_K GEMV: operates directly on raw Q6_K blocks on device.
    fn gemv_q6k             (&self, _p: &GemvQ6KParams)           -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_q6k".into()))
    }
    /// Quantize F32 activations to Q8_1 blocks for fast integer GEMV.
    fn quantize_q8          (&self, _p: &QuantizeQ8Params)        -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("quantize_q8".into()))
    }
    /// Q4_K × Q8_1 GEMV with dp4a integer dot product.
    fn gemv_q4k_q8          (&self, _p: &GemvQ4KQ8Params)        -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_q4k_q8".into()))
    }
    /// Q4_K × F32 GEMV — direct, no intermediate Q8 quantize step.
    /// 1 kernel instead of 2 (quantize + gemv). Key speedup for down projection.
    fn gemv_q4k_f32         (&self, _p: &GemvQ4KF32Params)       -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_q4k_f32".into()))
    }
    /// Q6_K × F32 GEMV — direct F32 input, no quantize_q8 step.
    /// Saves 1 kernel launch vs (quantize_q8 + gemv_q6k_q8) path.
    fn gemv_q6k_f32         (&self, _p: &GemvQ6KF32Params)       -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_q6k_f32".into()))
    }
    /// BitNet 1.58 ternary GEMV: 2-bit packed {-1,0,+1} weights × F32 activations.
    /// Pure add/subtract inner loop — zero multiply instructions.
    /// Achieves ~2× memory bandwidth reduction vs Q4K (2 bits vs 4 bits per weight).
    fn gemv_ternary_f32     (&self, _p: &GemvTernaryF32Params)   -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_ternary_f32".into()))
    }
    /// BF16 weights × F32 activations GEMV for decode (M=1).
    /// Replaces hipBLAS SGEMM for lm_head — eliminates library overhead.
    fn gemv_bf16_f32        (&self, _p: &GemvBf16F32Params)      -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_bf16_f32".into()))
    }
    /// W8A16 GEMV — INT8 weights × BF16 activations → F32 output (decode only).
    /// Symmetric per-row quantization: w_f32 = w_i8 * scale[row].
    /// ~2× memory bandwidth reduction vs BF16 → ~2× decode tok/s theoretically.
    fn gemv_w8a16           (&self, _p: &GemvW8A16Params)        -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_w8a16".into()))
    }
    /// SVD-factorized BF16 GEMM/GEMV (DNA V4 virtual tensor).
    /// Default impl chains two gemm_bf16_f32 calls via `scratch` — no kernel required.
    /// Handles both decode (m=1) and prefill (m>1) with the same math path.
    /// Backends may override with a fused kernel for throughput.
    ///
    /// Scratch buffer must hold `m * rank` F32 elements.
    fn gemv_svd_bf16(&self, p: &GemvSvdBf16Params) -> Result<(), BackendError> {
        // Step 1: scratch [m, rank] = X [m, cols] · Bᵀ [cols, rank]
        // gemm_bf16_f32 treats `b` as [n, k] (pre-transposed layout in VRAM), so passing
        // B tensor of shape [rank, cols] with n=rank, k=cols yields X · Bᵀ.
        self.gemm_bf16_f32(&GemmBf16Params {
            m: p.m,
            n: p.rank,
            k: p.cols,
            alpha: 1.0,
            beta: 0.0,
            a: p.x,
            a_offset: p.x_offset,
            b: p.b,
            c: p.scratch,
        })?;
        // Step 2: Y [m, rows] = scratch [m, rank] · Aᵀ [rank, rows]
        // A tensor is [rows, rank]; with n=rows, k=rank we get scratch · Aᵀ.
        self.gemm_bf16_f32(&GemmBf16Params {
            m: p.m,
            n: p.rows,
            k: p.rank,
            alpha: 1.0,
            beta: p.beta,
            a: p.scratch,
            a_offset: 0,
            b: p.a,
            c: p.out,
        })
    }
    /// Fused RMSNorm + Q/K/V Q4K projection — eliminates Q8 intermediate.
    /// Saves 1 kernel per attention layer (rms_norm_quantize + fused_qkv → single launch).
    fn fused_rms_qkv_q4k_f32(&self, _p: &FusedRmsQkvQ4KF32Params) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_rms_qkv_q4k_f32".into()))
    }
    /// Fused RMSNorm + QKV where wq=Q4K, wk=Q4K, wv=Q6K (Qwen2.5-1.5B GGUF layout).
    fn fused_rms_qkv_q4k_q6k_f32(&self, _p: &crate::params::FusedRmsQkvQ4KQ6KF32Params) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_rms_qkv_q4k_q6k_f32".into()))
    }
    /// Fused RMSNorm + gate+up+SiLU Q4K — eliminates Q8 intermediate for MLP.
    /// Saves 1 kernel per MLP layer (rms_norm_quantize + gate_up_silu → single launch).
    fn fused_rms_gate_up_silu_q4k_f32(&self, _p: &FusedRmsGateUpSiluQ4KF32Params) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_rms_gate_up_silu_q4k_f32".into()))
    }
    /// Fused RMSNorm + Q/K/V projection for IARA native Q4 format (saves 3 launches).
    fn fused_rms_qkv_q4(&self, _p: &FusedRmsQkvQ4Params) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_rms_qkv_q4".into()))
    }
    /// Fused RMSNorm + gate+up+SiLU for IARA native Q4 format (saves 3 launches).
    fn fused_rms_gate_up_silu_q4(&self, _p: &FusedRmsGateUpSiluQ4Params) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_rms_gate_up_silu_q4".into()))
    }
    /// Q6_K × Q8_1 GEMV with integer multiply.
    fn gemv_q6k_q8          (&self, _p: &GemvQ6KQ8Params)        -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_q6k_q8".into()))
    }

    fn gravity_score        (&self, p: &GravityScoreParams)       -> Result<(), BackendError>;
    fn cfc_forward          (&self, p: &CfcForwardParams)         -> Result<(), BackendError>;
    fn kv_cache_quantized_append(&self, p: &KvCacheQuantizedParams) -> Result<(), BackendError>;
    fn fused_norm_bias_rope (&self, p: &FusedNormBiasRopeParams)  -> Result<(), BackendError>;
    fn kv_cache_int8_append(&self, p: &KvCacheInt8AppendParams) -> Result<(), BackendError>;
    fn kv_cache_int8_read(&self, p: &KvCacheInt8ReadParams) -> Result<(), BackendError>;

    /// WHT+INT8 KV cache append (Item 4: PolarQuant).
    fn kv_cache_wht_append(&self, _p: &KvCacheWhtAppendParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("kv_cache_wht_append".into()))
    }

    /// WHT+INT8 KV cache read/dequant (Item 4: PolarQuant).
    fn kv_cache_wht_read(&self, _p: &KvCacheWhtReadParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("kv_cache_wht_read".into()))
    }

    /// Fused RoPE + KV cache copy: apply RoPE to Q/K, copy K+V to cache (saves 3 launches).
    fn fused_rope_kv_copy(&self, _p: &FusedRopeKvCopyParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_rope_kv_copy".into()))
    }

    /// Fused RMSNorm + Q8 quantize in one kernel (saves 1 launch per norm+quantize pair).
    fn fused_rms_norm_quantize(&self, _p: &FusedRmsNormQuantizeParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_rms_norm_quantize".into()))
    }

    /// Fused QKV projection: compute Q, K, V from same input in one kernel launch.
    /// Default: returns NotImplemented (fallback to separate Q + K + V projections).
    fn fused_qkv(&self, _p: &FusedQkvParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_qkv".into()))
    }

    /// Fused MLP gate+up+SiLU: out[i] = SiLU(dot(gate[i], x)) * dot(up[i], x) in one kernel.
    /// Default: returns NotImplemented (fallback to separate gate + up + silu_mul).
    fn fused_gate_up_silu(&self, _p: &FusedGateUpSiluParams, _is_q6k: bool) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_gate_up_silu".into()))
    }

    /// Fused single-token attention: Q×K^T → softmax → ×V in one kernel (decode n=1).
    /// Default: returns NotImplemented (fallback to separate GEMM+softmax+GEMM).
    fn fused_attention_decode(&self, _p: &FusedAttentionDecodeParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("fused_attention_decode".into()))
    }

    /// Soma dois tensores F32 in-place: a[i] += alpha * b[i]
    /// Usado para aplicar CfC correction ao hidden state.
    fn add_tensors_inplace(&self, a: &AnyTensor, b: &AnyTensor, n_elems: usize, alpha: f32) -> Result<(), BackendError>;

    // ── FP8 ──────────────────────────────────────────────────────────────────

    /// Upload FP8 data + per-row scales to device
    fn upload_fp8(&self, data: &[u8], scales: &[f32], shape: &[usize]) -> (AnyTensor, AnyTensor) {
        let data_tensor = self.upload_f32(
            &data.iter().map(|&b| b as f32).collect::<Vec<_>>(), shape
        );
        let scales_tensor = self.upload_f32(scales, &[shape[0]]);
        (data_tensor, scales_tensor)
    }

    /// Dequantize FP8 tensor to F32 on device
    fn dequant_fp8_to_f32(&self, _fp8_data: &AnyTensor, _scales: &AnyTensor, _output: &AnyTensor, _m: usize, _k: usize) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("dequant_fp8_to_f32".into()))
    }

    // ── BitNet Ternary ───────────────────────────────────────────────────────

    /// Upload ternary packed weights to device
    fn upload_ternary(&self, packed: &[u8], shape: &[usize]) -> AnyTensor {
        // Default: upload as raw bytes via u8→f32 conversion
        self.upload_f32(&packed.iter().map(|&b| b as f32).collect::<Vec<_>>(), shape)
    }

    /// Ternary GEMV: y = W_ternary @ x * scale (no multiplications, only add/sub)
    fn gemv_ternary(&self, _packed: &AnyTensor, _x: &AnyTensor, _output: &AnyTensor,
                    _m: usize, _n: usize, _k: usize, _scale: f32) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gemv_ternary".into()))
    }

    // ── Sparse MLP (gather/scatter for Kuramoto Router) ────────────────────

    /// Gather selected rows from BF16 weight matrix.
    fn gather_rows_bf16(&self, _p: &GatherRowsBf16Params) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gather_rows_bf16".into()))
    }

    /// Gather selected columns from BF16 weight matrix.
    fn gather_cols_bf16(&self, _p: &GatherColsBf16Params) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("gather_cols_bf16".into()))
    }

    /// GPU-side Kuramoto phase coherence: hidden × tuning_forks → coherences.
    fn kuramoto_coherence(&self, _p: &KuramotoCoherenceParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("kuramoto_coherence".into()))
    }

    /// GPU-side top-K selection from coherences → indices.
    fn topk_select(&self, _p: &TopkSelectParams) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("topk_select".into()))
    }

    // ── GPU-side sampling ─────────────────────────────────────────────────────

    /// GPU-side argmax: find index of max value without downloading logits.
    /// Returns (argmax_tensor_i32, optional_maxval_tensor_f32).
    /// Default: returns NotImplemented (fallback to CPU download + argmax).
    fn gpu_argmax(&self, _logits: &AnyTensor, _ncols: usize) -> Result<i32, BackendError> {
        Err(BackendError::NotImplemented("gpu_argmax".into()))
    }

    // ── Online BI measurement (adaptive layer skip) ───────────────────────────

    /// Compute cosine similarity between two F32 vectors (1D, length = dim).
    /// Writes result to `out[0]`. Used for Block Influence measurement.
    fn cosine_sim_1d(&self, _a: &AnyTensor, _b: &AnyTensor, _out: &AnyTensor, _dim: usize) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("cosine_sim_1d".into()))
    }

    /// Copy F32 tensor data on device (D2D). Used to snapshot hidden state before layer.
    fn copy_f32_d2d(&self, _src: &AnyTensor, _dst: &AnyTensor, _n: usize) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("copy_f32_d2d".into()))
    }

    /// Widen BF16 (u16 bits) → F32 on device. Used to avoid host-side conversion
    /// when only the widened form is needed device-side (e.g., dna_compress).
    fn widen_bf16_f32(&self, _src_bf16: &AnyTensor, _dst_f32: &AnyTensor, _n: usize) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("widen_bf16_f32".into()))
    }

    // ── Graph capture (HIP graphs / CUDA graphs) ──────────────────────────────

    /// Begin capturing kernel launches into a graph (instead of executing them).
    /// Returns false if graph capture is not supported by this backend.
    fn begin_graph_capture(&self) -> bool { false }

    /// End graph capture, instantiate/update executable, and launch the graph.
    /// Returns false if graph capture was not active.
    fn end_graph_capture_and_launch(&self) -> bool { false }

    // ── Dense linear algebra (hipSOLVER / future LAPACK-like) ─────────────────

    /// QR factorisation of an F32 matrix: A = Q · R, producing Q explicitly.
    ///
    /// **Layout**: the input is expected in **column-major** (LAPACK convention).
    /// `a` is `rows × cols` with leading dimension = rows. `q_out` must have
    /// capacity for `rows * cols` floats; on return it holds the orthonormal Q.
    /// `a` may be clobbered.
    fn qr_f32(&self, _a: &AnyTensor, _q_out: &AnyTensor, _rows: usize, _cols: usize) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("qr_f32".into()))
    }

    /// Thin SVD of an F32 matrix: A = U · diag(S) · Vᵀ.
    ///
    /// **Layout**: the input is expected in **column-major**. For an input of
    /// shape `rows × cols` with `k = min(rows, cols)`:
    /// - `u`  : column-major `rows × k`
    /// - `s`  : `k` singular values (non-increasing)
    /// - `vt` : column-major `k × cols`
    /// `a` is destroyed in-place.
    fn svd_f32(
        &self,
        _a: &AnyTensor,
        _u: &AnyTensor,
        _s: &AnyTensor,
        _vt: &AnyTensor,
        _rows: usize,
        _cols: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::NotImplemented("svd_f32".into()))
    }

    // ── Sync e informações ───────────────────────────────────────────────────

    /// Sincroniza o device (equivale a hipDeviceSynchronize / clFinish).
    fn sync(&self) -> Result<(), BackendError>;

    /// (free_bytes, total_bytes) — opcional, retorna None se não disponível.
    fn mem_info(&self) -> Option<(usize, usize)> { None }
}

#[cfg(test)]
mod tests {
    //! Verifies the SVD-factorized GEMM math that `gemv_svd_bf16`'s default impl
    //! encodes. Since no CPU ComputeBackend ships in this crate, we emulate the
    //! two gemm_bf16_f32 calls with a pure-F32 row-major matmul helper and
    //! confirm `(X · Bᵀ) · Aᵀ ≈ X · (A·B)ᵀ` — i.e., that the chained form
    //! produces the same output as computing against the reconstructed W.
    //!
    //! This locks in the math of the default impl; the full backend integration
    //! is covered by rocm-backend / opencl-backend smoke tests at runtime.
    //!
    //! Backend-level numerical tests against real hardware are deferred until a
    //! ROCm/OpenCL test harness is available — document only here.

    /// Row-major F32 matmul: C [m, n] = A [m, k] · B [k, n], beta-scaled.
    fn matmul_rm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// Treats `b` as `[n, k]` pre-transposed (matches gemm_bf16_f32's `b` layout).
    /// Computes C [m, n] = A [m, k] · Bᵀ, where B is stored [n, k] row-major.
    fn matmul_b_transposed(a: &[f32], b_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b_nk[j * k + p];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn run_case(m: usize) {
        let rows = 7usize;  // W.rows (output dim)
        let cols = 11usize; // W.cols (input dim)
        let rank = 3usize;

        // Deterministic pseudo-random fill.
        let mk = |n: usize, seed: u32| -> Vec<f32> {
            (0..n).map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((x & 0xFFFF) as f32 / 65535.0) - 0.5
            }).collect()
        };

        let a_mat = mk(rows * rank, 1); // A [rows, rank]
        let b_mat = mk(rank * cols, 2); // B [rank, cols]
        let x     = mk(m * cols, 3);    // X [m, cols]

        // Reference: reconstruct W [rows, cols] = A · B, then Y = X · Wᵀ.
        let w = matmul_rm(&a_mat, &b_mat, rows, cols, rank);
        // Y_ref [m, rows] = X · Wᵀ.  W stored [rows, cols] → treat as "b [n=rows, k=cols]".
        let y_ref = matmul_b_transposed(&x, &w, m, rows, cols);

        // Chained form — mirrors default impl:
        //   Step 1: scratch [m, rank] = X · Bᵀ, with B passed as [n=rank, k=cols].
        let scratch = matmul_b_transposed(&x, &b_mat, m, rank, cols);
        //   Step 2: Y [m, rows] = scratch · Aᵀ, with A passed as [n=rows, k=rank].
        let y_chained = matmul_b_transposed(&scratch, &a_mat, m, rows, rank);

        assert_eq!(y_ref.len(), y_chained.len());
        for (i, (r, c)) in y_ref.iter().zip(y_chained.iter()).enumerate() {
            let diff = (r - c).abs();
            assert!(
                diff < 1e-4,
                "m={m} idx={i} ref={r} chained={c} diff={diff}"
            );
        }
    }

    #[test]
    fn svd_chained_matches_reconstructed_m1() { run_case(1); }

    #[test]
    fn svd_chained_matches_reconstructed_m4() { run_case(4); }

    #[test]
    fn svd_chained_matches_reconstructed_m16() { run_case(16); }
}
