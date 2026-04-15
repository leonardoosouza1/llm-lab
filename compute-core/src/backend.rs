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

    // ── Graph capture (HIP graphs / CUDA graphs) ──────────────────────────────

    /// Begin capturing kernel launches into a graph (instead of executing them).
    /// Returns false if graph capture is not supported by this backend.
    fn begin_graph_capture(&self) -> bool { false }

    /// End graph capture, instantiate/update executable, and launch the graph.
    /// Returns false if graph capture was not active.
    fn end_graph_capture_and_launch(&self) -> bool { false }

    // ── Sync e informações ───────────────────────────────────────────────────

    /// Sincroniza o device (equivale a hipDeviceSynchronize / clFinish).
    fn sync(&self) -> Result<(), BackendError>;

    /// (free_bytes, total_bytes) — opcional, retorna None se não disponível.
    fn mem_info(&self) -> Option<(usize, usize)> { None }
}
