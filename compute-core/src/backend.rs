use crate::{AnyTensor, BackendError, params::*};
use crate::params::{KvCacheInt8AppendParams, KvCacheInt8ReadParams};

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

    /// Download de tensor F32 device → Vec<f32> host. Sincroniza o device.
    fn download_f32(&self, t: &AnyTensor) -> Vec<f32>;

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
    fn silu_mul_inplace     (&self, p: &SiluMulParams)            -> Result<(), BackendError>;
    fn gelu_mul_inplace     (&self, p: &GeluMulParams)            -> Result<(), BackendError>;
    fn copy_to_kv_cache     (&self, p: &KvCacheParams)            -> Result<(), BackendError>;
    fn add_bias_inplace     (&self, p: &AddBiasParams)            -> Result<(), BackendError>;
    fn gemv_q4              (&self, p: &GemvQ4Params)             -> Result<(), BackendError>;
    fn gravity_score        (&self, p: &GravityScoreParams)       -> Result<(), BackendError>;
    fn cfc_forward          (&self, p: &CfcForwardParams)         -> Result<(), BackendError>;
    fn kv_cache_quantized_append(&self, p: &KvCacheQuantizedParams) -> Result<(), BackendError>;
    fn fused_norm_bias_rope (&self, p: &FusedNormBiasRopeParams)  -> Result<(), BackendError>;
    fn kv_cache_int8_append(&self, p: &KvCacheInt8AppendParams) -> Result<(), BackendError>;
    fn kv_cache_int8_read(&self, p: &KvCacheInt8ReadParams) -> Result<(), BackendError>;

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

    // ── Sync e informações ───────────────────────────────────────────────────

    /// Sincroniza o device (equivale a hipDeviceSynchronize / clFinish).
    fn sync(&self) -> Result<(), BackendError>;

    /// (free_bytes, total_bytes) — opcional, retorna None se não disponível.
    fn mem_info(&self) -> Option<(usize, usize)> { None }
}
