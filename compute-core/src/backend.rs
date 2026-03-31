use crate::{AnyTensor, BackendError, params::*};

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

    /// Aloca buffer Q4 não inicializado no device.
    fn alloc_q4(&self, shape: &[usize]) -> AnyTensor;

    /// Download de tensor F32 device → Vec<f32> host. Sincroniza o device.
    fn download_f32(&self, t: &AnyTensor) -> Vec<f32>;

    // ── GEMM ─────────────────────────────────────────────────────────────────

    fn gemm_bf16_f32(&self, p: &GemmBf16Params) -> Result<(), BackendError>;
    fn sgemm(&self, p: &SgemmParams) -> Result<(), BackendError>;

    // ── Kernels customizados ──────────────────────────────────────────────────

    fn embed_lookup         (&self, p: &EmbedParams)              -> Result<(), BackendError>;
    fn rms_norm             (&self, p: &RmsNormParams)            -> Result<(), BackendError>;
    fn rope_inplace         (&self, p: &RopeParams)               -> Result<(), BackendError>;
    fn softmax_rows         (&self, p: &SoftmaxParams)            -> Result<(), BackendError>;
    fn silu_mul_inplace     (&self, p: &SiluMulParams)            -> Result<(), BackendError>;
    fn copy_to_kv_cache     (&self, p: &KvCacheParams)            -> Result<(), BackendError>;
    fn add_bias_inplace     (&self, p: &AddBiasParams)            -> Result<(), BackendError>;
    fn gemv_q4              (&self, p: &GemvQ4Params)             -> Result<(), BackendError>;
    fn gravity_score        (&self, p: &GravityScoreParams)       -> Result<(), BackendError>;
    fn cfc_forward          (&self, p: &CfcForwardParams)         -> Result<(), BackendError>;
    fn kv_cache_quantized_append(&self, p: &KvCacheQuantizedParams) -> Result<(), BackendError>;

    // ── Sync e informações ───────────────────────────────────────────────────

    /// Sincroniza o device (equivale a hipDeviceSynchronize / clFinish).
    fn sync(&self) -> Result<(), BackendError>;

    /// (free_bytes, total_bytes) — opcional, retorna None se não disponível.
    fn mem_info(&self) -> Option<(usize, usize)> { None }
}
