use crate::tensor::AnyTensor;

/// GEMM mixed precision: A(F32) × B(BF16) → C(F32)
/// Layout: row-major. B é transposta (pesos armazenados como [out_dim, in_dim]).
pub struct GemmBf16Params<'a> {
    pub m: usize,       // linhas de A e C (n_tokens)
    pub n: usize,       // colunas de C (out_dim)
    pub k: usize,       // colunas de A / linhas de B (in_dim)
    pub alpha: f32,
    pub beta: f32,      // 0.0 → sobrescreve C; 1.0 → acumula (residual)
    pub a: &'a AnyTensor,   // [m, k] F32 (ativações)
    /// Offset em elementos dentro de `a`. Permite apontar para o último token
    /// (ex: `(n_tokens - 1) * hidden_size`) sem criar sub-slice.
    pub a_offset: usize,
    pub b: &'a AnyTensor,   // [n, k] BF16 (pesos, já transpostos no layout)
    pub c: &'a AnyTensor,   // [m, n] F32 (saída — escrita no device)
}

/// GEMM F32 genérico com leading dimensions (para attention strided)
pub struct SgemmParams<'a> {
    pub trans_a: bool,
    pub trans_b: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub beta: f32,
    pub a: &'a AnyTensor,
    pub lda: usize,
    /// Offset em elementos dentro de `a` (para sub-views de cabeçote de atenção).
    pub a_offset: usize,
    pub b: &'a AnyTensor,
    pub ldb: usize,
    /// Offset em elementos dentro de `b` (para sub-views de cabeçote de atenção).
    pub b_offset: usize,
    pub c: &'a AnyTensor,
    pub ldc: usize,
    /// Offset em elementos dentro de `c` (para sub-views de cabeçote de atenção).
    pub c_offset: usize,
}

/// Embedding lookup: token_ids → embeddings BF16 → F32
pub struct EmbedParams<'a> {
    pub token_ids: &'a [u32],           // na CPU
    pub vocab: &'a AnyTensor,           // [vocab_size, hidden_size] BF16
    pub out: &'a AnyTensor,             // [n_tokens, hidden_size] F32 (saída)
    pub n_tokens: usize,
    pub hidden_size: usize,
}

/// RMSNorm: out = (input / rms(input)) * weight
pub struct RmsNormParams<'a> {
    pub input: &'a AnyTensor,   // [n_tokens, hidden] F32
    pub weight: &'a AnyTensor,  // [hidden] F32
    pub output: &'a AnyTensor,  // [n_tokens, hidden] F32 (pode ser == input para in-place)
    pub n_tokens: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

/// Rotary Position Embeddings (in-place)
pub struct RopeParams<'a> {
    pub tensor: &'a AnyTensor,  // [n_tokens, n_heads, head_dim] F32 (modificado in-place)
    pub pos_offset: usize,      // posição absoluta do primeiro token (para KV cache)
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
}

/// Softmax row-wise (in-place)
pub struct SoftmaxParams<'a> {
    pub scores: &'a AnyTensor,  // [n_rows, row_len] F32 (modificado in-place)
    /// Offset em elementos dentro de `scores` (para apontar para o bloco do
    /// cabeçote correto em buffers compartilhados de [n_heads * n_tokens, seq]).
    pub offset_elems: usize,
    pub n_rows: usize,
    pub row_len: usize,
}

/// SwiGLU: gate = silu(gate) * up (in-place em gate)
pub struct SiluMulParams<'a> {
    pub gate: &'a AnyTensor,    // [n_tokens, intermediate] F32 (modificado in-place)
    pub up: &'a AnyTensor,      // [n_tokens, intermediate] F32 (só leitura)
    pub n_tokens: usize,
    pub intermediate_size: usize,
}

/// Append de KV no cache com sliding window
pub struct KvCacheParams<'a> {
    pub cache: &'a AnyTensor,   // [max_seq, kv_dim] F32 (escrito na posição insert_pos)
    pub new_kv: &'a AnyTensor,  // [n_new_tokens, kv_dim] F32 (novos tokens)
    pub insert_pos: usize,      // posição de inserção (com wrap se sliding window)
    pub n_new_tokens: usize,
    pub kv_dim: usize,
    pub max_seq: usize,
}

/// Add bias in-place: tensor[tok, d] += bias[d]
pub struct AddBiasParams<'a> {
    pub tensor: &'a AnyTensor,  // [n_tokens, dim] F32 (modificado in-place)
    pub bias: &'a AnyTensor,    // [dim] F32
    pub n_tokens: usize,
    pub dim: usize,
}

/// GEMV dequantizado: pesos INT4 comprimido × ativações F32 → F32
/// Dequantiza on-the-fly: w_float = scale * (packed_val - zero)
pub struct GemvQ4Params<'a> {
    pub m: usize,                  // linhas (n_tokens, tipicamente 1 no decode)
    pub n: usize,                  // colunas de saída (out_dim)
    pub k: usize,                  // dimensão interna (in_dim)
    pub packed: &'a AnyTensor,     // [n, k/2] Q4 comprimido (2 valores por byte)
    pub scales: &'a AnyTensor,     // [n, k/group_size] F32 scales por grupo
    pub zeros: &'a AnyTensor,      // [n, k/group_size] F32 zero-points por grupo
    pub group_size: usize,         // tipicamente 128
    pub x: &'a AnyTensor,          // [m, k] F32 ativações
    pub out: &'a AnyTensor,        // [m, n] F32 saída
    pub beta: f32,                 // 0.0 sobrescreve, 1.0 acumula (residual)
}

/// Gravity Score: calcula importância semântica por token
/// gravity(i) = média(attn_score(i) * ||value_vec(i)||₂) sobre heads
pub struct GravityScoreParams<'a> {
    pub attn_weights: &'a AnyTensor,  // [n_heads, seq_len] F32 (após softmax)
    pub v_cache: &'a AnyTensor,       // [max_seq_len, n_kv_heads * head_dim] F32 (V cache)
    pub value_norms: &'a AnyTensor,   // [n_kv_heads, max_seq_len] F32 (buffer de saída para norms)
    pub gravity_out: &'a AnyTensor,   // [seq_len] F32 (média sobre heads)
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
}

/// CfC Forward: Closed-form Continuous liquid correction layer
/// h(t) = σ(f) · h_prev + (1-σ(f)) · tanh(g)
/// onde f = x @ wf + h_prev @ wf2 + bf
///      g = x @ wg + h_prev @ wg2 + bg
pub struct CfcForwardParams<'a> {
    pub x: &'a AnyTensor,         // [n_tokens, hidden] F32 — hidden state do transformer
    pub h_prev: &'a AnyTensor,    // [cfc_dim] F32 — estado CfC anterior
    pub wf: &'a AnyTensor,        // [cfc_dim, hidden] F32 — pesos do gate f
    pub wf2: &'a AnyTensor,       // [cfc_dim, cfc_dim] F32 — recurrence f
    pub bf: &'a AnyTensor,        // [cfc_dim] F32 — bias f
    pub wg: &'a AnyTensor,        // [cfc_dim, hidden] F32 — pesos do candidate g
    pub wg2: &'a AnyTensor,       // [cfc_dim, cfc_dim] F32 — recurrence g
    pub bg: &'a AnyTensor,        // [cfc_dim] F32 — bias g
    pub wout: &'a AnyTensor,      // [hidden, cfc_dim] F32 — projeção de saída
    pub bout: &'a AnyTensor,      // [hidden] F32 — bias de saída
    pub h_out: &'a AnyTensor,     // [cfc_dim] F32 — novo estado (escrita)
    pub correction: &'a AnyTensor, // [n_tokens, hidden] F32 — correção (escrita, somada ao residual)
    pub n_tokens: usize,
    pub hidden: usize,
    pub cfc_dim: usize,
}

/// Quantized KV Cache: append com quantização on-the-fly
pub struct KvCacheQuantizedParams<'a> {
    pub cache: &'a AnyTensor,     // [max_seq, kv_dim/2] Q4 comprimido
    pub scales: &'a AnyTensor,    // [max_seq, kv_dim/group_size] F16
    pub zeros: &'a AnyTensor,     // [max_seq, kv_dim/group_size] F16
    pub new_kv: &'a AnyTensor,    // [n_new, kv_dim] F32 (será quantizado)
    pub insert_pos: usize,
    pub n_new_tokens: usize,
    pub kv_dim: usize,
    pub max_seq: usize,
    pub group_size: usize,
}
