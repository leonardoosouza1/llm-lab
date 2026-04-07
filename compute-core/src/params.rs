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

/// Params para SGEMM batched (todas as cabeças de atenção em 1 chamada)
/// Usa hipblasSgemmStridedBatched para substituir 32 chamadas sequenciais por 1.
pub struct SgemmStridedBatchedParams<'a> {
    pub trans_a: bool,
    pub trans_b: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub beta: f32,
    pub a: &'a AnyTensor,
    pub lda: usize,
    pub stride_a: usize,  // stride entre batches em A
    pub b: &'a AnyTensor,
    pub ldb: usize,
    pub stride_b: usize,  // stride entre batches em B
    pub c: &'a AnyTensor,
    pub ldc: usize,
    pub stride_c: usize,  // stride entre batches em C
    pub batch_count: usize,
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

/// LayerNorm: out = (input - mean) / sqrt(variance + eps) * weight + bias
/// Usado por Phi, StableLM, Falcon (em vez de RMSNorm)
pub struct LayerNormParams<'a> {
    pub input: &'a AnyTensor,   // [n_tokens, hidden] F32
    pub weight: &'a AnyTensor,  // [hidden] F32
    pub bias: &'a AnyTensor,    // [hidden] F32
    pub output: &'a AnyTensor,  // [n_tokens, hidden] F32 (pode ser == input para in-place)
    pub n_tokens: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

/// GELU activation: gate = gelu(gate) * up (in-place em gate)
/// Usado por Phi, Gemma, Falcon (em vez de SiLU/SwiGLU)
pub struct GeluMulParams<'a> {
    pub gate: &'a AnyTensor,    // [n_tokens, intermediate] F32 (modificado in-place)
    pub up: &'a AnyTensor,      // [n_tokens, intermediate] F32 (só leitura)
    pub n_tokens: usize,
    pub intermediate_size: usize,
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

/// Params para kernel fused: RMSNorm + bias opcional + RoPE
/// Equivale a: rms_norm(input, weight) → add_bias(output, bias) → rope(output, pos)
/// Mas em um único kernel dispatch, evitando 2 round-trips de VRAM.
pub struct FusedNormBiasRopeParams<'a> {
    pub input: &'a AnyTensor,      // [n_tokens, hidden_size] F32
    pub norm_weight: &'a AnyTensor, // [hidden_size] F32
    pub bias: Option<&'a AnyTensor>, // [hidden_size] F32 — None se modelo sem bias (ex: Llama)
    pub output: &'a AnyTensor,     // [n_tokens, hidden_size] F32 (escrita no device)
    pub n_tokens: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub pos_offset: usize,         // posição no KV-cache para RoPE
    pub theta: f32,                // RoPE theta (10000.0 ou 1000000.0)
    pub eps: f32,                  // epsilon para RMSNorm (1e-5 ou 1e-6)
}

/// Params para quantizar F32→INT8 e inserir no KV-cache
/// Quantização por-cabeça: cada cabeça tem seu próprio scale factor.
pub struct KvCacheInt8AppendParams<'a> {
    pub cache_q: &'a AnyTensor,       // INT8 cache [max_seq, kv_dim]
    pub cache_scales: &'a AnyTensor,   // F32 scales [max_seq, n_kv_heads] — 1 scale per head per token
    pub new_kv: &'a AnyTensor,         // F32 input [n_new_tokens, kv_dim]
    pub insert_pos: usize,
    pub n_new_tokens: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub max_seq: usize,
}

/// Params para dequantizar INT8→F32 do KV-cache para atenção
pub struct KvCacheInt8ReadParams<'a> {
    pub cache_q: &'a AnyTensor,        // INT8 cache
    pub cache_scales: &'a AnyTensor,    // F32 scales
    pub output: &'a AnyTensor,          // F32 output [seq_len, kv_dim]
    pub seq_len: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
}

// ── Sparse MLP (Kuramoto Router) ────────────────────────────────────────────

/// Gather selected rows from a BF16 weight matrix into contiguous output.
/// src[total_rows, cols] → dst[n_indices, cols], selected by indices.
pub struct GatherRowsBf16Params<'a> {
    pub src: &'a AnyTensor,       // [total_rows, cols] BF16
    pub dst: &'a AnyTensor,       // [n_indices, cols] BF16
    pub indices: &'a AnyTensor,   // [n_indices] I32
    pub n_indices: usize,
    pub cols: usize,
}

/// Gather selected columns from a BF16 weight matrix into contiguous output.
/// src[rows, total_cols] → dst[rows, n_indices], selected by indices.
pub struct GatherColsBf16Params<'a> {
    pub src: &'a AnyTensor,       // [rows, total_cols] BF16
    pub dst: &'a AnyTensor,       // [rows, n_indices] BF16
    pub indices: &'a AnyTensor,   // [n_indices] I32
    pub rows: usize,
    pub total_cols: usize,
    pub n_indices: usize,
}

/// GPU-side Kuramoto coherence computation.
/// hidden[hidden_dim] F32 × tuning_forks[n_neurons, phase_dim] F32 → coherences[n_neurons] F32
pub struct KuramotoCoherenceParams<'a> {
    pub hidden: &'a AnyTensor,        // [hidden_dim] F32
    pub tuning_forks: &'a AnyTensor,  // [n_neurons, phase_dim] F32
    pub coherences: &'a AnyTensor,    // [n_neurons] F32 — output
    pub n_neurons: usize,
    pub hidden_dim: usize,
    pub phase_dim: usize,
}

/// GPU-side top-K selection from coherences.
/// coherences[n_neurons] F32 → indices[top_k] I32
pub struct TopkSelectParams<'a> {
    pub coherences: &'a AnyTensor,  // [n_neurons] F32
    pub indices: &'a AnyTensor,     // [top_k] I32 — output
    pub n_neurons: usize,
    pub top_k: usize,
}
