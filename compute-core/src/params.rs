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
///
/// Offsets: `a_offset`, `b_offset`, `c_offset` são em elementos e definem o ponto
/// inicial do batch 0 dentro do tensor. Batch `i` começa em `offset + i * stride`.
/// Para broadcast (mesma matriz para todos os batches), use `stride = 0`.
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
    pub a_offset: usize,  // offset em elementos do batch 0 em A
    pub b: &'a AnyTensor,
    pub ldb: usize,
    pub stride_b: usize,  // stride entre batches em B (0 = broadcast)
    pub b_offset: usize,  // offset em elementos do batch 0 em B
    pub c: &'a AnyTensor,
    pub ldc: usize,
    pub stride_c: usize,  // stride entre batches em C
    pub c_offset: usize,  // offset em elementos do batch 0 em C
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

/// Causal softmax row-wise (in-place) — masks future key positions
pub struct SoftmaxCausalParams<'a> {
    pub scores: &'a AnyTensor,
    pub offset_elems: usize,
    pub n_rows: usize,
    pub row_len: usize,
    pub n_tokens: usize,      // query tokens in this batch
    pub query_offset: usize,  // position of first query token in sequence
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
    pub x_offset: usize,           // offset em floats para x (para selecionar token específico)
    pub out: &'a AnyTensor,        // [m, n] F32 saída
    pub beta: f32,                 // 0.0 sobrescreve, 1.0 acumula (residual)
}

/// Native GGML Q4_K GEMV: raw Q4_K blocks × F32 activations → F32 output.
/// Dequantizes on-the-fly in F32 — no BF16 intermediate, no error compounding.
/// Each weight row is stored as consecutive Q4_K blocks (144 bytes per 256 values).
pub struct GemvQ4KParams<'a> {
    pub m: usize,                  // batch size (n_tokens, typically 1 for decode)
    pub n: usize,                  // output dimension (rows of weight matrix)
    pub k: usize,                  // input dimension (columns, must be multiple of 256)
    pub data: &'a AnyTensor,       // [n * (k/256) * 144] raw Q4_K block bytes on GPU
    pub x: &'a AnyTensor,          // [m, k] F32 input activations
    pub x_offset: usize,           // offset in floats into x (for selecting specific token)
    pub out: &'a AnyTensor,        // [m, n] F32 output
    pub beta: f32,                 // 0.0=overwrite, 1.0=accumulate
}

/// Native GGML Q6_K GEMV: raw Q6_K blocks × F32 activations → F32 output.
/// Same approach as Q4_K but for 6-bit quantization (210 bytes per 256 values).
pub struct GemvQ6KParams<'a> {
    pub m: usize,
    pub n: usize,
    pub k: usize,                  // must be multiple of 256
    pub data: &'a AnyTensor,       // [n * (k/256) * 210] raw Q6_K block bytes on GPU
    pub x: &'a AnyTensor,          // [m, k] F32 input activations
    pub x_offset: usize,
    pub out: &'a AnyTensor,        // [m, n] F32 output
    pub beta: f32,
}

///// Quantize F32 activations to Q8_1 blocks for integer GEMV.
pub struct QuantizeQ8Params<'a> {
    pub x: &'a AnyTensor,          // [k] F32 input
    pub x_offset: usize,           // offset in floats
    pub out: &'a AnyTensor,        // [(k/32)*36] u8 output Q8_1 blocks
    pub k: usize,                  // input dimension, must be multiple of 32
}

/// Q4_K × F32 GEMV — direct F32 activations, no Q8 intermediate.
/// Eliminates quantize step between fused_gate_up_silu output and down projection.
pub struct GemvQ4KF32Params<'a> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub data: &'a AnyTensor,   // [n * (k/256) * 144] raw Q4_K blocks
    pub x: &'a AnyTensor,      // [k] F32 activations
    pub x_offset: usize,       // element offset into x (for last-token select)
    pub out: &'a AnyTensor,    // [m, n] F32 output
    pub beta: f32,
}

/// BF16 weights × F32 activations GEMV — replaces hipBLAS for decode (M=1).
/// Used for lm_head: N=vocab_size, K=hidden_size.
pub struct GemvBf16F32Params<'a> {
    pub n: usize,              // output dim (vocab_size)
    pub k: usize,              // input dim (hidden_size)
    pub w: &'a AnyTensor,     // [n, k] BF16 weights
    pub x: &'a AnyTensor,     // [k] F32 activations (last token)
    pub x_offset: usize,       // element offset into x
    pub out: &'a AnyTensor,   // [n] F32 output
}

/// W8A32 GEMV — INT8 weights × F32 activations → F32 output.
/// Weights are quantized symmetrically per-row: w_f32 = w_i8 * scale[row].
/// Halves memory bandwidth vs BF16 GEMV (1 byte/weight vs 2 bytes).
/// Theoretical speedup on RX 6750 XT: ~2× decode tok/s vs BF16 (weight BW bound).
pub struct GemvW8A16Params<'a> {
    pub n: usize,              // output dim (rows of W)
    pub k: usize,              // input dim (cols of W)
    pub w: &'a AnyTensor,     // [n, k] INT8 weights (row-major, stored as u8 on device)
    pub scales: &'a AnyTensor, // [n] F32 per-row scales
    pub x: &'a AnyTensor,     // [k] F32 activations
    pub x_offset: usize,       // element offset into x (F32 elements)
    pub out: &'a AnyTensor,   // [n] F32 output
    pub beta: f32,             // 0.0 = overwrite; 1.0 = accumulate (residual add)
}

/// Fused RMSNorm + Q/K/V Q4K projection in one kernel.
/// Eliminates Q8 intermediate tensor: saves 1 launch per attention layer.
pub struct FusedRmsQkvQ4KF32Params<'a> {
    pub x: &'a AnyTensor,       // [k] F32 hidden state (input, read-only)
    pub norm_w: &'a AnyTensor,  // [k] F32 RMSNorm weights
    pub wq: &'a AnyTensor,      // [nq * (k/256) * 144] Q4K Q weights
    pub wk: &'a AnyTensor,      // [nk * (k/256) * 144] Q4K K weights
    pub wv: &'a AnyTensor,      // [nv * (k/256) * 144] Q4K V weights
    pub out_q: &'a AnyTensor,   // [nq] F32 Q output
    pub out_k: &'a AnyTensor,   // [nk] F32 K output
    pub out_v: &'a AnyTensor,   // [nv] F32 V output
    pub bias_q: Option<&'a AnyTensor>,
    pub bias_k: Option<&'a AnyTensor>,
    pub bias_v: Option<&'a AnyTensor>,
    pub nq: usize,
    pub nk: usize,
    pub nv: usize,
    pub k: usize,
    pub eps: f32,
    pub x_offset: usize,
}

/// Fused RMSNorm + gate+up+SiLU Q4K in one kernel.
/// Eliminates Q8 intermediate for MLP: saves 1 launch per MLP layer.
pub struct FusedRmsGateUpSiluQ4KF32Params<'a> {
    pub x: &'a AnyTensor,       // [k] F32 hidden state
    pub norm_w: &'a AnyTensor,  // [k] F32 RMSNorm weights
    pub gate: &'a AnyTensor,    // [n * (k/256) * 144] Q4K gate weights
    pub up: &'a AnyTensor,      // [n * (k/256) * 144] Q4K up weights
    pub out: &'a AnyTensor,     // [n] F32 output (SiLU(gate) * up)
    pub n: usize,
    pub k: usize,
    pub eps: f32,
    pub x_offset: usize,
}

/// Fused RMSNorm + Q/K/V projection for IARA native Q4 format.
/// Saves 3 kernel launches (norm+q8 + wq + wk + wv → 1 kernel).
pub struct FusedRmsQkvQ4Params<'a> {
    pub x: &'a AnyTensor,       // [k] F32 hidden state
    pub norm_w: &'a AnyTensor,  // [k] F32 RMSNorm weights
    // Q4 weight tensors (packed nibbles)
    pub wq_packed: &'a AnyTensor,
    pub wq_scales: &'a AnyTensor,
    pub wq_zeros:  &'a AnyTensor,
    pub wk_packed: &'a AnyTensor,
    pub wk_scales: &'a AnyTensor,
    pub wk_zeros:  &'a AnyTensor,
    pub wv_packed: &'a AnyTensor,
    pub wv_scales: &'a AnyTensor,
    pub wv_zeros:  &'a AnyTensor,
    // Outputs
    pub out_q: &'a AnyTensor,
    pub out_k: &'a AnyTensor,
    pub out_v: &'a AnyTensor,
    // Optional biases
    pub bias_q: Option<&'a AnyTensor>,
    pub bias_k: Option<&'a AnyTensor>,
    pub bias_v: Option<&'a AnyTensor>,
    pub nq: usize,
    pub nk: usize,
    pub nv: usize,
    pub k: usize,
    pub group_size: usize,
    pub eps: f32,
}

/// Fused RMSNorm + gate+up+SiLU for IARA native Q4 format.
/// Saves 3 kernel launches (norm+q8 + wgate + wup + silu → 1 kernel).
pub struct FusedRmsGateUpSiluQ4Params<'a> {
    pub x: &'a AnyTensor,
    pub norm_w: &'a AnyTensor,
    pub gate_packed: &'a AnyTensor,
    pub gate_scales: &'a AnyTensor,
    pub gate_zeros:  &'a AnyTensor,
    pub up_packed:   &'a AnyTensor,
    pub up_scales:   &'a AnyTensor,
    pub up_zeros:    &'a AnyTensor,
    pub out: &'a AnyTensor,
    pub n: usize,
    pub k: usize,
    pub group_size: usize,
    pub eps: f32,
}

/// Q4_K × Q8_1 GEMV with dp4a integer dot product.
pub struct GemvQ4KQ8Params<'a> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub data: &'a AnyTensor,       // [n * (k/256) * 144] raw Q4_K blocks
    pub x_q8: &'a AnyTensor,      // [(k/32)*36] Q8_1 blocks (pre-quantized)
    pub out: &'a AnyTensor,        // [m, n] F32 output
    pub beta: f32,
}

/// Q6_K × Q8_1 GEMV with integer multiply.
pub struct GemvQ6KQ8Params<'a> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub data: &'a AnyTensor,       // [n * (k/256) * 210] raw Q6_K blocks
    pub x_q8: &'a AnyTensor,      // [(k/32)*36] Q8_1 blocks (pre-quantized)
    pub out: &'a AnyTensor,        // [m, n] F32 output
    pub beta: f32,
}

/// Fused RoPE + KV cache copy: apply RoPE to Q and K, copy K+V to KV cache.
pub struct FusedRopeKvCopyParams<'a> {
    pub q: &'a AnyTensor,            // [n_heads * head_dim] — RoPE applied in-place
    pub k: &'a AnyTensor,            // [n_kv_heads * head_dim] — RoPE applied in-place
    pub v: &'a AnyTensor,            // [n_kv_heads * head_dim] — copied to cache
    pub kv_k_cache: &'a AnyTensor,   // [max_seq, kv_dim] — K cache
    pub kv_v_cache: &'a AnyTensor,   // [max_seq, kv_dim] — V cache
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub pos_start: usize,
    pub theta: f32,
    pub kv_dim: usize,
    pub max_seq: usize,
}

/// Fused RMSNorm + Quantize Q8_1: normalize and quantize in one kernel launch.
pub struct FusedRmsNormQuantizeParams<'a> {
    pub input: &'a AnyTensor,     // [hidden] F32 input
    pub weight: &'a AnyTensor,    // [hidden] F32 RMSNorm weight
    pub out_f32: &'a AnyTensor,   // [hidden] F32 normed output
    pub out_q8: &'a AnyTensor,    // [(hidden/32)*36] Q8_1 output
    pub hidden_size: usize,
    pub eps: f32,
}

/// Fused QKV projection: compute Q, K, V from same Q8_1 input in one kernel launch.
pub struct FusedQkvParams<'a> {
    pub wq_data: &'a AnyTensor,   // [Nq * nblk * 144] Q weights (Q4K)
    pub wk_data: &'a AnyTensor,   // [Nk * nblk * 144] K weights (Q4K)
    pub wv_data: &'a AnyTensor,   // [Nv * nblk * 144] V weights (Q4K)
    pub x_q8: &'a AnyTensor,     // [(K/32)*36] Q8_1 input
    pub out_q: &'a AnyTensor,    // [Nq] Q output
    pub out_k: &'a AnyTensor,    // [Nk] K output
    pub out_v: &'a AnyTensor,    // [Nv] V output
    pub bias_q: Option<&'a AnyTensor>,  // [Nq] Q bias or None
    pub bias_k: Option<&'a AnyTensor>,  // [Nk] K bias or None
    pub bias_v: Option<&'a AnyTensor>,  // [Nv] V bias or None
    pub nq: usize,               // Q output dim
    pub nk: usize,               // K output dim
    pub nv: usize,               // V output dim
    pub k: usize,                // input dim (hidden_size)
}

/// Fused MLP gate+up+SiLU kernel: out[i] = SiLU(gate_dot[i]) * up_dot[i]
/// Both gate and up weight matrices are Q4_K or Q6_K, input is Q8_1.
pub struct FusedGateUpSiluParams<'a> {
    pub gate_data: &'a AnyTensor,  // gate weight raw bytes
    pub up_data: &'a AnyTensor,    // up weight raw bytes
    pub x_q8: &'a AnyTensor,      // pre-quantized Q8_1 input
    pub out: &'a AnyTensor,        // [N] F32 output (gate buffer reused)
    pub n: usize,                  // intermediate_size (output dim)
    pub k: usize,                  // hidden_size (input dim)
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

/// Params para WHT+INT8 quantize+append (Item 4: PolarQuant KV cache)
pub struct KvCacheWhtAppendParams<'a> {
    pub cache_q: &'a AnyTensor,       // INT8 cache [max_seq, kv_dim]
    pub cache_scales: &'a AnyTensor,   // F32 scales [max_seq, n_kv_heads]
    pub new_kv: &'a AnyTensor,         // F32 input [n_new_tokens, kv_dim]
    pub insert_pos: usize,
    pub n_new_tokens: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub max_seq: usize,
}

/// Params para dequantizar WHT INT8 KV cache
pub struct KvCacheWhtReadParams<'a> {
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

/// Fused single-token attention (decode only, n=1).
/// Combines Q×K^T → softmax → ×V into a single kernel, eliminating
/// the scores tensor and 3 separate kernel launches per layer.
/// Supports GQA via kv_head mapping.
pub struct FusedAttentionDecodeParams<'a> {
    pub q: &'a AnyTensor,           // [1, n_heads * head_dim] F32
    pub k_cache: &'a AnyTensor,     // [max_seq, n_kv_heads * head_dim] F32
    pub v_cache: &'a AnyTensor,     // [max_seq, n_kv_heads * head_dim] F32
    pub out: &'a AnyTensor,         // [1, n_heads * head_dim] F32 (output)
    pub seq_len: usize,             // current filled positions in KV cache
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub scale: f32,                 // 1/sqrt(head_dim)
}

/// Ternary (BitNet 1.58) × F32 GEMV — zero-multiply inference.
///
/// Weight format: 2-bit packed, 4 weights per byte, row-major.
///   Encoding: -1 = 0b00, 0 = 0b01, +1 = 0b10
///   packed[row * ceil(k/4) + col/4], bits [(col%4)*2 +: 2]
///
/// Operation: out[m,n] = scale * Σ_k { w[n,k] * x[m,k] }
/// Dequantization is purely add/sub — no multiplies in the inner loop.
pub struct GemvTernaryF32Params<'a> {
    pub m: usize,              // batch / token count (typically 1 for decode)
    pub n: usize,              // output dimension (number of weight rows)
    pub k: usize,              // inner dimension (hidden size, weight columns)
    pub packed: &'a AnyTensor, // [n, ceil(k/4)] 2-bit ternary weights on GPU
    pub x: &'a AnyTensor,     // [m, k] F32 activations on GPU
    pub x_offset: usize,      // element offset into x (for last-token select)
    pub out: &'a AnyTensor,   // [m, n] F32 output on GPU
    pub scale: f32,           // per-tensor dequant scale (TernaryWeight::scale)
    pub beta: f32,            // 0.0 = overwrite, 1.0 = accumulate into out
}
