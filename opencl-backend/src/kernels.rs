use opencl3::context::Context;
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use compute_core::BackendError;

// BF16 é armazenado como ushort (u16) — convertido com as_float
pub const KERNEL_SOURCE: &str = r#"

// Converte BF16 (ushort) → F32
inline float bf16_to_f32(ushort x) {
    return as_float((uint)x << 16);
}

// ─── Kernel 1: embed_lookup ─────────────────────────────────────────────────
// embed_bf16 : [vocab, hidden] ushort (BF16)
// token_ids  : [n_tokens] uint
// out        : [n_tokens, hidden] float
// Grid: (n_tokens, local_size) — um group por token, local_size threads por hidden
__kernel void embed_lookup(
    __global const ushort* embed_bf16,
    __global const uint*   token_ids,
    __global       float*  out,
    int hidden
) {
    int tok = get_global_id(0);
    uint id  = token_ids[tok];
    int base_in  = id  * hidden;
    int base_out = tok * hidden;
    for (int i = get_local_id(1); i < hidden; i += get_local_size(1)) {
        out[base_out + i] = bf16_to_f32(embed_bf16[base_in + i]);
    }
}

// ─── Kernel 2: rms_norm ────────────────────────────────────────────────────
// out     : [n_tokens, hidden] float
// x       : [n_tokens, hidden] float
// w       : [hidden] float (scale)
// Grid: (n_tokens * local_size,) com group size = local_size — um group por token
// Local: (local_size,) — redução com memória local
__kernel void rms_norm(
    __global       float* out,
    __global const float* x,
    __global const float* w,
    int   hidden,
    float eps,
    __local float* scratch
) {
    int tok = get_group_id(0);
    int tid = get_local_id(0);
    int lsz = get_local_size(0);

    // Soma de quadrados com redução paralela
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += lsz) {
        float v = x[tok * hidden + i];
        sum_sq += v * v;
    }
    scratch[tid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(scratch[0] / (float)hidden + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = tid; i < hidden; i += lsz) {
        out[tok * hidden + i] = x[tok * hidden + i] * inv_rms * w[i];
    }
}

// ─── Kernel 3: add_bias_inplace ────────────────────────────────────────────
// x    : [n_tokens, dim] float (modificado in-place)
// bias : [dim] float
// Grid: (n_tokens, dim)
__kernel void add_bias_inplace(
    __global float*       x,
    __global const float* bias,
    int dim
) {
    int tok = get_global_id(0);
    int d   = get_global_id(1);
    if (d < dim) {
        x[tok * dim + d] += bias[d];
    }
}

// ─── Kernel 4: rope_inplace ────────────────────────────────────────────────
// x         : [n_tokens, n_heads, head_dim] float (modificado in-place)
// Grid: (n_tokens * half, n_heads) — um work-item por (token, head, par_de_dims)
// Onde half = head_dim / 2
__kernel void rope_inplace(
    __global float* x,
    int n_heads,
    int head_dim,
    int pos_start,
    float theta
) {
    int tok  = get_global_id(0) / (head_dim / 2);
    int i    = get_global_id(0) % (head_dim / 2);
    int head = get_global_id(1);
    int half_dim = head_dim / 2;

    int base = (tok * n_heads + head) * head_dim;
    float pos = (float)(pos_start + tok);

    float freq  = 1.0f / pow(theta, (float)(2 * i) / (float)head_dim);
    float angle = pos * freq;
    float c = cos(angle);
    float s = sin(angle);

    float x0 = x[base + i];
    float x1 = x[base + i + half_dim];

    x[base + i]            = x0 * c - x1 * s;
    x[base + i + half_dim] = x0 * s + x1 * c;
}

// ─── Kernel 5: silu_mul_inplace ────────────────────────────────────────────
// gate : [n_tokens, inter] float (gate = silu(gate) * up, in-place)
// up   : [n_tokens, inter] float
// Grid: (total_elements,) alinhado a múltiplo de 256
__kernel void silu_mul_inplace(
    __global       float* gate,
    __global const float* up,
    int total
) {
    int i = get_global_id(0);
    if (i < total) {
        float g = gate[i];
        float silu_g = g / (1.0f + exp(-g));
        gate[i] = silu_g * up[i];
    }
}

// ─── Kernel 6: softmax_rows ────────────────────────────────────────────────
// scores : [n_rows, row_len] float (in-place)
// Grid: (n_rows * local_size,) com group size = local_size — um group por row
// Local: (local_size,) com memória local para redução
__kernel void softmax_rows(
    __global float* scores,
    int row_len,
    __local float* scratch,
    int score_offset
) {
    int row = get_group_id(0);
    int tid = get_local_id(0);
    int lsz = get_local_size(0);
    int base = score_offset + row * row_len;

    // Encontra o máximo (estabilidade numérica)
    float max_val = -INFINITY;
    for (int i = tid; i < row_len; i += lsz) {
        max_val = max(max_val, scores[base + i]);
    }
    scratch[tid] = max_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] = max(scratch[tid], scratch[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Soma de exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < row_len; i += lsz) {
        sum_exp += exp(scores[base + i] - row_max);
    }
    scratch[tid] = sum_exp;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = tid; i < row_len; i += lsz) {
        scores[base + i] = exp(scores[base + i] - row_max) * inv_sum;
    }
}

// ─── Kernel 7: copy_to_kv_cache ────────────────────────────────────────────
// cache   : [max_seq, kv_dim] float
// new_kv  : [n_tokens, kv_dim] float
// Grid: (total,) alinhado a múltiplo de 256 onde total = n_tokens * kv_dim
__kernel void copy_to_kv_cache(
    __global       float* cache,
    __global const float* new_kv,
    int n_tokens,
    int kv_dim,
    int insert_pos
) {
    int idx = get_global_id(0);
    int total = n_tokens * kv_dim;
    if (idx < total) {
        int tok_local = idx / kv_dim;
        int dim_idx   = idx % kv_dim;
        int cache_row = insert_pos + tok_local;
        cache[cache_row * kv_dim + dim_idx] = new_kv[idx];
    }
}

// ─── Kernel 8: gemm_bf16_f32 ───────────────────────────────────────────────
// A : [M, K] float (ativações)
// B : [N, K] ushort BF16 (pesos — B é transposta logicamente: B[n, k])
// C : [M, N] float (saída)
// Grid: (M, N)
__kernel void gemm_bf16_f32(
    __global const float*  A,
    __global const ushort* B,
    __global       float*  C,
    int M, int N, int K,
    float alpha, float beta,
    int a_offset
) {
    int row = get_global_id(0);  // M
    int col = get_global_id(1);  // N
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[a_offset + row * K + k] * bf16_to_f32(B[col * K + k]);
    }
    int out_idx = row * N + col;
    C[out_idx] = alpha * sum + beta * C[out_idx];
}

// ─── Kernel 9: sgemm ───────────────────────────────────────────────────────
// GEMM F32 genérico com transpose e leading dims
// Grid: (M, N)
__kernel void sgemm(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    int M, int N, int K,
    int trans_a, int trans_b,
    int lda, int ldb, int ldc,
    float alpha, float beta,
    int a_offset, int b_offset, int c_offset
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a_val = trans_a ? A[a_offset + k * lda + row] : A[a_offset + row * lda + k];
        float b_val = trans_b ? B[b_offset + col * ldb + k] : B[b_offset + k * ldb + col];
        sum += a_val * b_val;
    }
    int out_idx = c_offset + row * ldc + col;
    C[out_idx] = alpha * sum + beta * C[out_idx];
}

"#;

pub struct OclKernels {
    // Mantém o programa vivo enquanto os kernels existirem
    _program: Program,
    pub embed_lookup:  Kernel,
    pub rms_norm:      Kernel,
    pub add_bias:      Kernel,
    pub rope_inplace:  Kernel,
    pub silu_mul:      Kernel,
    pub softmax_rows:  Kernel,
    pub copy_kv_cache: Kernel,
    pub gemm_bf16:     Kernel,
    pub sgemm:         Kernel,
}

impl OclKernels {
    pub fn build(ctx: &Context) -> Result<Self, BackendError> {
        let program = Program::create_and_build_from_source(ctx, KERNEL_SOURCE, "")
            .map_err(|e| BackendError::NotAvailable(
                format!("Falha ao compilar kernels OpenCL: {e}")
            ))?;

        macro_rules! kernel {
            ($name:literal) => {
                Kernel::create(&program, $name)
                    .map_err(|e| BackendError::Kernel(
                        format!("Kernel '{}' não encontrado: {}", $name, e)
                    ))?
            };
        }

        Ok(Self {
            embed_lookup:  kernel!("embed_lookup"),
            rms_norm:      kernel!("rms_norm"),
            add_bias:      kernel!("add_bias_inplace"),
            rope_inplace:  kernel!("rope_inplace"),
            silu_mul:      kernel!("silu_mul_inplace"),
            softmax_rows:  kernel!("softmax_rows"),
            copy_kv_cache: kernel!("copy_to_kv_cache"),
            gemm_bf16:     kernel!("gemm_bf16_f32"),
            sgemm:         kernel!("sgemm"),
            _program: program,
        })
    }
}
