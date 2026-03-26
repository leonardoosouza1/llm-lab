// llm_kernels.hip — HIP GPU kernels for LLM inference on AMD RDNA2 (GFX1032).
// Compiled with: hipcc --offload-arch=gfx1032 -O3 -o llm_kernels.hsaco llm_kernels.hip
//
// All kernels are extern "C" __global__ so Rust can resolve them by name via
// hipModuleGetFunction after loading the .hsaco with hipModuleLoadData.
//
// RDNA2 notes:
//   - Wavefront (warp) size: 64 threads
//   - Max shared memory per block: 64 KB
//   - Block sizes must be powers of 2 (64, 128, 256, 512, 1024)

#include <hip/hip_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helper: BF16 (stored as uint16_t) → float
// BF16 is the upper 16 bits of the IEEE-754 f32 representation.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    // Shift the 16-bit pattern into the upper half of a 32-bit float.
    unsigned int u = (unsigned int)bits << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Kernel 1: embed_lookup
//
// For each token in [n_tokens], copy the corresponding embedding row from the
// BF16 weight matrix into a float output matrix.
//
// Grid : (n_tokens,  1, 1)
// Block: (min(hidden, 256), 1, 1)
//
// Each block handles exactly one token; threads walk the hidden dimension in
// strides of blockDim.x.
// ---------------------------------------------------------------------------
extern "C" __global__ void embed_lookup(
    const unsigned short* __restrict__ embed,   // [vocab, hidden] in BF16 (u16 bits)
    const unsigned int*   __restrict__ token_ids, // [n_tokens]
    float*                __restrict__ out,       // [n_tokens, hidden] in F32
    int hidden
) {
    int tok = blockIdx.x;           // which token this block processes
    unsigned int tid = threadIdx.x; // local thread index

    // Fetch the vocabulary index for this token.
    unsigned int vocab_idx = token_ids[tok];

    // Base pointers into the embedding table and output.
    const unsigned short* embed_row = embed + (size_t)vocab_idx * hidden;
    float*                out_row   = out   + (size_t)tok       * hidden;

    // Strided loop: each thread handles indices tid, tid+blockDim.x, ...
    for (int i = (int)tid; i < hidden; i += (int)blockDim.x) {
        out_row[i] = bf16_to_f32(embed_row[i]);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: rms_norm
//
// RMSNorm: out[i] = (x[i] / rms) * w[i]
//   where rms = sqrt( mean(x^2) + eps )
//
// Grid : (n_tokens, 1, 1)
// Block: (min(hidden, 1024), 1, 1)  — must be power of 2
//
// Uses dynamic shared memory (2 * blockDim.x floats are NOT needed here;
// we only need blockDim.x floats for the parallel reduction).
// ---------------------------------------------------------------------------
__launch_bounds__(1024)
extern "C" __global__ void rms_norm(
    float*       __restrict__ out,  // [n_tokens, hidden]
    const float* __restrict__ x,    // [n_tokens, hidden]
    const float* __restrict__ w,    // [hidden]  — per-element scale (gamma)
    int hidden,
    float eps
) {
    extern __shared__ float sdata[];  // blockDim.x floats

    int tok = blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    const float* x_row   = x   + (size_t)tok * hidden;
    float*       out_row = out + (size_t)tok * hidden;

    // ── Step 1: accumulate x[i]^2 into shared memory (strided) ──────────────
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += bsz) {
        float v = x_row[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // ── Step 2: parallel tree reduction in shared memory ─────────────────────
    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // ── Step 3: compute rms from the total sum ────────────────────────────────
    // sdata[0] now holds sum of x^2 over all hidden elements.
    float rms_inv = rsqrtf(sdata[0] / (float)hidden + eps);
    __syncthreads();  // let all threads see rms_inv (it's in a register from tid==0)
    // Actually rms_inv is computed by all threads from the same sdata[0],
    // so broadcast via shared memory is not needed — every thread does the
    // identical scalar computation after the sync.

    // ── Step 4: write normalized output ──────────────────────────────────────
    for (int i = tid; i < hidden; i += bsz) {
        out_row[i] = x_row[i] * rms_inv * w[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: add_bias_inplace
//
// x[tok, d] += bias[d]  for each token and each dimension.
//
// Grid : (n_tokens, 1, 1)
// Block: (min(dim, 1024), 1, 1)
// ---------------------------------------------------------------------------
__launch_bounds__(1024)
extern "C" __global__ void add_bias_inplace(
    float*       __restrict__ x,    // [n_tokens, dim]
    const float* __restrict__ bias, // [dim]
    int dim
) {
    int tok = blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    float* row = x + (size_t)tok * dim;

    for (int d = tid; d < dim; d += bsz) {
        row[d] += bias[d];
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: rope_inplace
//
// Apply Rotary Position Embeddings in-place using the "halves" convention
// used by Qwen: the rotation pair is (x[i], x[i + head_dim/2]).
//
// Grid : (n_tokens, n_heads, 1)
// Block: (head_dim/2, 1, 1)  — max 128 threads (head_dim ≤ 256)
//
//   blockIdx.x = token index
//   blockIdx.y = head index
//   threadIdx.x = i  in [0, half_dim)
// ---------------------------------------------------------------------------
extern "C" __global__ void rope_inplace(
    float* __restrict__ x,  // [n_tokens, n_heads, head_dim]
    int n_heads,
    int head_dim,
    int pos_start,
    float theta
) {
    int tok      = blockIdx.x;
    int head     = blockIdx.y;
    int i        = (int)threadIdx.x;
    int half_dim = head_dim / 2;

    // Guard: should not happen if grid is sized correctly, but be safe.
    if (i >= half_dim) return;

    // Base index into the flat [n_tokens, n_heads, head_dim] buffer.
    size_t base = ((size_t)tok * n_heads + head) * head_dim;

    // Rotary frequency for this dimension pair.
    float freq  = 1.0f / powf(theta, (2.0f * (float)i) / (float)head_dim);
    float angle = (float)(pos_start + tok) * freq;
    float c     = cosf(angle);
    float s     = sinf(angle);

    float x0 = x[base + i];
    float x1 = x[base + half_dim + i];

    x[base + i]          = x0 * c - x1 * s;
    x[base + half_dim + i] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// Kernel 5: silu_mul_inplace
//
// SwiGLU: gate[i] = silu(gate[i]) * up[i]
//   silu(x) = x / (1 + exp(-x))  = x * sigmoid(x)
//
// Grid : ( ceil(n_tokens * inter / 256), 1, 1 )
// Block: ( 256, 1, 1 )
// ---------------------------------------------------------------------------
extern "C" __global__ void silu_mul_inplace(
    float*       __restrict__ gate,       // [n_tokens, inter] — modified in-place
    const float* __restrict__ up,         // [n_tokens, inter]
    int total_elems                       // n_tokens * inter
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total_elems) return;

    float g = gate[idx];
    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    gate[idx] = silu_g * up[idx];
}

// ---------------------------------------------------------------------------
// Kernel 6: softmax_rows
//
// Numerically-stable softmax over each row of the scores matrix.
// One block per row (one attention head × one query token).
//
// Grid : (n_query_tokens * n_heads, 1, 1)
// Block: ( min(seq_len_padded_to_pow2, 1024), 1, 1 )
//
// Dynamic shared memory layout (two passes, reuse buffer with sync):
//   First  use: blockDim.x floats for max reduction
//   Second use: blockDim.x floats for exp-sum reduction
// We allocate 2 * blockDim.x floats so both passes can overlap if needed,
// but in practice we reuse the same sdata[] with __syncthreads() separating
// the passes to keep the allocation at blockDim.x floats.
// (Caller allocates blockDim.x * sizeof(float) bytes of shared memory.)
// ---------------------------------------------------------------------------
__launch_bounds__(1024)
extern "C" __global__ void softmax_rows(
    float* __restrict__ scores, // [n_rows, seq_len]
    int seq_len
) {
    extern __shared__ float sdata[];  // blockDim.x floats

    int row = blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    float* row_ptr = scores + (size_t)row * seq_len;

    // ── Pass 1: find row max ──────────────────────────────────────────────────
    float local_max = -1e38f;
    for (int j = tid; j < seq_len; j += bsz) {
        float v = row_ptr[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // ── Pass 2: compute exp(x - max) and accumulate sum ───────────────────────
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += bsz) {
        float e = expf(row_ptr[j] - row_max);
        row_ptr[j] = e;          // write back immediately to save registers
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    // ── Pass 3: normalize ─────────────────────────────────────────────────────
    for (int j = tid; j < seq_len; j += bsz) {
        row_ptr[j] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: copy_to_kv_cache
//
// Append n_tokens new K (or V) rows to the pre-allocated KV-cache on GPU.
//   cache[current_len + tok, :] = new_kv[tok, :]  for tok in [0, n_tokens)
//
// Grid : (1, 1, 1)
// Block: (min(n_tokens * kv_dim, 1024), 1, 1)
//
// For very large n_tokens * kv_dim the single block would be too small;
// in practice KV heads are fewer than Q heads (GQA), so kv_dim is moderate.
// The strided loop handles any total size.
// ---------------------------------------------------------------------------
__launch_bounds__(1024)
extern "C" __global__ void copy_to_kv_cache(
    float*       __restrict__ cache,   // [max_seq, kv_dim] — pre-allocated GPU buffer
    const float* __restrict__ new_kv,  // [n_tokens, kv_dim] — new slice to append
    int n_tokens,
    int kv_dim,
    int current_len                    // row index in cache to start writing
) {
    int tid   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = n_tokens * kv_dim;

    for (int idx = tid; idx < total; idx += (int)(blockDim.x * gridDim.x)) {
        int tok = idx / kv_dim;
        int dim = idx % kv_dim;
        cache[(size_t)(current_len + tok) * kv_dim + dim] = new_kv[(size_t)tok * kv_dim + dim];
    }
}
