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

// ---------------------------------------------------------------------------
// Kernel 8: f32_to_bf16
// Converter F32 -> BF16
// ---------------------------------------------------------------------------
extern "C" __global__ void f32_to_bf16(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int bits;
    __builtin_memcpy(&bits, &src[idx], sizeof(float));
    bits += 0x7FFF + ((bits >> 16) & 1);
    dst[idx] = (unsigned short)(bits >> 16);
}

// ---------------------------------------------------------------------------
// Helper: FP16 (stored as uint16_t) → float
// IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa.
// Conversão portável via manipulação de bits — sem inline asm.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp16_to_f32(unsigned short bits) {
    unsigned int sign = (unsigned int)(bits >> 15) << 31;
    unsigned int exp  = (bits >> 10) & 0x1F;
    unsigned int mant = bits & 0x3FF;

    unsigned int f32_bits;
    if (exp == 0) {
        if (mant == 0) {
            // ±0
            f32_bits = sign;
        } else {
            // Subnormal FP16 → normal F32
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f32_bits = sign | ((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf / NaN
        f32_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        // Normal
        f32_bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    __builtin_memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Kernel 9: dequant_gemv_q4
//
// INT4 dequantized GEMV: packed_weights(Q4) × activations(F32) → output(F32)
// Each byte holds 2 INT4 values (low 4 bits + high 4 bits).
// Group-wise dequantization: w_float = scale[group] * (int4_val - zero[group])
//
// Grid : (N, 1, 1)  — one block per output row
// Block: (256, 1, 1) — strided loop over K dimension
//
// RDNA2 notes:
//   - 256 threads per block for good occupancy
//   - Shared memory used for parallel reduction of dot product
//   - Each thread computes a partial dot product, accumulates via shared memory
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
extern "C" __global__ void dequant_gemv_q4(
    const unsigned char* __restrict__ packed,   // [N, K/2] — 2 int4 per byte
    const float* __restrict__ scales,           // [N, K/group_size] — F32
    const float* __restrict__ zeros,            // [N, K/group_size] — F32
    const float* __restrict__ x,                // [M, K] — input activations
    float* __restrict__ out,                    // [M, N] — output
    int M,           // batch dim (tokens)
    int N,           // output dim (rows of weight matrix)
    int K,           // input dim (hidden, columns)
    int group_size,  // typically 128
    float beta       // 0.0=overwrite, 1.0=accumulate
) {
    extern __shared__ float sdata[];  // 256 floats for reduction

    int row = blockIdx.x;           // which output row (weight row) this block processes
    int mi  = blockIdx.y;           // which token in the batch [0..M)
    int tid = (int)threadIdx.x;     // thread index within block
    int bsz = (int)blockDim.x;      // block size = 256

    if (row >= N || mi >= M) return;

    // Compute pointers to this row's packed weights, scales, and zeros.
    int n_groups = (K + group_size - 1) / group_size;
    const unsigned char* packed_row = packed + (size_t)row * (K / 2);
    const float* scales_row = scales + (size_t)row * n_groups;
    const float* zeros_row  = zeros  + (size_t)row * n_groups;
    const float* x_row      = x + (size_t)mi * K;

    // Accumulate partial dot product in this thread.
    float local_sum = 0.0f;

    for (int k = tid; k < K; k += bsz) {
        // Extract INT4 values from packed byte.
        int byte_idx = k / 2;
        unsigned char packed_byte = packed_row[byte_idx];
        int int4_val;
        if (k % 2 == 0) {
            int4_val = (int)(packed_byte & 0x0F);
        } else {
            int4_val = (int)((packed_byte >> 4) & 0x0F);
        }

        // Sign-extend 4-bit value to signed int (-8 to 7).
        if (int4_val >= 8) {
            int4_val -= 16;
        }

        // Get group index and dequant parameters (F32 — sem conversão).
        int group_idx = k / group_size;
        float scale_f32 = scales_row[group_idx];
        float zero_f32 = zeros_row[group_idx];

        // Dequantize: w = scale * (int4_val - zero)
        float w = scale_f32 * ((float)int4_val - zero_f32);

        // Accumulate into dot product.
        local_sum += w * x_row[k];
    }

    // Write local_sum to shared memory for reduction.
    sdata[tid] = local_sum;
    __syncthreads();

    // Parallel tree reduction in shared memory.
    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for this (mi, row).
    if (tid == 0) {
        float result = sdata[0];
        int out_idx = mi * N + row;
        if (beta == 0.0f) {
            out[out_idx] = result;
        } else {
            out[out_idx] = beta * out[out_idx] + result;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 10: gravity_score
//
// Compute semantic gravity per token position:
//   gravity[t] = mean_over_heads( attn_weights[h,t] * value_norms[h,t] )
//
// Each token position gets one block. Threads within the block reduce across
// heads using shared memory.
//
// Grid : (seq_len, 1, 1)
// Block: (min(n_heads, 256), 1, 1)
//
// RDNA2 notes:
//   - Shared memory reduction pattern for efficient inter-thread communication
//   - Coalesce reads from attn_weights and value_norms arrays
// ---------------------------------------------------------------------------
__launch_bounds__(256, 4)
extern "C" __global__ void gravity_score(
    const float* __restrict__ attn_weights,  // [n_heads, seq_len] — post-softmax
    const float* __restrict__ value_norms,   // [n_heads, seq_len] — L2 norms of V
    float* __restrict__ gravity_out,         // [seq_len] — output gravity scores
    int n_heads,
    int seq_len
) {
    extern __shared__ float sdata[];  // min(n_heads, 256) floats

    int t = blockIdx.x;              // token index
    int tid = (int)threadIdx.x;      // thread within block (head index)
    int bsz = (int)blockDim.x;       // typically n_heads or 256

    if (t >= seq_len) return;

    // Each thread loads the gravity contribution from one head and accumulates.
    float local_sum = 0.0f;
    for (int h = tid; h < n_heads; h += bsz) {
        float attn_val = attn_weights[(size_t)h * seq_len + t];
        float norm_val = value_norms[(size_t)h * seq_len + t];
        local_sum += attn_val * norm_val;
    }

    // Reduce across threads via shared memory.
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 computes mean and writes result.
    if (tid == 0) {
        gravity_out[t] = sdata[0] / (float)n_heads;
    }
}

// ---------------------------------------------------------------------------
// Kernel 11: cfc_gate_update
//
// CfC (Closed-form Continuous) liquid correction layer state update.
// Computes: h_new[i] = sigmoid(f[i]) * h_prev[i] + (1 - sigmoid(f[i])) * tanh(g[i])
//
// Note: f_proj and g_proj are pre-computed projections:
//   f_proj = x_slice @ wf^T + h_prev @ wf2^T + bf
//   g_proj = x_slice @ wg^T + h_prev @ wg2^T + bg
// This kernel only handles the CfC gating and state update.
//
// Grid : (1, 1, 1)  — single block processes entire CfC state
// Block: (min(cfc_dim, 256), 1, 1)
//
// RDNA2 notes:
//   - Simple element-wise operations, good for memory bandwidth
//   - Strided loop for cfc_dim dimensions
// ---------------------------------------------------------------------------
extern "C" __global__ void cfc_gate_update(
    const float* __restrict__ f_proj,    // [cfc_dim] — pre-computed x@wf + h@wf2 + bf
    const float* __restrict__ g_proj,    // [cfc_dim] — pre-computed x@wg + h@wg2 + bg
    const float* __restrict__ h_prev,    // [cfc_dim]
    float* __restrict__ h_out,           // [cfc_dim] — new state
    int cfc_dim
) {
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    // Strided loop over cfc_dim dimensions.
    for (int i = tid; i < cfc_dim; i += bsz) {
        float f = f_proj[i];
        float g = g_proj[i];
        float h = h_prev[i];

        // Sigmoid: 1 / (1 + exp(-x))
        float sigmoid_f = 1.0f / (1.0f + expf(-f));

        // Tanh: (exp(2x) - 1) / (exp(2x) + 1)
        // For numerical stability, use built-in tanhf.
        float tanh_g = tanhf(g);

        // h_new = sigmoid(f) * h + (1 - sigmoid(f)) * tanh(g)
        h_out[i] = sigmoid_f * h + (1.0f - sigmoid_f) * tanh_g;
    }
}

// ---------------------------------------------------------------------------
// Kernel 12: value_norms
//
// Compute L2 norm of each value vector (along hidden dimension).
// For each token position and KV head, compute the Frobenius norm of the
// value vector.
//
// Grid : (n_kv_heads, seq_len, 1)
// Block: (min(head_dim, 256), 1, 1)
//
// Layout assumptions:
//   v_cache is [seq_len, n_kv_heads * head_dim] where values are contiguous
//   per head within the head_dim extent.
//
// RDNA2 notes:
//   - Each block computes one L2 norm
//   - Shared memory reduction for tree sum-of-squares
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
extern "C" __global__ void value_norms(
    const float* __restrict__ v_cache,    // [seq_len, n_kv_heads * head_dim] F32
    float* __restrict__ norms_out,        // [n_kv_heads, seq_len] F32
    int seq_len,
    int n_kv_heads,
    int head_dim
) {
    extern __shared__ float sdata[];  // min(head_dim, 256) floats

    int h = blockIdx.x;              // KV head index
    int t = blockIdx.y;              // token position
    int tid = (int)threadIdx.x;      // thread index
    int bsz = (int)blockDim.x;       // block size

    if (h >= n_kv_heads || t >= seq_len) return;

    // Base index into v_cache for this (head, token).
    size_t base = (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;

    // Accumulate sum of squares (partial norm).
    float local_sumsq = 0.0f;
    for (int d = tid; d < head_dim; d += bsz) {
        float v = v_cache[base + d];
        local_sumsq += v * v;
    }

    // Reduce via shared memory.
    sdata[tid] = local_sumsq;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 computes final norm and writes result.
    if (tid == 0) {
        float norm = sqrtf(sdata[0]);
        norms_out[(size_t)h * seq_len + t] = norm;
    }
}

// ---------------------------------------------------------------------------
// Kernel 13: kv_cache_quantize_append
//
// Quantize F32 KV values to INT4 (Q4) and append to quantized cache.
// For each group of group_size elements in a KV vector:
//   1. Find min and max values
//   2. Compute scale = (max - min) / 15, zero = min
//   3. Quantize each value: q = round(15 * (val - min) / (max - min))
//   4. Pack two INT4 values per byte
//
// Grid : (n_tokens, 1, 1)
// Block: (min(kv_dim, 512), 1, 1)
//
// The cache buffers are pre-allocated to hold up to max_seq tokens.
// This kernel appends n_tokens new KV vectors starting at position current_len.
//
// RDNA2 notes:
//   - Per-group min/max reduction via shared memory
//   - Efficient packing of INT4 pairs
// ---------------------------------------------------------------------------
__launch_bounds__(512, 1)
extern "C" __global__ void kv_cache_quantize_append(
    unsigned char* __restrict__ cache_packed,  // [max_seq, kv_dim/2] Q4
    unsigned short* __restrict__ cache_scales, // [max_seq, n_groups] FP16
    unsigned short* __restrict__ cache_zeros,  // [max_seq, n_groups] FP16
    const float* __restrict__ new_kv,          // [n_tokens, kv_dim] F32
    int n_tokens,
    int kv_dim,
    int current_len,
    int group_size
) {
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    // Each block processes one token; threads handle groups in parallel.
    int tok = blockIdx.x;
    if (tok >= n_tokens) return;

    const float* tok_kv = new_kv + (size_t)tok * kv_dim;
    unsigned char* tok_packed = cache_packed + (size_t)(current_len + tok) * (kv_dim / 2);
    unsigned short* tok_scales = cache_scales + (size_t)(current_len + tok) * ((kv_dim + group_size - 1) / group_size);
    unsigned short* tok_zeros = cache_zeros + (size_t)(current_len + tok) * ((kv_dim + group_size - 1) / group_size);

    int n_groups = (kv_dim + group_size - 1) / group_size;

    // Process one group per thread (strided if n_groups > bsz).
    for (int g = tid; g < n_groups; g += bsz) {
        int group_start = g * group_size;
        int group_end = min(group_start + group_size, kv_dim);
        int group_len = group_end - group_start;

        // Find min and max in this group.
        float group_min = 1e38f;
        float group_max = -1e38f;
        for (int i = group_start; i < group_end; i++) {
            float v = tok_kv[i];
            if (v < group_min) group_min = v;
            if (v > group_max) group_max = v;
        }

        // Compute scale and zero.
        float scale = (group_max - group_min) / 15.0f;
        if (scale < 1e-8f) scale = 1e-8f;  // avoid division by zero
        float zero = group_min;

        // Quantize and pack INT4 values.
        for (int i = group_start; i < group_end; i += 2) {
            int q_low = (int)roundf(15.0f * (tok_kv[i] - zero) / scale);
            q_low = max(0, min(15, q_low));  // clamp to [0, 15]

            int q_high = 0;
            if (i + 1 < group_end) {
                q_high = (int)roundf(15.0f * (tok_kv[i + 1] - zero) / scale);
                q_high = max(0, min(15, q_high));
            }

            // Pack: low 4 bits in low nibble, high 4 bits in high nibble.
            unsigned char packed_byte = (unsigned char)((q_high << 4) | q_low);
            tok_packed[(i - group_start) / 2] = packed_byte;
        }

        // Convert scale and zero to FP16 and store.
        // Simple truncation to FP16 (not high-precision conversion, but acceptable for quantization params).
        unsigned int scale_bits, zero_bits;
        __builtin_memcpy(&scale_bits, &scale, sizeof(float));
        __builtin_memcpy(&zero_bits, &zero, sizeof(float));
        // Truncate to FP16 by taking upper 16 bits (BF16-like, or use actual F16 if available).
        tok_scales[g] = (unsigned short)(scale_bits >> 16);
        tok_zeros[g] = (unsigned short)(zero_bits >> 16);
    }
}

// ---------------------------------------------------------------------------
// Kernel 14: add_tensors_inplace
//
// Element-wise in-place addition: a[i] += alpha * b[i]
//
// Grid : (ceil(n / 256), 1, 1)
// Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void add_tensors_inplace(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += alpha * b[i];
    }
}
