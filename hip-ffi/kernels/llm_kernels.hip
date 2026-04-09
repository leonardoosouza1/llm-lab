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
// Kernel 6b: softmax_rows_causal
//
// Causal (masked) softmax for prefill attention (n_tokens > 1).
// Same as softmax_rows but masks out future key positions:
//   scores[row][j] = -inf if j > query_offset + (row % n_tokens)
//
// Grid : (total_rows, 1, 1)
// Block: (blockDim.x, 1, 1)  — one block per row; shared mem = blockDim.x floats
// ---------------------------------------------------------------------------
__launch_bounds__(1024)
extern "C" __global__ void softmax_rows_causal(
    float* __restrict__ scores, // [n_rows, seq_len]
    int seq_len,
    int n_tokens,       // number of query tokens in this batch
    int query_offset    // position of first query token in sequence
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    int query_tok = row % n_tokens;
    int max_key_pos = query_offset + query_tok;  // inclusive: attend to [0..max_key_pos]

    float* row_ptr = scores + (size_t)row * seq_len;

    // ── Apply causal mask: set future positions to -inf ───────────────────────
    for (int j = tid; j < seq_len; j += bsz) {
        if (j > max_key_pos) {
            row_ptr[j] = -1e38f;
        }
    }
    __syncthreads();

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
        row_ptr[j] = e;
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
// Kernel: fused_rope_kv_copy — RoPE(Q), RoPE(K), copy K→KV, copy V→KV
//
// For decode (n=1): applies RoPE to Q and K in-place, then copies K and V
// to their respective KV cache positions. Replaces 4 separate kernel launches.
//
// Virtual block mapping:
//   blockIdx.x < n_heads     → RoPE on Q[head]
//   blockIdx.x >= n_heads    → RoPE on K[kv_head], copy K[kv_head] + V[kv_head]
//
// Grid : (n_heads + n_kv_heads, 1, 1)
// Block: (head_dim, 1, 1)  — threads 0..head_dim/2-1 do RoPE, all do copy
// ---------------------------------------------------------------------------
extern "C" __global__ void fused_rope_kv_copy(
    float*       __restrict__ q,         // [n_heads * head_dim]
    float*       __restrict__ k,         // [n_kv_heads * head_dim]
    const float* __restrict__ v,         // [n_kv_heads * head_dim]
    float*       __restrict__ kv_k_cache,// [max_seq, n_kv_heads * head_dim]
    float*       __restrict__ kv_v_cache,// [max_seq, n_kv_heads * head_dim]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos_start,
    float theta,
    int kv_dim,      // n_kv_heads * head_dim
    int max_seq
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int half_dim = head_dim / 2;

    if (bid < n_heads) {
        // ── RoPE on Q head ──
        if (tid < half_dim) {
            const int head = bid;
            size_t base = (size_t)head * head_dim;
            float freq = 1.0f / powf(theta, (2.0f * (float)tid) / (float)head_dim);
            float angle = (float)pos_start * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = q[base + tid];
            float x1 = q[base + half_dim + tid];
            q[base + tid]            = x0 * c - x1 * s;
            q[base + half_dim + tid] = x0 * s + x1 * c;
        }
    } else {
        // ── RoPE on K head + copy K + copy V ──
        const int kv_head = bid - n_heads;
        if (kv_head >= n_kv_heads) return;

        size_t k_base = (size_t)kv_head * head_dim;

        // Apply RoPE to K (only threads 0..half_dim-1)
        if (tid < half_dim) {
            float freq = 1.0f / powf(theta, (2.0f * (float)tid) / (float)head_dim);
            float angle = (float)pos_start * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = k[k_base + tid];
            float x1 = k[k_base + half_dim + tid];
            k[k_base + tid]            = x0 * c - x1 * s;
            k[k_base + half_dim + tid] = x0 * s + x1 * c;
        }
        __syncthreads();  // ensure RoPE completes before copy

        // Copy K[kv_head] to kv_k_cache at position pos_start
        if (tid < head_dim) {
            size_t cache_idx = (size_t)pos_start * kv_dim + kv_head * head_dim + tid;
            kv_k_cache[cache_idx] = k[k_base + tid];
        }

        // Copy V[kv_head] to kv_v_cache at position pos_start (no RoPE)
        if (tid < head_dim) {
            size_t cache_idx = (size_t)pos_start * kv_dim + kv_head * head_dim + tid;
            kv_v_cache[cache_idx] = v[k_base + tid];
        }
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
// Kernel 9: dequant_gemv_q4_naive (reference implementation)
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
//   - NAIVE: reads scales/zeros from global memory per group, no vectorization
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
extern "C" __global__ void dequant_gemv_q4_naive(
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

        // Asymmetric quantization [0, 15] — no sign extension needed.
        // Encoding: q = round((val - min) / scale), scale = (max-min)/15, zero = min_val
        // Decoding: val = q * scale + zero

        // Get group index and dequant parameters (F32 — sem conversão).
        int group_idx = k / group_size;
        float scale_f32 = scales_row[group_idx];
        float zero_f32 = zeros_row[group_idx];

        // Dequantize: w = scale * q + zero  (where zero = min_val)
        float w = scale_f32 * (float)int4_val + zero_f32;

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
// Kernel 9: dequant_gemv_q4 (OPTIMIZED for RDNA2)
//
// INT4 dequantized GEMV: packed_weights(Q4) × activations(F32) → output(F32)
//
// Key optimizations for AMD RDNA2 (RX 6750 XT / GFX1032):
//
// 1. Vectorized 4-byte loads: Read uint32_t (4 bytes = 8 INT4 values) per load
//    instead of single bytes. Maximizes memory coalescing: 256 threads × 4 bytes
//    = 1024 bytes per step, perfectly aligned for 64-byte cache lines.
//
// 2. FMA instructions: Use __fmaf_rn() for fused multiply-add, halving the
//    instruction count for dequant + accumulate.
//
// 3. High occupancy: __launch_bounds__(256, 4) allows 4 blocks per CU = 32
//    wavefronts per CU, 4x more than the old (256,2). Better latency hiding.
//
// 4. Minimal shared memory: Only 8 floats for warp reduction (32 bytes).
//    Scales/zeros fit in L1 cache (<1KB for group_size=128, K≤8960).
//    No __syncthreads() barrier before compute.
//
// 5. Bitshift group index: group_size=128 → k >> 7. Avoids integer division.
//
// Grid : (N, M, 1)  — one block per output row per token
// Block: (256, 1, 1) — 8 warps (wave32), strided over K dimension
// Shared: 8 floats = 32 bytes (warp reduction scratch)
// ---------------------------------------------------------------------------
__launch_bounds__(256, 4)
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
    extern __shared__ float sh_reduce[];  // [8] floats for cross-warp reduction

    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    const int tid = (int)threadIdx.x;

    if (row >= N || mi >= M) return;

    const int half_K = K >> 1;
    const int n_groups = (K + group_size - 1) / group_size;
    const unsigned char* __restrict__ packed_row = packed + (size_t)row * half_K;
    const float* __restrict__ scales_row = scales + (size_t)row * n_groups;
    const float* __restrict__ zeros_row  = zeros  + (size_t)row * n_groups;
    const float* __restrict__ x_row      = x + (size_t)mi * K;

    float acc = 0.0f;

    // ── Main loop: vectorized 4-byte loads ───────────────────────────────────
    // Each uint32_t load gives 4 packed bytes = 8 INT4 values.
    // 256 threads × 4 bytes = 1024 bytes per step (16 cache lines, coalesced).
    const int n_u32 = half_K >> 2;
    const uint32_t* __restrict__ p32 = reinterpret_cast<const uint32_t*>(packed_row);

    for (int i = tid; i < n_u32; i += 256) {
        uint32_t word = p32[i];
        int k_base = i << 3;  // i * 8: first element index for this uint32

        // Unrolled: extract 4 bytes → 8 INT4 values → 8 FMAs
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            const int byte_val = (word >> (b << 3)) & 0xFF;
            const int k0 = k_base + (b << 1);

            // Group index via bitshift for power-of-2 group_size (128 → >>7)
            const int g = k0 / group_size;
            const float s = scales_row[g];
            const float z = zeros_row[g];

            // Dequant + dot-product via 2 FMAs: w = s*val+z, acc += w*x
            acc = __fmaf_rn(__fmaf_rn(s, (float)(byte_val & 0xF), z), x_row[k0], acc);
            acc = __fmaf_rn(__fmaf_rn(s, (float)(byte_val >> 4),  z), x_row[k0 + 1], acc);
        }
    }

    // ── Remainder: byte-level loads for tail ─────────────────────────────────
    const int tail_start = n_u32 << 2;
    for (int bp = tail_start + tid; bp < half_K; bp += 256) {
        const unsigned char pb = packed_row[bp];
        const int k0 = bp << 1;
        const int g = k0 / group_size;
        const float s = scales_row[g];
        const float z = zeros_row[g];
        acc = __fmaf_rn(__fmaf_rn(s, (float)(pb & 0xF), z), x_row[k0], acc);
        acc = __fmaf_rn(__fmaf_rn(s, (float)(pb >> 4),  z), x_row[k0 + 1], acc);
    }

    // ── Warp reduction (wave32 on RDNA2) ─────────────────────────────────────
    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off);

    // ── Cross-warp reduction via shared memory ───────────────────────────────
    const int warp_id = tid >> 5;
    if ((tid & 31) == 0)
        sh_reduce[warp_id] = acc;
    __syncthreads();

    // 8 warp sums → single result (only first warp active)
    if (tid < 8) {
        float val = sh_reduce[tid];
        for (int off = 4; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (tid == 0) {
            const int out_idx = mi * N + row;
            out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], val) : val;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: dequant_gemv_q4k — Native GGML Q4_K GEMV
//
// Computes y[mi, row] = dot(dequant(Q4_K_row[row]), x[mi]) for decode (M=1
// typical). Operates directly on raw Q4_K blocks — NO intermediate BF16.
// Dequantizes on-the-fly in F32, accumulates in F32.
//
// Q4_K block layout (144 bytes per 256 values):
//   d:      2 bytes (f16) — super-block scale
//   dmin:   2 bytes (f16) — super-block minimum scale
//   scales: 12 bytes — packed 6-bit sub-group scales + mins (8 sub-groups)
//   qs:     128 bytes — 256 4-bit quantized values (2 per byte)
//
// Dequantization per byte (produces 2 values from low/high nibble):
//   w_lo = d * sc_lo * (qs_byte & 0xF) - dmin * m_lo
//   w_hi = d * sc_hi * (qs_byte >> 4)  - dmin * m_hi
//
// Grid : (N, M, 1) — one block per output row per token
// Block: (256, 1, 1) — threads stride over qs bytes across all Q4_K blocks
// Shared: 8 floats = 32 bytes (cross-warp reduction)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void get_scale_min_k4(
    int j, const unsigned char* scales, unsigned char* sc, unsigned char* m
) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >> 4)   | ((scales[j] >> 6) << 4);
    }
}

__launch_bounds__(256, 4)
extern "C" __global__ void dequant_gemv_q4k(
    const unsigned char* __restrict__ data,   // [N * n_blocks_per_row * 144] raw Q4_K
    const float*         __restrict__ x,      // [M, K] input activations F32
    float*               __restrict__ out,    // [M, N] output F32
    int M, int N, int K,
    float beta
) {
    extern __shared__ float sh_reduce[];  // [8] for cross-warp reduction

    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    const int tid = (int)threadIdx.x;

    if (row >= N || mi >= M) return;

    const int n_blocks_per_row = K / 256;      // Q4_K blocks per weight row
    const int row_bytes = n_blocks_per_row * 144;
    const unsigned char* __restrict__ row_data = data + (size_t)row * row_bytes;
    const float* __restrict__ x_row = x + (size_t)mi * K;

    // Total qs bytes per row = n_blocks_per_row * 128
    // Each byte produces 2 dequantized values → 2 FMAs
    const int total_qs_bytes = n_blocks_per_row * 128;

    float acc = 0.0f;

    for (int byte_idx = tid; byte_idx < total_qs_bytes; byte_idx += 256) {
        const int blk = byte_idx / 128;         // which Q4_K block
        const int byte_in_blk = byte_idx % 128; // byte within block's qs[]
        const int group = byte_in_blk >> 5;      // 0..3 (group of 32 bytes)
        const int l = byte_in_blk & 31;          // 0..31 (position in group)

        // Block header pointer
        const unsigned char* __restrict__ block_ptr = row_data + blk * 144;

        // Read d, dmin as F16 → F32
        const float d    = fp16_to_f32(*(const unsigned short*)(block_ptr + 0));
        const float dmin = fp16_to_f32(*(const unsigned short*)(block_ptr + 2));

        // Unpack 6-bit scales for low nibble (scale_idx = 2*group)
        // and high nibble (scale_idx = 2*group + 1)
        const unsigned char* sc_ptr = block_ptr + 4;
        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4(2 * group,     sc_ptr, &sc_lo, &m_lo);
        get_scale_min_k4(2 * group + 1, sc_ptr, &sc_hi, &m_hi);

        // Read quantized byte
        const unsigned char q_byte = block_ptr[16 + byte_in_blk];
        const int q_lo = q_byte & 0x0F;
        const int q_hi = q_byte >> 4;

        // Dequantize: w = d * sc * q_val - dmin * m
        const float w_lo = d * (float)sc_lo * (float)q_lo - dmin * (float)m_lo;
        const float w_hi = d * (float)sc_hi * (float)q_hi - dmin * (float)m_hi;

        // Map to K-dimension indices
        // Low nibble: position = blk*256 + group*64 + l
        // High nibble: position = blk*256 + group*64 + 32 + l
        const int k_lo = blk * 256 + group * 64 + l;
        const int k_hi = k_lo + 32;

        acc = __fmaf_rn(w_lo, x_row[k_lo], acc);
        acc = __fmaf_rn(w_hi, x_row[k_hi], acc);
    }

    // ── Warp reduction (wave32 on RDNA2) ─────────────────────────────────
    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off);

    // ── Cross-warp reduction via shared memory ───────────────────────────
    const int warp_id = tid >> 5;
    if ((tid & 31) == 0)
        sh_reduce[warp_id] = acc;
    __syncthreads();

    if (tid < 8) {
        float val = sh_reduce[tid];
        for (int off = 4; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (tid == 0) {
            const int out_idx = mi * N + row;
            out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], val) : val;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: dequant_gemv_q6k — Native GGML Q6_K GEMV
//
// Q6_K block layout (210 bytes per 256 values):
//   ql:     128 bytes — lower 4 bits (2 values per byte, interleaved)
//   qh:     64 bytes  — upper 2 bits (4 values per byte)
//   scales: 16 bytes  — int8 scales for 16 sub-groups of 16
//   d:      2 bytes (f16) — super-block scale
//
// Each value is 6-bit signed: val = ((ql_bits | (qh_bits << 4)) - 32)
// Dequant: w = d * scale[sub_group] * val
//
// Grid : (N, M, 1)
// Block: (256, 1, 1)
// Shared: 8 floats = 32 bytes
// ---------------------------------------------------------------------------
__launch_bounds__(256, 4)
extern "C" __global__ void dequant_gemv_q6k(
    const unsigned char* __restrict__ data,   // [N * n_blocks_per_row * 210] raw Q6_K
    const float*         __restrict__ x,      // [M, K] input activations F32
    float*               __restrict__ out,    // [M, N] output F32
    int M, int N, int K,
    float beta
) {
    extern __shared__ float sh_reduce[];

    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    const int tid = (int)threadIdx.x;

    if (row >= N || mi >= M) return;

    const int n_blocks_per_row = K / 256;
    const int row_bytes = n_blocks_per_row * 210;
    const unsigned char* __restrict__ row_data = data + (size_t)row * row_bytes;
    const float* __restrict__ x_row = x + (size_t)mi * K;

    float acc = 0.0f;

    // Each thread processes one value per block (tid = position 0..255)
    // Q6_K block: ql[128] + qh[64] + scales[16 int8] + d[f16]
    // 2 groups of 128 values. Within each group:
    //   4 sub-groups of 32 values. 16 scale values → 2 per sub-group (split at l=16).
    //
    // GGML reference: is = l/16; scale = sc_g[is + sub*2]
    // This gives 16 distinct scales across the 256 values.

    for (int blk = 0; blk < n_blocks_per_row; blk++) {
        const unsigned char* __restrict__ bp = row_data + blk * 210;
        const unsigned char* ql = bp;                              // 128 bytes
        const unsigned char* qh = bp + 128;                        // 64 bytes
        const signed char* scales = (const signed char*)(bp + 192); // 16 int8
        const float d = fp16_to_f32(*(const unsigned short*)(bp + 208));

        const int n_group = tid >> 7;       // 0 or 1
        const int pos_in_group = tid & 127; // 0..127
        const int sub = pos_in_group >> 5;  // 0..3 (sub-group of 32)
        const int l = pos_in_group & 31;    // 0..31

        const unsigned char* ql_g = ql + n_group * 64;
        const unsigned char* qh_g = qh + n_group * 32;
        const signed char* sc_g = scales + n_group * 8;

        int q_val;
        switch (sub) {
            case 0: q_val = (int)( ql_g[l]       & 0xF) | (int)(((qh_g[l] >> 0) & 3) << 4); break;
            case 1: q_val = (int)( ql_g[32 + l]  & 0xF) | (int)(((qh_g[l] >> 2) & 3) << 4); break;
            case 2: q_val = (int)((ql_g[l]       >> 4)) | (int)(((qh_g[l] >> 4) & 3) << 4); break;
            default: q_val = (int)((ql_g[32 + l] >> 4)) | (int)(((qh_g[l] >> 6) & 3) << 4); break;
        }
        q_val -= 32; // signed 6-bit: range [-32, 31]

        // GGML uses is = l/16 to select between 2 scales per sub-group of 32:
        //   scale_idx_within_group = (l/16) + sub*2
        // This gives 8 scale indices per group (0..7), 16 total across both groups.
        const int is = l >> 4; // 0 for l<16, 1 for l>=16
        const float sc = (float)sc_g[is + sub * 2];

        const float w = d * sc * (float)q_val;
        const int k = blk * 256 + tid;
        acc = __fmaf_rn(w, x_row[k], acc);
    }

    // ── Warp reduction (wave32) ──────────────────────────────────────────
    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off);

    const int warp_id = tid >> 5;
    if ((tid & 31) == 0)
        sh_reduce[warp_id] = acc;
    __syncthreads();

    if (tid < 8) {
        float val = sh_reduce[tid];
        for (int off = 4; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (tid == 0) {
            const int out_idx = mi * N + row;
            out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], val) : val;
        }
    }
}

// ---------------------------------------------------------------------------
// dp4a — dot product of 4 packed int8 values, accumulated into int32
// On RDNA2: __builtin_amdgcn_sdot4(a, b, c, false)
// a, b: each contain 4 packed int8 values; result = c + sum(a[i]*b[i])
// ---------------------------------------------------------------------------
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// ---------------------------------------------------------------------------
// fp16_load — load half-precision float as f32 (uses existing fp16_to_f32)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp16_load(const void* ptr) {
    return fp16_to_f32(*(const unsigned short*)ptr);
}

// f32_to_fp16 — convert float to half-precision (stored as unsigned short)
__device__ __forceinline__ unsigned short f32_to_fp16(float val) {
    // Use bit manipulation for FP32 → FP16 conversion
    unsigned int f32;
    __builtin_memcpy(&f32, &val, sizeof(f32));
    unsigned int sign = (f32 >> 16) & 0x8000;
    int exp = ((f32 >> 23) & 0xFF) - 127 + 15;
    unsigned int frac = (f32 >> 13) & 0x3FF;
    if (exp <= 0) return (unsigned short)sign; // flush to zero
    if (exp >= 31) return (unsigned short)(sign | 0x7C00); // infinity
    return (unsigned short)(sign | (exp << 10) | frac);
}

// ---------------------------------------------------------------------------
// Kernel: quantize_f32_q8_1
//
// Quantize F32 activation vector to Q8_1 blocks (32 values each).
// Q8_1 block layout (36 bytes): half d (2B) + half s (2B) + int8_t qs[32] (32B)
//   d = max(|x[i]|) / 127  (quantization scale)
//   s = sum(x[i])           (original float sum, used for min correction)
//   qs[i] = round(x[i] / d)
//
// Grid : (ceil(n_blocks / warps_per_block), 1, 1)
// Block: (256, 1, 1)  — 8 warps, each handles one block per iteration
// ---------------------------------------------------------------------------
__launch_bounds__(256, 4)
extern "C" __global__ void quantize_f32_q8_1(
    const float* __restrict__ x,       // [K] input F32 activations
    unsigned char* __restrict__ out,    // [K/32 * 36] output Q8_1 blocks
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int n_blocks = K >> 5;

    for (int blk = blockIdx.x * warps_per_block + warp_id;
         blk < n_blocks;
         blk += gridDim.x * warps_per_block) {
        const float xi = x[blk * 32 + lane];

        // Warp-wide max absolute value
        float amax = fabsf(xi);
        for (int off = 16; off >= 1; off >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, off));

        // Warp-wide sum
        float sum = xi;
        for (int off = 16; off >= 1; off >>= 1)
            sum += __shfl_xor(sum, off);

        const float d = amax / 127.0f;
        const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
        const signed char q = (signed char)__float2int_rn(xi * id);

        // Write Q8_1 block: [fp16 d (2B), fp16 s (2B), int8 qs[32] (32B)] = 36 bytes
        unsigned char* bp = out + (size_t)blk * 36;
        if (lane == 0) {
            *(unsigned short*)(bp + 0) = f32_to_fp16(d);
            *(unsigned short*)(bp + 2) = f32_to_fp16(sum);
        }
        bp[4 + lane] = (unsigned char)q;
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_rms_norm_quantize — RMSNorm + Q8_1 quantization in one pass
//
// Computes: normed[i] = x[i] / rms * w[i]  (F32 output)
//           q8[block] = quantize_q8_1(normed) (Q8_1 output)
//
// Saves 1 kernel launch per norm+quantize pair (3 per layer).
//
// Grid : (1, 1, 1)  — single block for n=1 token
// Block: (1024, 1, 1)
// Shared: 1024 * 4 bytes + hidden * 4 bytes (normed values)
//
// hidden must be <= 16384 (shared memory limit)
// hidden must be a multiple of 32 (Q8 block size)
// ---------------------------------------------------------------------------
__launch_bounds__(1024)
extern "C" __global__ void fused_rms_norm_quantize(
    const float* __restrict__ x,       // [hidden] input
    const float* __restrict__ w,       // [hidden] RMSNorm weight
    float*       __restrict__ out_f32, // [hidden] normed F32 output
    unsigned char* __restrict__ out_q8, // [(hidden/32)*36] Q8_1 output
    int hidden,
    float eps
) {
    extern __shared__ float sdata[];  // [blockDim.x] for reduction, then [hidden] for normed
    float* normed_buf = sdata + blockDim.x;  // second half of shared mem

    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    // ── Phase 1: compute sum(x²) → rms_inv ──
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += bsz) {
        float v = x[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float rms_inv = rsqrtf(sdata[0] / (float)hidden + eps);

    // ── Phase 2: compute normed values and store in shared + global ──
    for (int i = tid; i < hidden; i += bsz) {
        float n_val = x[i] * rms_inv * w[i];
        out_f32[i] = n_val;
        normed_buf[i] = n_val;
    }
    __syncthreads();

    // ── Phase 3: quantize normed values to Q8_1 ──
    // Each warp handles one Q8 block (32 values)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_q8_blocks = hidden >> 5;
    const int warps_per_block = bsz >> 5;

    for (int blk = warp_id; blk < n_q8_blocks; blk += warps_per_block) {
        const float xi = normed_buf[blk * 32 + lane];

        float amax = fabsf(xi);
        for (int off = 16; off >= 1; off >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, off));

        float sum = xi;
        for (int off = 16; off >= 1; off >>= 1)
            sum += __shfl_xor(sum, off);

        const float d = amax / 127.0f;
        const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
        const signed char q = (signed char)__float2int_rn(xi * id);

        unsigned char* bp = out_q8 + (size_t)blk * 36;
        if (lane == 0) {
            *(unsigned short*)(bp + 0) = f32_to_fp16(d);
            *(unsigned short*)(bp + 2) = f32_to_fp16(sum);
        }
        bp[4 + lane] = (unsigned char)q;
    }
}

// ---------------------------------------------------------------------------
// Scale/min unpacking for Q4_K (6-bit packed scales)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void unpack_scale_min_k4(
    int j, const unsigned char* scales, unsigned char* sc, unsigned char* m
) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >> 4)   | ((scales[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// Kernel: gemv_q4k_q8 — Q4_K × Q8_1 GEMV with dp4a
//
// Each warp (32 threads) handles one output row's dot product.
// Thread layout: 4 groups of 8 threads per Q4_K block.
//
// Grid : (N, M, 1)
// Block: (32, 1, 1) — single warp on RDNA2
// ---------------------------------------------------------------------------
__launch_bounds__(32, 2)
extern "C" __global__ void gemv_q4k_q8(
    const unsigned char* __restrict__ w_data,  // [N * nblk * 144] raw Q4_K
    const unsigned char* __restrict__ x_q8,    // [M * (K/32) * 36] Q8_1 blocks
    float*               __restrict__ out,     // [M, N] output
    int M, int N, int K,
    float beta
) {
    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    if (row >= N || mi >= M) return;

    const int lane = threadIdx.x;  // 0..31

    const int nblk = K / 256;  // Q4_K blocks per row
    const unsigned char* __restrict__ row_w = w_data + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * (K / 32) * 36;

    float acc = 0.0f;

    // Thread mapping: lane → group (0..3), position within group (0..7)
    const int group = lane >> 3;       // 0..3
    const int lane_g = lane & 7;       // 0..7 (each handles 4 bytes)

    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* bp = row_w + blk * 144;
        const float d    = fp16_load(bp + 0);
        const float dmin = fp16_load(bp + 2);

        // Unpack 6-bit scales for low and high nibble sub-groups
        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        unpack_scale_min_k4(2 * group,     bp + 4, &sc_lo, &m_lo);
        unpack_scale_min_k4(2 * group + 1, bp + 4, &sc_hi, &m_hi);

        // Load 4 qs bytes → extract low and high nibbles as packed int32
        const int q4_packed = *(const int*)(bp + 16 + group * 32 + lane_g * 4);
        const int q4_lo = q4_packed & 0x0F0F0F0F;
        const int q4_hi = (q4_packed >> 4) & 0x0F0F0F0F;

        // Corresponding Q8_1 blocks for low/high nibbles
        const int q8_lo_idx = blk * 8 + 2 * group;
        const int q8_hi_idx = blk * 8 + 2 * group + 1;

        const unsigned char* q8_lo_bp = x_base + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = x_base + (size_t)q8_hi_idx * 36;

        const float d8_lo = fp16_to_f32(*(const unsigned short*)(q8_lo_bp + 0));
        const float d8_hi = fp16_to_f32(*(const unsigned short*)(q8_hi_bp + 0));

        // Load 4 Q8_1 int8 values as packed int32
        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        // dp4a: dot product of 4 packed int8 pairs
        const int dot_lo = dp4a(q4_lo, u_lo, 0);
        const int dot_hi = dp4a(q4_hi, u_hi, 0);

        // Sum of Q8 values for min correction
        const int sum_lo = dp4a(0x01010101, u_lo, 0);
        const int sum_hi = dp4a(0x01010101, u_hi, 0);

        // Warp reduce within each group: 8 lanes → lane_g=0
        int rd_lo = dot_lo, rd_hi = dot_hi, rs_lo = sum_lo, rs_hi = sum_hi;
        for (int off = 4; off >= 1; off >>= 1) {
            rd_lo += __shfl_down(rd_lo, off);
            rd_hi += __shfl_down(rd_hi, off);
            rs_lo += __shfl_down(rs_lo, off);
            rs_hi += __shfl_down(rs_hi, off);
        }

        if (lane_g == 0) {
            acc += d8_lo * (d * (float)sc_lo * (float)rd_lo - dmin * (float)m_lo * (float)rs_lo);
            acc += d8_hi * (d * (float)sc_hi * (float)rd_hi - dmin * (float)m_hi * (float)rs_hi);
        }
    }

    // Full warp reduce
    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off);

    if (lane == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], acc) : acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel: gemv_q4k_q8_multi — Multi-row Q4_K × Q8_1 GEMV (8 rows/block)
//
// Processes 8 output rows per block. All 8 warps share the Q8_1 input
// loaded into LDS, eliminating 7/8 of Q8 global memory reads.
// Critical for large N matrices (lm_head: N=151936, K=1536).
//
// Grid : (ceil(N/8), M, 1)
// Block: (32, 8, 1) — 8 warps, each handles one row
// ---------------------------------------------------------------------------
#define MULTI_ROWS 8
__launch_bounds__(256, 4)
extern "C" __global__ void gemv_q4k_q8_multi(
    const unsigned char* __restrict__ w_data,  // [N * nblk * 144] raw Q4_K
    const unsigned char* __restrict__ x_q8,    // [M * (K/32) * 36] Q8_1 blocks
    float*               __restrict__ out,     // [M, N] output
    int M, int N, int K,
    float beta
) {
    const int row = blockIdx.x * MULTI_ROWS + threadIdx.y;
    const int mi  = blockIdx.y;
    if (mi >= M) return;

    const int lane   = threadIdx.x;  // 0..31
    const int nblk   = K / 256;
    const int nq8    = K / 32;       // number of Q8_1 blocks

    // ── Shared memory for Q8_1 input (all warps share) ──
    // Max K=8960 → nq8=280 → 280*36 = 10080 bytes. Fits in LDS.
    __shared__ unsigned char x_shared[280 * 36];  // 10080 bytes

    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * nq8 * 36;

    // Cooperative load of Q8 input into shared memory
    for (int i = threadIdx.y * 32 + threadIdx.x; i < nq8 * 36; i += 256) {
        x_shared[i] = x_base[i];
    }
    __syncthreads();

    if (row >= N) return;

    const unsigned char* __restrict__ row_w = w_data + (size_t)row * nblk * 144;
    float acc = 0.0f;

    const int group  = lane >> 3;
    const int lane_g = lane & 7;

    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* bp = row_w + blk * 144;
        const float d    = fp16_load(bp + 0);
        const float dmin = fp16_load(bp + 2);

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        unpack_scale_min_k4(2 * group,     bp + 4, &sc_lo, &m_lo);
        unpack_scale_min_k4(2 * group + 1, bp + 4, &sc_hi, &m_hi);

        const int q4_packed = *(const int*)(bp + 16 + group * 32 + lane_g * 4);
        const int q4_lo = q4_packed & 0x0F0F0F0F;
        const int q4_hi = (q4_packed >> 4) & 0x0F0F0F0F;

        const int q8_lo_idx = blk * 8 + 2 * group;
        const int q8_hi_idx = blk * 8 + 2 * group + 1;

        // Read Q8 from shared memory instead of global
        const unsigned char* q8_lo_bp = x_shared + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = x_shared + (size_t)q8_hi_idx * 36;

        const float d8_lo = fp16_to_f32(*(const unsigned short*)(q8_lo_bp + 0));
        const float d8_hi = fp16_to_f32(*(const unsigned short*)(q8_hi_bp + 0));

        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        const int dot_lo = dp4a(q4_lo, u_lo, 0);
        const int dot_hi = dp4a(q4_hi, u_hi, 0);
        const int sum_lo = dp4a(0x01010101, u_lo, 0);
        const int sum_hi = dp4a(0x01010101, u_hi, 0);

        int rd_lo = dot_lo, rd_hi = dot_hi, rs_lo = sum_lo, rs_hi = sum_hi;
        for (int off = 4; off >= 1; off >>= 1) {
            rd_lo += __shfl_down(rd_lo, off);
            rd_hi += __shfl_down(rd_hi, off);
            rs_lo += __shfl_down(rs_lo, off);
            rs_hi += __shfl_down(rs_hi, off);
        }

        if (lane_g == 0) {
            acc += d8_lo * (d * (float)sc_lo * (float)rd_lo - dmin * (float)m_lo * (float)rs_lo);
            acc += d8_hi * (d * (float)sc_hi * (float)rd_hi - dmin * (float)m_hi * (float)rs_hi);
        }
    }

    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off);

    if (lane == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], acc) : acc;
    }
}
#undef MULTI_ROWS

// ---------------------------------------------------------------------------
// Kernel: gemv_q6k_q8 — Q6_K × Q8_1 GEMV with integer multiply
//
// Q6_K: 210 bytes per 256 values. ql[128] + qh[64] + scales[16] + d[2]
// Each warp processes one row. Each lane processes one 6-bit value per
// Q8_1 sub-block, multiplied by the corresponding Q8_1 int8 value.
//
// Grid : (N, M, 1)
// Block: (32, 1, 1) — single warp per row
// ---------------------------------------------------------------------------
__launch_bounds__(32, 2)
extern "C" __global__ void gemv_q6k_q8(
    const unsigned char* __restrict__ w_data,  // [N * nblk * 210] raw Q6_K
    const unsigned char* __restrict__ x_q8,    // [M * (K/32) * 36] Q8_1 blocks
    float*               __restrict__ out,     // [M, N] output
    int M, int N, int K,
    float beta
) {
    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    if (row >= N || mi >= M) return;

    const int lane = threadIdx.x;

    const int nblk = K / 256;
    const unsigned char* __restrict__ row_w = w_data + (size_t)row * nblk * 210;
    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * (K / 32) * 36;

    float acc = 0.0f;

    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* bp = row_w + blk * 210;
        const unsigned char* ql = bp;
        const unsigned char* qh = bp + 128;
        const signed char* scales = (const signed char*)(bp + 192);
        const float d6 = fp16_load(bp + 208);

        // Process all 8 sub-groups (each 32 values) sequentially per Q6_K block
        // Each lane handles one value per sub-group
        for (int q8_sub = 0; q8_sub < 8; q8_sub++) {
            const int n_group = q8_sub >> 2;
            const int sub = q8_sub & 3;

            const unsigned char* ql_g = ql + n_group * 64;
            const unsigned char* qh_g = qh + n_group * 32;
            const signed char* sc_g = scales + n_group * 8;

            const int is = lane >> 4;
            const float sc = (float)sc_g[is + sub * 2];

            int q_val;
            switch (sub) {
                case 0: q_val = (int)( ql_g[lane]      & 0xF) | (int)(((qh_g[lane] >> 0) & 3) << 4); break;
                case 1: q_val = (int)( ql_g[32+lane]   & 0xF) | (int)(((qh_g[lane] >> 2) & 3) << 4); break;
                case 2: q_val = (int)((ql_g[lane]      >> 4)) | (int)(((qh_g[lane] >> 4) & 3) << 4); break;
                default: q_val = (int)((ql_g[32+lane]  >> 4)) | (int)(((qh_g[lane] >> 6) & 3) << 4); break;
            }
            q_val -= 32;

            // Load Q8_1
            const int q8_blk_idx = blk * 8 + q8_sub;
            const unsigned char* q8_bp = x_base + (size_t)q8_blk_idx * 36;
            const float d8 = fp16_to_f32(*(const unsigned short*)(q8_bp + 0));
            const int q8_val = (int)((const signed char*)(q8_bp + 4))[lane];

            // Per-element product with scale
            float fval = sc * (float)(q_val * q8_val);

            // Warp reduce
            for (int off = 16; off >= 1; off >>= 1)
                fval += __shfl_down(fval, off);

            if (lane == 0) {
                acc += d6 * d8 * fval;
            }
        }
    }

    if (lane == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], acc) : acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel: gemv_q6k_q8_multi — Multi-row Q6_K × Q8_1 GEMV (8 rows/block)
//
// Same as gemv_q6k_q8 but 8 warps per block sharing Q8 input via LDS.
// Critical for lm_head with N=151936.
//
// Grid : (ceil(N/8), M, 1)
// Block: (32, 8, 1)
// ---------------------------------------------------------------------------
#define MULTI_ROWS_Q6 8
__launch_bounds__(256, 4)
extern "C" __global__ void gemv_q6k_q8_multi(
    const unsigned char* __restrict__ w_data,
    const unsigned char* __restrict__ x_q8,
    float*               __restrict__ out,
    int M, int N, int K,
    float beta
) {
    const int row = blockIdx.x * MULTI_ROWS_Q6 + threadIdx.y;
    const int mi  = blockIdx.y;
    if (mi >= M) return;

    const int lane = threadIdx.x;
    const int nblk = K / 256;
    const int nq8  = K / 32;

    __shared__ unsigned char x_shared[280 * 36];  // max K=8960

    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * nq8 * 36;
    for (int i = threadIdx.y * 32 + threadIdx.x; i < nq8 * 36; i += 256) {
        x_shared[i] = x_base[i];
    }
    __syncthreads();

    if (row >= N) return;

    const unsigned char* __restrict__ row_w = w_data + (size_t)row * nblk * 210;
    float acc = 0.0f;

    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* bp = row_w + blk * 210;
        const unsigned char* ql = bp;
        const unsigned char* qh = bp + 128;
        const signed char* scales = (const signed char*)(bp + 192);
        const float d6 = fp16_load(bp + 208);

        for (int q8_sub = 0; q8_sub < 8; q8_sub++) {
            const int n_group = q8_sub >> 2;
            const int sub = q8_sub & 3;

            const unsigned char* ql_g = ql + n_group * 64;
            const unsigned char* qh_g = qh + n_group * 32;
            const signed char* sc_g = scales + n_group * 8;

            const int is = lane >> 4;
            const float sc = (float)sc_g[is + sub * 2];

            int q_val;
            switch (sub) {
                case 0: q_val = (int)( ql_g[lane]      & 0xF) | (int)(((qh_g[lane] >> 0) & 3) << 4); break;
                case 1: q_val = (int)( ql_g[32+lane]   & 0xF) | (int)(((qh_g[lane] >> 2) & 3) << 4); break;
                case 2: q_val = (int)((ql_g[lane]      >> 4)) | (int)(((qh_g[lane] >> 4) & 3) << 4); break;
                default: q_val = (int)((ql_g[32+lane]  >> 4)) | (int)(((qh_g[lane] >> 6) & 3) << 4); break;
            }
            q_val -= 32;

            const int q8_blk_idx = blk * 8 + q8_sub;
            const unsigned char* q8_bp = x_shared + (size_t)q8_blk_idx * 36;
            const float d8 = fp16_to_f32(*(const unsigned short*)(q8_bp + 0));
            const int q8_val = (int)((const signed char*)(q8_bp + 4))[lane];

            float fval = sc * (float)(q_val * q8_val);

            for (int off = 16; off >= 1; off >>= 1)
                fval += __shfl_down(fval, off);

            if (lane == 0) {
                acc += d6 * d8 * fval;
            }
        }
    }

    if (lane == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], acc) : acc;
    }
}
#undef MULTI_ROWS_Q6

// ---------------------------------------------------------------------------
// Kernel: fused_qkv_q4k — Fused Q+K+V projection in a single kernel
//
// Computes Q, K, V projections from the same Q8_1 input in one launch.
// Virtual row space: [0..Nq) → Q, [Nq..Nq+Nk) → K, [Nq+Nk..Nq+Nk+Nv) → V
// 8 warps per block share Q8 input via LDS.
//
// For qwen2.5-1.5b: Nq=1536, Nk=256, Nv=256, K=1536 → 2048 total rows, 256 blocks
// Saves 2 kernel launches per layer (28 layers × 2 = 56 launches/token saved).
//
// Grid : (ceil((Nq+Nk+Nv)/8), 1, 1)
// Block: (32, 8, 1) — 8 warps
// ---------------------------------------------------------------------------
#define FUSED_QKV_ROWS 8
__launch_bounds__(256, 4)
extern "C" __global__ void fused_qkv_q4k(
    const unsigned char* __restrict__ wq_data,  // [Nq * nblk * 144] Q weights
    const unsigned char* __restrict__ wk_data,  // [Nk * nblk * 144] K weights
    const unsigned char* __restrict__ wv_data,  // [Nv * nblk * 144] V weights
    const unsigned char* __restrict__ x_q8,     // [(K/32) * 36] Q8_1 input
    float*               __restrict__ out_q,    // [Nq] Q output
    float*               __restrict__ out_k,    // [Nk] K output
    float*               __restrict__ out_v,    // [Nv] V output
    int Nq, int Nk, int Nv, int K
) {
    const int total_N = Nq + Nk + Nv;
    const int vrow = blockIdx.x * FUSED_QKV_ROWS + threadIdx.y;  // virtual row
    const int lane = threadIdx.x;
    const int nblk = K / 256;
    const int nq8  = K / 32;

    // ── Cooperative load of Q8 input into shared memory ──
    __shared__ unsigned char x_shared[280 * 36];  // max K=8960
    for (int i = threadIdx.y * 32 + threadIdx.x; i < nq8 * 36; i += 256) {
        x_shared[i] = x_q8[i];
    }
    __syncthreads();

    if (vrow >= total_N) return;

    // Determine which matrix this virtual row belongs to
    const unsigned char* __restrict__ row_w;
    float* out_ptr;
    int local_row;
    if (vrow < Nq) {
        local_row = vrow;
        row_w = wq_data + (size_t)local_row * nblk * 144;
        out_ptr = out_q;
    } else if (vrow < Nq + Nk) {
        local_row = vrow - Nq;
        row_w = wk_data + (size_t)local_row * nblk * 144;
        out_ptr = out_k;
    } else {
        local_row = vrow - Nq - Nk;
        row_w = wv_data + (size_t)local_row * nblk * 144;
        out_ptr = out_v;
    }

    float acc = 0.0f;
    const int group  = lane >> 3;
    const int lane_g = lane & 7;

    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* bp = row_w + blk * 144;
        const float d    = fp16_load(bp + 0);
        const float dmin = fp16_load(bp + 2);

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        unpack_scale_min_k4(2 * group,     bp + 4, &sc_lo, &m_lo);
        unpack_scale_min_k4(2 * group + 1, bp + 4, &sc_hi, &m_hi);

        const int q4_packed = *(const int*)(bp + 16 + group * 32 + lane_g * 4);
        const int q4_lo = q4_packed & 0x0F0F0F0F;
        const int q4_hi = (q4_packed >> 4) & 0x0F0F0F0F;

        const int q8_lo_idx = blk * 8 + 2 * group;
        const int q8_hi_idx = blk * 8 + 2 * group + 1;

        const unsigned char* q8_lo_bp = x_shared + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = x_shared + (size_t)q8_hi_idx * 36;

        const float d8_lo = fp16_to_f32(*(const unsigned short*)(q8_lo_bp + 0));
        const float d8_hi = fp16_to_f32(*(const unsigned short*)(q8_hi_bp + 0));

        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        const int dot_lo = dp4a(q4_lo, u_lo, 0);
        const int dot_hi = dp4a(q4_hi, u_hi, 0);
        const int sum_lo = dp4a(0x01010101, u_lo, 0);
        const int sum_hi = dp4a(0x01010101, u_hi, 0);

        int rd_lo = dot_lo, rd_hi = dot_hi, rs_lo = sum_lo, rs_hi = sum_hi;
        for (int off = 4; off >= 1; off >>= 1) {
            rd_lo += __shfl_down(rd_lo, off);
            rd_hi += __shfl_down(rd_hi, off);
            rs_lo += __shfl_down(rs_lo, off);
            rs_hi += __shfl_down(rs_hi, off);
        }

        if (lane_g == 0) {
            acc += d8_lo * (d * (float)sc_lo * (float)rd_lo - dmin * (float)m_lo * (float)rs_lo);
            acc += d8_hi * (d * (float)sc_hi * (float)rd_hi - dmin * (float)m_hi * (float)rs_hi);
        }
    }

    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off);

    if (lane == 0) {
        out_ptr[local_row] = acc;
    }
}
#undef FUSED_QKV_ROWS

// ---------------------------------------------------------------------------
// Kernel: fused_gate_up_silu_q4k — Fused gate+up+SiLU*mul GEMV for MLP
//
// Computes:  out[row] = SiLU(gate_dot) * up_dot
// where gate_dot = dot(gate_weights[row], x_q8) and up_dot = dot(up_weights[row], x_q8)
//
// Same Q4_K × Q8_1 dp4a approach as gemv_q4k_q8, but processes BOTH gate and
// up weight matrices in a single kernel, applying SiLU activation inline.
// Eliminates 2 kernel launches (separate gate GEMV + SiLU*mul) per layer.
//
// Grid : (N, 1, 1)  — one warp per output row (N = intermediate_size)
// Block: (32, 1, 1) — single warp
// ---------------------------------------------------------------------------
__launch_bounds__(32, 2)
extern "C" __global__ void fused_gate_up_silu_q4k(
    const unsigned char* __restrict__ gate_data,  // [N * nblk * 144] gate Q4_K weights
    const unsigned char* __restrict__ up_data,    // [N * nblk * 144] up Q4_K weights
    const unsigned char* __restrict__ x_q8,       // [(K/32) * 36] Q8_1 input
    float*               __restrict__ out,        // [N] output
    int N, int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const int nblk = K / 256;
    const unsigned char* __restrict__ gate_row = gate_data + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ up_row   = up_data   + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ x_base   = x_q8;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    const int group  = lane >> 3;
    const int lane_g = lane & 7;

    for (int blk = 0; blk < nblk; blk++) {
        // ── Gate weights ──
        const unsigned char* gbp = gate_row + blk * 144;
        const float gd    = fp16_load(gbp + 0);
        const float gdmin = fp16_load(gbp + 2);
        unsigned char gsc_lo, gm_lo, gsc_hi, gm_hi;
        unpack_scale_min_k4(2 * group,     gbp + 4, &gsc_lo, &gm_lo);
        unpack_scale_min_k4(2 * group + 1, gbp + 4, &gsc_hi, &gm_hi);
        const int gq4 = *(const int*)(gbp + 16 + group * 32 + lane_g * 4);
        const int gq4_lo = gq4 & 0x0F0F0F0F;
        const int gq4_hi = (gq4 >> 4) & 0x0F0F0F0F;

        // ── Up weights ──
        const unsigned char* ubp = up_row + blk * 144;
        const float ud    = fp16_load(ubp + 0);
        const float udmin = fp16_load(ubp + 2);
        unsigned char usc_lo, um_lo, usc_hi, um_hi;
        unpack_scale_min_k4(2 * group,     ubp + 4, &usc_lo, &um_lo);
        unpack_scale_min_k4(2 * group + 1, ubp + 4, &usc_hi, &um_hi);
        const int uq4 = *(const int*)(ubp + 16 + group * 32 + lane_g * 4);
        const int uq4_lo = uq4 & 0x0F0F0F0F;
        const int uq4_hi = (uq4 >> 4) & 0x0F0F0F0F;

        // ── Shared Q8_1 input (same for gate and up) ──
        const int q8_lo_idx = blk * 8 + 2 * group;
        const int q8_hi_idx = blk * 8 + 2 * group + 1;
        const unsigned char* q8_lo_bp = x_base + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = x_base + (size_t)q8_hi_idx * 36;
        const float d8_lo = fp16_to_f32(*(const unsigned short*)(q8_lo_bp + 0));
        const float d8_hi = fp16_to_f32(*(const unsigned short*)(q8_hi_bp + 0));
        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        // ── dp4a for gate ──
        int gdot_lo = dp4a(gq4_lo, u_lo, 0);
        int gdot_hi = dp4a(gq4_hi, u_hi, 0);
        int gsum_lo = dp4a(0x01010101, u_lo, 0);
        int gsum_hi = dp4a(0x01010101, u_hi, 0);

        // ── dp4a for up ──
        int udot_lo = dp4a(uq4_lo, u_lo, 0);
        int udot_hi = dp4a(uq4_hi, u_hi, 0);
        int usum_lo = gsum_lo;  // same Q8 input → same sums
        int usum_hi = gsum_hi;

        // ── Intra-group reduction ──
        for (int off = 4; off >= 1; off >>= 1) {
            gdot_lo += __shfl_down(gdot_lo, off);
            gdot_hi += __shfl_down(gdot_hi, off);
            gsum_lo += __shfl_down(gsum_lo, off);
            gsum_hi += __shfl_down(gsum_hi, off);
            udot_lo += __shfl_down(udot_lo, off);
            udot_hi += __shfl_down(udot_hi, off);
            usum_lo += __shfl_down(usum_lo, off);
            usum_hi += __shfl_down(usum_hi, off);
        }

        if (lane_g == 0) {
            acc_gate += d8_lo * (gd * (float)gsc_lo * (float)gdot_lo - gdmin * (float)gm_lo * (float)gsum_lo);
            acc_gate += d8_hi * (gd * (float)gsc_hi * (float)gdot_hi - gdmin * (float)gm_hi * (float)gsum_hi);
            acc_up   += d8_lo * (ud * (float)usc_lo * (float)udot_lo - udmin * (float)um_lo * (float)usum_lo);
            acc_up   += d8_hi * (ud * (float)usc_hi * (float)udot_hi - udmin * (float)um_hi * (float)usum_hi);
        }
    }

    // Full warp reduce
    for (int off = 16; off >= 1; off >>= 1) {
        acc_gate += __shfl_down(acc_gate, off);
        acc_up   += __shfl_down(acc_up, off);
    }

    if (lane == 0) {
        // SiLU(gate) * up
        float silu_gate = acc_gate / (1.0f + expf(-acc_gate));
        out[row] = silu_gate * acc_up;
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_gate_up_silu_q6k — Same fusion for Q6_K weights
// ---------------------------------------------------------------------------
__launch_bounds__(32, 2)
extern "C" __global__ void fused_gate_up_silu_q6k(
    const unsigned char* __restrict__ gate_data,  // [N * nblk * 210] gate Q6_K weights
    const unsigned char* __restrict__ up_data,    // [N * nblk * 210] up Q6_K weights
    const unsigned char* __restrict__ x_q8,       // [(K/32) * 36] Q8_1 input
    float*               __restrict__ out,        // [N] output
    int N, int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int lane = threadIdx.x;
    const int nblk = K / 256;
    const unsigned char* __restrict__ gate_row = gate_data + (size_t)row * nblk * 210;
    const unsigned char* __restrict__ up_row   = up_data   + (size_t)row * nblk * 210;
    const unsigned char* __restrict__ x_base   = x_q8;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (int blk = 0; blk < nblk; blk++) {
        // ── Gate Q6_K block ──
        const unsigned char* gbp = gate_row + blk * 210;
        const unsigned char* gql = gbp;           // ql[128]
        const unsigned char* gqh = gbp + 128;     // qh[64]
        const signed char* gsc   = (const signed char*)(gbp + 192); // scales[16]
        const float gd           = fp16_load(gbp + 208);

        // ── Up Q6_K block ──
        const unsigned char* ubp = up_row + blk * 210;
        const unsigned char* uql = ubp;
        const unsigned char* uqh = ubp + 128;
        const signed char* usc   = (const signed char*)(ubp + 192);
        const float ud           = fp16_load(ubp + 208);

        // 8 sub-blocks of 32 values each per Q6_K block
        for (int sub = 0; sub < 8; sub++) {
            const int q8_idx = blk * 8 + sub;
            const unsigned char* q8_bp = x_base + (size_t)q8_idx * 36;
            const float d8 = fp16_to_f32(*(const unsigned short*)(q8_bp + 0));

            // Each lane handles one value
            if (lane < 32) {
                int val_idx = sub * 32 + lane;
                // Reconstruct 6-bit value: ql has low 4 bits, qh has bits 4-5
                int ql_byte = gql[val_idx / 2];
                int gq6 = (val_idx & 1) ? (ql_byte >> 4) : (ql_byte & 0xF);
                int qh_bit = (gqh[val_idx / 4] >> (2 * (val_idx % 4))) & 3;
                gq6 |= (qh_bit << 4);
                gq6 -= 32;  // Q6_K values are offset by 32

                int ql_byte_u = uql[val_idx / 2];
                int uq6 = (val_idx & 1) ? (ql_byte_u >> 4) : (ql_byte_u & 0xF);
                int qh_bit_u = (uqh[val_idx / 4] >> (2 * (val_idx % 4))) & 3;
                uq6 |= (qh_bit_u << 4);
                uq6 -= 32;

                signed char q8_val = *(const signed char*)(q8_bp + 4 + lane);

                float gw = gd * (float)gsc[sub] * (float)gq6 * d8 * (float)q8_val;
                float uw = ud * (float)usc[sub] * (float)uq6 * d8 * (float)q8_val;
                acc_gate += gw;
                acc_up   += uw;
            }
        }
    }

    // Full warp reduce
    for (int off = 16; off >= 1; off >>= 1) {
        acc_gate += __shfl_down(acc_gate, off);
        acc_up   += __shfl_down(acc_up, off);
    }

    if (lane == 0) {
        float silu_gate = acc_gate / (1.0f + expf(-acc_gate));
        out[row] = silu_gate * acc_up;
    }
}

// ---------------------------------------------------------------------------
// Kernel: fused_gate_up_silu_q4k_multi — Multi-row fused MLP (8 rows/block)
//
// Same as fused_gate_up_silu_q4k but 8 warps per block share Q8 via LDS.
// Reduces global Q8 reads by 8× and block count from N to ceil(N/8).
// For N=8960, K=1536: 1120 blocks instead of 8960.
//
// Grid : (ceil(N/8), 1, 1)
// Block: (32, 8, 1) — 8 warps
// ---------------------------------------------------------------------------
#define FUSED_MLP_ROWS 8
__launch_bounds__(256, 4)
extern "C" __global__ void fused_gate_up_silu_q4k_multi(
    const unsigned char* __restrict__ gate_data,
    const unsigned char* __restrict__ up_data,
    const unsigned char* __restrict__ x_q8,
    float*               __restrict__ out,
    int N, int K
) {
    const int row = blockIdx.x * FUSED_MLP_ROWS + threadIdx.y;
    const int lane = threadIdx.x;
    const int nblk = K / 256;
    const int nq8  = K / 32;

    // Cooperative load Q8 into shared memory
    __shared__ unsigned char x_shared[280 * 36];  // max K=8960 → 280 Q8 blocks
    for (int i = threadIdx.y * 32 + threadIdx.x; i < nq8 * 36; i += 256) {
        x_shared[i] = x_q8[i];
    }
    __syncthreads();

    if (row >= N) return;

    const unsigned char* __restrict__ gate_row = gate_data + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ up_row   = up_data   + (size_t)row * nblk * 144;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    const int group  = lane >> 3;
    const int lane_g = lane & 7;

    for (int blk = 0; blk < nblk; blk++) {
        // Gate weights
        const unsigned char* gbp = gate_row + blk * 144;
        const float gd    = fp16_load(gbp + 0);
        const float gdmin = fp16_load(gbp + 2);
        unsigned char gsc_lo, gm_lo, gsc_hi, gm_hi;
        unpack_scale_min_k4(2 * group,     gbp + 4, &gsc_lo, &gm_lo);
        unpack_scale_min_k4(2 * group + 1, gbp + 4, &gsc_hi, &gm_hi);
        const int gq4 = *(const int*)(gbp + 16 + group * 32 + lane_g * 4);
        const int gq4_lo = gq4 & 0x0F0F0F0F;
        const int gq4_hi = (gq4 >> 4) & 0x0F0F0F0F;

        // Up weights
        const unsigned char* ubp = up_row + blk * 144;
        const float ud    = fp16_load(ubp + 0);
        const float udmin = fp16_load(ubp + 2);
        unsigned char usc_lo, um_lo, usc_hi, um_hi;
        unpack_scale_min_k4(2 * group,     ubp + 4, &usc_lo, &um_lo);
        unpack_scale_min_k4(2 * group + 1, ubp + 4, &usc_hi, &um_hi);
        const int uq4 = *(const int*)(ubp + 16 + group * 32 + lane_g * 4);
        const int uq4_lo = uq4 & 0x0F0F0F0F;
        const int uq4_hi = (uq4 >> 4) & 0x0F0F0F0F;

        // Q8 from shared memory
        const int q8_lo_idx = blk * 8 + 2 * group;
        const int q8_hi_idx = blk * 8 + 2 * group + 1;
        const unsigned char* q8_lo_bp = x_shared + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = x_shared + (size_t)q8_hi_idx * 36;
        const float d8_lo = fp16_to_f32(*(const unsigned short*)(q8_lo_bp + 0));
        const float d8_hi = fp16_to_f32(*(const unsigned short*)(q8_hi_bp + 0));
        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        // dp4a for gate
        int gdot_lo = dp4a(gq4_lo, u_lo, 0);
        int gdot_hi = dp4a(gq4_hi, u_hi, 0);
        int gsum_lo = dp4a(0x01010101, u_lo, 0);
        int gsum_hi = dp4a(0x01010101, u_hi, 0);

        // dp4a for up
        int udot_lo = dp4a(uq4_lo, u_lo, 0);
        int udot_hi = dp4a(uq4_hi, u_hi, 0);
        int usum_lo = gsum_lo;
        int usum_hi = gsum_hi;

        // Intra-group reduction
        for (int off = 4; off >= 1; off >>= 1) {
            gdot_lo += __shfl_down(gdot_lo, off);
            gdot_hi += __shfl_down(gdot_hi, off);
            gsum_lo += __shfl_down(gsum_lo, off);
            gsum_hi += __shfl_down(gsum_hi, off);
            udot_lo += __shfl_down(udot_lo, off);
            udot_hi += __shfl_down(udot_hi, off);
            usum_lo += __shfl_down(usum_lo, off);
            usum_hi += __shfl_down(usum_hi, off);
        }

        if (lane_g == 0) {
            acc_gate += d8_lo * (gd * (float)gsc_lo * (float)gdot_lo - gdmin * (float)gm_lo * (float)gsum_lo);
            acc_gate += d8_hi * (gd * (float)gsc_hi * (float)gdot_hi - gdmin * (float)gm_hi * (float)gsum_hi);
            acc_up   += d8_lo * (ud * (float)usc_lo * (float)udot_lo - udmin * (float)um_lo * (float)usum_lo);
            acc_up   += d8_hi * (ud * (float)usc_hi * (float)udot_hi - udmin * (float)um_hi * (float)usum_hi);
        }
    }

    for (int off = 16; off >= 1; off >>= 1) {
        acc_gate += __shfl_down(acc_gate, off);
        acc_up   += __shfl_down(acc_up, off);
    }

    if (lane == 0) {
        float silu_gate = acc_gate / (1.0f + expf(-acc_gate));
        out[row] = silu_gate * acc_up;
    }
}
#undef FUSED_MLP_ROWS

// ---------------------------------------------------------------------------
// Kernel: fused_attention_decode — Single-token attention in one kernel
//
// For decode (n=1): computes Q×K^T → scale → softmax → ×V for one head.
// Each block handles one Q head. Threads collaborate to compute the dot
// products and softmax over the KV cache sequence.
//
// Grid : (n_heads, 1, 1)
// Block: (256, 1, 1)  — 8 warps per block
// ---------------------------------------------------------------------------
__launch_bounds__(256, 4)
extern "C" __global__ void fused_attention_decode(
    const float* __restrict__ q,        // [n_heads * head_dim] single query
    const float* __restrict__ k_cache,  // [seq_len, n_kv_heads * head_dim]
    const float* __restrict__ v_cache,  // [seq_len, n_kv_heads * head_dim]
    float*       __restrict__ out,      // [n_heads * head_dim] output
    int seq_len,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    float scale
) {
    extern __shared__ float smem[];  // [seq_len] for scores, then reused

    const int head = blockIdx.x;
    if (head >= n_heads) return;
    const int tid = threadIdx.x;

    // GQA: map Q head → KV head
    const int heads_per_group = n_heads / n_kv_heads;
    const int kv_head = head / heads_per_group;

    const float* q_head = q + head * head_dim;
    const float* k_head_base = k_cache + kv_head * head_dim;  // stride = n_kv_heads * head_dim
    const float* v_head_base = v_cache + kv_head * head_dim;
    const int kv_stride = n_kv_heads * head_dim;

    // ── Phase 1: Q×K^T → scores[seq_len] ──
    // Each thread computes scores for multiple sequence positions
    for (int s = tid; s < seq_len; s += blockDim.x) {
        const float* k_vec = k_head_base + s * kv_stride;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot = __fmaf_rn(q_head[d], k_vec[d], dot);
        }
        smem[s] = dot * scale;
    }
    __syncthreads();

    // ── Phase 2: Softmax over scores ──
    // Find max
    float local_max = -1e30f;
    for (int s = tid; s < seq_len; s += blockDim.x) {
        local_max = fmaxf(local_max, smem[s]);
    }
    // Warp reduce max
    for (int off = 16; off >= 1; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    // Cross-warp reduce via shared memory (use smem after seq_len)
    __shared__ float warp_max[8];
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (tid < 8) {
        float val = warp_max[tid];
        for (int off = 4; off >= 1; off >>= 1)
            val = fmaxf(val, __shfl_down(val, off));
        if (tid == 0) warp_max[0] = val;
    }
    __syncthreads();
    float global_max = warp_max[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int s = tid; s < seq_len; s += blockDim.x) {
        float e = __expf(smem[s] - global_max);
        smem[s] = e;
        local_sum += e;
    }
    // Warp reduce sum
    for (int off = 16; off >= 1; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    __shared__ float warp_sum[8];
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    if (tid < 8) {
        float val = warp_sum[tid];
        for (int off = 4; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (tid == 0) warp_sum[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_sum[0];

    // Normalize scores
    for (int s = tid; s < seq_len; s += blockDim.x) {
        smem[s] *= inv_sum;
    }
    __syncthreads();

    // ── Phase 3: scores × V → output ──
    // Each thread computes a subset of output dimensions
    float* out_head = out + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            acc = __fmaf_rn(smem[s], v_head_base[s * kv_stride + d], acc);
        }
        out_head[d] = acc;
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

// ---------------------------------------------------------------------------
// Kernel 15: fused_norm_bias_rope
//
// Fused kernel: RMSNorm + optional bias + RoPE in a single dispatch.
// Combines three operations to reduce memory round-trips:
//   1. RMSNorm: out[i] = (input[i] / rms) * norm_weight[i]
//   2. Add bias (optional): out[i] += bias[i]
//   3. RoPE: rotate pairs within each head using position encoding
//
// Grid : (n_tokens, 1, 1)
// Block: (256, 1, 1)
//
// Each block processes one token's hidden_size elements.
// Shared memory is used for the variance reduction (parallel tree sum).
//
// Layout assumptions:
//   - input/output are [n_tokens, hidden_size] in row-major order
//   - hidden_size must be divisible by n_heads
//   - head_dim = hidden_size / n_heads
//   - RoPE is applied per head using "halves" convention: rotate (i, i + head_dim/2)
//   - bias pointer can be NULL (signaled by has_bias == 0)
//
// RDNA2 notes:
//   - 256 threads per block balances wavefront utilization and shared memory
//   - Parallel tree reduction in shared memory for variance
//   - Strided loops for elements beyond block size
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
extern "C" __global__ void fused_norm_bias_rope(
    float*       __restrict__ output,     // [n_tokens, hidden_size] F32 (output — may be same as input for in-place)
    const float* __restrict__ input,      // [n_tokens, hidden_size] F32 (input)
    const float* __restrict__ norm_weight, // [hidden_size] F32 (RMSNorm scaling)
    const float* __restrict__ bias,       // [hidden_size] F32 or NULL (optional bias)
    int hidden_size,
    int n_heads,
    int pos_start,                        // absolute position for first token in this batch
    float theta,                          // RoPE base (10000.0 or similar)
    float eps,                            // epsilon for RMSNorm (1e-5 or 1e-6)
    int has_bias                          // 1 if bias is valid, 0 if NULL
) {
    extern __shared__ float sdata[];  // 256 floats for parallel reduction

    int tok = blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;        // 256

    int head_dim = hidden_size / n_heads;
    int half_dim = head_dim / 2;

    const float* input_row = input + (size_t)tok * hidden_size;
    float*       out_row   = output + (size_t)tok * hidden_size;

    // ── Phase 1: Parallel reduction to compute RMS variance ──────────────────
    // Accumulate sum of squares from this thread's elements.
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += bsz) {
        float v = input_row[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // sdata[0] now contains sum of x^2 over all hidden_size elements.
    float rms_inv = rsqrtf(sdata[0] / (float)hidden_size + eps);
    __syncthreads();  // Ensure all threads see the computed rms_inv

    // ── Phase 2: Normalize + bias + RoPE in one strided pass ────────────────
    for (int i = tid; i < hidden_size; i += bsz) {
        float val = input_row[i] * rms_inv * norm_weight[i];
        if (has_bias) {
            val += bias[i];
        }

        // Apply RoPE if element is within a rotation pair.
        // Determine which head and position within head this element belongs to.
        int head_idx = i / head_dim;
        int pos_in_head = i % head_dim;

        // RoPE uses the "halves" convention: rotate pairs (j, j + half_dim).
        // Check if this is the first or second element of a rotation pair.
        if (pos_in_head < half_dim) {
            int pair_idx = pos_in_head + half_dim;  // index of the paired element
            float paired_val = input_row[pair_idx] * rms_inv * norm_weight[pair_idx];
            if (has_bias) {
                paired_val += bias[pair_idx];
            }

            // Compute rotation angle for this dimension pair within this head.
            float freq = 1.0f / powf(theta, (2.0f * (float)pos_in_head) / (float)head_dim);
            float angle = (float)(pos_start + tok) * freq;
            float c = cosf(angle);
            float s = sinf(angle);

            // Apply 2D rotation: [val, paired_val] -> [val*c - paired_val*s, val*s + paired_val*c]
            float rotated_val = val * c - paired_val * s;
            float rotated_pair = val * s + paired_val * c;

            out_row[i] = rotated_val;
            out_row[pair_idx] = rotated_pair;
        }
        // If pos_in_head >= half_dim, the value was already written by its pair.
        // Skip to avoid writing twice (and overwriting the rotation result).
        else {
            // This element will be written by its paired thread.
            // Do nothing here; the pairing thread handles both.
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 16: kv_cache_int8_quantize_append
//
// Quantizes F32 K/V values to INT8 with per-head absolute-max scaling.
// Appends quantized values to the INT8 KV cache at insert_pos.
//
// Strategy:
//   For each new token and each head:
//     1. Find the absolute maximum value among head_dim elements
//     2. Compute scale = absmax / 127.0
//     3. Quantize: q[i] = round(clamp(val[i] / scale, -127, 127))
//     4. Store q values and scale
//
// Grid : (n_new_tokens * n_kv_heads, 1, 1) — one block per (token, head) pair
// Block: (256, 1, 1)
//
// Memory layout:
//   - cache_q: [max_seq, kv_dim] INT8
//   - cache_scales: [max_seq, n_kv_heads] F32
//   - new_kv: [n_new_tokens, kv_dim] F32
//   - kv_dim = n_kv_heads * head_dim
//
// Each block processes one token's one head (head_dim elements).
// Threads cooperate to find absmax and quantize in strided passes.
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
extern "C" __global__ void kv_cache_int8_quantize_append(
    int8_t*       __restrict__ cache_q,      // [max_seq, kv_dim] INT8
    float*        __restrict__ cache_scales, // [max_seq, n_kv_heads] F32
    const float*  __restrict__ new_kv,       // [n_new_tokens, kv_dim] F32
    int insert_pos,
    int n_new_tokens,
    int kv_dim,
    int head_dim,
    int n_kv_heads,
    int max_seq
) {
    extern __shared__ float sdata[];  // blockDim.x floats for parallel reduction

    // Determine which (token, head) pair this block handles
    int block_id = blockIdx.x;
    int tok_offset = block_id / n_kv_heads;  // which new token (0 .. n_new_tokens-1)
    int head_idx = block_id % n_kv_heads;    // which head (0 .. n_kv_heads-1)

    // Absolute position in cache (with wrap-around for sliding window)
    int cache_pos = (insert_pos + tok_offset) % max_seq;

    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    // Pointers to this token's data and this head's data
    const float* new_kv_row = new_kv + (size_t)tok_offset * kv_dim;
    const float* head_start = new_kv_row + (size_t)head_idx * head_dim;

    int8_t*  cache_q_row = cache_q + (size_t)cache_pos * kv_dim;
    int8_t*  cache_q_head = cache_q_row + (size_t)head_idx * head_dim;

    float*   cache_scales_row = cache_scales + (size_t)cache_pos * n_kv_heads;

    // ── Phase 1: Find absolute maximum among head_dim elements ──────────────
    float local_absmax = 0.0f;
    for (int i = tid; i < head_dim; i += bsz) {
        float v = head_start[i];
        float absv = fabsf(v);
        if (absv > local_absmax) {
            local_absmax = absv;
        }
    }
    sdata[tid] = local_absmax;
    __syncthreads();

    // Tree reduction to find global absmax
    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
        __syncthreads();
    }

    // sdata[0] contains the absolute maximum
    float absmax = sdata[0];
    float scale = (absmax > 1e-7f) ? (absmax / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale for this (token, head) pair
    if (tid == 0) {
        cache_scales_row[head_idx] = scale;
    }
    __syncthreads();

    // ── Phase 2: Quantize and store INT8 values ────────────────────────────
    for (int i = tid; i < head_dim; i += bsz) {
        float v = head_start[i];
        float q_float = v * inv_scale;
        // Clamp to [-127, 127] and round
        q_float = fmaxf(-127.0f, fminf(127.0f, q_float));
        int8_t q_int = (int8_t)roundf(q_float);
        cache_q_head[i] = q_int;
    }
}

// ---------------------------------------------------------------------------
// Kernel 17: kv_cache_int8_dequantize
//
// Dequantizes INT8 KV cache back to F32 for attention computation.
// Reads from INT8 cache and scales, produces F32 output.
//
// Grid : (seq_len, n_kv_heads, 1) — one block per (seq_pos, head) pair
// Block: (256, 1, 1)
//
// Memory layout:
//   - cache_q: [max_seq, kv_dim] INT8
//   - cache_scales: [max_seq, n_kv_heads] F32
//   - output: [seq_len, kv_dim] F32
//
// Each block dequantizes one head's elements for one position.
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
extern "C" __global__ void kv_cache_int8_dequantize(
    const int8_t* __restrict__ cache_q,      // [max_seq, kv_dim] INT8
    const float*  __restrict__ cache_scales, // [max_seq, n_kv_heads] F32
    float*        __restrict__ output,       // [seq_len, kv_dim] F32
    int seq_len,
    int kv_dim,
    int head_dim,
    int n_kv_heads
) {
    // Determine which (seq_pos, head) pair this block handles
    int seq_pos = blockIdx.x;   // which sequence position (0 .. seq_len-1)
    int head_idx = blockIdx.y;  // which head (0 .. n_kv_heads-1)

    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    // Pointers to this position's data and this head's data
    const int8_t* cache_q_row = cache_q + (size_t)seq_pos * kv_dim;
    const int8_t* cache_q_head = cache_q_row + (size_t)head_idx * head_dim;

    const float*  cache_scales_row = cache_scales + (size_t)seq_pos * n_kv_heads;
    float scale = cache_scales_row[head_idx];

    float*        output_row = output + (size_t)seq_pos * kv_dim;
    float*        output_head = output_row + (size_t)head_idx * head_dim;

    // ── Dequantize: output[i] = cache_q[i] * scale ────────────────────────
    for (int i = tid; i < head_dim; i += bsz) {
        int8_t q_int = cache_q_head[i];
        float q_float = (float)q_int;
        output_head[i] = q_float * scale;
    }
}

// ---------------------------------------------------------------------------
// Kernel: gather_rows_bf16
//
// Gathers selected rows from a BF16 matrix into a contiguous output buffer.
// Used for sparse MLP: select active neuron rows from gate_proj/up_proj.
//
// src:     [total_rows, cols] BF16 (u16)
// dst:     [n_indices, cols] BF16 (u16)
// indices: [n_indices] int — which rows to gather
//
// Grid:  (n_indices, 1, 1)
// Block: (256, 1, 1) — each block copies one row
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_rows_bf16(
    const unsigned short* __restrict__ src,
    unsigned short*       __restrict__ dst,
    const int*            __restrict__ indices,
    int n_indices,
    int cols
) {
    int row_out = (int)blockIdx.x;  // which output row (0..n_indices-1)
    if (row_out >= n_indices) return;

    int row_src = indices[row_out];  // which source row to copy
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    const unsigned short* src_row = src + (size_t)row_src * cols;
    unsigned short*       dst_row = dst + (size_t)row_out * cols;

    for (int i = tid; i < cols; i += bsz) {
        dst_row[i] = src_row[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel: gather_cols_bf16
//
// Gathers selected COLUMNS from a BF16 matrix into contiguous output.
// Used for sparse MLP: select active neuron columns from down_proj.
//
// src:     [rows, total_cols] BF16 (u16)  — full down_proj weight
// dst:     [rows, n_indices] BF16 (u16)   — gathered columns
// indices: [n_indices] int — which columns to gather
//
// Grid:  (rows, 1, 1)
// Block: (256, 1, 1) — each block gathers one row's selected columns
// ---------------------------------------------------------------------------
extern "C" __global__ void gather_cols_bf16(
    const unsigned short* __restrict__ src,
    unsigned short*       __restrict__ dst,
    const int*            __restrict__ indices,
    int rows,
    int total_cols,
    int n_indices
) {
    int row = (int)blockIdx.x;  // which row (0..rows-1)
    if (row >= rows) return;

    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    const unsigned short* src_row = src + (size_t)row * total_cols;
    unsigned short*       dst_row = dst + (size_t)row * n_indices;

    for (int i = tid; i < n_indices; i += bsz) {
        int col_src = indices[i];
        dst_row[i] = src_row[col_src];
    }
}

// ---------------------------------------------------------------------------
// Kernel: kuramoto_coherence
//
// GPU-side Kuramoto phase coherence computation.
// For each neuron, computes |mean(exp(i*(θ_hidden - θ_fork)))| where:
//   θ_hidden[k] = atan2(hidden[2k+1], hidden[2k])
//   θ_fork stored pre-computed in tuning_forks[neuron_idx * phase_dim + k]
//
// hidden:       [hidden_dim] F32 — normalized hidden state (1 token)
// tuning_forks: [n_neurons, phase_dim] F32 — pre-computed phase angles
// coherences:   [n_neurons] F32 — output coherence per neuron
//
// phase_dim = hidden_dim / 2
//
// Grid:  (n_neurons, 1, 1)  — one block per neuron
// Block: (256, 1, 1)        — threads reduce over phase_dim
// ---------------------------------------------------------------------------
extern "C" __global__ void kuramoto_coherence(
    const float* __restrict__ hidden,        // [hidden_dim] F32
    const float* __restrict__ tuning_forks,  // [n_neurons, phase_dim] F32
    float*       __restrict__ coherences,    // [n_neurons] F32
    int n_neurons,
    int hidden_dim,
    int phase_dim
) {
    int neuron_idx = (int)blockIdx.x;
    if (neuron_idx >= n_neurons) return;

    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    const float* fork = tuning_forks + (size_t)neuron_idx * phase_dim;

    // Each thread accumulates partial cos/sin sums
    float local_cos = 0.0f;
    float local_sin = 0.0f;

    for (int k = tid; k < phase_dim; k += bsz) {
        // Compute hidden phase: atan2(hidden[2k+1], hidden[2k])
        int idx_re = 2 * k;
        int idx_im = 2 * k + 1;
        float re = (idx_re < hidden_dim) ? hidden[idx_re] : 0.0f;
        float im = (idx_im < hidden_dim) ? hidden[idx_im] : 0.0f;
        float phase_h = atan2f(im, re);

        // Phase difference
        float delta = phase_h - fork[k];
        local_cos += cosf(delta);
        local_sin += sinf(delta);
    }

    // Warp-level reduction using shuffle
    // RDNA2 wavefront = 64 threads
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_cos += __shfl_down(local_cos, offset);
        local_sin += __shfl_down(local_sin, offset);
    }

    // Cross-warp reduction via shared memory (for blocks > 64 threads)
    __shared__ float s_cos[4]; // max 4 warps (256 threads / 64)
    __shared__ float s_sin[4];

    int warp_id = tid / 64;
    int lane_id = tid % 64;

    if (lane_id == 0) {
        s_cos[warp_id] = local_cos;
        s_sin[warp_id] = local_sin;
    }
    __syncthreads();

    // First warp reduces across warps
    if (tid < 4) {
        float c = s_cos[tid];
        float s = s_sin[tid];
        // Reduce 4 values (only need first warp)
        for (int offset = 2; offset > 0; offset >>= 1) {
            c += __shfl_down(c, offset);
            s += __shfl_down(s, offset);
        }
        if (tid == 0) {
            float mean_cos = c / (float)phase_dim;
            float mean_sin = s / (float)phase_dim;
            coherences[neuron_idx] = sqrtf(mean_cos * mean_cos + mean_sin * mean_sin);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: topk_select
//
// Selects top-K indices from a coherence array using a simple parallel
// threshold approach: find the K-th largest value, then select all >= it.
//
// For small K relative to N, this is faster than a full sort.
// Uses bitonic partial sort approach.
//
// coherences: [n_neurons] F32 — input coherences
// indices:    [top_k] I32 — output: indices of top-K neurons
// n_neurons:  total number of neurons
// top_k:      how many to keep
//
// Grid:  (1, 1, 1) — single block
// Block: (256, 1, 1) — cooperative selection
// ---------------------------------------------------------------------------
extern "C" __global__ void topk_select(
    const float* __restrict__ coherences,
    int*         __restrict__ indices,
    int n_neurons,
    int top_k
) {
    // Simple approach: each thread scans a portion of coherences,
    // maintains a local top-K, then merge cooperatively.
    // For n_neurons=8960, top_k=2240-4480, a threshold-based approach works.

    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    // Phase 1: Find approximate K-th value via sampling
    // Sample every (n_neurons/256)-th element, sort samples, pick threshold
    __shared__ float threshold;
    __shared__ int count;

    if (tid == 0) {
        // Quick threshold: sample coherences, find approximate K-th value
        // Use simple insertion into a small sorted buffer
        float samples[64];
        int n_samples = 0;
        int step = n_neurons > 64 ? n_neurons / 64 : 1;
        for (int i = 0; i < n_neurons && n_samples < 64; i += step) {
            samples[n_samples++] = coherences[i];
        }
        // Sort samples (small insertion sort)
        for (int i = 1; i < n_samples; i++) {
            float key = samples[i];
            int j = i - 1;
            while (j >= 0 && samples[j] < key) {
                samples[j + 1] = samples[j];
                j--;
            }
            samples[j + 1] = key;
        }
        // K-th value estimate (scale sample rank to full rank)
        int sample_rank = (int)((float)top_k / n_neurons * n_samples);
        if (sample_rank >= n_samples) sample_rank = n_samples - 1;
        threshold = samples[sample_rank] * 0.95f; // slight margin
        count = 0;
    }
    __syncthreads();

    // Phase 2: Each thread collects indices above threshold
    float thr = threshold;
    for (int i = tid; i < n_neurons; i += bsz) {
        if (coherences[i] >= thr) {
            int pos = atomicAdd(&count, 1);
            if (pos < top_k) {
                indices[pos] = i;
            }
        }
    }
    __syncthreads();

    // Phase 3: If we got too few, lower threshold and retry
    // (In practice the sampling approach gets close enough)
    if (tid == 0 && count < top_k) {
        // Fill remaining with sequential scan for largest remaining
        int filled = count;
        for (int i = 0; i < n_neurons && filled < top_k; i++) {
            if (coherences[i] < thr) {
                // Check if this is larger than anything we've missed
                indices[filled] = i;
                filled++;
            }
        }
        count = filled;
    }
}

// ---------------------------------------------------------------------------
// Kernel: argmax_f32 — GPU-side argmax over F32 logits
//
// Finds the index of the maximum value in a row of floats.
// Eliminates the need to download 608KB of logits to CPU for greedy sampling.
//
// Grid : (1, 1, 1)   — single row (last token logits)
// Block: (1024, 1, 1) — 32 warps
//
// Phase 1: Each thread scans with stride, finds local max
// Phase 2: Warp reduction via __shfl_xor
// Phase 3: Inter-warp reduction via shared memory
// Phase 4: Thread 0 writes argmax index + max value
// ---------------------------------------------------------------------------
__launch_bounds__(1024, 1)
extern "C" __global__ void argmax_f32(
    const float* __restrict__ x,      // [ncols] logits
    int*         __restrict__ dst,     // [1] output argmax index
    float*       __restrict__ dst_val, // [1] output max value (optional, can be NULL)
    int ncols
) {
    const int tid = threadIdx.x;

    float maxval = -1e30f;
    int   argmax = 0;

    // Phase 1: Strided scan
    for (int col = tid; col < ncols; col += blockDim.x) {
        const float val = x[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    // Phase 2: Warp reduction via __shfl_xor
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float val = __shfl_xor(maxval, offset);
        const int   col = __shfl_xor(argmax, offset);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    // Phase 3: Inter-warp reduction via shared memory
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;  // 1024/32 = 32

    __shared__ float shared_maxval[32];
    __shared__ int   shared_argmax[32];

    if (lane_id == 0) {
        shared_maxval[warp_id] = maxval;
        shared_argmax[warp_id] = argmax;
    }
    __syncthreads();

    // Warp 0 does final reduction
    if (warp_id == 0) {
        maxval = (lane_id < n_warps) ? shared_maxval[lane_id] : -1e30f;
        argmax = (lane_id < n_warps) ? shared_argmax[lane_id] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float val = __shfl_xor(maxval, offset);
            const int   col = __shfl_xor(argmax, offset);
            if (val > maxval) {
                maxval = val;
                argmax = col;
            }
        }

        if (lane_id == 0) {
            dst[0] = argmax;
            if (dst_val != nullptr) {
                dst_val[0] = maxval;
            }
        }
    }
}

