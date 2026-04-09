// gemv_w64.hip — Wave64 GEMV kernels for Q4_K/Q6_K (ported from llama.cpp)
//
// Compiled with: hipcc --offload-arch=gfx1031 -O3 -ffast-math --genco
// (NO -mno-wavefrontsize64, so wave64 is the default on RDNA2)
//
// Key insight: wave64 (64 threads/warp) achieves 2.26x better bandwidth than
// wave32 (32 threads/warp) for the same GEMV kernel, because 64 concurrent
// threads generate more outstanding memory requests per row.
//
// Benchmark result (standalone):
//   wave32: 161 GB/s, 247 tok/s (GEMV only)
//   wave64: 364 GB/s, 559 tok/s (GEMV only)
//
// These kernels replace gemv_q4k_q8 and fused_gate_up_silu_q4k from the
// wave32 file for the decode (n=1) hot path.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ─── dp4a: 4-element int8 dot product (RDNA2 native) ───────────────────────
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// ─── Q4_K vec_dot (llama.cpp port) ─────────────────────────────────────────
// Each thread handles VDR=2 sub-iterations within one Q4_K block.
// With 64 threads and qi/vdr=16 threads per block, 4 blocks are processed
// simultaneously per loop iteration.
//
// Q4_K block layout (144 bytes):
//   dm:     4 bytes (fp16 d + fp16 dmin)
//   scales: 12 bytes (6-bit packed scales and mins)
//   qs:     128 bytes (4-bit quantized values)
//
// Q8_1 block layout (36 bytes):
//   ds:     4 bytes (fp16 d + fp16 sum)
//   qs:     32 bytes (int8 quantized values)

#define QK_K    256
#define QI4_K   32    // QK_K / (4 * QR4_K)
#define QR4_K   2
#define QK8_1   32
#define VDR     2

__device__ __forceinline__ float vec_dot_q4k_q8_1(
    const unsigned char* __restrict__ bq4_data,  // pointer to this Q4K block (144 bytes)
    const unsigned char* __restrict__ bq8_base,  // pointer to corresponding Q8_1 blocks
    int iqs  // sub-position within block (0,2,4,...,30)
) {
    int v[2];
    int u[2 * QR4_K];  // 4 ints
    float d8[QR4_K];   // 2 floats

    // Which pair of Q8_1 blocks within this Q4_K super-block
    // QK8_1/4 = 8, so bq8_offset cycles through 0,2,4,6 for iqs 0-30
    const int bq8_offset = QR4_K * ((iqs / 2) / (QK8_1 / 4));

    // Load 4-bit quantized values (two int32s = 8 bytes = 32 x 4-bit values)
    const int* q4 = (const int*)(bq4_data + 16 + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];  // +16 bytes offset (second half of sub-block)

    // Unpack scales using uint16 method (llama.cpp: branchless for j < 2)
    const unsigned short* scales = (const unsigned short*)(bq4_data + 4);
    unsigned short aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3fu;
        aux[1] = scales[j + 2] & 0x3f3fu;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0fu) | ((scales[j - 2] & 0xc0c0u) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0fu) | ((scales[j + 0] & 0xc0c0u) >> 2);
    }
    const unsigned char* sc = (const unsigned char*)aux;
    const unsigned char* m  = sc + 2;

    // Load Q8_1 data for QR4_K=2 blocks
    for (int i = 0; i < QR4_K; ++i) {
        const unsigned char* bq8i = bq8_base + (size_t)(bq8_offset + i) * 36;
        d8[i] = __half2float(*(const __half*)bq8i);
        const int* q8 = (const int*)(bq8i + 4 + 4 * ((iqs / 2) % 4));
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    // vec_dot_q4_K_q8_1_impl_vmmq: compute dot product
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float dm_d = __half2float(*(const __half*)(bq4_data + 0));
    const float dm_m = __half2float(*(const __half*)(bq4_data + 2));
    return dm_d * sumf_d - dm_m * sumf_m;
}

// ─── gemv_q4k_q8_w64: Q4_K × Q8_1 GEMV (wave64, 64 threads/row) ──────────
//
// Direct port of llama.cpp mul_mat_vec_q for RDNA2 (ncols_dst=1, nwarps=1).
// Each wavefront of 64 threads processes one output row.
// 64 threads / (qi/vdr=16) = 4 Q4K blocks per loop iteration.
//
// Grid:  (N, M, 1)
// Block: (64, 1, 1) — one wave64 per row
// ────────────────────────────────────────────────────────────────────────────

__launch_bounds__(64, 1)
extern "C" __global__ void gemv_q4k_q8_w64(
    const unsigned char* __restrict__ w_data,  // [N * nblk * 144]
    const unsigned char* __restrict__ x_q8,    // [M * (K/32) * 36]
    float*               __restrict__ out,     // [M, N]
    int M, int N, int K,
    float beta
) {
    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    if (row >= N || mi >= M) return;

    const int tid = threadIdx.x;  // 0..63
    const int nblk = K / QK_K;
    const int blocks_per_iter = VDR * 64 / QI4_K;  // = 2 * 64 / 32 = 4

    const unsigned char* __restrict__ row_w = w_data + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * (K / 32) * 36;

    float tmp = 0.0f;

    for (int kbx = tid / (QI4_K / VDR); kbx < nblk; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K / QK8_1);  // = kbx * 8
        const int kqs = VDR * (tid % (QI4_K / VDR));  // = 2 * (tid % 16)
        tmp += vec_dot_q4k_q8_1(
            row_w + (size_t)kbx * 144,
            x_base + (size_t)kby * 36,
            kqs
        );
    }

    // wave64 reduction: 64-wide shuffle
    for (int off = 32; off >= 1; off >>= 1)
        tmp += __shfl_xor(tmp, off);

    if (tid == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], tmp) : tmp;
    }
}

// ─── fused_gate_up_silu_q4k_w64: Fused MLP gate+up+SiLU (wave64) ──────────
//
// Computes: out[row] = SiLU(dot(gate_w[row], x)) * dot(up_w[row], x)
// Both gate and up projections in a single kernel, sharing the Q8 input.
// Each wave64 (64 threads) handles one output row.
//
// Grid:  (N, 1, 1)
// Block: (64, 1, 1) — one wave64 per row
// ────────────────────────────────────────────────────────────────────────────

__launch_bounds__(64, 1)
extern "C" __global__ void fused_gate_up_silu_q4k_w64(
    const unsigned char* __restrict__ gate_data,  // [N * nblk * 144]
    const unsigned char* __restrict__ up_data,    // [N * nblk * 144]
    const unsigned char* __restrict__ x_q8,       // [(K/32) * 36]
    float*               __restrict__ out,        // [N]
    int N, int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;
    const int nblk = K / QK_K;
    const int blocks_per_iter = VDR * 64 / QI4_K;  // = 4

    const unsigned char* __restrict__ gate_row = gate_data + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ up_row   = up_data   + (size_t)row * nblk * 144;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (int kbx = tid / (QI4_K / VDR); kbx < nblk; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K / QK8_1);
        const int kqs = VDR * (tid % (QI4_K / VDR));
        const unsigned char* q8_base = x_q8 + (size_t)kby * 36;

        acc_gate += vec_dot_q4k_q8_1(gate_row + (size_t)kbx * 144, q8_base, kqs);
        acc_up   += vec_dot_q4k_q8_1(up_row   + (size_t)kbx * 144, q8_base, kqs);
    }

    // wave64 reduction
    for (int off = 32; off >= 1; off >>= 1) {
        acc_gate += __shfl_xor(acc_gate, off);
        acc_up   += __shfl_xor(acc_up, off);
    }

    if (tid == 0) {
        float silu_gate = acc_gate / (1.0f + expf(-acc_gate));
        out[row] = silu_gate * acc_up;
    }
}

// ─── fused_qkv_q4k_w64: Fused Q/K/V projection (wave64) ──────────────────
//
// Computes all three projections (Q, K, V) in a single kernel launch.
// Virtual row indexing maps rows to the appropriate weight matrix.
// Each wave64 handles one output row, with Q8 input shared via LDS.
//
// Grid:  (ceil(total_N / ROWS_PER_BLOCK), 1, 1)
// Block: (64, ROWS_PER_BLOCK, 1)
// Shared: nq8 * 36 bytes for Q8 input
// ────────────────────────────────────────────────────────────────────────────

#define FUSED_QKV_ROWS_W64 4  // 4 rows per block × 64 threads = 256 threads

__launch_bounds__(256, 1)
extern "C" __global__ void fused_qkv_q4k_w64(
    const unsigned char* __restrict__ wq_data,
    const unsigned char* __restrict__ wk_data,
    const unsigned char* __restrict__ wv_data,
    const unsigned char* __restrict__ x_q8,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    int Nq, int Nk, int Nv, int K
) {
    const int total_N = Nq + Nk + Nv;
    const int vrow = blockIdx.x * FUSED_QKV_ROWS_W64 + threadIdx.y;
    const int tid = threadIdx.x;  // 0..63
    const int nblk = K / QK_K;
    const int nq8 = K / 32;

    // Cooperative load of Q8 input into shared memory
    extern __shared__ unsigned char x_shared[];
    const int flat_tid = threadIdx.y * 64 + tid;  // 0..255
    for (int i = flat_tid; i < nq8 * 36; i += FUSED_QKV_ROWS_W64 * 64) {
        x_shared[i] = x_q8[i];
    }
    __syncthreads();

    if (vrow >= total_N) return;

    // Determine which matrix this virtual row belongs to
    const unsigned char* __restrict__ row_w;
    float* out_ptr;
    const float* bias_ptr;
    int local_row;
    if (vrow < Nq) {
        local_row = vrow;
        row_w = wq_data + (size_t)local_row * nblk * 144;
        out_ptr = out_q;
        bias_ptr = bias_q;
    } else if (vrow < Nq + Nk) {
        local_row = vrow - Nq;
        row_w = wk_data + (size_t)local_row * nblk * 144;
        out_ptr = out_k;
        bias_ptr = bias_k;
    } else {
        local_row = vrow - Nq - Nk;
        row_w = wv_data + (size_t)local_row * nblk * 144;
        out_ptr = out_v;
        bias_ptr = bias_v;
    }

    const int blocks_per_iter = VDR * 64 / QI4_K;  // = 4
    float tmp = 0.0f;

    for (int kbx = tid / (QI4_K / VDR); kbx < nblk; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K / QK8_1);
        const int kqs = VDR * (tid % (QI4_K / VDR));
        // Use shared memory for Q8 input
        tmp += vec_dot_q4k_q8_1(
            row_w + (size_t)kbx * 144,
            x_shared + (size_t)kby * 36,
            kqs
        );
    }

    // wave64 reduction
    for (int off = 32; off >= 1; off >>= 1)
        tmp += __shfl_xor(tmp, off);

    if (tid == 0) {
        float result = tmp;
        if (bias_ptr) result += bias_ptr[local_row];
        out_ptr[local_row] = result;
    }
}
#undef FUSED_QKV_ROWS_W64

// ─── gemv_q6k_q8_w64: Q6_K × Q8_1 GEMV (wave64) ──────────────────────────
//
// Q6_K: 210 bytes per 256 values. ql[128] + qh[64] + scales[16] + d[2]
// Each wave64 processes one row, iterating over all blocks.
//
// Grid:  (N, M, 1)
// Block: (64, 1, 1)
// ────────────────────────────────────────────────────────────────────────────

__launch_bounds__(64, 1)
extern "C" __global__ void gemv_q6k_q8_w64(
    const unsigned char* __restrict__ w_data,
    const unsigned char* __restrict__ x_q8,
    float*               __restrict__ out,
    int M, int N, int K,
    float beta
) {
    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    if (row >= N || mi >= M) return;

    const int tid = threadIdx.x;  // 0..63
    const int nblk = K / 256;
    const unsigned char* __restrict__ row_w = w_data + (size_t)row * nblk * 210;
    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * (K / 32) * 36;

    float acc = 0.0f;

    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* bp = row_w + blk * 210;
        const unsigned char* ql = bp;
        const unsigned char* qh = bp + 128;
        const signed char* scales = (const signed char*)(bp + 192);
        const float d6 = __half2float(*(const __half*)(bp + 208));

        // Each thread handles positions across sub-groups
        // 8 sub-groups × 32 values = 256. With 64 threads, each handles ~4 values.
        // But simpler: iterate over sub-groups, each thread handles one value per sub
        // Only first 32 threads have work in each sub-group (32 values each)
        // With 64 threads: process 2 sub-groups per iteration
        for (int q8_pair = tid / 32; q8_pair < 4; q8_pair += 1) {
            const int lane = tid & 31;
            const int q8_sub_a = q8_pair * 2;
            const int q8_sub_b = q8_pair * 2 + 1;

            // Sub-group A
            {
                const int n_group = q8_sub_a >> 2;
                const int sub = q8_sub_a & 3;
                const unsigned char* ql_g = ql + n_group * 64;
                const unsigned char* qh_g = qh + n_group * 32;
                const signed char* sc_g = scales + n_group * 8;
                const int is = lane >> 4;
                const float sc = (float)sc_g[is + sub * 2];
                int q_val;
                switch (sub) {
                    case 0: q_val = (int)(ql_g[lane] & 0xF) | (int)(((qh_g[lane] >> 0) & 3) << 4); break;
                    case 1: q_val = (int)(ql_g[32 + lane] & 0xF) | (int)(((qh_g[lane] >> 2) & 3) << 4); break;
                    case 2: q_val = (int)((ql_g[lane] >> 4)) | (int)(((qh_g[lane] >> 4) & 3) << 4); break;
                    default: q_val = (int)((ql_g[32 + lane] >> 4)) | (int)(((qh_g[lane] >> 6) & 3) << 4); break;
                }
                q_val -= 32;
                const int q8_blk_idx = blk * 8 + q8_sub_a;
                const unsigned char* q8_bp = x_base + (size_t)q8_blk_idx * 36;
                const float d8 = __half2float(*(const __half*)(q8_bp + 0));
                const int q8_val = (int)((const signed char*)(q8_bp + 4))[lane];
                float fval = sc * (float)(q_val * q8_val);

                // Partial warp reduce (within 32 lanes of our half-wave)
                for (int off = 16; off >= 1; off >>= 1)
                    fval += __shfl_down(fval, off);

                // Only lane 0 of each 32-thread group contributes
                if (lane == 0) acc += d6 * d8 * fval;
            }

            // Sub-group B
            {
                const int n_group = q8_sub_b >> 2;
                const int sub = q8_sub_b & 3;
                const unsigned char* ql_g = ql + n_group * 64;
                const unsigned char* qh_g = qh + n_group * 32;
                const signed char* sc_g = scales + n_group * 8;
                const int is = lane >> 4;
                const float sc = (float)sc_g[is + sub * 2];
                int q_val;
                switch (sub) {
                    case 0: q_val = (int)(ql_g[lane] & 0xF) | (int)(((qh_g[lane] >> 0) & 3) << 4); break;
                    case 1: q_val = (int)(ql_g[32 + lane] & 0xF) | (int)(((qh_g[lane] >> 2) & 3) << 4); break;
                    case 2: q_val = (int)((ql_g[lane] >> 4)) | (int)(((qh_g[lane] >> 4) & 3) << 4); break;
                    default: q_val = (int)((ql_g[32 + lane] >> 4)) | (int)(((qh_g[lane] >> 6) & 3) << 4); break;
                }
                q_val -= 32;
                const int q8_blk_idx = blk * 8 + q8_sub_b;
                const unsigned char* q8_bp = x_base + (size_t)q8_blk_idx * 36;
                const float d8 = __half2float(*(const __half*)(q8_bp + 0));
                const int q8_val = (int)((const signed char*)(q8_bp + 4))[lane];
                float fval = sc * (float)(q_val * q8_val);

                for (int off = 16; off >= 1; off >>= 1)
                    fval += __shfl_down(fval, off);

                if (lane == 0) acc += d6 * d8 * fval;
            }
        }
    }

    // Cross-half-wave reduction (tid 0 and tid 32 have partial sums)
    // In wave64: shuffle across all 64 threads
    acc += __shfl_xor(acc, 32);

    if (tid == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], acc) : acc;
    }
}
