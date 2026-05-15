// mmvq_llama.hip — Q4_K × Q8_1 GEMV kernel ported from llama.cpp (MIT)
//
// Source: https://github.com/ggml-org/llama.cpp
//   ggml/src/ggml-cuda/{mmvq.cu,vecdotq.cuh,common.cuh,ggml-common.h}
// Adapted for RDNA2 (gfx1030/gfx1031), wave32 native, 1 warp per row, 1 row per block.
// Decode-only path: ncols_dst=1, no fusion, no batch.
//
// Signature matches llm-lab's existing `gemv_q4k_q8_w64` for drop-in replacement:
//   void mmvq_q4k_llama(const unsigned char* w, const unsigned char* x_q8,
//                       float* out, int M, int N, int K, float beta)
//   w:     [N, K/256] blocks of block_q4_K (144 bytes each)
//   x_q8:  [M, K/32]  blocks of block_q8_1 (36 bytes each)
//   out:   [M, N] f32; layout out[mi*N + row]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ─── block formats (must match llama.cpp ggml-common.h) ─────────────────────

#define QK_K          256
#define K_SCALE_SIZE  12
#define QR4_K         2
#define QI4_K         (QK_K / (4 * QR4_K))   // = 32
#define QK8_1         32
#define QI8_1         (QK8_1 / 4)            // = 8 ints per block

// Q4_K super-block: 144 bytes total
struct block_q4_K {
    __half  d;
    __half  dmin;
    uint8_t scales[K_SCALE_SIZE];             // 12
    uint8_t qs[QK_K / 2];                     // 128 (4-bit packed)
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K layout mismatch");

// Q8_1 block: 36 bytes total
struct block_q8_1 {
    __half d;
    __half s;
    int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 layout mismatch");

// ─── helpers ────────────────────────────────────────────────────────────────

__device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

__device__ __forceinline__ float warp_reduce_sum_f32(float x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, 32);
    }
    return x;
}

// ─── vec_dot kernel bodies (from llama.cpp vecdotq.cuh, MIT) ────────────────

__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
    const __half2 dm4, const float* __restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i + 1], ggml_cuda_dp4a(v0i, u[2*i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i + 1], ggml_cuda_dp4a(0x01010101, u[2*i + 0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
    int kbx, int iqs)
{
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;

    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t* scales = (const uint16_t*) bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __half2float(bq8i->d);

        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2*i + 0] = q8[0];
        u[2*i + 1] = q8[4];
    }

    // Load dm as half2 (two halves packed)
    const __half2 dm4 = *reinterpret_cast<const __half2*>(&bq4_K->d);
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm4, d8);
}

// ─── kernel: 1 warp per row, 32 threads, decode (ncols=1, no fusion) ────────
//
// Grid:  (N, M, 1)     one block per output row, per batch row
// Block: (32, 1, 1)    single wave32
// Each thread computes partial sum over a subset of Q4_K super-blocks.
// Then warp-reduce and thread 0 writes result.

__launch_bounds__(32, 8)
extern "C" __global__ void mmvq_q4k_llama(
    const unsigned char* __restrict__ w_data,   // Q4_K blocks
    const unsigned char* __restrict__ x_q8,     // Q8_1 blocks
    float*               __restrict__ out,
    int M, int N, int K, float beta)
{
    const int row = blockIdx.x;   // 0..N-1
    const int mi  = blockIdx.y;   // 0..M-1
    if (row >= N || mi >= M) return;

    const int tid = threadIdx.x;  // 0..31

    const int qk  = QK_K;
    const int qi  = QI4_K;           // 32
    const int vdr = 2;                // VDR_Q4_K_Q8_1_MMVQ
    const int blocks_per_row_x = K / qk;
    const int blocks_per_iter  = vdr * 32 / qi;  // = 2

    const block_q4_K* vx = (const block_q4_K*)(w_data + (size_t) row * blocks_per_row_x * sizeof(block_q4_K));
    const block_q8_1*  y = (const block_q8_1*)(x_q8   + (size_t) mi  * (K / QK8_1) * sizeof(block_q8_1));

    float tmp = 0.0f;

    // tid / (qi/vdr) = tid / 16 = 0 or 1   → kbx index within iter
    // tid % (qi/vdr) = tid % 16            → kqs offset
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);   // y block aligned with kbx (= kbx * 8)
        const int kqs = vdr * (tid % (qi / vdr));

        tmp += vec_dot_q4_K_q8_1(vx, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum_f32(tmp);

    if (tid == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], tmp) : tmp;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q6_K × Q8_1 GEMV — ported from llama.cpp vecdotq.cuh (MIT).
// Q6_K block: ql[128] + qh[64] + scales[16 int8] + d[fp16] = 210 bytes.
// QR6_K=2, QI6_K=32. VDR_Q6_K_Q8_1_MMVQ=1.
// Used for down_proj (the largest MLP weight, K=intermediate_size).
// ═══════════════════════════════════════════════════════════════════════════

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))   // = 32

struct block_q6_K {
    uint8_t ql[QK_K/2];      // 128 — lower 4 bits
    uint8_t qh[QK_K/4];      // 64  — upper 2 bits
    int8_t  scales[QK_K/16]; // 16  — int8 scales
    __half  d;               // 2   — super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K layout mismatch");

__device__ __forceinline__ int mmvq_get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    return (int)x16[2*i32 + 0] | ((int)x16[2*i32 + 1] << 16);
}
__device__ __forceinline__ int mmvq_get_int_b4(const void* x, int i32) {
    return ((const int*) x)[i32];
}

// __vsubss4: signed saturating per-byte subtract. RDNA has no native op; emulate.
__device__ __forceinline__ int mmvq_vsubss4(int a, int b) {
    signed char* pa = (signed char*)&a;
    signed char* pb = (signed char*)&b;
    int r;
    signed char* pr = (signed char*)&r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int v = (int)pa[i] - (int)pb[i];
        pr[i] = (signed char)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
    return r;
}

__device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    int vl, int vh, const int* __restrict__ u, const int8_t* __restrict__ scales,
    const __half d, const float* __restrict__ d8)
{
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc  = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi  = mmvq_vsubss4((vil | vih), 0x20202020); // (vil|vih) - 32
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }
    return __half2float(d) * sumf;
}

__device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
    int kbx, int iqs)
{
    const block_q6_K* bq6_K = (const block_q6_K*) vbq + kbx;

    const int bq8_offset  = 2*QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift    = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = mmvq_get_int_b2(bq6_K->ql, iqs);
    const int vh = mmvq_get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t* scales = bq6_K->scales + scale_offset;

    int   u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = mmvq_get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __half2float(bq8_1[bq8_offset + 2*i].d);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// Grid (N, M, 1)  Block (32, 1, 1). 1 warp/row, decode (ncols=1).
__launch_bounds__(32, 8)
extern "C" __global__ void mmvq_q6k_llama(
    const unsigned char* __restrict__ w_data,   // Q6_K blocks
    const unsigned char* __restrict__ x_q8,     // Q8_1 blocks
    float*               __restrict__ out,
    int M, int N, int K, float beta)
{
    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    if (row >= N || mi >= M) return;

    const int tid = threadIdx.x;  // 0..31

    const int qi  = QI6_K;            // 32
    const int vdr = 1;                // VDR_Q6_K_Q8_1_MMVQ
    const int blocks_per_row_x = K / QK_K;
    const int blocks_per_iter  = vdr * 32 / qi;  // = 1

    const block_q6_K* vx = (const block_q6_K*)(w_data + (size_t) row * blocks_per_row_x * sizeof(block_q6_K));
    const block_q8_1*  y = (const block_q8_1*)(x_q8   + (size_t) mi  * (K / QK8_1) * sizeof(block_q8_1));

    float tmp = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K / QK8_1);   // = kbx * 8
        const int kqs = vdr * (tid % (qi / vdr));   // = tid
        tmp += vec_dot_q6_K_q8_1(vx, &y[kby], kbx, kqs);
    }

    tmp = warp_reduce_sum_f32(tmp);

    if (tid == 0) {
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], tmp) : tmp;
    }
}
