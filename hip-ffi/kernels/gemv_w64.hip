// gemv_w64.hip — Wave64 GEMV kernels using IARA's data access pattern
//
// Compiled with -mwavefrontsize64 (explicit wave64 on RDNA2).
// Uses the same Q4_K/Q8_1 data access pattern as the proven wave32 kernels,
// but with 64 threads per row for better memory bandwidth utilization.
//
// Cross-half communication uses LDS (shared memory) which is guaranteed
// correct on all RDNA2 hardware regardless of wavefront mode.
// readlane-based approach was abandoned due to compiler codegen issues
// (v_trunc_f32 emission inside conditional blocks).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ─── Helpers ────────────────────────────────────────────────────────────────

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

__device__ __forceinline__ float fp16_load(const void* ptr) {
    return __half2float(*(const __half*)ptr);
}

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

// ─── gemv_q4k_q8_w64 ────────────────────────────────────────────────────────
// Grid:  (N, M, 1)  Block: (64, 1, 1)
// Half 0 (tid 0-31): even blocks. Half 1 (tid 32-63): odd blocks.
// Cross-half sum via LDS (2 floats).

__launch_bounds__(64, 1)
extern "C" __global__ void gemv_q4k_q8_w64(
    const unsigned char* __restrict__ w_data,
    const unsigned char* __restrict__ x_q8,
    float*               __restrict__ out,
    int M, int N, int K,
    float beta
) {
    const int row = blockIdx.x;
    const int mi  = blockIdx.y;
    if (row >= N || mi >= M) return;

    const int tid   = threadIdx.x;   // 0..63
    const int half  = tid >> 5;      // 0 or 1
    const int lane  = tid & 31;      // 0..31 within half
    const int group  = lane >> 3;    // 0..3
    const int lane_g = lane & 7;     // 0..7
    const int nblk  = K / 256;

    const unsigned char* __restrict__ row_w  = w_data + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * (K / 32) * 36;

    float acc = 0.0f;

    for (int blk = half; blk < nblk; blk += 2) {
        const unsigned char* bp = row_w + blk * 144;
        const float d    = fp16_load(bp + 0);
        const float dmin = fp16_load(bp + 2);

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        unpack_scale_min_k4(2 * group,     bp + 4, &sc_lo, &m_lo);
        unpack_scale_min_k4(2 * group + 1, bp + 4, &sc_hi, &m_hi);

        const int q4    = *(const int*)(bp + 16 + group * 32 + lane_g * 4);
        const int q4_lo = q4 & 0x0F0F0F0F;
        const int q4_hi = (q4 >> 4) & 0x0F0F0F0F;

        const int q8_lo_idx = blk * 8 + 2 * group;
        const unsigned char* q8_lo_bp = x_base + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = q8_lo_bp + 36;

        const float d8_lo = fp16_load(q8_lo_bp);
        const float d8_hi = fp16_load(q8_hi_bp);
        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        const int dot_lo = dp4a(q4_lo, u_lo, 0);
        const int dot_hi = dp4a(q4_hi, u_hi, 0);
        const int sum_lo = dp4a(0x01010101, u_lo, 0);
        const int sum_hi = dp4a(0x01010101, u_hi, 0);

        acc += d8_lo * (d * (float)sc_lo * (float)dot_lo - dmin * (float)m_lo * (float)sum_lo);
        acc += d8_hi * (d * (float)sc_hi * (float)dot_hi - dmin * (float)m_hi * (float)sum_hi);
    }

    // Step 1: Reduce within each 32-lane half
    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off, 32);

    // Step 2: cross-half sum via LDS (2 floats: slot 0=half0 sum, slot 1=half1 sum)
    __shared__ float lds[2];
    if (lane == 0) lds[half] = acc;
    __syncthreads();

    if (tid == 0) {
        acc = lds[0] + lds[1];
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], acc) : acc;
    }
}

// ─── fused_gate_up_silu_q4k_w64 ─────────────────────────────────────────────
// Grid: (N, 1, 1)  Block: (64, 1, 1)

__launch_bounds__(64, 1)
extern "C" __global__ void fused_gate_up_silu_q4k_w64(
    const unsigned char* __restrict__ gate_data,
    const unsigned char* __restrict__ up_data,
    const unsigned char* __restrict__ x_q8,
    float*               __restrict__ out,
    int N, int K
) {
    const int row   = blockIdx.x;
    if (row >= N) return;

    const int tid    = threadIdx.x;
    const int half   = tid >> 5;
    const int lane   = tid & 31;
    const int group  = lane >> 3;
    const int lane_g = lane & 7;
    const int nblk   = K / 256;

    const unsigned char* __restrict__ gate_row = gate_data + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ up_row   = up_data   + (size_t)row * nblk * 144;
    const unsigned char* __restrict__ x_base   = x_q8;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (int blk = half; blk < nblk; blk += 2) {
        // Gate weights
        const unsigned char* gbp = gate_row + blk * 144;
        const float gd    = fp16_load(gbp + 0);
        const float gdmin = fp16_load(gbp + 2);
        unsigned char gsc_lo, gm_lo, gsc_hi, gm_hi;
        unpack_scale_min_k4(2 * group,     gbp + 4, &gsc_lo, &gm_lo);
        unpack_scale_min_k4(2 * group + 1, gbp + 4, &gsc_hi, &gm_hi);
        const int gq4 = *(const int*)(gbp + 16 + group * 32 + lane_g * 4);

        // Up weights
        const unsigned char* ubp = up_row + blk * 144;
        const float ud    = fp16_load(ubp + 0);
        const float udmin = fp16_load(ubp + 2);
        unsigned char usc_lo, um_lo, usc_hi, um_hi;
        unpack_scale_min_k4(2 * group,     ubp + 4, &usc_lo, &um_lo);
        unpack_scale_min_k4(2 * group + 1, ubp + 4, &usc_hi, &um_hi);
        const int uq4 = *(const int*)(ubp + 16 + group * 32 + lane_g * 4);

        // Shared Q8 input
        const int q8_lo_idx = blk * 8 + 2 * group;
        const unsigned char* q8_lo_bp = x_base + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = q8_lo_bp + 36;
        const float d8_lo = fp16_load(q8_lo_bp);
        const float d8_hi = fp16_load(q8_hi_bp);
        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        const int sum_lo = dp4a(0x01010101, u_lo, 0);
        const int sum_hi = dp4a(0x01010101, u_hi, 0);

        // Gate
        const int gdot_lo = dp4a(gq4 & 0x0F0F0F0F, u_lo, 0);
        const int gdot_hi = dp4a((gq4 >> 4) & 0x0F0F0F0F, u_hi, 0);
        acc_gate += d8_lo * (gd * (float)gsc_lo * (float)gdot_lo - gdmin * (float)gm_lo * (float)sum_lo);
        acc_gate += d8_hi * (gd * (float)gsc_hi * (float)gdot_hi - gdmin * (float)gm_hi * (float)sum_hi);

        // Up
        const int udot_lo = dp4a(uq4 & 0x0F0F0F0F, u_lo, 0);
        const int udot_hi = dp4a((uq4 >> 4) & 0x0F0F0F0F, u_hi, 0);
        acc_up += d8_lo * (ud * (float)usc_lo * (float)udot_lo - udmin * (float)um_lo * (float)sum_lo);
        acc_up += d8_hi * (ud * (float)usc_hi * (float)udot_hi - udmin * (float)um_hi * (float)sum_hi);
    }

    // Reduce within each 32-lane half
    for (int off = 16; off >= 1; off >>= 1) {
        acc_gate += __shfl_down(acc_gate, off, 32);
        acc_up   += __shfl_down(acc_up,   off, 32);
    }

    // Cross-half sum via LDS (4 floats: gate[half] and up[half])
    __shared__ float lds_gate[2], lds_up[2];
    if (lane == 0) {
        lds_gate[half] = acc_gate;
        lds_up[half]   = acc_up;
    }
    __syncthreads();

    if (tid == 0) {
        acc_gate = lds_gate[0] + lds_gate[1];
        acc_up   = lds_up[0]   + lds_up[1];
        float silu_gate = acc_gate / (1.0f + expf(-acc_gate));
        out[row] = silu_gate * acc_up;
    }
}

// ─── fused_qkv_q4k_w64 ───────────────────────────────────────────────────────
// Grid: (ceil(total_N/4), 1, 1)  Block: (64, 4, 1)
// threadIdx.y = row within block; threadIdx.x = 0..63 per row (wave64)
// Cross-half via LDS: 4 rows × 2 halves = 8 float slots (appended after x_shared)

#define FUSED_QKV_ROWS_W64 4

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
    const int tid  = threadIdx.x;  // 0..63
    const int half  = tid >> 5;
    const int lane  = tid & 31;
    const int group  = lane >> 3;
    const int lane_g = lane & 7;
    const int nblk  = K / 256;
    const int nq8   = K / 32;

    // Cooperative load of Q8 input into shared memory (all 256 threads)
    // Layout: [nq8*36 bytes x_shared] + [4*2*4 bytes LDS for cross-half sums]
    extern __shared__ unsigned char x_shared[];

    const int flat_tid = threadIdx.y * 64 + tid;
    for (int i = flat_tid; i < nq8 * 36; i += FUSED_QKV_ROWS_W64 * 64)
        x_shared[i] = x_q8[i];
    __syncthreads();

    // LDS slots for cross-half sums: [row][half]
    float* lds_acc = (float*)(x_shared + nq8 * 36);

    if (vrow >= total_N) {
        // Still need to participate in syncthreads below
        // Write dummy value to prevent uninitialized read
        if (lane == 0) lds_acc[threadIdx.y * 2 + half] = 0.0f;
        __syncthreads();
        return;
    }

    const unsigned char* __restrict__ row_w;
    float* out_ptr;
    const float* bias_ptr;
    int local_row;
    if (vrow < Nq) {
        local_row = vrow;
        row_w    = wq_data + (size_t)local_row * nblk * 144;
        out_ptr  = out_q;
        bias_ptr = bias_q;
    } else if (vrow < Nq + Nk) {
        local_row = vrow - Nq;
        row_w    = wk_data + (size_t)local_row * nblk * 144;
        out_ptr  = out_k;
        bias_ptr = bias_k;
    } else {
        local_row = vrow - Nq - Nk;
        row_w    = wv_data + (size_t)local_row * nblk * 144;
        out_ptr  = out_v;
        bias_ptr = bias_v;
    }

    float acc = 0.0f;

    for (int blk = half; blk < nblk; blk += 2) {
        const unsigned char* bp = row_w + blk * 144;
        const float d    = fp16_load(bp + 0);
        const float dmin = fp16_load(bp + 2);

        unsigned char sc_lo, m_lo, sc_hi, m_hi;
        unpack_scale_min_k4(2 * group,     bp + 4, &sc_lo, &m_lo);
        unpack_scale_min_k4(2 * group + 1, bp + 4, &sc_hi, &m_hi);

        const int q4    = *(const int*)(bp + 16 + group * 32 + lane_g * 4);
        const int q8_lo_idx = blk * 8 + 2 * group;
        const unsigned char* q8_lo_bp = x_shared + (size_t)q8_lo_idx * 36;
        const unsigned char* q8_hi_bp = q8_lo_bp + 36;

        const float d8_lo = fp16_load(q8_lo_bp);
        const float d8_hi = fp16_load(q8_hi_bp);
        const int u_lo = *(const int*)(q8_lo_bp + 4 + lane_g * 4);
        const int u_hi = *(const int*)(q8_hi_bp + 4 + lane_g * 4);

        const int dot_lo = dp4a(q4 & 0x0F0F0F0F, u_lo, 0);
        const int dot_hi = dp4a((q4 >> 4) & 0x0F0F0F0F, u_hi, 0);
        const int sum_lo = dp4a(0x01010101, u_lo, 0);
        const int sum_hi = dp4a(0x01010101, u_hi, 0);

        acc += d8_lo * (d * (float)sc_lo * (float)dot_lo - dmin * (float)m_lo * (float)sum_lo);
        acc += d8_hi * (d * (float)sc_hi * (float)dot_hi - dmin * (float)m_hi * (float)sum_hi);
    }

    // Reduce within each 32-lane half
    for (int off = 16; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off, 32);

    // Cross-half via LDS
    if (lane == 0) lds_acc[threadIdx.y * 2 + half] = acc;
    __syncthreads();

    if (tid == 0) {
        acc = lds_acc[threadIdx.y * 2 + 0] + lds_acc[threadIdx.y * 2 + 1];
        float result = acc;
        if (bias_ptr) result += bias_ptr[local_row];
        out_ptr[local_row] = result;
    }
}
#undef FUSED_QKV_ROWS_W64

// ─── gemv_q6k_q8_w64 ────────────────────────────────────────────────────────
// Grid: (N, M, 1)  Block: (64, 1, 1)
// Half 0: sub-groups 0-3; Half 1: sub-groups 4-7 per Q6K block.

__launch_bounds__(64, 1)
extern "C" __global__ void gemv_q6k_q8_w64(
    const unsigned char* __restrict__ w_data,
    const unsigned char* __restrict__ x_q8,
    float*               __restrict__ out,
    int M, int N, int K,
    float beta
) {
    const int row   = blockIdx.x;
    const int mi    = blockIdx.y;
    if (row >= N || mi >= M) return;

    const int tid   = threadIdx.x;
    const int half  = tid >> 5;
    const int lane  = tid & 31;
    const int nblk  = K / 256;

    const unsigned char* __restrict__ row_w  = w_data + (size_t)row * nblk * 210;
    const unsigned char* __restrict__ x_base = x_q8 + (size_t)mi * (K / 32) * 36;

    float acc = 0.0f;

    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* bp     = row_w + blk * 210;
        const unsigned char* ql     = bp;
        const unsigned char* qh     = bp + 128;
        const signed char*   scales = (const signed char*)(bp + 192);
        const float d6 = fp16_load(bp + 208);

        // Half 0 handles sub-groups 0-3; Half 1 handles sub-groups 4-7
        for (int q8_sub = half * 4; q8_sub < half * 4 + 4; q8_sub++) {
            const int n_group = q8_sub >> 2;
            const int sub     = q8_sub & 3;

            const unsigned char* ql_g = ql + n_group * 64;
            const unsigned char* qh_g = qh + n_group * 32;
            const signed char*   sc_g = scales + n_group * 8;

            const int is = lane >> 4;
            const float sc = (float)sc_g[is + sub * 2];

            int q_val;
            switch (sub) {
                case 0:  q_val = (int)( ql_g[lane]      & 0xF) | (int)(((qh_g[lane] >> 0) & 3) << 4); break;
                case 1:  q_val = (int)( ql_g[32 + lane] & 0xF) | (int)(((qh_g[lane] >> 2) & 3) << 4); break;
                case 2:  q_val = (int)((ql_g[lane]      >> 4)) | (int)(((qh_g[lane] >> 4) & 3) << 4); break;
                default: q_val = (int)((ql_g[32 + lane] >> 4)) | (int)(((qh_g[lane] >> 6) & 3) << 4); break;
            }
            q_val -= 32;

            const int q8_blk_idx = blk * 8 + q8_sub;
            const unsigned char* q8_bp = x_base + (size_t)q8_blk_idx * 36;
            const float d8 = fp16_load(q8_bp);
            const int q8_val = (int)((const signed char*)(q8_bp + 4))[lane];

            float fval = sc * (float)(q_val * q8_val);
            // Reduce within 32-lane half (width=32 prevents crossing into other half)
            for (int off = 16; off >= 1; off >>= 1)
                fval += __shfl_down(fval, off, 32);

            if (lane == 0) acc += d6 * d8 * fval;
        }
    }

    // Cross-half sum via LDS
    __shared__ float lds[2];
    if (lane == 0) lds[half] = acc;
    __syncthreads();

    if (tid == 0) {
        acc = lds[0] + lds[1];
        const int out_idx = mi * N + row;
        out[out_idx] = (beta != 0.0f) ? __fmaf_rn(beta, out[out_idx], acc) : acc;
    }
}
