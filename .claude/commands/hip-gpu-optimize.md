# HIP/GPU Kernel Optimization for RDNA2 (RX 6750 XT / gfx1030)

You are an expert GPU kernel engineer optimizing HIP kernels for AMD RDNA2 hardware.
Apply the following knowledge when writing, reviewing, or optimizing HIP kernels.

## Hardware Specs — RX 6750 XT (Navi 22, gfx1030)

| Parameter | Value |
|---|---|
| Compute Units (CUs) | 40 |
| SIMDs per CU | 2 (SIMD32 units) |
| VGPRs per SIMD | 1024 (alloc granularity 8) |
| SGPRs per CU | 800 (max 106 per wavefront) |
| LDS per CU | 64 KB, 32 banks x 4 bytes |
| L0 Vector Cache | 16 KB per CU, 128-byte lines |
| L1 Cache | 128 KB per Shader Array (4 SAs total) |
| **L2 Infinity Cache** | **96 MB total, ~1.5-2 TB/s internal BW** |
| VRAM | 12 GB GDDR6, 192-bit, **432 GB/s peak** |
| Peak FP32 | 13.3 TFLOPS (40 CUs x 128 FLOPS/cyc x 2600 MHz) |
| Native Wave Size | 32 (SIMD32 units) |
| Cache Line | 128 bytes |
| dp4a (INT8 dot) | **YES** — `__builtin_amdgcn_sdot4(a, b, c, false)` |

## Critical Rule #1: Wave64 IS Better for Memory-Bound GEMV

Despite RDNA2 having 32-wide SIMDs, **wave64 outperforms wave32 for memory-bound kernels** (benchmarked: 88 vs 68 tok/s):

- **Scalar amortization**: Loop control, pointer math, branch instructions execute ONCE for 64 threads (vs once per 32). For GEMV with significant per-block overhead (unpack_scale_min_k4, index calculations), this saves ~30% instruction overhead.
- **Lower cache pressure**: `__launch_bounds__(64, 1)` = fewer concurrent waves = less L0 cache thrashing (L0 is only 16KB/CU).
- **Register freedom**: The compiler can use up to 256 VGPRs with unconstrained launch bounds, enabling better scheduling and zero register spills.
- **Memory coalescing**: The coalescer sees 64 addresses at once, merging requests more efficiently.

**When to use wave32:** Compute-bound kernels, kernels with very low scalar overhead, or when occupancy must be maximized (e.g., elementwise ops).

**When to use wave64:** Memory-bound GEMV/GEMM (Q4K, Q6K inference decode), kernels with significant per-block scalar overhead.

## Critical Rule #2: Occupancy — Higher Is NOT Always Better

AMD GPUOpen documentation explicitly states: "Peak occupancy does not always mean peak performance."

For memory-bound GEMV:
- **Too many waves per CU → L0 cache thrashing** (16 KB shared across all waves)
- The memory controller is already saturated; more waves just add contention
- `__launch_bounds__(64, 1)` (wave64, low occupancy) gives near-exclusive L0 access per wave

For compute-bound kernels: higher occupancy helps hide ALU latency.

## dp4a FOR QUANTIZED INFERENCE
```cpp
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}
```
- Works with both wave32 and wave64 on gfx1030
- Q4_K pattern: extract nibbles `(q4 >> (4*i)) & 0x0F0F0F0F`, dp4a against Q8 data
- Keep data as packed int32 — dequantize in registers after dp4a

## INFINITY CACHE (96 MB L2)
- Model weights (~830 MB for Qwen2.5-1.5B) do NOT fit → weight reads are VRAM-bound
- KV cache + activations DO fit → attention and activation reuse benefit from Infinity Cache
- Individual layer weights (7-22 MB) may temporarily fit, but are evicted by next layer
- Consider **nontemporal loads** for weight streaming to avoid polluting cache

## KEY BOTTLENECK: KERNEL LAUNCH OVERHEAD
- Current: 395 launches/token, ~1.5µs each = 0.6ms dispatch overhead (5.4% of 11ms total)
- Plus GPU idle gaps between launches: estimated additional 3-5%
- **Total launch overhead: ~9-10% of runtime**
- Solutions:
  - Fuse operations (RMSNorm+Quantize already fused)
  - HIP Graph (tested: slower due to graph update overhead for changing seq_len)
  - Persistent kernel approach (advanced)

## BANDWIDTH ANALYSIS
- Effective BW at 88 tok/s: ~73 GB/s = 17% of 432 GB/s peak
- llama.cpp at ~149 tok/s: ~164 GB/s = 38% of peak
- Theoretical min at 432 GB/s: 1.92 ms/token = 520 tok/s
- Gap is due to: launch overhead, cache behavior, instruction overhead per byte loaded

## OPTIMIZATION PRIORITIES (ordered by expected impact)

1. **Reduce kernel launches** — fuse more operations, reduce the 395 launches/token
2. **Nontemporal loads for weight data** — prevent weight data from polluting L0/L1 cache
3. **Prefetch next block** — overlap memory load latency with compute
4. **Section profiling** — `IARA_SECTION_PROF=1` to identify exact bottleneck kernels
5. **Vectorized loads** — `float4`/`int4` to reduce memory instruction count

## WARP REDUCTION PATTERN (for both wave32 and wave64)
```cpp
// Wave64 reduction: 6 steps
for (int off = 32; off >= 1; off >>= 1)
    acc += __shfl_down(acc, off, 64);

// Wave32 reduction: 5 steps
for (int off = 16; off >= 1; off >>= 1)
    acc += __shfl_down(acc, off);
```

## PROFILING
```bash
# Section profiling (per-section timing):
IARA_SECTION_PROF=1 ./run_iara.sh native

# ROCm profiling:
rocprof --stats --hip-trace ./binary
# Key: kernel duration, FetchSize, WriteSize, MemUnitBusy, ALUBusy, L2CacheHit
```

## Anti-Patterns to AVOID
1. **High-occupancy multi-row wave32 for memory-bound GEMV** — causes L0 thrashing, slower than wave64
2. **LDS for small data that fits in L0** — adds sync overhead + bank conflicts
3. **`__launch_bounds__(N, M)` with large M for memory-bound** — forces compiler to reduce VGPRs, may hurt scheduling
4. **Byte-by-byte memory copies** — use int32/int4 for LDS fills
5. **Assuming "fewer instructions = faster"** — for memory-bound, instruction count is secondary to memory access pattern
