# AMD Composable Kernel (CK) GEMM Kernels

This directory contains C++ kernel source files for fused GEMM + activation operations using AMD Composable Kernel.

## Files

- `gemm_silu.cpp` — GEMM with SiLU activation (gate projection path for MLP)
- `gemm_gelu.cpp` — GEMM with GELU activation
- `gemm_passthrough.cpp` — GEMM with no activation (attention layers)

## Prerequisites

To compile these kernels, you need:

1. **ROCm** — AMD's compute platform
2. **hipcc** — HIP compiler (comes with ROCm)
3. **Composable Kernel headers** — CK library from ROCm

## Build Integration

The `build.rs` script will:

1. Check if `composable-kernel` feature is enabled
2. Locate CK headers via environment variables or standard ROCm paths
3. Invoke hipcc to compile kernel instantiations
4. Link the compiled objects into the Rust binary

If CK headers are not found, compilation fails gracefully and `cargo check` passes without the feature flag.

## Example Kernel Structure

```cpp
// Simplified template instantiation (actual CK code is more complex)
#include "ck/tensor_operation/gpu/device/gemm_operator.hpp"

using GemmConfig = ck::tensor_operation::device::Gemm<
    ck::tensor_operation::device::GemmSpec::GemmWithSiLU,
    float,  // input type
    float   // accumulation type
>;

void launch_gemm_silu(float* a, float* b, float* c, uint32_t m, uint32_t n, uint32_t k) {
    // Launch kernel via CK device
}
```

## Conditional Compilation

- **Without feature**: Rust FFI stubs return `CkError::NotAvailable`
- **With feature**: Rust FFI calls compiled GPU kernels

This allows `cargo check` to pass even if CK is not installed, while full GPU support is available when compiling with the feature flag and CK headers present.
