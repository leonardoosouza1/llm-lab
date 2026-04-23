// Host-side HIP runtime API — parseado como C++17 com Clang do ROCm.
// O --resource-dir no build.rs aponta para o Clang do ROCm que tem <array>, <cstdint> etc.
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include <hipsolver/hipsolver.h>
