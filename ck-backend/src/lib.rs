//! AMD Composable Kernel GEMM Backend
//!
//! Provides fused GEMM + activation kernels via AMD CK.
//! Requires ROCm + CK headers to build with actual GPU support.

pub mod gemm;

pub use gemm::{CkActivation, CkGemmConfig};
