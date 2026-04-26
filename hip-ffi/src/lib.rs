//! Minimal HIP FFI for AMD RDNA2 (GFX1032 — RX 6750 XT).
//! Raw bindings via bindgen from hip_runtime_api.h (host-side only).
//! Sem compilação de kernels GPU, sem target amdgcn, sem subcrates.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

pub mod raw {
    include!(concat!(env!("OUT_DIR"), "/hip_bindings.rs"));
}

pub mod hsa_raw {
    include!(concat!(env!("OUT_DIR"), "/hsa_bindings.rs"));
}

pub mod safe;
pub mod tensor;
pub mod blas;
pub mod solver;
pub mod kernels;
pub mod hsa_dispatch;

pub use safe::*;
pub use tensor::{GpuTensor, GpuTensorBF16};
pub use blas::{BlasHandle, gemm_bf16_f32, sgemm_strided, sgemm_strided_batched};
pub use solver::{SolverHandle, qr_f32, svd_f32};
pub use kernels::GpuKernels;
