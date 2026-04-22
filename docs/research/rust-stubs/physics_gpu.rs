// ============================================================================
// physics_gpu.rs — Kernels de Física para GPU AMD (GFX1032 / RDNA2)
// ============================================================================
// Os kernels HIP reais ficam em iara-engine/kernels/physics.hip
// Compilados pelo build.rs com hipcc --offload-arch=gfx1032
// Carregados em runtime via hip_ffi::hip_module_load_data
// ============================================================================

#[cfg(feature = "rocm")]
use hip_ffi::{DeviceBuffer, HipError, hip_module_load_data, hip_module_get_function, hip_launch_kernel, hip_stream_create, hip_stream_sync};

/// Binário HSA do kernel de física — compilado pelo build.rs
/// Caminho: iara-engine/kernels/physics.co (gerado em compile time)
/// Por enquanto: placeholder até o kernel .hip estar pronto
pub const PHYSICS_KERNEL: Option<&[u8]> = None; // será: Some(include_bytes!("../kernels/physics.co"))

#[cfg(feature = "rocm")]
use hip_ffi::HipError;

#[cfg(not(feature = "rocm"))]
type HipError = std::io::Error;

#[cfg(feature = "rocm")]
pub struct PhysicsGpu {
    stream: hip_ffi::raw::hipStream_t,
}

#[cfg(not(feature = "rocm"))]
pub struct PhysicsGpu {
    _phantom: std::marker::PhantomData<()>,
}

impl PhysicsGpu {
    #[cfg(feature = "rocm")]
    pub fn new() -> Result<Self, HipError> {
        let stream = hip_stream_create()?;
        Ok(Self { stream })
    }

    #[cfg(not(feature = "rocm"))]
    pub fn new() -> Result<Self, HipError> {
        Ok(Self { _phantom: std::marker::PhantomData })
    }

    /// Kernel: Repulsão de Pauli entre neurônios (CPU fallback por enquanto)
    /// TODO: mover para kernel HIP em kernels/physics.hip
    pub fn pauli_repulsion_cpu(
        states: &[f32],
        k_pauli: f32,
    ) -> Vec<f32> {
        let n = states.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut force = 0.0f32;
            for j in 0..n {
                if i != j {
                    let dist = (states[i] - states[j]).abs();
                    if dist > 0.0 && dist < 0.05 {
                        force += k_pauli / (dist * dist + 1e-4);
                    }
                }
            }
            out[i] = force;
        }
        out
    }

    /// Kernel: Malha KAN Elástica (CPU fallback por enquanto)
    /// TODO: mover para kernel HIP em kernels/physics.hip
    pub fn kan_spline_cpu(states: &[f32], weights_dna: &[f32]) -> Vec<f32> {
        let n = states.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut coupling = 0.0f32;
            for j in 0..n {
                let w = weights_dna[i * n + j];
                coupling += w * states[j] * (-states[j].abs()).exp();
            }
            out[i] = coupling;
        }
        out
    }
}

#[cfg(feature = "rocm")]
impl Drop for PhysicsGpu {
    fn drop(&mut self) {
        unsafe { hip_ffi::hip_stream_destroy(self.stream).ok(); }
    }
}
