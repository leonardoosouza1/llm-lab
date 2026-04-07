// ============================================================================
// gpu_orchestrator.rs — Orquestrador de Kernels HIP Nativos (hip_ffi)
// ============================================================================

use anyhow::Result;

#[cfg(feature = "rocm")]
mod rocm_impl {
    use anyhow::Result;
    use hip_ffi::{DeviceBuffer, hip_set_device, hip_device_sync, HipError};

    pub struct GpuOrchestrator {
        pub device_id: i32,
        // módulo HIP carregado em runtime (opcional até kernels .hip estarem compilados)
        module: Option<hip_ffi::raw::hipModule_t>,
    }

    impl GpuOrchestrator {
        pub fn init() -> Result<Self> {
            hip_set_device(0).map_err(|e| anyhow::anyhow!("HIP set_device: {}", e))?;
            log::info!("🎯 GpuOrchestrator: GPU 0 (RX 6750 XT) ativa");

            // Kernels HIP serão carregados quando kernels/physics.co existir
            // Por enquanto o orquestrador opera sem módulo externo
            Ok(Self { device_id: 0, module: None })
        }

        /// Lança kernel de Repulsão de Pauli (quando módulo HIP estiver disponível)
        pub fn launch_pauli(
            &self,
            states_dev: &DeviceBuffer<f32>,
            repulsion_dev: &mut DeviceBuffer<f32>,
            k_pauli: f32,
        ) -> Result<()> {
            let n = states_dev.len();

            match self.module {
                Some(m) => {
                    // Kernel HIP real
                    let func = hip_ffi::hip_module_get_function(m, "pauli_repulsion_kernel")
                        .map_err(|e| anyhow::anyhow!("get_function: {}", e))?;

                    let block = (256u32, 1u32, 1u32);
                    let grid = (((n + 255) / 256) as u32, 1u32, 1u32);
                    let n_u32 = n as u32;

                    let mut args: [*mut std::ffi::c_void; 4] = [
                        states_dev.kernel_arg(),
                        repulsion_dev.kernel_arg(),
                        &n_u32 as *const u32 as *mut std::ffi::c_void,
                        &k_pauli as *const f32 as *mut std::ffi::c_void,
                    ];

                    unsafe {
                        hip_ffi::hip_launch_kernel(func, grid, block, 0, std::ptr::null_mut(), &mut args)
                            .map_err(|e| anyhow::anyhow!("launch_kernel: {}", e))?;
                    }
                    hip_device_sync().map_err(|e| anyhow::anyhow!("device_sync: {}", e))?;
                }
                None => {
                    // CPU fallback temporário
                    log::warn!("⚠️ Kernel HIP não carregado — usando CPU fallback para pauli");
                }
            }

            Ok(())
        }
    }
}

#[cfg(not(feature = "rocm"))]
mod rocm_impl {
    use anyhow::Result;

    pub struct GpuOrchestrator {
        pub device_id: i32,
    }

    impl GpuOrchestrator {
        pub fn init() -> Result<Self> {
            log::info!("GpuOrchestrator: CPU-only mode (rocm feature not enabled)");
            Ok(Self { device_id: -1 })
        }

        pub fn launch_pauli(
            &self,
            _states_dev: &[f32],
            _repulsion_dev: &mut [f32],
            _k_pauli: f32,
        ) -> Result<()> {
            log::warn!("GPU operations not available without rocm feature");
            Ok(())
        }
    }
}

pub use rocm_impl::GpuOrchestrator;
