// ============================================================================
// gpu_orchestrator.rs — Orquestrador de Kernels Nativos AMD (Direct Point)
// ============================================================================
// Gerencia o carregamento de módulos HIP e a execução de kernels em Rust.
// ============================================================================

use anyhow::Result;
use rocm_rs::hip::*;
use std::path::PathBuf;
use crate::physics_gpu;

pub struct GpuOrchestrator {
    pub device: Device,
    pub module: Module,
}

impl GpuOrchestrator {
    /// Inicializa a GPU e carrega os kernels compilados do physics_gpu.rs
    pub fn init() -> Result<Self> {
        // Garante que o dispositivo 0 (RX 6750 XT) está ativo
        let device = Device::new(0)?;
        device.set_current()?;

        // O binário do kernel é gerado pela macro no physics_gpu.rs em tempo de compilação/runtime
        let kernel_path = PathBuf::from(physics_gpu::AMDGPU_KERNEL_BINARY_PATH);
        
        if !kernel_path.exists() {
            anyhow::bail!("❌ Binário do kernel GPU não encontrado em: {:?}", kernel_path);
        }

        log::info!("🎯 Carregando Módulo GPU Nativo: {:?}", kernel_path);
        let module = Module::load(kernel_path)?;

        Ok(Self { device, module })
    }

    /// Executa o kernel de Repulsão de Pauli na GPU
    pub fn launch_pauli(
        &self,
        states_dev: &DeviceMemory<f32>,
        repulsion_dev: &mut DeviceMemory<f32>,
        n: usize,
        k_pauli: f32
    ) -> Result<()> {
        let function = self.module.get_function("pauli_repulsion_kernel")?;

        // Configura argumentos para o kernel (Direct Point)
        let mut kernel_args = [
            states_dev.as_kernel_arg(),
            repulsion_dev.as_kernel_arg(),
            (n as u32).as_kernel_arg(),
            k_pauli.as_kernel_arg(),
        ];

        // Configuração de threads (1D Grid)
        // Usamos blocos de 256 threads para alinhar com a arquitetura AMD RDNA2
        let block_dim = Dim3 { x: 256, y: 1, z: 1 };
        let grid_dim = Dim3 { 
            x: ((n + 255) / 256) as u32, 
            y: 1, 
            z: 1 
        };

        unsafe {
            function.launch(grid_dim, block_dim, 0, None, &mut kernel_args)?;
        }

        Ok(())
    }
}
