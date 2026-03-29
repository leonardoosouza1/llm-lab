// ============================================================================
// context.rs — Contexto OpenCL com detecção de arquitetura de memória
// ============================================================================
//
// Detecta automaticamente se o dispositivo usa UMA (Unified Memory Architecture):
//   - Intel iGPU (Iris Plus, UHD, Arc): CPU e GPU compartilham a mesma RAM física
//   - GPU discreta: memória separada, requer transferências explícitas via PCIe
//
// Na UMA: buffers alocados com CL_MEM_ALLOC_HOST_PTR vivem em RAM acessível
// tanto pela CPU quanto pela GPU — não há barramento no caminho.
// ============================================================================

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::platform::get_platforms;
use compute_core::BackendError;

pub struct OclContext {
    pub device:  Device,
    pub context: Context,
    pub queue:   CommandQueue,
    /// true = Intel iGPU (ou outro UMA): CPU e GPU compartilham RAM física.
    /// false = GPU discreta: memória separada, copies via PCIe.
    pub is_uma:  bool,
}

impl OclContext {
    /// Detecta o primeiro GPU OpenCL disponível.
    /// Prefere Intel iGPU (UMA), mas aceita qualquer GPU OpenCL.
    pub fn detect() -> Result<Self, BackendError> {
        let platforms = get_platforms()
            .map_err(|e| BackendError::NotAvailable(format!("Sem plataformas OpenCL: {e}")))?;

        let mut best: Option<(Device, String)> = None;

        'outer: for platform in &platforms {
            let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) else {
                continue;
            };
            for dev_id in device_ids {
                let device = Device::new(dev_id);
                let name   = device.name().unwrap_or_default();
                let vendor = device.vendor().unwrap_or_default();

                // Prefere Intel iGPU (UMA nativo)
                if vendor.to_ascii_lowercase().contains("intel") {
                    best = Some((device, name));
                    break 'outer;
                }
                if best.is_none() {
                    best = Some((device, name));
                }
            }
        }

        let (device, name) = best
            .ok_or_else(|| BackendError::NotAvailable("Nenhuma GPU OpenCL encontrada".into()))?;

        // ── Detecta UMA via CL_DEVICE_HOST_UNIFIED_MEMORY ────────────────────
        // Retorna cl_uint (u32): 0 = não-UMA, 1 = UMA (memória unificada)
        let is_uma = device
            .host_unified_memory()
            .unwrap_or(false);

        log::info!(
            "OpenCL GPU: {} | Arquitetura: {}",
            name,
            if is_uma { "UMA (memória unificada — zero-copy)" } else { "Discreta (PCIe)" }
        );

        if is_uma {
            log::info!(
                "  → UMA ativado: uploads/downloads serão zero-copy via CL_MEM_ALLOC_HOST_PTR"
            );
        }

        let context = Context::from_device(&device)
            .map_err(|e| BackendError::NotAvailable(format!("Falha ao criar contexto OpenCL: {e}")))?;

        // Fila principal — serializa todos os dispatches de kernels
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|e| BackendError::NotAvailable(format!("Falha ao criar CommandQueue: {e}")))?;

        Ok(Self { device, context, queue, is_uma })
    }
}
