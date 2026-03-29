// ============================================================================
// hardware/mod.rs — Backend de Compute: detecção automática (ROCm → OpenCL)
// ============================================================================
// Não importa hip-ffi diretamente. Usa o trait ComputeBackend de compute-core.
// O backend concreto é selecionado via feature flags:
//   --features rocm    → tenta RocmBackend primeiro
//   --features opencl  → tenta OpenClBackend em seguida
// ============================================================================

use std::sync::Arc;
use compute_core::ComputeBackend;

pub struct HardwareManager {
    pub device_id: i32,
    pub gpu_name: String,
    pub backend: Arc<dyn ComputeBackend>,
}

impl HardwareManager {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let backend = detect_backend()?;
        let name = backend.name().to_string();
        log::info!("🎯 Backend: {}", name);
        if let Some((free, total)) = backend.mem_info() {
            log::info!("   VRAM livre: {:.1} GB / {:.1} GB",
                free as f64 / 1e9, total as f64 / 1e9);
        }
        Ok(Self { device_id: 0, gpu_name: name, backend })
    }
}

pub fn detect_backend() -> Result<Arc<dyn ComputeBackend>, Box<dyn std::error::Error>> {
    // cl3 faz dlopen("libOpenCL.so") mas Ubuntu instala só libOpenCL.so.1 (sem symlink).
    // Se OPENCL_DYLIB_PATH não foi setado externamente, tentamos o caminho versionado.
    #[cfg(feature = "opencl")]
    if std::env::var("OPENCL_DYLIB_PATH").is_err() {
        for candidate in &[
            "/lib/x86_64-linux-gnu/libOpenCL.so.1",
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
            "/usr/lib/libOpenCL.so.1",
        ] {
            if std::path::Path::new(candidate).exists() {
                // SAFETY: chamado antes de qualquer thread adicional; sem race.
                unsafe { std::env::set_var("OPENCL_DYLIB_PATH", candidate); }
                log::info!("OPENCL_DYLIB_PATH → {}", candidate);
                break;
            }
        }
    }

    // Tente ROCm primeiro
    #[cfg(feature = "rocm")]
    {
        match rocm_backend::RocmBackend::probe() {
            Ok(b) => {
                log::info!("✅ ROCm backend detectado: {}", b.name());
                return Ok(Arc::new(b));
            }
            Err(e) => log::warn!("ROCm não disponível: {e}"),
        }
    }

    // Tente OpenCL (Intel iGPU, etc.)
    #[cfg(feature = "opencl")]
    {
        match opencl_backend::OpenClBackend::probe() {
            Ok(b) => {
                log::info!("✅ OpenCL backend detectado: {}", b.name());
                return Ok(Arc::new(b));
            }
            Err(e) => log::warn!("OpenCL não disponível: {e}"),
        }
    }

    Err("Nenhum backend de compute disponível. Compile com --features rocm ou --features opencl".into())
}
