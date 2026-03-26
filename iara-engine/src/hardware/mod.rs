// ============================================================================
// hardware/mod.rs — Hardware Management — HIP Nativo (sem candle)
// ============================================================================

use std::env;

pub struct HardwareManager {
    pub device_id: i32,
    pub gpu_name: String,
}

impl HardwareManager {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        if env::var("ROCM_PATH").is_err() {
            env::set_var("ROCM_PATH", "/opt/rocm-6.4.4");
        }
        // GFX override para RX 6750 XT (Navi 22 / GFX1032)
        if env::var("HSA_OVERRIDE_GFX_VERSION").is_err() {
            env::set_var("HSA_OVERRIDE_GFX_VERSION", "10.3.0");
        }

        hip_ffi::hip_set_device(0)?;

        let (free, total) = hip_ffi::hip_mem_info()?;
        log::info!(
            "🎯 GPU: AMD RX 6750 XT | VRAM livre: {:.1} GB / {:.1} GB",
            free as f64 / 1e9,
            total as f64 / 1e9
        );

        Ok(Self {
            device_id: 0,
            gpu_name: "AMD RX 6750 XT (GFX1032 / RDNA2)".to_string(),
        })
    }
}
