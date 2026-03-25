use std::process::Command;
use anyhow::Result;
use rocm_rs::rocm_smi::RocmSmi;

/// Propriocepção: O motor sentindo o próprio hardware (AMD ROCm Nativo via rocm-rs).
pub struct Proprioception;

impl Proprioception {
    /// Retorna o estresse térmico e de memória do sistema (0.0 a 1.0).
    pub fn get_hardware_stress() -> f64 {
        // Tenta usar a biblioteca nativa ROCm-SMI (Só funciona no container ROCm)
        if let Ok(smi) = RocmSmi::init() {
            if let Ok(devices) = smi.get_device_count() {
                if devices > 0 {
                    let device = 0;
                    let temp = smi.get_device_temperature(device).unwrap_or(45) as f64;
                    let mem_info = smi.get_device_memory_usage(device).unwrap_or((0, 1));
                    
                    let temp_stress = ((temp - 40.0) / 45.0).clamp(0.0, 1.0);
                    let mem_stress = (mem_info.0 as f64 / mem_info.1 as f64).clamp(0.0, 1.0);
                    
                    return (temp_stress * 0.6 + mem_stress * 0.4).clamp(0.0, 1.0);
                }
            }
        }

        // Fallback robusto via comando se a lib falhar por qualquer motivo
        let mem = Self::get_mem_usage().unwrap_or(0.5);
        (mem * 0.4 + 0.3 * 0.6).clamp(0.0, 1.0)
    }

    fn get_mem_usage() -> Result<f64> {
        let output = Command::new("free").arg("-m").output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        if lines.len() > 1 {
            let parts: Vec<&str> = lines[1].split_whitespace().collect();
            if parts.len() > 2 {
                let total: f64 = parts[1].parse()?;
                let used: f64 = parts[2].parse()?;
                return Ok(used / total);
            }
        }
        anyhow::bail!("Parse error")
    }
}
