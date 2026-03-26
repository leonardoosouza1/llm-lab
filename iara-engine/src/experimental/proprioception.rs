use std::process::Command;
use anyhow::Result;

/// Propriocepção: O motor sentindo o próprio hardware.
/// Agora utiliza chamadas de sistema diretas para `rocm-smi` e `free` 
/// para evitar dependências pesadas de compilação de kernel.
pub struct Proprioception;

impl Proprioception {
    /// Retorna o estresse térmico e de memória do sistema (0.0 a 1.0).
    pub fn get_hardware_stress() -> f64 {
        // 1. Tentar obter dados da GPU via rocm-smi (AMD)
        if let Ok(gpu_stress) = Self::get_gpu_stress() {
            return gpu_stress;
        }

        // 2. Fallback: usa `free` do sistema quando ROCm não está disponível
        let mem = Self::get_mem_usage().unwrap_or(0.5);
        (mem * 0.4 + 0.3 * 0.6).clamp(0.0, 1.0)
    }

    /// Executa `rocm-smi` e extrai temperatura e uso de VRAM.
    fn get_gpu_stress() -> Result<f64> {
        // Usamos --showtemp e --showmemuse. O formato JSON é mais robusto mas 
        // as versões antigas do rocm-smi podem não suportar bem. 
        // Vamos usar a saída padrão e capturar os valores.
        let output = Command::new("rocm-smi")
            .args(["--showtemp", "--showmemuse"])
            .output()?;

        if !output.status.success() {
            anyhow::bail!("rocm-smi failed");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut temp = 45.0_f64;
        let mut mem_stress = 0.5_f64;

        for line in stdout.lines() {
            // Exemplo de linha: "0: Temperature (Sensor edge): 42.0 c"
            if line.contains("Temperature") {
                if let Some(val_str) = line.split(':').last() {
                    if let Some(val) = val_str.trim().split_whitespace().next() {
                        temp = val.parse::<f64>().unwrap_or(45.0_f64);
                    }
                }
            }
            // Exemplo: "0: VRAM Total Memory (B): 8589934592, VRAM Total Used Memory (B): 1073741824"
            if line.contains("VRAM Total Used Memory") {
                let parts: Vec<&str> = line.split(',').collect();
                let mut total = 1.0_f64;
                let mut used = 0.0_f64;
                for p in parts {
                    if p.contains("Total Memory") {
                        total = p.split(':').last().unwrap_or("1").trim().parse::<f64>().unwrap_or(1.0_f64);
                    }
                    if p.contains("Used Memory") {
                        used = p.split(':').last().unwrap_or("0").trim().parse::<f64>().unwrap_or(0.0_f64);
                    }
                }
                mem_stress = (used / total).clamp(0.0_f64, 1.0_f64);
            }
        }

        let temp_stress = ((temp - 40.0_f64) / 45.0_f64).clamp(0.0_f64, 1.0_f64);
        Ok((temp_stress * 0.6 + mem_stress * 0.4).clamp(0.0_f64, 1.0_f64))
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
