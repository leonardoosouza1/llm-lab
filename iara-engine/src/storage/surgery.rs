// ============================================================================
// storage/surgery.rs — Tensor Surgery: Pruning and Requantization
// ============================================================================
//
// Tools for post-hoc modification of .iara v2 files:
//   - Prune: Remove dead tensors and repack the file
//   - Requantize: Change dtype of specific tensors
//   - Export: Write modified model to new .iara file
//
// ============================================================================

use super::iara_format::{IaraDtype, IaraFormatError, IaraResult, IaraV2Reader, IaraV2Writer};
use std::path::Path;
use std::collections::{HashMap, HashSet};

/// Performs surgery on .iara v2 files: prune, requantize, reexport
pub struct TensorSurgeon {
    reader: IaraV2Reader,
    /// Tensors marked for removal
    pruned_tensors: HashSet<String>,
    /// Tensor name → new dtype overrides
    requantized: HashMap<String, IaraDtype>,
}

impl TensorSurgeon {
    /// Open an existing .iara v2 file for modification
    pub fn from_iara(path: &Path) -> IaraResult<Self> {
        let reader = IaraV2Reader::open(path)?;
        Ok(Self {
            reader,
            pruned_tensors: HashSet::new(),
            requantized: HashMap::new(),
        })
    }

    /// Mark tensors for removal
    pub fn prune(&mut self, dead_tensors: &[String]) -> &mut Self {
        for tensor_name in dead_tensors {
            if self.reader.header.tensors.contains_key(tensor_name) {
                self.pruned_tensors.insert(tensor_name.clone());
            }
        }
        self
    }

    /// Change quantization dtype for a specific tensor
    pub fn requantize(&mut self, tensor_name: &str, new_dtype: IaraDtype) -> &mut Self {
        if self.reader.header.tensors.contains_key(tensor_name) {
            self.requantized.insert(tensor_name.to_string(), new_dtype);
        }
        self
    }

    /// Export to a new .iara v2 file with modifications
    /// Note: This implementation copies existing quantized data without re-quantizing
    /// (actual re-quantization would require full decompression and recompression)
    pub fn export(&self, output_path: &Path) -> IaraResult<u64> {
        // Create output metadata
        let mut output_metadata = self.reader.header.metadata.clone();

        // Update metadata to reflect changes
        let original_tensor_count = self.reader.header.tensors.len();
        let new_tensor_count = original_tensor_count - self.pruned_tensors.len();
        let compression_estimate = (original_tensor_count as f64) / (new_tensor_count.max(1) as f64);

        output_metadata.original_size_bytes = (output_metadata.original_size_bytes as f64 / compression_estimate) as u64;

        // Create writer
        let mut writer = IaraV2Writer::new(output_path, output_metadata)?;

        // Copy tensors, skipping pruned ones
        for (tensor_name, tensor_meta) in &self.reader.header.tensors {
            if self.pruned_tensors.contains(tensor_name) {
                log::info!("  ✂️  Pruned: {}", tensor_name);
                continue;
            }

            // Get tensor data from mmap
            let tensor_data = self.reader.tensor_data(tensor_name)?;

            // Determine dtype (use requantized if specified, otherwise keep original)
            let target_dtype = self.requantized
                .get(tensor_name)
                .copied()
                .unwrap_or(tensor_meta.dtype);

            match target_dtype {
                IaraDtype::Bf16 => {
                    writer.write_bf16_tensor(tensor_name, &tensor_meta.shape, tensor_data)?;
                }
                IaraDtype::F32 => {
                    writer.write_bf16_tensor(tensor_name, &tensor_meta.shape, tensor_data)?;
                }
                IaraDtype::Fp8E4m3 => {
                    let scales = self.reader.tensor_scales(tensor_name)?;
                    writer.write_fp8_tensor(tensor_name, &tensor_meta.shape, tensor_data, &convert_scales(scales)?)?;
                }
                IaraDtype::Q4 => {
                    let scales = self.reader.tensor_scales(tensor_name)?;
                    let zeros = self.reader.tensor_zeros(tensor_name)?;
                    writer.write_q4_tensor(
                        tensor_name,
                        &tensor_meta.shape,
                        tensor_data,
                        &convert_scales(scales)?,
                        &convert_scales(zeros)?,
                        tensor_meta.group_size,
                    )?;
                }
                IaraDtype::Ternary => {
                    let scales = self.reader.tensor_scales(tensor_name)?;
                    let scale_f32 = if scales.len() == 4 {
                        f32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]])
                    } else {
                        1.0
                    };

                    let sparsity = tensor_meta.quant_params
                        .get("sparsity")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(0.0);

                    writer.write_ternary_tensor(tensor_name, &tensor_meta.shape, tensor_data, scale_f32, sparsity)?;
                }
                IaraDtype::Ptt => {
                    // Copy P+T+T tensor as-is (can't re-encode efficiently)
                    let scales = self.reader.tensor_scales(tensor_name)?;
                    let pin_indices_data = self.reader.tensor_pin_indices(tensor_name)?;
                    let pin_values_data = self.reader.tensor_pin_values(tensor_name)?;

                    let pin_threshold = tensor_meta.quant_params
                        .get("pin_threshold_sigma")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(2.0);
                    let pin_ratio = tensor_meta.quant_params
                        .get("pin_ratio")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(0.0);
                    let terrain_sparsity = tensor_meta.quant_params
                        .get("terrain_sparsity")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(0.0);

                    // Parse pin indices as u32 array
                    let pin_indices: Vec<u32> = pin_indices_data.chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    // Parse pin values as u16 array (BF16)
                    let pin_values: Vec<u16> = pin_values_data.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();

                    writer.write_ptt_tensor(
                        tensor_name, &tensor_meta.shape,
                        tensor_data, &convert_scales(scales)?,
                        &pin_indices, &pin_values,
                        pin_threshold, pin_ratio, terrain_sparsity,
                    )?;
                }
            }

            if self.requantized.contains_key(tensor_name) {
                log::info!("  🔄 Requantized: {} → {}", tensor_name, target_dtype.name());
            }
        }

        // Finalize and return size
        let total_size = writer.finalize()?;

        log::info!("📦 Surgery complete: {} → {} (removed {} tensors)",
            self.reader.file_size(),
            total_size,
            self.pruned_tensors.len());

        Ok(total_size)
    }

    /// Generate a human-readable report of planned modifications
    pub fn report(&self) -> SurgeryReport {
        let mut original_size = 0u64;
        let mut pruned_size = 0u64;
        let mut requantized_size = 0u64;

        for (tensor_name, tensor_meta) in &self.reader.header.tensors {
            let tensor_bytes = tensor_meta.data_length
                + tensor_meta.scales_length.unwrap_or(0)
                + tensor_meta.zeros_length.unwrap_or(0);

            original_size += tensor_bytes;

            if self.pruned_tensors.contains(tensor_name) {
                pruned_size += tensor_bytes;
            } else if self.requantized.contains_key(tensor_name) {
                requantized_size += tensor_bytes;
            }
        }

        let estimated_final_size = original_size - pruned_size;

        SurgeryReport {
            original_size,
            pruned_size,
            estimated_final_size,
            tensors_removed: self.pruned_tensors.len(),
            tensors_requantized: self.requantized.len(),
            estimated_savings_percent: if original_size > 0 {
                100.0 * pruned_size as f64 / original_size as f64
            } else {
                0.0
            },
        }
    }
}

/// Summary of planned tensor surgery modifications
#[derive(Debug, Clone)]
pub struct SurgeryReport {
    pub original_size: u64,
    pub pruned_size: u64,
    pub estimated_final_size: u64,
    pub tensors_removed: usize,
    pub tensors_requantized: usize,
    pub estimated_savings_percent: f64,
}

impl SurgeryReport {
    /// Pretty-print the report
    pub fn display(&self) -> String {
        vec![
            format!("Tensor Surgery Report"),
            format!("─────────────────────────────────"),
            format!("Original size: {:.1} MB", self.original_size as f64 / 1e6),
            format!("After pruning: {:.1} MB", self.estimated_final_size as f64 / 1e6),
            format!("Space saved: {:.1} MB ({:.1}%)",
                self.pruned_size as f64 / 1e6,
                self.estimated_savings_percent),
            format!(""),
            format!("Changes:"),
            format!("  Removed: {} tensors ({:.1} MB)",
                self.tensors_removed,
                self.pruned_size as f64 / 1e6),
            format!("  Requantized: {} tensors", self.tensors_requantized),
            format!(""),
            format!("Result: {:.1} MB ({:.2}× compression)",
                self.estimated_final_size as f64 / 1e6,
                self.original_size as f64 / self.estimated_final_size.max(1) as f64),
        ].join("\n")
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert u8 scales to f32 vector
fn convert_scales(scales_bytes: &[u8]) -> IaraResult<Vec<f32>> {
    scales_bytes
        .chunks_exact(4)
        .map(|c| {
            if c.len() != 4 {
                return Err(IaraFormatError::InvalidFormat("Invalid scale chunk".into()));
            }
            Ok(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_scales() {
        let data = 1.5f32.to_le_bytes().to_vec();
        let scales = convert_scales(&data).unwrap();
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_surgery_report_display() {
        let report = SurgeryReport {
            original_size: 100_000_000, // 100 MB exactly
            pruned_size: 20_000_000,    // 20 MB
            estimated_final_size: 80_000_000, // 80 MB
            tensors_removed: 5,
            tensors_requantized: 3,
            estimated_savings_percent: 20.0,
        };

        let display = report.display();
        assert!(display.contains("100.0 MB"), "Display should contain '100.0 MB', got: {}", display);
        assert!(display.contains("80.0 MB"), "Display should contain '80.0 MB', got: {}", display);
        assert!(display.contains("Removed: 5"));
    }
}
