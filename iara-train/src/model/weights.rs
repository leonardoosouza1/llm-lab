// ============================================================================
// weights.rs — CPU weight loading for iara-train
// ============================================================================
// Loads safetensors weights into CPU f32 buffers for training.
// No GPU tensors here — training forward/backward runs on CPU f32.
// AnyTensor is kept for interface compatibility with the ComputeBackend trait
// but the real training weights live in `cpu_tensors`.
// ============================================================================

use std::collections::HashMap;
use std::path::Path;

use compute_core::safetensors::{parse_header, convert_bf16_to_f32, convert_f16_to_f32, convert_f32_direct, data_offset};

/// All weights for a transformer model, stored as CPU f32 tensors.
/// Key format mirrors HuggingFace safetensors naming:
///   "model.embed_tokens.weight"
///   "model.layers.{i}.self_attn.q_proj.weight"
///   etc.
pub struct ModelWeights {
    /// Flat f32 data for each named tensor.
    pub tensors: HashMap<String, Vec<f32>>,
    /// Shape info for each named tensor (for bounds checking).
    pub shapes: HashMap<String, Vec<usize>>,
    /// Model configuration parsed from config.json.
    pub config: compute_core::config::ModelConfig,
}

impl ModelWeights {
    /// Load weights from a directory containing model.safetensors + config.json.
    ///
    /// Supports single-file models (model.safetensors) and sharded models
    /// (model-00001-of-NNNNN.safetensors). All tensors are converted to f32.
    pub fn from_dir(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Parse config.json (no seq_len cap for training)
        let config_path = model_dir.join("config.json");
        let config = compute_core::config::ModelConfig::from_json_with_cap(&config_path, usize::MAX)?;

        log::info!("ModelWeights: loading Qwen2.5 {}L h={} vocab={}",
            config.num_hidden_layers, config.hidden_size, config.vocab_size);

        // Collect all .safetensors files in the directory
        let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        st_files.sort();

        if st_files.is_empty() {
            return Err(format!("no .safetensors files found in {}", model_dir.display()).into());
        }

        log::info!("Found {} safetensors file(s)", st_files.len());

        let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for st_path in &st_files {
            log::info!("  Loading {}", st_path.file_name().unwrap().to_string_lossy());
            let (header, mmap) = parse_header(st_path)?;
            let data_start = data_offset(&mmap)?;

            for (name, meta) in &header {
                let numel = meta.numel();
                let abs_start = data_start + meta.data_offsets[0];
                let abs_end   = data_start + meta.data_offsets[1];
                let slice = &mmap[abs_start..abs_end];

                let data = match meta.dtype.as_str() {
                    "BF16" => convert_bf16_to_f32(slice, numel),
                    "F16"  => convert_f16_to_f32(slice, numel),
                    "F32"  => convert_f32_direct(slice, numel),
                    other  => {
                        log::debug!("Skipping tensor {} with unsupported dtype {}", name, other);
                        continue;
                    }
                };

                shapes.insert(name.clone(), meta.shape.clone());
                tensors.insert(name.clone(), data);
            }
        }

        log::info!("Loaded {} tensors", tensors.len());

        Ok(Self { tensors, shapes, config })
    }

    /// Get a tensor by name, panicking with a useful message if not found.
    pub fn get(&self, name: &str) -> &[f32] {
        self.tensors.get(name)
            .unwrap_or_else(|| panic!("tensor '{}' not found in model weights", name))
    }

    /// Get a tensor by name, returning None if not found.
    pub fn get_opt(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    /// Get shape for a tensor.
    pub fn shape(&self, name: &str) -> &[usize] {
        self.shapes.get(name)
            .unwrap_or_else(|| panic!("shape for '{}' not found", name))
    }
}
