use compute_core::backend::ComputeBackend;
use compute_core::tensor::AnyTensor;
use std::collections::HashMap;

/// All weights for a transformer model, stored as GPU tensors
pub struct ModelWeights {
    pub tensors: HashMap<String, AnyTensor>,
    pub config: compute_core::config::ModelConfig,
}

impl ModelWeights {
    /// Load from safetensors file using compute-core parser
    pub fn from_safetensors(
        path: &std::path::Path,
        backend: &dyn ComputeBackend,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Will use compute_core::safetensors::parse_header
        todo!("Wire to compute-core safetensors loader + backend.upload_bf16()")
    }
}
