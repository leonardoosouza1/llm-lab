use compute_core::tensor::AnyTensor;
use compute_core::backend::ComputeBackend;
use std::collections::HashMap;

/// LoRA adapter: stores A and B matrices per target projection
pub struct LoraAdapter {
    pub rank: usize,
    pub alpha: f64,
    pub adapters: HashMap<String, LoraWeight>,
}

pub struct LoraWeight {
    pub a: AnyTensor,  // [rank, in_features]
    pub b: AnyTensor,  // [out_features, rank]
}

impl LoraAdapter {
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            adapters: HashMap::new(),
        }
    }

    /// Initialize LoRA for attention projections (Q, K, V, O)
    pub fn init_for_attention(
        &mut self,
        config: &compute_core::config::ModelConfig,
        backend: &dyn ComputeBackend,
    ) {
        todo!("Allocate A (kaiming) and B (zeros) on GPU via backend")
    }

    /// Merge LoRA into base weights: W' = W + (alpha/rank) * B @ A
    pub fn merge_into(
        &self,
        weights: &mut crate::model::weights::ModelWeights,
        backend: &dyn ComputeBackend,
    ) {
        todo!("GPU GEMM to merge LoRA deltas")
    }
}
