use compute_core::backend::ComputeBackend;
use crate::model::weights::ModelWeights;

/// Forward pass through the transformer, returning logits
pub fn forward(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    input_ids: &[u32],
    seq_pos: usize,
) -> Vec<f32> {
    // Will dispatch to backend.gemm_bf16_f32(), backend.rms_norm(), etc.
    // Same flow as iara-engine but with gradient checkpointing hooks
    todo!("Implement native forward pass using ComputeBackend trait")
}
