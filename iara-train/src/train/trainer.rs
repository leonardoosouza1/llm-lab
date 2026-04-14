use compute_core::backend::ComputeBackend;
use crate::model::weights::ModelWeights;
use crate::model::lora::LoraAdapter;
use crate::train::dataset::TextDataset;

pub struct TrainConfig {
    pub lr: f64,
    pub num_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub checkpoint_every: usize,
    pub output_dir: std::path::PathBuf,
    pub lora_rank: usize,
    pub lora_alpha: f64,
}

pub struct Trainer {
    config: TrainConfig,
}

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }

    pub fn train(
        &self,
        weights: &ModelWeights,
        lora: &mut LoraAdapter,
        dataset: &TextDataset,
        backend: &dyn ComputeBackend,
    ) {
        // Native training loop:
        // 1. Forward pass via crate::model::forward::forward()
        // 2. CE loss computation
        // 3. Backward pass (manual gradient computation via HIP kernels)
        // 4. LoRA weight update (SGD/AdamW)
        // 5. Checkpoint every N steps
        todo!("Implement native training loop with HIP backward pass")
    }
}
