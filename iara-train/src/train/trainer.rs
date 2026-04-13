//! Main training loop for LoRA fine-tuning with optional knowledge distillation.
//!
//! Orchestrates:
//! 1. Dataset loading and batching via burn's `DataLoader`
//! 2. Forward pass through the student model
//! 3. Loss computation (CE and/or KL-divergence)
//! 4. Backward pass (burn autodiff)
//! 5. Optimizer step (AdamW)
//! 6. Periodic checkpointing
//! 7. Final LoRA weight merge and export

use burn::data::dataloader::DataLoaderBuilder;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{DefaultRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::dataset::{TextDataset, TrainingBatch, TrainingBatcher};
use super::loss;

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

/// Configuration for a training run.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Learning rate for AdamW.
    pub lr: f64,
    /// AdamW beta_1.
    pub beta_1: f32,
    /// AdamW beta_2.
    pub beta_2: f32,
    /// AdamW weight decay.
    pub weight_decay: f32,
    /// Total number of training steps.
    pub num_steps: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Sequence length (context window).
    pub seq_len: usize,
    /// Checkpoint every N steps (0 = no intermediate checkpoints).
    pub checkpoint_every: usize,
    /// Output directory for checkpoints and final model.
    pub output_dir: PathBuf,
    /// Distillation alpha (1.0 = pure CE, 0.0 = pure KL).
    pub distill_alpha: f64,
    /// Distillation temperature.
    pub distill_temperature: f64,
    /// Shuffle seed for the data loader.
    pub shuffle_seed: u64,
    /// Number of data loader worker threads.
    pub num_workers: usize,
    /// LoRA rank (for logging; the model should already have LoRA applied).
    pub lora_rank: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta_1: 0.9,
            beta_2: 0.999,
            weight_decay: 0.01,
            num_steps: 500,
            batch_size: 1,
            seq_len: 512,
            checkpoint_every: 100,
            output_dir: PathBuf::from("checkpoints"),
            distill_alpha: 1.0,
            distill_temperature: 2.0,
            shuffle_seed: 42,
            num_workers: 0,
            lora_rank: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Training metrics (accumulated per logging interval)
// ---------------------------------------------------------------------------

/// Metrics collected during training.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f32,
    pub lr: f64,
    pub tokens_per_sec: f64,
    pub elapsed_ms: u128,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// The core training loop.
///
/// Generic over:
/// - `B`: an `AutodiffBackend` (e.g., `Autodiff<Wgpu>`)
/// - `M`: a model that implements `AutodiffModule<B>` and produces logits
///
/// The model's forward signature is abstracted through a closure so that
/// any architecture (Qwen2, LLaMA, etc.) can be trained without the trainer
/// knowing the model internals.
pub struct Trainer<B: AutodiffBackend> {
    config: TrainConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(config: TrainConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    /// Run the training loop.
    ///
    /// # Arguments
    ///
    /// * `model` — The model with LoRA adapters already applied. Must impl
    ///   `AutodiffModule<B>` and be able to produce logits from input token IDs.
    /// * `dataset` — Pre-tokenized training data.
    /// * `forward_fn` — Closure that runs the model forward pass:
    ///   `(model, input_ids) -> logits [batch, seq_len, vocab_size]`
    /// * `teacher_logits_fn` — Optional closure that returns pre-computed teacher
    ///   logits for a batch of input IDs. If `None`, pure CE training is used.
    ///
    /// # Returns
    ///
    /// The trained model (with LoRA weights updated).
    pub fn train<M, F, T>(
        &self,
        mut model: M,
        dataset: TextDataset,
        forward_fn: F,
        teacher_logits_fn: Option<T>,
    ) -> M
    where
        M: AutodiffModule<B> + std::fmt::Debug + 'static,
        F: Fn(&M, Tensor<B, 2, Int>) -> Tensor<B, 3>,
        T: Fn(&Tensor<B, 2, Int>) -> Tensor<B, 3>,
    {
        let cfg = &self.config;

        log::info!(
            "Starting training: steps={}, batch_size={}, seq_len={}, lr={}, lora_rank={}",
            cfg.num_steps, cfg.batch_size, cfg.seq_len, cfg.lr, cfg.lora_rank,
        );
        log::info!(
            "Dataset: {} tokens, {} windows",
            dataset.num_samples() + cfg.seq_len,
            dataset.num_samples(),
        );

        if teacher_logits_fn.is_some() {
            log::info!(
                "Distillation enabled: alpha={}, temperature={}",
                cfg.distill_alpha, cfg.distill_temperature,
            );
        }

        // Ensure output directory exists
        std::fs::create_dir_all(&cfg.output_dir).unwrap_or_else(|e| {
            panic!("cannot create output dir {}: {e}", cfg.output_dir.display())
        });

        // Build optimizer
        let mut optimizer = AdamWConfig::new()
            .with_beta_1(cfg.beta_1)
            .with_beta_2(cfg.beta_2)
            .with_weight_decay(cfg.weight_decay)
            .init::<B, M>();

        // Build data loader
        let dataloader = DataLoaderBuilder::new(TrainingBatcher::new())
            .batch_size(cfg.batch_size)
            .shuffle(cfg.shuffle_seed)
            .num_workers(cfg.num_workers)
            .set_device(self.device.clone())
            .build(dataset);

        let total_items = dataloader.num_items();
        log::info!("DataLoader: {} items, batch_size={}", total_items, cfg.batch_size);

        // Training loop
        let mut global_step = 0usize;
        let mut total_tokens = 0u64;
        let train_start = Instant::now();

        'outer: loop {
            for batch in dataloader.iter() {
                if global_step >= cfg.num_steps {
                    break 'outer;
                }

                let step_start = Instant::now();
                let batch: TrainingBatch<B> = batch;
                let [batch_size, seq_len] = batch.input_ids.dims();

                // Forward pass
                let logits = forward_fn(&model, batch.input_ids.clone());

                // Compute loss
                let loss = if let Some(ref teacher_fn) = teacher_logits_fn {
                    let teacher_logits = teacher_fn(&batch.input_ids);
                    loss::distillation_loss(
                        logits,
                        teacher_logits,
                        batch.target_ids,
                        cfg.distill_alpha,
                        cfg.distill_temperature,
                    )
                } else {
                    loss::cross_entropy_loss(logits, batch.target_ids)
                };

                // Read loss: use inner (non-autodiff) backend to read the scalar
                let loss_val = {
                    let data = loss.clone().inner().into_data();
                    data.as_slice::<f32>().map(|s| s[0] as f64).unwrap_or(-1.0)
                };

                // Backward pass
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);

                // Optimizer step
                model = optimizer.step(cfg.lr, model, grads);

                // Metrics
                let step_ms = step_start.elapsed().as_millis();
                let tokens_this_step = (batch_size * seq_len) as u64;
                total_tokens += tokens_this_step;

                global_step += 1;

                // Log every step
                if global_step % 10 == 0 || global_step == 1 {
                    let tokens_per_sec = if step_ms > 0 {
                        tokens_this_step as f64 / (step_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };

                    log::info!(
                        "step {}/{} | loss: {:.4} | lr: {:.2e} | {:.0} tok/s | {}ms",
                        global_step, cfg.num_steps, loss_val, cfg.lr, tokens_per_sec, step_ms,
                    );
                }

                // Checkpoint (TODO: implement once Module<InnerBackend> bounds are resolved)
                // if cfg.checkpoint_every > 0 && global_step % cfg.checkpoint_every == 0 {
                //     self.save_checkpoint(&model, global_step);
                // }
            }
        }

        let total_ms = train_start.elapsed().as_millis();
        let avg_tok_per_sec = if total_ms > 0 {
            total_tokens as f64 / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };

        log::info!(
            "Training complete: {} steps, {} tokens, {:.1}s, {:.0} avg tok/s",
            global_step, total_tokens, total_ms as f64 / 1000.0, avg_tok_per_sec,
        );

        // TODO: Save final model once checkpoint bounds are resolved
        // self.save_checkpoint(&model, global_step);

        model
    }

    /// Save a checkpoint of the model.
    fn save_checkpoint<M>(&self, model: &M, step: usize)
    where
        M: Module<B::InnerBackend> + std::fmt::Debug,
    {
        let path = self.config.output_dir.join(format!("checkpoint-{step}"));
        log::info!("Saving checkpoint: {}", path.display());

        let recorder = DefaultRecorder::new();
        match recorder.record(model.clone().into_record(), path.clone()) {
            Ok(()) => log::info!("Checkpoint saved: {}", path.display()),
            Err(e) => log::error!("Failed to save checkpoint: {e:?}"),
        }
    }
}

/// Convenience function to run training with a pre-tokenized binary dataset.
///
/// This is the highest-level entry point: give it a model, a token file path,
/// and a config, and it will run the full training loop.
pub fn train_from_token_file<B, M, F>(
    model: M,
    token_file: &Path,
    config: TrainConfig,
    device: B::Device,
    forward_fn: F,
) -> M
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + std::fmt::Debug + 'static,
    F: Fn(&M, Tensor<B, 2, Int>) -> Tensor<B, 3>,
{
    let dataset = TextDataset::from_token_file(token_file, config.seq_len);
    let trainer = Trainer::<B>::new(config, device);
    trainer.train(model, dataset, forward_fn, None::<fn(&Tensor<B, 2, Int>) -> Tensor<B, 3>>)
}
