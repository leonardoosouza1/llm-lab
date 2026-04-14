// ============================================================================
// trainer.rs — LoRA fine-tuning training loop for Qwen2.5
// ============================================================================
// CPU-only training loop (correctness first, speed second).
// Uses the CPU forward pass from model::forward and CPU LoRA from model::lora.
//
// Note: This is a simplified LoRA training loop. The LoRA backward pass is
// approximated: we compute gradients w.r.t. the full logits and propagate
// them through the lm_head back into the hidden states, then use the saved
// layer inputs to compute LoRA gradients for q_proj and v_proj.
//
// A proper full backward pass requires storing all intermediate activations,
// which we skip for now in favor of a "last-layer LoRA" approximation.
// ============================================================================

use crate::model::weights::ModelWeights;
use crate::model::lora::LoraAdapter;
use crate::model::forward::{forward, top_k_last_token};
use crate::train::dataset::TextDataset;
use crate::train::loss::cross_entropy_loss;
use std::time::Instant;

pub struct TrainConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub adam_eps: f32,
    pub num_steps: usize,
    pub batch_size: usize,   // number of sequences per gradient step
    pub seq_len: usize,
    pub checkpoint_every: usize,
    pub output_dir: std::path::PathBuf,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub log_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            beta1: 0.9,
            beta2: 0.999,
            adam_eps: 1e-8,
            num_steps: 100,
            batch_size: 1,
            seq_len: 128,
            checkpoint_every: 50,
            output_dir: std::path::PathBuf::from("lora_output"),
            lora_rank: 16,
            lora_alpha: 32.0,
            log_every: 10,
        }
    }
}

pub struct Trainer {
    pub config: TrainConfig,
}

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }

    /// Run the full training loop.
    ///
    /// Forward pass → CE loss → simplified backward → Adam step on LoRA weights.
    pub fn train(
        &self,
        weights: &ModelWeights,
        lora: &mut LoraAdapter,
        dataset: &TextDataset,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let vocab_size = weights.config.vocab_size;
        let n_layers = weights.config.num_hidden_layers;
        let hidden_size = weights.config.hidden_size;

        log::info!("Starting LoRA training: {} steps, lr={}, rank={}",
            self.config.num_steps, self.config.lr, self.config.lora_rank);
        log::info!("LoRA trainable params: {}", lora.num_params());

        let t_start = Instant::now();
        let mut step = 0usize;
        let mut sample_idx = 0usize;
        let num_samples = dataset.num_samples();

        if num_samples == 0 {
            return Err("Dataset is empty".into());
        }

        while step < self.config.num_steps {
            let t_step = Instant::now();
            let mut total_loss = 0.0f32;

            // Gradient accumulation over batch_size sequences
            for _b in 0..self.config.batch_size {
                let window = dataset.get(sample_idx % num_samples)
                    .ok_or("dataset.get returned None")?;
                sample_idx += 1;

                let input_ids = &window.input_ids;
                let targets   = &window.target_ids;
                let seq_len   = input_ids.len();

                // Forward pass (collect layer inputs for LoRA backward)
                let result = forward(weights, input_ids, true);

                // CE loss + gradient w.r.t. logits
                let (loss, grad_logits) = cross_entropy_loss(&result.logits, targets, vocab_size);
                total_loss += loss;

                // Simplified backward pass for LoRA:
                //
                // For the full chain:
                //   logits = hidden_final @ lm_head^T
                //   hidden_final = rms_norm(hidden_last_layer)
                //   ... complex chain through all layers ...
                //
                // Approximation: we use the gradient signal from the logits
                // projected back through lm_head to get grad_hidden, then
                // for each layer we use the saved layer input and the
                // gradient of the output (via the residual) to compute LoRA grads.
                //
                // This is an approximation — a proper BPTT is needed for full accuracy,
                // but this gives a training signal for the LoRA parameters.

                // grad_hidden = grad_logits @ lm_head  [seq_len, hidden_size]
                let lm_head_w = weights.get_opt("lm_head.weight")
                    .unwrap_or_else(|| weights.get("model.embed_tokens.weight"));
                let mut grad_hidden = vec![0.0f32; seq_len * hidden_size];

                // grad_logits: [seq_len, vocab_size], lm_head: [vocab_size, hidden_size]
                // grad_hidden[t, h] = sum_v grad_logits[t, v] * lm_head[v, h]
                for t in 0..seq_len {
                    for h in 0..hidden_size {
                        let mut s = 0.0f32;
                        for v in 0..vocab_size {
                            s += grad_logits[t * vocab_size + v] * lm_head_w[v * hidden_size + h];
                        }
                        grad_hidden[t * hidden_size + h] = s;
                    }
                }

                // Propagate gradient through each layer's LoRA adapters.
                // We use the saved layer inputs (pre-norm, pre-attention inputs).
                // For q_proj: input is normed hidden state, output is q vectors.
                // We approximate the norm gradient as identity (common simplification).
                for layer_idx in 0..n_layers {
                    if layer_idx >= result.layer_inputs.len() {
                        break;
                    }

                    let layer_input = &result.layer_inputs[layer_idx];

                    // q_proj LoRA grad:
                    // Approximate: grad_q ≈ grad_hidden (via residual connection)
                    // in_features = hidden_size, out_features = q_dim
                    if let Some(q_lora) = lora.get_mut(&format!("{}.q_proj", layer_idx)) {
                        let q_dim = q_lora.out_features;
                        // Project grad_hidden down to q_dim via q_proj weights
                        let wq = weights.get(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx));
                        let mut grad_q = vec![0.0f32; seq_len * q_dim];
                        for t in 0..seq_len {
                            for o in 0..q_dim {
                                let mut s = 0.0f32;
                                for h in 0..hidden_size {
                                    // wq[o, h] (stored [out, in])
                                    s += grad_hidden[t * hidden_size + h] * wq[o * hidden_size + h];
                                }
                                grad_q[t * q_dim + o] = s;
                            }
                        }
                        q_lora.accumulate_grad(layer_input, &grad_q, seq_len);
                    }

                    // v_proj LoRA grad:
                    if let Some(v_lora) = lora.get_mut(&format!("{}.v_proj", layer_idx)) {
                        let kv_dim = v_lora.out_features;
                        let wv = weights.get(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx));
                        let mut grad_v = vec![0.0f32; seq_len * kv_dim];
                        for t in 0..seq_len {
                            for o in 0..kv_dim {
                                let mut s = 0.0f32;
                                for h in 0..hidden_size {
                                    s += grad_hidden[t * hidden_size + h] * wv[o * hidden_size + h];
                                }
                                grad_v[t * kv_dim + o] = s;
                            }
                        }
                        v_lora.accumulate_grad(layer_input, &grad_v, seq_len);
                    }
                }
            } // end batch loop

            // Normalize gradients by batch size
            if self.config.batch_size > 1 {
                let inv_batch = 1.0 / self.config.batch_size as f32;
                for adapter in lora.adapters.values_mut() {
                    for g in &mut adapter.grad_a { *g *= inv_batch; }
                    for g in &mut adapter.grad_b { *g *= inv_batch; }
                }
            }

            // Adam optimizer step
            lora.adam_step_all(
                self.config.lr,
                self.config.beta1,
                self.config.beta2,
                self.config.adam_eps,
            );

            let avg_loss = total_loss / self.config.batch_size as f32;
            step += 1;

            if step % self.config.log_every == 0 || step == 1 {
                let elapsed = t_step.elapsed().as_secs_f32();
                let total_elapsed = t_start.elapsed().as_secs_f32();
                log::info!(
                    "step {:4}/{} | loss={:.4} | step_time={:.1}s | total={:.1}s",
                    step, self.config.num_steps, avg_loss, elapsed, total_elapsed
                );
            }

            if step % self.config.checkpoint_every == 0 {
                let ckpt_dir = self.config.output_dir.join(format!("step_{}", step));
                lora.save(&ckpt_dir)?;
                log::info!("Checkpoint saved to {}", ckpt_dir.display());
            }
        }

        // Final checkpoint
        lora.save(&self.config.output_dir)?;
        log::info!("Training complete in {:.1}s", t_start.elapsed().as_secs_f32());
        Ok(())
    }

    /// Run a quick sanity check: one forward pass and one loss computation.
    pub fn sanity_check(
        weights: &ModelWeights,
        dataset: &TextDataset,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let window = dataset.get(0).ok_or("dataset is empty")?;
        let result = forward(weights, &window.input_ids, false);
        let (loss, _) = cross_entropy_loss(
            &result.logits,
            &window.target_ids,
            weights.config.vocab_size,
        );
        Ok(loss)
    }
}

/// Log the top-k next tokens after a forward pass (for debugging).
pub fn log_top_tokens(
    weights: &ModelWeights,
    input_ids: &[u32],
    k: usize,
) {
    let result = forward(weights, input_ids, false);
    let top = top_k_last_token(&result.logits, weights.config.vocab_size, k);
    log::info!("Forward pass: {:.1}ms, top-{} tokens:", result.elapsed_ms, k);
    for (token_id, prob) in &top {
        log::info!("  token_id={} prob={:.4}", token_id, prob);
    }
}
