// ============================================================================
// lora.rs — CPU LoRA adapters for Qwen2.5 attention projections
// ============================================================================
// Pure f32 CPU LoRA (Low-Rank Adaptation).
//
// For each adapted projection W (e.g. q_proj, v_proj):
//   output = x @ W^T + (x @ A^T) @ B^T * (alpha / rank)
//
// A: [rank, in_features]  — initialized Kaiming uniform, trained
// B: [out_features, rank] — initialized zeros, trained
//
// Adapts only q_proj and v_proj per layer for efficiency.
// Uses simple SGD with gradient accumulation. Adam is planned.
// ============================================================================

use std::collections::HashMap;
use rand::Rng;

/// Single LoRA weight pair for one projection.
pub struct LoraWeight {
    /// A matrix: [rank, in_features] — row-major
    pub a: Vec<f32>,
    /// B matrix: [out_features, rank] — row-major
    pub b: Vec<f32>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub scale: f32,  // alpha / rank

    // Gradient accumulators (for Adam)
    pub grad_a: Vec<f32>,
    pub grad_b: Vec<f32>,
    // Adam first moment
    pub m_a: Vec<f32>,
    pub m_b: Vec<f32>,
    // Adam second moment
    pub v_a: Vec<f32>,
    pub v_b: Vec<f32>,
    pub step: usize,
}

impl LoraWeight {
    /// Create a new LoRA weight pair.
    /// A is initialized Kaiming uniform (fan_in = in_features).
    /// B is initialized to zeros.
    pub fn new(rank: usize, in_features: usize, out_features: usize, alpha: f32) -> Self {
        let mut rng = rand::thread_rng();
        // Kaiming uniform: U(-sqrt(1/fan_in), sqrt(1/fan_in)) * sqrt(3) — matches PyTorch default
        let bound = (1.0 / in_features as f32).sqrt() * (3.0f32).sqrt();
        let a: Vec<f32> = (0..rank * in_features)
            .map(|_| rng.gen_range(-bound..bound))
            .collect();
        let b = vec![0.0f32; out_features * rank];
        let numel_a = rank * in_features;
        let numel_b = out_features * rank;

        Self {
            a,
            b,
            rank,
            in_features,
            out_features,
            scale: alpha / rank as f32,
            grad_a: vec![0.0f32; numel_a],
            grad_b: vec![0.0f32; numel_b],
            m_a: vec![0.0f32; numel_a],
            m_b: vec![0.0f32; numel_b],
            v_a: vec![0.0f32; numel_a],
            v_b: vec![0.0f32; numel_b],
            step: 0,
        }
    }

    /// Apply LoRA delta to an existing output.
    ///
    /// `x`: input [seq_len, in_features]
    /// `w_out`: output buffer [seq_len, out_features], will have LoRA delta added
    pub fn apply(&self, x: &[f32], w_out: &mut [f32], seq_len: usize) {
        // lora_out = (x @ A^T) @ B^T * scale
        // Step 1: h = x @ A^T  → [seq_len, rank]
        let mut h = vec![0.0f32; seq_len * self.rank];
        for t in 0..seq_len {
            for r in 0..self.rank {
                let mut s = 0.0f32;
                for k in 0..self.in_features {
                    s += x[t * self.in_features + k] * self.a[r * self.in_features + k];
                }
                h[t * self.rank + r] = s;
            }
        }

        // Step 2: delta = h @ B^T * scale → [seq_len, out_features]
        for t in 0..seq_len {
            for o in 0..self.out_features {
                let mut s = 0.0f32;
                for r in 0..self.rank {
                    s += h[t * self.rank + r] * self.b[o * self.rank + r];
                }
                w_out[t * self.out_features + o] += s * self.scale;
            }
        }
    }

    /// Accumulate gradients for A and B given upstream gradient.
    ///
    /// `x`: input [seq_len, in_features]
    /// `grad_out`: upstream gradient [seq_len, out_features]
    pub fn accumulate_grad(&mut self, x: &[f32], grad_out: &[f32], seq_len: usize) {
        // LoRA forward: y = (x @ A^T) @ B^T * scale
        // Backward:
        //   grad_h = grad_out @ B * scale   [seq_len, rank]
        //   grad_A += grad_h^T @ x          [rank, in_features]
        //   grad_B += grad_out^T @ h * scale [out_features, rank]
        //   (where h = x @ A^T)

        // Recompute h = x @ A^T [seq_len, rank]
        let mut h = vec![0.0f32; seq_len * self.rank];
        for t in 0..seq_len {
            for r in 0..self.rank {
                let mut s = 0.0f32;
                for k in 0..self.in_features {
                    s += x[t * self.in_features + k] * self.a[r * self.in_features + k];
                }
                h[t * self.rank + r] = s;
            }
        }

        // grad_h = grad_out @ B * scale [seq_len, rank]
        let mut grad_h = vec![0.0f32; seq_len * self.rank];
        for t in 0..seq_len {
            for r in 0..self.rank {
                let mut s = 0.0f32;
                for o in 0..self.out_features {
                    s += grad_out[t * self.out_features + o] * self.b[o * self.rank + r];
                }
                grad_h[t * self.rank + r] = s * self.scale;
            }
        }

        // grad_A += grad_h^T @ x [rank, in_features]
        for r in 0..self.rank {
            for k in 0..self.in_features {
                let mut s = 0.0f32;
                for t in 0..seq_len {
                    s += grad_h[t * self.rank + r] * x[t * self.in_features + k];
                }
                self.grad_a[r * self.in_features + k] += s;
            }
        }

        // grad_B += grad_out^T @ h * scale [out_features, rank]
        for o in 0..self.out_features {
            for r in 0..self.rank {
                let mut s = 0.0f32;
                for t in 0..seq_len {
                    s += grad_out[t * self.out_features + o] * h[t * self.rank + r];
                }
                self.grad_b[o * self.rank + r] += s * self.scale;
            }
        }
    }

    /// Adam optimizer step.
    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        self.step += 1;
        let step = self.step as f32;
        let bc1 = 1.0 - beta1.powf(step);
        let bc2 = 1.0 - beta2.powf(step);

        // Update A
        for i in 0..self.grad_a.len() {
            let g = self.grad_a[i];
            self.m_a[i] = beta1 * self.m_a[i] + (1.0 - beta1) * g;
            self.v_a[i] = beta2 * self.v_a[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_a[i] / bc1;
            let v_hat = self.v_a[i] / bc2;
            self.a[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            self.grad_a[i] = 0.0;
        }

        // Update B
        for i in 0..self.grad_b.len() {
            let g = self.grad_b[i];
            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * g;
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_b[i] / bc1;
            let v_hat = self.v_b[i] / bc2;
            self.b[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            self.grad_b[i] = 0.0;
        }
    }

    /// SGD step (simpler alternative to Adam).
    pub fn sgd_step(&mut self, lr: f32) {
        for i in 0..self.a.len() {
            self.a[i] -= lr * self.grad_a[i];
            self.grad_a[i] = 0.0;
        }
        for i in 0..self.b.len() {
            self.b[i] -= lr * self.grad_b[i];
            self.grad_b[i] = 0.0;
        }
    }
}

/// LoRA adapter: manages all LoRA weights across layers.
pub struct LoraAdapter {
    pub rank: usize,
    pub alpha: f32,
    /// Key: "{layer_idx}.{proj}" e.g. "0.q_proj", "0.v_proj"
    pub adapters: HashMap<String, LoraWeight>,
}

impl LoraAdapter {
    /// Create LoRA adapters for q_proj and v_proj in all layers.
    pub fn new_for_qwen2(
        rank: usize,
        alpha: f32,
        config: &compute_core::config::ModelConfig,
    ) -> Self {
        let hidden_size = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let mut adapters = HashMap::new();

        for layer_idx in 0..config.num_hidden_layers {
            // q_proj: [hidden_size, q_dim] (in_features=hidden, out_features=q_dim)
            adapters.insert(
                format!("{}.q_proj", layer_idx),
                LoraWeight::new(rank, hidden_size, q_dim, alpha),
            );
            // v_proj: [hidden_size, kv_dim]
            adapters.insert(
                format!("{}.v_proj", layer_idx),
                LoraWeight::new(rank, hidden_size, kv_dim, alpha),
            );
        }

        log::info!("LoRA adapters: rank={} alpha={} layers={} projs=[q,v] params={}",
            rank, alpha, config.num_hidden_layers,
            adapters.values().map(|w| w.a.len() + w.b.len()).sum::<usize>());

        Self { rank, alpha, adapters }
    }

    /// Get mutable reference to a specific adapter.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut LoraWeight> {
        self.adapters.get_mut(key)
    }

    /// Get reference to a specific adapter.
    pub fn get(&self, key: &str) -> Option<&LoraWeight> {
        self.adapters.get(key)
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.adapters.values().map(|w| w.a.len() + w.b.len()).sum()
    }

    /// Run Adam step on all adapters.
    pub fn adam_step_all(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        for adapter in self.adapters.values_mut() {
            adapter.adam_step(lr, beta1, beta2, eps);
        }
    }

    /// Save LoRA weights to a directory as individual .bin files.
    pub fn save(&self, output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(output_dir)?;
        for (key, w) in &self.adapters {
            let a_path = output_dir.join(format!("{}_a.bin", key.replace('.', "_")));
            let b_path = output_dir.join(format!("{}_b.bin", key.replace('.', "_")));
            let a_bytes: Vec<u8> = w.a.iter().flat_map(|v| v.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = w.b.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&a_path, &a_bytes)?;
            std::fs::write(&b_path, &b_bytes)?;
        }
        log::info!("LoRA weights saved to {}", output_dir.display());
        Ok(())
    }
}
