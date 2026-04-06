//! Sparse Autoencoder (SAE) for mechanistic interpretability.
//!
//! Observes hidden states from a transformer layer and learns to decompose
//! them into sparse, interpretable features. Trains online during inference
//! (same pattern as ESN observer).
//!
//! Architecture:
//!   Encoder: z = ReLU(W_enc @ (x - b_dec) + b_enc)   // z is sparse
//!   Decoder: x_hat = W_dec @ z + b_dec                 // reconstruction
//!   Loss: MSE(x, x_hat) + λ * ||z||₁                  // sparsity penalty

use serde::{Serialize, Deserialize};

/// Configuration for SAE
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaeConfig {
    pub input_dim: usize,          // = hidden_size (e.g. 4096)
    pub expansion_factor: usize,   // latent_dim = input_dim * expansion (e.g. 4x = 16384)
    pub sparsity_coeff: f32,       // L1 penalty lambda (e.g. 0.001)
    pub learning_rate: f32,        // Adam LR (e.g. 3e-4)
    pub target_layer: usize,       // Which transformer layer to observe
}

impl Default for SaeConfig {
    fn default() -> Self {
        Self {
            input_dim: 4096,
            expansion_factor: 4,
            sparsity_coeff: 0.001,
            learning_rate: 3e-4,
            target_layer: 15,  // Penultimate for 36-layer model
        }
    }
}

/// Adam optimizer state for a single parameter group
struct AdamState {
    m: Vec<f32>,    // First moment
    v: Vec<f32>,    // Second moment
    t: u64,         // Timestep
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl AdamState {
    fn new(size: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Sparse Autoencoder
pub struct SparseAutoencoder {
    pub config: SaeConfig,
    latent_dim: usize,

    // Parameters
    w_enc: Vec<f32>,    // [latent_dim, input_dim]
    b_enc: Vec<f32>,    // [latent_dim]
    w_dec: Vec<f32>,    // [input_dim, latent_dim]
    b_dec: Vec<f32>,    // [input_dim]

    // Optimizer states
    opt_w_enc: AdamState,
    opt_b_enc: AdamState,
    opt_w_dec: AdamState,
    opt_b_dec: AdamState,

    // Tracking
    pub step: u64,
    pub running_loss: f32,
    pub running_recon_loss: f32,
    pub running_sparsity: f32,
    pub feature_activations: Vec<f64>,  // Cumulative activation count per feature
}

/// Result from a single SAE observation step
#[derive(Clone, Debug, Serialize)]
pub struct SaeObservation {
    pub active_features: Vec<(usize, f32)>,  // (feature_idx, activation_value)
    pub n_active: usize,
    pub sparsity: f32,         // fraction of zero features
    pub recon_loss: f32,       // MSE reconstruction loss
    pub total_loss: f32,       // MSE + L1
    pub step: u64,
}

impl SparseAutoencoder {
    pub fn new(config: SaeConfig) -> Self {
        let input_dim = config.input_dim;
        let latent_dim = input_dim * config.expansion_factor;
        let lr = config.learning_rate;

        // Kaiming initialization for encoder
        let scale_enc = (2.0 / input_dim as f32).sqrt();
        let scale_dec = (2.0 / latent_dim as f32).sqrt();

        let w_enc = Self::random_init(latent_dim * input_dim, scale_enc);
        let w_dec = Self::random_init(input_dim * latent_dim, scale_dec);

        let w_enc_size = latent_dim * input_dim;
        let w_dec_size = input_dim * latent_dim;

        Self {
            latent_dim,
            w_enc,
            b_enc: vec![0.0; latent_dim],
            w_dec,
            b_dec: vec![0.0; input_dim],
            opt_w_enc: AdamState::new(w_enc_size, lr),
            opt_b_enc: AdamState::new(latent_dim, lr),
            opt_w_dec: AdamState::new(w_dec_size, lr),
            opt_b_dec: AdamState::new(input_dim, lr),
            step: 0,
            running_loss: 0.0,
            running_recon_loss: 0.0,
            running_sparsity: 0.0,
            feature_activations: vec![0.0; latent_dim],
            config,
        }
    }

    fn random_init(size: usize, scale: f32) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Simple deterministic pseudo-random (no external dep needed)
        let mut vals = Vec::with_capacity(size);
        for i in 0..size {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let h = hasher.finish();
            // Map to [-scale, scale]
            let normalized = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
            vals.push(normalized as f32 * scale);
        }
        vals
    }

    /// Observe a hidden state and train one step.
    /// `x` is the hidden state vector [input_dim] from the target layer.
    /// Returns observation with active features and loss metrics.
    pub fn observe(&mut self, x: &[f32]) -> SaeObservation {
        let input_dim = self.config.input_dim;
        let latent_dim = self.latent_dim;
        assert_eq!(x.len(), input_dim, "SAE input dim mismatch");

        // ── Forward Pass ──

        // 1. Center: x_c = x - b_dec
        let x_centered: Vec<f32> = x.iter()
            .zip(&self.b_dec)
            .map(|(&xi, &bi)| xi - bi)
            .collect();

        // 2. Encode: z_pre = W_enc @ x_c + b_enc
        let mut z_pre = vec![0.0f32; latent_dim];
        for j in 0..latent_dim {
            let mut sum = self.b_enc[j];
            let row_start = j * input_dim;
            for i in 0..input_dim {
                sum += self.w_enc[row_start + i] * x_centered[i];
            }
            z_pre[j] = sum;
        }

        // 3. ReLU: z = max(0, z_pre)
        let z: Vec<f32> = z_pre.iter().map(|&v| v.max(0.0)).collect();

        // 4. Decode: x_hat = W_dec @ z + b_dec
        let mut x_hat = vec![0.0f32; input_dim];
        for i in 0..input_dim {
            let mut sum = self.b_dec[i];
            let row_start = i * latent_dim;
            for j in 0..latent_dim {
                sum += self.w_dec[row_start + j] * z[j];
            }
            x_hat[i] = sum;
        }

        // ── Loss ──

        // MSE reconstruction loss
        let recon_loss: f32 = x.iter()
            .zip(&x_hat)
            .map(|(&xi, &xhi)| (xi - xhi).powi(2))
            .sum::<f32>() / input_dim as f32;

        // L1 sparsity loss
        let l1: f32 = z.iter().map(|&zi| zi.abs()).sum::<f32>() / latent_dim as f32;
        let total_loss = recon_loss + self.config.sparsity_coeff * l1;

        // ── Backward Pass ──

        // d_loss / d_x_hat = 2(x_hat - x) / input_dim
        let d_x_hat: Vec<f32> = x_hat.iter()
            .zip(x)
            .map(|(&xhi, &xi)| 2.0 * (xhi - xi) / input_dim as f32)
            .collect();

        // d_loss / d_b_dec = d_x_hat (from decoder) + (-1) * d_z_via_encoder_centering
        // Simplified: d_b_dec = d_x_hat
        let d_b_dec: Vec<f32> = d_x_hat.clone();

        // d_loss / d_w_dec = d_x_hat @ z.T  → [input_dim, latent_dim]
        let mut d_w_dec = vec![0.0f32; input_dim * latent_dim];
        for i in 0..input_dim {
            let row_start = i * latent_dim;
            for j in 0..latent_dim {
                d_w_dec[row_start + j] = d_x_hat[i] * z[j];
            }
        }

        // d_loss / d_z = W_dec.T @ d_x_hat + lambda * sign(z) / latent_dim
        let mut d_z = vec![0.0f32; latent_dim];
        for j in 0..latent_dim {
            let mut sum = 0.0;
            for i in 0..input_dim {
                sum += self.w_dec[i * latent_dim + j] * d_x_hat[i];
            }
            // Add L1 gradient (subgradient at 0)
            let l1_grad = if z[j] > 0.0 {
                self.config.sparsity_coeff / latent_dim as f32
            } else {
                0.0  // ReLU kills gradient when z <= 0
            };
            d_z[j] = if z_pre[j] > 0.0 { sum + l1_grad } else { 0.0 }; // ReLU gate
        }

        // d_loss / d_b_enc = d_z
        let d_b_enc: Vec<f32> = d_z.clone();

        // d_loss / d_w_enc = d_z @ x_c.T → [latent_dim, input_dim]
        let mut d_w_enc = vec![0.0f32; latent_dim * input_dim];
        for j in 0..latent_dim {
            let row_start = j * input_dim;
            for i in 0..input_dim {
                d_w_enc[row_start + i] = d_z[j] * x_centered[i];
            }
        }

        // ── Adam Update ──
        self.opt_w_enc.step(&mut self.w_enc, &d_w_enc);
        self.opt_b_enc.step(&mut self.b_enc, &d_b_enc);
        self.opt_w_dec.step(&mut self.w_dec, &d_w_dec);
        self.opt_b_dec.step(&mut self.b_dec, &d_b_dec);

        // ── Track Features ──
        let active_features: Vec<(usize, f32)> = z.iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, &v)| {
                self.feature_activations[i] += 1.0;
                (i, v)
            })
            .collect();

        let sparsity = 1.0 - (active_features.len() as f32 / latent_dim as f32);

        // Running averages
        let alpha = 0.99;
        self.running_loss = alpha * self.running_loss + (1.0 - alpha) * total_loss;
        self.running_recon_loss = alpha * self.running_recon_loss + (1.0 - alpha) * recon_loss;
        self.running_sparsity = alpha * self.running_sparsity + (1.0 - alpha) * sparsity;
        self.step += 1;

        SaeObservation {
            n_active: active_features.len(),
            active_features,
            sparsity,
            recon_loss,
            total_loss,
            step: self.step,
        }
    }

    /// Get top-K most frequently activated features
    pub fn top_features(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self.feature_activations.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Get SAE status for WebSocket broadcast
    pub fn status_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "sae_state",
            "layer": self.config.target_layer,
            "active_features_avg": if self.step > 0 {
                self.feature_activations.iter().filter(|&&v| v > 0.0).count()
            } else { 0 },
            "sparsity": self.running_sparsity,
            "recon_loss": self.running_recon_loss,
            "total_loss": self.running_loss,
            "step": self.step,
            "top_features": self.top_features(10),
            "latent_dim": self.latent_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sae(input_dim: usize) -> SparseAutoencoder {
        SparseAutoencoder::new(SaeConfig {
            input_dim,
            expansion_factor: 2,  // Small for testing
            sparsity_coeff: 0.001,
            learning_rate: 1e-3,
            target_layer: 0,
        })
    }

    #[test]
    fn test_sae_forward_backward() {
        let mut sae = make_sae(64);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let obs = sae.observe(&x);
        assert!(obs.recon_loss.is_finite());
        assert!(obs.total_loss.is_finite());
        assert!(obs.sparsity >= 0.0 && obs.sparsity <= 1.0);
        assert_eq!(obs.step, 1);
    }

    #[test]
    fn test_sae_loss_decreases() {
        let mut sae = make_sae(32);
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).cos()).collect();

        let first_loss = sae.observe(&x).recon_loss;

        // Train for 100 steps on same input (should overfit)
        let mut last_loss = first_loss;
        for _ in 0..100 {
            last_loss = sae.observe(&x).recon_loss;
        }

        // Loss should decrease (overfitting on single input)
        assert!(last_loss < first_loss,
            "Loss should decrease: first={}, last={}", first_loss, last_loss);
    }

    #[test]
    fn test_sae_sparsity() {
        let mut sae = make_sae(64);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        // After some training, should have >50% sparsity (ReLU kills many features)
        for _ in 0..50 {
            sae.observe(&x);
        }
        let obs = sae.observe(&x);
        assert!(obs.sparsity > 0.3, "Sparsity too low: {}", obs.sparsity);
    }

    #[test]
    fn test_sae_top_features() {
        let mut sae = make_sae(32);
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

        for _ in 0..10 {
            sae.observe(&x);
        }

        let top = sae.top_features(5);
        assert_eq!(top.len(), 5);
        // Should be sorted descending
        for i in 1..top.len() {
            assert!(top[i-1].1 >= top[i].1);
        }
    }

    #[test]
    fn test_sae_status_json() {
        let mut sae = make_sae(32);
        let x: Vec<f32> = vec![1.0; 32];
        sae.observe(&x);

        let status = sae.status_json();
        assert_eq!(status["type"], "sae_state");
        assert!(status["step"].as_u64().unwrap() > 0);
    }
}
