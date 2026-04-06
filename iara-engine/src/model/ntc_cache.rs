//! NTC-compressed KV Cache
//! Instead of [max_seq, kv_dim], stores:
//!   - latent_grid: [grid_size, latent_dim] ≈ 64×32 = 2KB per layer/kv
//!   - mlp_weights: ~8KB per layer/kv (2 linear layers)
//!   - Total: ~10KB per layer/kv vs ~4MB (F32) or ~1MB (INT8)

use serde_json::json;
use rand::Rng;

/// Adam optimizer state for online training
pub struct AdamState {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,
    // Moment buffers — lazily allocated
    pub m: Vec<f32>,  // first moment
    pub v: Vec<f32>,  // second moment
}

impl AdamState {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Update parameters using Adam
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        if params.len() == 0 {
            return;
        }

        // Lazy allocation
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }

        self.t += 1;
        let t_f32 = self.t as f32;

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t_f32));
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t_f32));

            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

/// Helper struct for forward pass intermediates (for backprop)
struct ForwardIntermediates {
    pe: Vec<f32>,
    latent: Vec<f32>,
    input: Vec<f32>,       // concat of pe + latent
    pre_relu: Vec<f32>,    // w1 @ input + b1
    hidden: Vec<f32>,      // relu(pre_relu)
    grid_pos: f32,
    g0: usize,
    g1: usize,
    alpha: f32,
}

/// Gradient accumulator for NTC parameters
struct NtcGradients {
    d_latent_grid: Vec<f32>,
    d_w1: Vec<f32>,
    d_b1: Vec<f32>,
    d_w2: Vec<f32>,
    d_b2: Vec<f32>,
}

impl NtcGradients {
    fn new(grid_size: usize, latent_dim: usize, input_dim: usize, hidden_dim: usize, kv_dim: usize) -> Self {
        Self {
            d_latent_grid: vec![0.0; grid_size * latent_dim],
            d_w1: vec![0.0; input_dim * hidden_dim],
            d_b1: vec![0.0; hidden_dim],
            d_w2: vec![0.0; hidden_dim * kv_dim],
            d_b2: vec![0.0; kv_dim],
        }
    }

    fn zero(&mut self) {
        for v in self.d_latent_grid.iter_mut() { *v = 0.0; }
        for v in self.d_w1.iter_mut() { *v = 0.0; }
        for v in self.d_b1.iter_mut() { *v = 0.0; }
        for v in self.d_w2.iter_mut() { *v = 0.0; }
        for v in self.d_b2.iter_mut() { *v = 0.0; }
    }
}

/// NTC KV Cache for a single layer, single K or V
pub struct NtcKvCache {
    pub grid_size: usize,          // Number of grid points (e.g. 64)
    pub latent_dim: usize,         // Latent dimension (e.g. 32)
    pub pe_freqs: usize,           // Number of positional encoding frequencies (e.g. 16)
    pub hidden_dim: usize,         // MLP hidden dim (e.g. 128)
    pub kv_dim: usize,             // Output dim (= n_kv_heads * head_dim)

    // Trainable parameters
    pub latent_grid: Vec<f32>,     // [grid_size, latent_dim]
    pub w1: Vec<f32>,              // [input_dim, hidden_dim] where input_dim = 2*pe_freqs + latent_dim
    pub b1: Vec<f32>,              // [hidden_dim]
    pub w2: Vec<f32>,              // [hidden_dim, kv_dim]
    pub b2: Vec<f32>,              // [kv_dim]

    // Recent buffer (uncompressed, exact values for last N tokens)
    pub recent_buffer: Vec<f32>,   // [recent_window, kv_dim]
    pub recent_window: usize,      // e.g. 32
    pub current_len: usize,

    // All seen K/V vectors for training (cleared after compression)
    pub training_buffer: Vec<f32>, // accumulated vectors for next training round
    pub training_positions: Vec<usize>,

    // Optimizer
    pub optimizer: AdamState,
}

impl NtcKvCache {
    pub fn new(grid_size: usize, latent_dim: usize, pe_freqs: usize, hidden_dim: usize, kv_dim: usize) -> Self {
        // input_dim = 2 * pe_freqs + latent_dim
        let input_dim = 2 * pe_freqs + latent_dim;
        let recent_window = 32;

        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        let scale_w1 = (2.0 / ((input_dim + hidden_dim) as f32)).sqrt();
        let scale_w2 = (2.0 / ((hidden_dim + kv_dim) as f32)).sqrt();

        let mut rng = rand::thread_rng();

        let mut w1 = vec![0.0; input_dim * hidden_dim];
        for v in w1.iter_mut() {
            *v = (rng.gen::<f32>() - 0.5) * 2.0 * scale_w1;
        }

        let mut w2 = vec![0.0; hidden_dim * kv_dim];
        for v in w2.iter_mut() {
            *v = (rng.gen::<f32>() - 0.5) * 2.0 * scale_w2;
        }

        // Small random latent grid
        let mut latent_grid = vec![0.0; grid_size * latent_dim];
        let scale_grid = 0.1;
        for v in latent_grid.iter_mut() {
            *v = (rng.gen::<f32>() - 0.5) * 2.0 * scale_grid;
        }

        Self {
            grid_size,
            latent_dim,
            pe_freqs,
            hidden_dim,
            kv_dim,
            latent_grid,
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: vec![0.0; kv_dim],
            recent_buffer: vec![0.0; recent_window * kv_dim],
            recent_window,
            current_len: 0,
            training_buffer: Vec::new(),
            training_positions: Vec::new(),
            optimizer: AdamState::new(0.001),
        }
    }

    /// Append a new K or V vector (called during forward pass)
    pub fn append(&mut self, kv_vector: &[f32]) {
        assert_eq!(kv_vector.len(), self.kv_dim);

        // 1. Add to recent_buffer (circular)
        let pos_in_recent = self.current_len % self.recent_window;
        let start = pos_in_recent * self.kv_dim;
        self.recent_buffer[start..start + self.kv_dim].copy_from_slice(kv_vector);

        // 2. Add to training buffer
        self.training_buffer.extend_from_slice(kv_vector);
        self.training_positions.push(self.current_len);

        self.current_len += 1;

        // 3. Every recent_window tokens, compress training buffer into NTC
        if self.current_len > 0 && self.current_len % self.recent_window == 0 {
            self.compress_batch();
        }
    }

    /// Compress accumulated training buffer into NTC representation
    fn compress_batch(&mut self) {
        let n_samples = self.training_positions.len();
        if n_samples == 0 {
            return;
        }

        // Mini-training: 50 steps of Adam
        for _step in 0..50 {
            let mut all_grads = NtcGradients::new(
                self.grid_size,
                self.latent_dim,
                2 * self.pe_freqs + self.latent_dim,
                self.hidden_dim,
                self.kv_dim,
            );

            for i in 0..n_samples {
                let pos = self.training_positions[i];
                let target = &self.training_buffer[i * self.kv_dim..(i + 1) * self.kv_dim];

                // Forward
                let (predicted, intermediates) = self.forward_with_intermediates(pos);

                // Backward: accumulate gradients
                self.backward(&predicted, target, &intermediates, &mut all_grads);
            }

            // Adam update
            self.apply_gradients(&all_grads, n_samples);
        }

        // Clear training buffer
        self.training_buffer.clear();
        self.training_positions.clear();
    }

    /// Reconstruct K or V for position `pos`
    pub fn decode(&self, pos: usize) -> Vec<f32> {
        // If pos is in recent_buffer, return exact value (zero loss)
        if self.current_len > 0 && pos >= self.current_len.saturating_sub(self.recent_window) {
            let idx = pos % self.recent_window;
            return self.recent_buffer[idx * self.kv_dim..(idx + 1) * self.kv_dim].to_vec();
        }

        // Otherwise, decode via NTC MLP
        self.forward_single(pos)
    }

    /// Forward pass of MLP decoder
    fn forward_single(&self, pos: usize) -> Vec<f32> {
        // 1. Positional Encoding
        let pe = positional_encoding(pos, self.pe_freqs, self.current_len.max(1));

        // 2. Latent Grid Lookup (linear interpolation)
        let latent = self.grid_lookup(pos);

        // 3. Concat: [pe, latent]
        let input: Vec<f32> = pe.iter().chain(latent.iter()).cloned().collect();

        // 4. MLP: Linear → ReLU → Linear
        let hidden = linear_relu(&input, &self.w1, &self.b1, self.hidden_dim);
        linear(&hidden, &self.w2, &self.b2, self.kv_dim)
    }

    /// Forward with intermediates for backprop
    fn forward_with_intermediates(&self, pos: usize) -> (Vec<f32>, ForwardIntermediates) {
        // 1. Positional Encoding
        let pe = positional_encoding(pos, self.pe_freqs, self.current_len.max(1));

        // 2. Grid lookup with interpolation info
        let grid_pos = (pos as f32 / self.current_len.max(1) as f32) * (self.grid_size - 1) as f32;
        let g0 = grid_pos.floor() as usize;
        let g1 = (g0 + 1).min(self.grid_size - 1);
        let alpha = grid_pos.fract();

        let latent: Vec<f32> = (0..self.latent_dim).map(|d| {
            self.latent_grid[g0 * self.latent_dim + d] * (1.0 - alpha)
            + self.latent_grid[g1 * self.latent_dim + d] * alpha
        }).collect();

        // 3. Concat
        let input: Vec<f32> = pe.iter().chain(latent.iter()).cloned().collect();

        // 4. First layer: Linear → ReLU
        let pre_relu: Vec<f32> = (0..self.hidden_dim).map(|j| {
            let sum: f32 = (0..input.len()).map(|i| input[i] * self.w1[i * self.hidden_dim + j]).sum::<f32>() + self.b1[j];
            sum
        }).collect();

        let hidden: Vec<f32> = pre_relu.iter().map(|x| x.max(0.0)).collect();

        // 5. Output layer: Linear
        let output: Vec<f32> = (0..self.kv_dim).map(|j| {
            let sum: f32 = (0..self.hidden_dim).map(|i| hidden[i] * self.w2[i * self.kv_dim + j]).sum::<f32>() + self.b2[j];
            sum
        }).collect();

        let intermediates = ForwardIntermediates {
            pe,
            latent,
            input,
            pre_relu,
            hidden,
            grid_pos,
            g0,
            g1,
            alpha,
        };

        (output, intermediates)
    }

    /// Latent grid bilinear interpolation
    fn grid_lookup(&self, pos: usize) -> Vec<f32> {
        let grid_pos = (pos as f32 / self.current_len.max(1) as f32) * (self.grid_size - 1) as f32;
        let g0 = grid_pos.floor() as usize;
        let g1 = (g0 + 1).min(self.grid_size - 1);
        let alpha = grid_pos.fract();

        (0..self.latent_dim).map(|d| {
            self.latent_grid[g0 * self.latent_dim + d] * (1.0 - alpha)
            + self.latent_grid[g1 * self.latent_dim + d] * alpha
        }).collect()
    }

    /// Compute gradients via manual backprop
    fn backward(&self, predicted: &[f32], target: &[f32], intermediates: &ForwardIntermediates, grads: &mut NtcGradients) {
        // Output layer gradient: d_loss/d_output = 2 * (predicted - target) / kv_dim
        let mut d_output = vec![0.0; self.kv_dim];
        for j in 0..self.kv_dim {
            d_output[j] = 2.0 * (predicted[j] - target[j]) / self.kv_dim as f32;
        }

        // Backprop through output layer (w2, b2, hidden)
        let mut d_hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            for j in 0..self.kv_dim {
                d_hidden[i] += d_output[j] * self.w2[i * self.kv_dim + j];
                grads.d_w2[i * self.kv_dim + j] += d_output[j] * intermediates.hidden[i];
            }
        }
        for j in 0..self.kv_dim {
            grads.d_b2[j] += d_output[j];
        }

        // Backprop through ReLU
        let mut d_pre_relu = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            if intermediates.pre_relu[i] > 0.0 {
                d_pre_relu[i] = d_hidden[i];
            }
        }

        // Backprop through first layer (w1, b1, input)
        let mut d_input = vec![0.0; intermediates.input.len()];
        for i in 0..intermediates.input.len() {
            for j in 0..self.hidden_dim {
                d_input[i] += d_pre_relu[j] * self.w1[i * self.hidden_dim + j];
                grads.d_w1[i * self.hidden_dim + j] += d_pre_relu[j] * intermediates.input[i];
            }
        }
        for j in 0..self.hidden_dim {
            grads.d_b1[j] += d_pre_relu[j];
        }

        // Backprop through PE and latent concat (input = [pe; latent])
        let d_pe = &d_input[0..2 * self.pe_freqs];
        let d_latent = &d_input[2 * self.pe_freqs..];

        // Backprop latent gradients to grid through bilinear interpolation
        for d in 0..self.latent_dim {
            let grad = d_latent[d];
            // Grid point g0
            grads.d_latent_grid[intermediates.g0 * self.latent_dim + d] += grad * (1.0 - intermediates.alpha);
            // Grid point g1
            grads.d_latent_grid[intermediates.g1 * self.latent_dim + d] += grad * intermediates.alpha;
        }
    }

    /// Apply accumulated gradients with Adam
    fn apply_gradients(&mut self, grads: &NtcGradients, _n_samples: usize) {
        // Update latent_grid
        self.optimizer.step(&mut self.latent_grid, &grads.d_latent_grid);

        // Update w1
        self.optimizer.step(&mut self.w1, &grads.d_w1);

        // Update b1
        self.optimizer.step(&mut self.b1, &grads.d_b1);

        // Update w2
        self.optimizer.step(&mut self.w2, &grads.d_w2);

        // Update b2
        self.optimizer.step(&mut self.b2, &grads.d_b2);
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let grid = self.grid_size * self.latent_dim * 4;
        let input_dim = 2 * self.pe_freqs + self.latent_dim;
        let w1 = input_dim * self.hidden_dim * 4;
        let b1 = self.hidden_dim * 4;
        let w2 = self.hidden_dim * self.kv_dim * 4;
        let b2 = self.kv_dim * 4;
        let recent = self.recent_window * self.kv_dim * 4;
        grid + w1 + b1 + w2 + b2 + recent
    }

    /// Status for dashboard/telemetry
    pub fn status_json(&self) -> serde_json::Value {
        json!({
            "type": "ntc_kv_cache",
            "current_len": self.current_len,
            "grid_size": self.grid_size,
            "latent_dim": self.latent_dim,
            "memory_bytes": self.memory_usage(),
            "recent_window": self.recent_window,
            "compression_ratio": if self.current_len > 0 {
                (self.current_len * self.kv_dim * 4) as f64 / self.memory_usage() as f64
            } else { 0.0 },
        })
    }
}

// ── Helper functions ──

fn positional_encoding(pos: usize, n_freqs: usize, max_len: usize) -> Vec<f32> {
    let norm_pos = pos as f32 / max_len.max(1) as f32;
    let mut pe = Vec::with_capacity(2 * n_freqs);
    for f in 0..n_freqs {
        let freq = (2.0f32).powi(f as i32) * std::f32::consts::PI;
        pe.push((freq * norm_pos).sin());
        pe.push((freq * norm_pos).cos());
    }
    pe
}

fn linear_relu(input: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    (0..out_dim).map(|j| {
        let sum: f32 = (0..in_dim).map(|i| input[i] * w[i * out_dim + j]).sum::<f32>() + b[j];
        sum.max(0.0)  // ReLU
    }).collect()
}

fn linear(input: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    (0..out_dim).map(|j| {
        let sum: f32 = (0..in_dim).map(|i| input[i] * w[i * out_dim + j]).sum::<f32>() + b[j];
        sum
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntc_creation() {
        let cache = NtcKvCache::new(16, 8, 8, 32, 64);

        assert_eq!(cache.grid_size, 16);
        assert_eq!(cache.latent_dim, 8);
        assert_eq!(cache.pe_freqs, 8);
        assert_eq!(cache.hidden_dim, 32);
        assert_eq!(cache.kv_dim, 64);
        assert_eq!(cache.current_len, 0);
        assert_eq!(cache.recent_window, 32);
    }

    #[test]
    fn test_ntc_append_single() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        let kv = vec![0.1; 64];
        cache.append(&kv);

        assert_eq!(cache.current_len, 1);
        assert_eq!(cache.training_positions.len(), 1);
        assert_eq!(cache.training_buffer.len(), 64);
    }

    #[test]
    fn test_ntc_recent_buffer_exact() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        // Append several vectors
        for i in 0..5 {
            let mut kv = vec![0.0; 64];
            kv[0] = i as f32;  // Mark with position
            cache.append(&kv);
        }

        // Recent buffer should return exact values for recent positions
        let decoded = cache.decode(4);  // Last position (in recent window)
        assert_eq!(decoded[0], 4.0, "Recent buffer should return exact values");
    }

    #[test]
    fn test_ntc_append_and_compress() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        // Append exactly recent_window (32) tokens to trigger compression
        for i in 0..32 {
            let mut kv = vec![0.0; 64];
            kv[0] = (i as f32) / 32.0;  // Normalized values
            cache.append(&kv);
        }

        // Training buffer should be cleared after compression
        assert_eq!(cache.training_buffer.len(), 0);
        assert_eq!(cache.training_positions.len(), 0);
        assert_eq!(cache.current_len, 32);
    }

    #[test]
    fn test_ntc_decode_after_compression() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        // Create training data with a simple pattern
        for i in 0..32 {
            let mut kv = vec![0.0; 64];
            let val = (i as f32) / 32.0;
            for j in 0..64 {
                kv[j] = val;  // All elements have same value based on position
            }
            cache.append(&kv);
        }

        // After compression, decode an older position
        let decoded = cache.decode(15);  // Middle of sequence
        assert_eq!(decoded.len(), 64);

        // The decoded values should be reasonable (not NaN/Inf)
        for &v in &decoded {
            assert!(v.is_finite(), "Decoded values should be finite");
        }
    }

    #[test]
    fn test_ntc_compression_ratio() {
        // Use large kv_dim and many tokens so uncompressed data >> NTC model size
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 512);

        // 256 tokens × 512 kv_dim × 4 bytes = 512KB uncompressed
        for i in 0..256 {
            let mut kv = vec![0.0; 512];
            kv[0] = (i as f32) / 256.0;
            cache.append(&kv);
        }

        let ratio = cache.status_json()["compression_ratio"].as_f64().unwrap();
        assert!(ratio > 1.0, "Compression ratio should be > 1.0, got {}", ratio);
    }

    #[test]
    fn test_ntc_memory_usage() {
        let cache = NtcKvCache::new(16, 8, 8, 32, 64);

        let mem = cache.memory_usage();

        // Verify components:
        // latent_grid: 16 * 8 * 4 = 512 bytes
        // input_dim = 2*8 + 8 = 24
        // w1: 24 * 32 * 4 = 3072 bytes
        // b1: 32 * 4 = 128 bytes
        // w2: 32 * 64 * 4 = 8192 bytes
        // b2: 64 * 4 = 256 bytes
        // recent: 32 * 64 * 4 = 8192 bytes
        // Total: 512 + 3072 + 128 + 8192 + 256 + 8192 = 20352 bytes

        let expected = 512 + 3072 + 128 + 8192 + 256 + 8192;
        assert_eq!(mem, expected, "Memory usage should match expected calculation");
    }

    #[test]
    fn test_ntc_status_json() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        for i in 0..10 {
            let mut kv = vec![0.0; 64];
            kv[0] = i as f32;
            cache.append(&kv);
        }

        let status = cache.status_json();

        assert_eq!(status["type"].as_str().unwrap(), "ntc_kv_cache");
        assert_eq!(status["current_len"].as_u64().unwrap(), 10);
        assert_eq!(status["grid_size"].as_u64().unwrap(), 16);
        assert_eq!(status["latent_dim"].as_u64().unwrap(), 8);
        assert!(status["memory_bytes"].is_u64());
        assert!(status["compression_ratio"].is_f64());
    }

    #[test]
    fn test_ntc_edge_case_single_token() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        let kv = vec![0.5; 64];
        cache.append(&kv);

        assert_eq!(cache.current_len, 1);

        // Decode should return from recent buffer (exact)
        let decoded = cache.decode(0);
        assert_eq!(decoded.len(), 64);
        for &v in &decoded {
            assert_eq!(v, 0.5);
        }
    }

    #[test]
    fn test_ntc_edge_case_empty() {
        let cache = NtcKvCache::new(16, 8, 8, 32, 64);

        assert_eq!(cache.current_len, 0);
        assert_eq!(cache.memory_usage(), 512 + 3072 + 128 + 8192 + 256 + 8192);

        let status = cache.status_json();
        assert_eq!(status["compression_ratio"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn test_ntc_recent_window_circular() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        // Add 64 vectors (2x recent_window)
        for i in 0..64 {
            let mut kv = vec![0.0; 64];
            kv[0] = i as f32;
            cache.append(&kv);
        }

        // After compression, recent positions (48-63) should be in recent buffer
        // Positions < 32 should be decoded from NTC

        // Last position should be exact
        let decoded_63 = cache.decode(63);
        assert_eq!(decoded_63[0], 63.0);

        // Earlier position should be decoded (approximately)
        let decoded_16 = cache.decode(16);
        assert!(decoded_16.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ntc_mlp_forward_finite() {
        let mut cache = NtcKvCache::new(8, 4, 4, 16, 32);

        // Append some data to enable decoding
        for i in 0..16 {
            let mut kv = vec![0.1; 32];
            kv[0] = i as f32;
            cache.append(&kv);
        }

        // Decode position should produce finite values
        for pos in 0..16 {
            let decoded = cache.decode(pos);
            for &v in &decoded {
                assert!(v.is_finite(), "Position {} produced non-finite value", pos);
            }
        }
    }

    #[test]
    fn test_ntc_training_accumulation() {
        let mut cache = NtcKvCache::new(16, 8, 8, 32, 64);

        // Add vectors one by one before compression
        for i in 0..16 {
            let mut kv = vec![0.0; 64];
            kv[0] = i as f32;
            cache.append(&kv);

            // Training buffer should accumulate
            assert_eq!(cache.training_buffer.len(), (i + 1) * 64);
            assert_eq!(cache.training_positions.len(), i + 1);
        }

        // Add more to trigger compression at position 32
        for i in 16..32 {
            let mut kv = vec![0.0; 64];
            kv[0] = i as f32;
            cache.append(&kv);
        }

        // After compression, training buffers should be cleared
        assert_eq!(cache.training_buffer.len(), 0);
        assert_eq!(cache.training_positions.len(), 0);
    }

    #[test]
    fn test_ntc_large_kv_dim() {
        let mut cache = NtcKvCache::new(32, 16, 16, 64, 256);

        for i in 0..32 {
            let mut kv = vec![0.0; 256];
            kv[0] = i as f32;
            cache.append(&kv);
        }

        // Should compress without issue
        assert_eq!(cache.training_buffer.len(), 0);

        // Memory should be reasonable
        let mem = cache.memory_usage();
        // latent: 32*16*4 = 2048
        // input_dim = 2*16+16 = 48
        // w1: 48*64*4 = 12288
        // b1: 64*4 = 256
        // w2: 64*256*4 = 65536
        // b2: 256*4 = 1024
        // recent: 32*256*4 = 32768
        let expected = 2048 + 12288 + 256 + 65536 + 1024 + 32768;
        assert_eq!(mem, expected);
    }

    #[test]
    fn test_ntc_small_grid() {
        let mut cache = NtcKvCache::new(4, 2, 2, 8, 16);

        for i in 0..8 {
            let kv = vec![i as f32 / 8.0; 16];
            cache.append(&kv);
        }

        // Should work with minimal parameters
        assert_eq!(cache.current_len, 8);
    }
}
