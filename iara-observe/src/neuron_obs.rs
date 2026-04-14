use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronSnapshot {
    pub layer_idx: usize,
    pub step: u64,
    pub timestamp_us: u64,

    // Attention metrics
    pub attn_entropy: f32,          // Shannon entropy of attention weights
    pub attn_top_positions: Vec<(usize, f32)>,  // top-5 attended positions
    pub attn_head_energies: Vec<f32>,            // per-head energy (sum Q^2)

    // MLP metrics
    pub mlp_activation_rate: f32,   // fraction non-zero post-SiLU
    pub mlp_top_neurons: Vec<(usize, f32)>,     // top-10 most active
    pub mlp_dead_neuron_count: u32,              // neurons with |act| < 1e-6
    pub mlp_gate_sparsity: f32,                  // gate_proj zero fraction

    // Residual stream
    pub residual_norm_pre: f32,     // L2 norm before layer
    pub residual_norm_post: f32,    // L2 norm after layer
    pub residual_delta: f32,        // L2(post - pre) -- how much the layer changed

    // Phase coherence (Kuramoto)
    pub phase_coherence: f32,       // 0.0=incoherent, 1.0=synchronized

    // Gradient info (training only)
    pub gradient_norm: Option<f32>,
    pub gradient_max: Option<f32>,
}

impl NeuronSnapshot {
    pub fn new(layer_idx: usize, step: u64) -> Self {
        Self {
            layer_idx,
            step,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            attn_entropy: 0.0,
            attn_top_positions: Vec::new(),
            attn_head_energies: Vec::new(),
            mlp_activation_rate: 0.0,
            mlp_top_neurons: Vec::new(),
            mlp_dead_neuron_count: 0,
            mlp_gate_sparsity: 0.0,
            residual_norm_pre: 0.0,
            residual_norm_post: 0.0,
            residual_delta: 0.0,
            phase_coherence: 0.0,
            gradient_norm: None,
            gradient_max: None,
        }
    }

    /// Layer is "dead" if residual delta is negligible
    pub fn is_dead_layer(&self) -> bool {
        self.residual_delta < 1e-4
    }

    /// Layer has anomalous attention (too focused or too diffuse)
    pub fn has_attn_anomaly(&self, seq_len: usize) -> bool {
        let max_entropy = (seq_len as f32).ln();
        self.attn_entropy < 0.1 || self.attn_entropy > max_entropy * 0.95
    }
}
