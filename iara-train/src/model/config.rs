use serde::Deserialize;
use std::path::Path;

/// Qwen2.5 model configuration.
///
/// Mirrors compute_core::config::ModelConfig fields using HuggingFace naming.
/// Parsing is delegated to compute_core::config::ModelConfig::from_json_with_cap.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
}

fn default_head_dim() -> usize { 128 }

impl From<compute_core::config::ModelConfig> for Qwen2Config {
    fn from(c: compute_core::config::ModelConfig) -> Self {
        Self {
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads,
            vocab_size: c.vocab_size,
            max_position_embeddings: c.max_position_embeddings,
            rms_norm_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
            head_dim: c.head_dim,
        }
    }
}

impl Qwen2Config {
    /// Parse config.json via compute_core::config::ModelConfig.
    /// No seq_len cap applied (training manages its own context length).
    pub fn from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mc = compute_core::config::ModelConfig::from_json_with_cap(path, usize::MAX)?;
        Ok(Self::from(mc))
    }

    pub fn kv_heads_per_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
