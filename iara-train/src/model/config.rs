use serde::Deserialize;
use std::path::Path;

/// Qwen2.5 model configuration, loaded from config.json
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

impl Qwen2Config {
    pub fn from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&content)?;
        Ok(cfg)
    }

    pub fn kv_heads_per_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
