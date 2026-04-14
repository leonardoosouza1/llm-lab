// ============================================================================
// config.rs — Unified model configuration for transformer LLMs
// ============================================================================
// Parses config.json (HuggingFace format) into a single struct used by both
// iara-engine (inference) and iara-train (fine-tuning).
// ============================================================================

use serde::Deserialize;
use std::path::Path;

/// Unified model configuration covering all fields needed by inference and training.
///
/// Field names follow the HuggingFace convention where possible.
/// For iara-engine compatibility, convenience aliases are provided.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub head_dim: usize,
}

fn default_max_position_embeddings() -> usize { 2048 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 1_000_000.0 }

impl ModelConfig {
    /// Parse a HuggingFace config.json, supporting nested `text_config` and
    /// architecture validation. Caps `max_seq_len` to `cap` (default 1024) to
    /// prevent VRAM bloat on long-context models.
    pub fn from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_json_with_cap(path, 1024)
    }

    /// Same as `from_json` but with a custom max_seq_len cap.
    pub fn from_json_with_cap(path: &Path, cap: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&content)?;

        let arch = v["architectures"].as_array()
            .and_then(|a| a.first())
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        if !arch.contains("CausalLM") && !arch.contains("ForCausalLM")
            && !v["hidden_size"].is_number() && !v["text_config"]["hidden_size"].is_number()
        {
            return Err(format!(
                "Architecture '{}' does not appear to be a CausalLM. Check config.json.",
                arch
            ).into());
        }

        let params = if v["hidden_size"].is_number() { &v }
                     else if v["text_config"]["hidden_size"].is_number() { &v["text_config"] }
                     else { return Err(format!("hidden_size missing in {}", path.display()).into()); };

        let uint = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
            let val = &params[key];
            val.as_u64().or_else(|| val.as_f64().map(|f| f as u64))
               .map(|n| n as usize)
               .ok_or_else(|| format!("'{}' missing in config.json", key).into())
        };

        let hidden_size = uint("hidden_size")?;
        let num_attention_heads = uint("num_attention_heads")?;
        let head_dim = params["head_dim"].as_u64()
            .or_else(|| params["head_dim"].as_f64().map(|f| f as u64))
            .map(|n| n as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        let max_position_embeddings = params["max_position_embeddings"].as_u64()
            .map(|n| n as usize)
            .unwrap_or(2048)
            .min(cap);

        Ok(Self {
            hidden_size,
            intermediate_size: uint("intermediate_size")?,
            num_hidden_layers: uint("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: uint("num_key_value_heads")?,
            vocab_size: uint("vocab_size")?,
            max_position_embeddings,
            rms_norm_eps: params["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            rope_theta: params["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            head_dim,
        })
    }

    // ── Convenience aliases (iara-engine compatibility) ──────────────────────

    #[inline] pub fn num_layers(&self) -> usize { self.num_hidden_layers }
    #[inline] pub fn num_heads(&self) -> usize { self.num_attention_heads }
    #[inline] pub fn num_kv_heads(&self) -> usize { self.num_key_value_heads }
    #[inline] pub fn rms_eps(&self) -> f32 { self.rms_norm_eps as f32 }
    #[inline] pub fn rope_theta_f32(&self) -> f32 { self.rope_theta as f32 }
    #[inline] pub fn max_seq_len(&self) -> usize { self.max_position_embeddings }

    /// Number of Q heads per KV group (for GQA).
    #[inline]
    pub fn kv_heads_per_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json_roundtrip() {
        let dir = std::env::temp_dir().join("compute_core_test_config");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "head_dim": 128
        }"#).unwrap();

        let cfg = ModelConfig::from_json(&path).unwrap();
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.max_position_embeddings, 1024); // capped
        assert_eq!(cfg.kv_heads_per_group(), 6);
        assert_eq!(cfg.num_layers(), 28);
        assert_eq!(cfg.rms_eps(), 1e-6_f32);

        std::fs::remove_dir_all(&dir).ok();
    }
}
