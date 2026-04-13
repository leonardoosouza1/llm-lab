//! Qwen2.5 transformer architecture implemented in burn.
//!
//! Supports the full Qwen2.5 family (0.5B through 72B) via configurable
//! parameters. Implements Grouped Query Attention (GQA), RoPE, SwiGLU MLP,
//! and RMSNorm — matching the HuggingFace `Qwen2ForCausalLM` layout.

use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
    RotaryEncodingConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};

use super::config::Qwen2Config;

// ---------------------------------------------------------------------------
// Model output
// ---------------------------------------------------------------------------

/// Output of a full forward pass through the Qwen2 model.
pub struct Qwen2Output<B: Backend> {
    /// Logits over the vocabulary: `[batch, seq_len, vocab_size]`.
    pub logits: Tensor<B, 3>,
    /// Per-layer hidden states *after* the residual add, if requested.
    /// Entry `i` has shape `[batch, seq_len, hidden_size]`.
    pub hidden_states: Option<Vec<Tensor<B, 3>>>,
}

// ---------------------------------------------------------------------------
// Qwen2 MLP (SwiGLU)
// ---------------------------------------------------------------------------

/// SwiGLU MLP block: `down_proj(silu(gate_proj(x)) * up_proj(x))`
#[derive(Module, Debug)]
pub struct Qwen2Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Qwen2Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ---------------------------------------------------------------------------
// Qwen2 Attention (GQA + RoPE)
// ---------------------------------------------------------------------------

/// Grouped-Query Attention with Rotary Position Embeddings.
#[derive(Module, Debug)]
pub struct Qwen2Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rotary: RotaryEncoding<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Qwen2Attention<B> {
    /// Run self-attention over `hidden_states` of shape `[batch, seq_len, hidden_size]`.
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = hidden_states.dims();
        let device = hidden_states.device();

        // Project to Q, K, V
        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE to Q and K
        // RotaryEncoding::forward expects (..., seq_len, head_dim)
        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);

        // GQA: expand K/V heads to match Q heads
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads);
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        // Causal mask: positions can only attend to earlier positions
        let causal_mask =
            Tensor::<B, 2, Bool>::triu_mask([seq_len, seq_len], 1, &device);
        // triu_mask with offset=1 marks everything *above* the diagonal as true
        // mask_fill sets those positions to -inf
        let attn_weights = attn_weights.mask_fill(
            causal_mask.unsqueeze_dims::<4>(&[0, 1]),
            f32::NEG_INFINITY,
        );

        let attn_weights = softmax(attn_weights, 3);

        // Weighted sum of values
        let attn_output = attn_weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }
}

/// Repeat K/V heads to match the number of Q heads for GQA.
/// Input shape: `[batch, num_kv_heads, seq_len, head_dim]`
/// Output shape: `[batch, num_kv_heads * n_rep, seq_len, head_dim]`
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([batch, num_kv_heads, n_rep, seq_len, head_dim])
        .reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
}

// ---------------------------------------------------------------------------
// Qwen2 Decoder Layer
// ---------------------------------------------------------------------------

/// A single transformer decoder layer.
#[derive(Module, Debug)]
pub struct Qwen2DecoderLayer<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: Qwen2Attention<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: Qwen2Mlp<B>,
}

impl<B: Backend> Qwen2DecoderLayer<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm attention
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states);
        let hidden_states = residual + hidden_states;

        // Pre-norm MLP
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }
}

// ---------------------------------------------------------------------------
// Qwen2 Model (without lm_head)
// ---------------------------------------------------------------------------

/// Core Qwen2 model: embedding + N decoder layers + final RMSNorm.
#[derive(Module, Debug)]
pub struct Qwen2Model<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<Qwen2DecoderLayer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> Qwen2Model<B> {
    /// Forward pass returning the final hidden state `[batch, seq_len, hidden_size]`.
    /// When `return_all_hidden_states` is true, collects per-layer outputs.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        return_all_hidden_states: bool,
    ) -> (Tensor<B, 3>, Option<Vec<Tensor<B, 3>>>) {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        let mut all_hidden_states = if return_all_hidden_states {
            Some(Vec::with_capacity(self.layers.len()))
        } else {
            None
        };

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
            if let Some(ref mut states) = all_hidden_states {
                states.push(hidden_states.clone());
            }
        }

        hidden_states = self.norm.forward(hidden_states);
        (hidden_states, all_hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Qwen2 For Causal LM (full model with lm_head)
// ---------------------------------------------------------------------------

/// Complete Qwen2.5 causal language model.
#[derive(Module, Debug)]
pub struct Qwen2ForCausalLM<B: Backend> {
    model: Qwen2Model<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> Qwen2ForCausalLM<B> {
    /// Full forward pass.
    ///
    /// - `input_ids`: `[batch, seq_len]` token indices.
    /// - `return_all_hidden_states`: collect per-layer hidden states for distillation.
    ///
    /// Returns `Qwen2Output` with logits `[batch, seq_len, vocab_size]`.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        return_all_hidden_states: bool,
    ) -> Qwen2Output<B> {
        let (hidden_states, all_hidden_states) =
            self.model.forward(input_ids, return_all_hidden_states);
        let logits = self.lm_head.forward(hidden_states);
        Qwen2Output {
            logits,
            hidden_states: all_hidden_states,
        }
    }
}

// ---------------------------------------------------------------------------
// Initialization from Qwen2Config
// ---------------------------------------------------------------------------

impl Qwen2Config {
    /// Build the full `Qwen2ForCausalLM` from this config, with randomly
    /// initialized weights on `device`.
    pub fn init_model<B: Backend>(&self, device: &B::Device) -> Qwen2ForCausalLM<B> {
        let model = self.init_base_model(device);
        let lm_head = LinearConfig::new(self.hidden_size, self.vocab_size)
            .with_bias(false)
            .init(device);
        Qwen2ForCausalLM { model, lm_head }
    }

    /// Build just the base `Qwen2Model` (no lm_head).
    pub fn init_base_model<B: Backend>(&self, device: &B::Device) -> Qwen2Model<B> {
        let embed_tokens =
            EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);

        let layers = (0..self.num_hidden_layers)
            .map(|_| self.init_decoder_layer(device))
            .collect();

        let norm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.rms_norm_eps)
            .init(device);

        Qwen2Model {
            embed_tokens,
            layers,
            norm,
        }
    }

    fn init_decoder_layer<B: Backend>(&self, device: &B::Device) -> Qwen2DecoderLayer<B> {
        let input_layernorm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.rms_norm_eps)
            .init(device);
        let post_attention_layernorm = RmsNormConfig::new(self.hidden_size)
            .with_epsilon(self.rms_norm_eps)
            .init(device);
        let self_attn = self.init_attention(device);
        let mlp = self.init_mlp(device);

        Qwen2DecoderLayer {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    fn init_attention<B: Backend>(&self, device: &B::Device) -> Qwen2Attention<B> {
        let kv_dim = self.num_key_value_heads * self.head_dim;

        let q_proj = LinearConfig::new(self.hidden_size, self.num_attention_heads * self.head_dim)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(self.hidden_size, kv_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(self.hidden_size, kv_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(self.num_attention_heads * self.head_dim, self.hidden_size)
            .with_bias(false)
            .init(device);

        let rotary = RotaryEncodingConfig::new(self.max_position_embeddings, self.head_dim)
            .with_theta(self.rope_theta as f32)
            .init(device);

        Qwen2Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
        }
    }

    fn init_mlp<B: Backend>(&self, device: &B::Device) -> Qwen2Mlp<B> {
        let gate_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(self.intermediate_size, self.hidden_size)
            .with_bias(false)
            .init(device);

        Qwen2Mlp {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight loading from safetensors
// ---------------------------------------------------------------------------

impl Qwen2Config {
    /// Load a pre-trained model from a safetensors file.
    ///
    /// Creates the model structure, then replaces all random weights with
    /// the pre-trained values from the safetensors file.
    pub fn load_from_safetensors<B: Backend>(
        &self,
        st_path: &std::path::Path,
        device: &B::Device,
    ) -> Result<Qwen2ForCausalLM<B>, Box<dyn std::error::Error>> {
        use crate::export::safetensors::{load_safetensors_header, load_safetensors_tensor};
        use burn::module::{Param, ParamId};
        use burn::tensor::TensorData;

        log::info!("Loading weights from {}", st_path.display());
        let t0 = std::time::Instant::now();

        let header = load_safetensors_header(st_path)?;
        log::info!("  Header: {} tensors", header.len());

        // Helper: load a 2D tensor as a burn Param
        // IMPORTANT: PyTorch stores Linear weights as [out, in] but burn expects [in, out]
        // So we transpose when loading.
        let load_2d = |name: &str| -> Param<Tensor<B, 2>> {
            let meta = header.get(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
            let data = load_safetensors_tensor(st_path, name)
                .unwrap_or_else(|e| panic!("failed to load {name}: {e}"));
            let shape: [usize; 2] = [meta.shape[0], meta.shape[1]];
            let tensor = Tensor::from_data(TensorData::new(data, shape), device);
            // Transpose [out, in] → [in, out] for burn's Linear
            let tensor = tensor.transpose();
            Param::initialized(ParamId::new(), tensor)
        };

        // Helper: load a 1D tensor as a burn Param
        let load_1d = |name: &str| -> Param<Tensor<B, 1>> {
            let meta = header.get(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
            let data = load_safetensors_tensor(st_path, name)
                .unwrap_or_else(|e| panic!("failed to load {name}: {e}"));
            let shape: [usize; 1] = [meta.shape[0]];
            let tensor = Tensor::from_data(TensorData::new(data, shape), device);
            Param::initialized(ParamId::new(), tensor)
        };

        // Create model with random weights, then replace
        let mut model = self.init_model(device);

        // Embedding (NOT transposed — shape [vocab, hidden] is same in PyTorch and burn)
        {
            let name = "model.embed_tokens.weight";
            let meta = header.get(name).unwrap();
            let data = load_safetensors_tensor(st_path, name).unwrap();
            let shape: [usize; 2] = [meta.shape[0], meta.shape[1]];
            let tensor = Tensor::from_data(TensorData::new(data, shape), device);
            model.model.embed_tokens.weight = Param::initialized(ParamId::new(), tensor);
        }

        // Decoder layers
        for i in 0..self.num_hidden_layers {
            let p = format!("model.layers.{i}");

            // Norms
            model.model.layers[i].input_layernorm.gamma =
                load_1d(&format!("{p}.input_layernorm.weight"));
            model.model.layers[i].post_attention_layernorm.gamma =
                load_1d(&format!("{p}.post_attention_layernorm.weight"));

            // Attention
            model.model.layers[i].self_attn.q_proj.weight =
                load_2d(&format!("{p}.self_attn.q_proj.weight"));
            model.model.layers[i].self_attn.k_proj.weight =
                load_2d(&format!("{p}.self_attn.k_proj.weight"));
            model.model.layers[i].self_attn.v_proj.weight =
                load_2d(&format!("{p}.self_attn.v_proj.weight"));
            model.model.layers[i].self_attn.o_proj.weight =
                load_2d(&format!("{p}.self_attn.o_proj.weight"));

            // MLP
            model.model.layers[i].mlp.gate_proj.weight =
                load_2d(&format!("{p}.mlp.gate_proj.weight"));
            model.model.layers[i].mlp.up_proj.weight =
                load_2d(&format!("{p}.mlp.up_proj.weight"));
            model.model.layers[i].mlp.down_proj.weight =
                load_2d(&format!("{p}.mlp.down_proj.weight"));

            if i % 7 == 0 {
                log::info!("  Loaded layer {i}/{}", self.num_hidden_layers);
            }
        }

        // Final norm
        model.model.norm.gamma = load_1d("model.norm.weight");

        // LM head (may share weights with embed_tokens)
        if header.contains_key("lm_head.weight") {
            model.lm_head.weight = load_2d("lm_head.weight");
        } else {
            // Weight tying: lm_head shares embed_tokens weight (transposed)
            // embed_tokens is [vocab, hidden], lm_head needs [hidden, vocab]
            log::info!("  lm_head.weight not found, tying with embed_tokens (transposed)");
            let embed_w = model.model.embed_tokens.weight.val();
            let lm_head_w = embed_w.transpose();
            model.lm_head.weight = Param::initialized(ParamId::new(), lm_head_w);
        }

        log::info!(
            "Model loaded in {:.1}s ({} layers)",
            t0.elapsed().as_secs_f32(),
            self.num_hidden_layers,
        );

        Ok(model)
    }
}
