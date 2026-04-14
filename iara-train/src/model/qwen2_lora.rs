//! Qwen2.5 with LoRA adapters on attention projections.
//!
//! Wraps Q, K, V, O projections with `LoraLinear` for parameter-efficient
//! fine-tuning. MLP and norms remain frozen. After training, call
//! `merge_lora()` to fold LoRA weights back into the base model.

use burn::module::{Module, Param, ParamId};
use burn::nn::{Embedding, Linear, RmsNorm, RotaryEncoding};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::lora::LoraLinear;
use super::qwen2::{Qwen2ForCausalLM, Qwen2Mlp};

// ---------------------------------------------------------------------------
// LoRA-wrapped Attention
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Qwen2AttentionLoRA<B: Backend> {
    pub q_proj: LoraLinear<B>,
    pub k_proj: LoraLinear<B>,
    pub v_proj: LoraLinear<B>,
    pub o_proj: LoraLinear<B>,
    pub rotary: RotaryEncoding<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Qwen2AttentionLoRA<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = hidden_states.dims();
        let device = hidden_states.device();

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads);
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads);

        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        let causal_mask: Tensor<B, 2, Bool> = {
            use burn::tensor::Int;
            let ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
            let row_ids = ids.clone().reshape([seq_len, 1]).expand([seq_len, seq_len]);
            let col_ids = ids.reshape([1, seq_len]).expand([seq_len, seq_len]);
            col_ids.greater(row_ids)
        };
        let attn_weights = attn_weights.mask_fill(
            causal_mask.unsqueeze_dims::<4>(&[0, 1]),
            f32::NEG_INFINITY,
        );

        let attn_weights = softmax(attn_weights, 3);

        let attn_output = attn_weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }
}

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
// LoRA Decoder Layer (attn = LoRA, MLP = frozen)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Qwen2DecoderLayerLoRA<B: Backend> {
    pub input_layernorm: RmsNorm<B>,
    pub self_attn: Qwen2AttentionLoRA<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub mlp: Qwen2Mlp<B>,
}

impl<B: Backend> Qwen2DecoderLayerLoRA<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }
}

// ---------------------------------------------------------------------------
// Full LoRA Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Qwen2ModelLoRA<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Qwen2DecoderLayerLoRA<B>>,
    pub norm: RmsNorm<B>,
}

impl<B: Backend> Qwen2ModelLoRA<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
        }
        self.norm.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Qwen2ForCausalLMLoRA<B: Backend> {
    pub model: Qwen2ModelLoRA<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> Qwen2ForCausalLMLoRA<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let hidden_states = self.model.forward(input_ids);
        self.lm_head.forward(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Conversion: base model → LoRA model
// ---------------------------------------------------------------------------

impl<B: Backend> Qwen2ForCausalLMLoRA<B> {
    /// Wrap a pre-trained `Qwen2ForCausalLM` with LoRA adapters on all
    /// attention projections (Q, K, V, O). MLP and norms are kept frozen.
    pub fn from_pretrained(
        base: Qwen2ForCausalLM<B>,
        rank: usize,
        alpha: f64,
        device: &B::Device,
    ) -> Self {
        let layers = base.model.layers.into_iter().map(|layer| {
            let self_attn = Qwen2AttentionLoRA {
                q_proj: LoraLinear::new(layer.self_attn.q_proj, rank, alpha, device),
                k_proj: LoraLinear::new(layer.self_attn.k_proj, rank, alpha, device),
                v_proj: LoraLinear::new(layer.self_attn.v_proj, rank, alpha, device),
                o_proj: LoraLinear::new(layer.self_attn.o_proj, rank, alpha, device),
                rotary: layer.self_attn.rotary,
                num_heads: layer.self_attn.num_heads,
                num_kv_heads: layer.self_attn.num_kv_heads,
                head_dim: layer.self_attn.head_dim,
            };
            Qwen2DecoderLayerLoRA {
                input_layernorm: layer.input_layernorm.no_grad(),
                self_attn,
                post_attention_layernorm: layer.post_attention_layernorm.no_grad(),
                mlp: layer.mlp.no_grad(),
            }
        }).collect();

        let model = Qwen2ModelLoRA {
            embed_tokens: base.model.embed_tokens.no_grad(),
            layers,
            norm: base.model.norm.no_grad(),
        };

        Qwen2ForCausalLMLoRA {
            model,
            lm_head: base.lm_head.no_grad(),
        }
    }

    /// Merge LoRA weights back into base Linear layers.
    /// Returns a standard `Qwen2ForCausalLM` ready for inference/export.
    pub fn merge_lora(self) -> Qwen2ForCausalLM<B> {
        use super::qwen2::{Qwen2Attention, Qwen2DecoderLayer, Qwen2Model};

        let layers = self.model.layers.into_iter().map(|layer| {
            Qwen2DecoderLayer {
                input_layernorm: layer.input_layernorm,
                self_attn: Qwen2Attention {
                    q_proj: layer.self_attn.q_proj.merge(),
                    k_proj: layer.self_attn.k_proj.merge(),
                    v_proj: layer.self_attn.v_proj.merge(),
                    o_proj: layer.self_attn.o_proj.merge(),
                    rotary: layer.self_attn.rotary,
                    num_heads: layer.self_attn.num_heads,
                    num_kv_heads: layer.self_attn.num_kv_heads,
                    head_dim: layer.self_attn.head_dim,
                },
                post_attention_layernorm: layer.post_attention_layernorm,
                mlp: layer.mlp,
            }
        }).collect();

        let model = Qwen2Model {
            embed_tokens: self.model.embed_tokens,
            layers,
            norm: self.model.norm,
        };

        Qwen2ForCausalLM {
            model,
            lm_head: self.lm_head,
        }
    }
}
