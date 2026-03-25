// ============================================================================
// model.rs — Estrutura Funcional do Qwen 8B em Candle
// ============================================================================
// Define as camadas do modelo e os métodos de forward pass reais (GPU).
// Inclui otimização de KV-Cache para performance máxima.
// ============================================================================

use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};
use std::path::Path;

pub struct QwenLayer {
    pub attention_wq: Linear,
    pub attention_wk: Linear,
    pub attention_wv: Linear,
    pub attention_wo: Linear,
    pub mlp_gate: Linear,
    pub mlp_up: Linear,
    pub mlp_down: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    // Otimização: Cache de Chaves e Valores (KV-Cache)
    pub kv_cache: Option<(Tensor, Tensor)>,
}

pub struct QwenModel {
    pub embed_tokens: Tensor,
    pub layers: Vec<QwenLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub device: Device,
}

impl QwenModel {
    /// Carrega o modelo real para a GPU (VRAM) a partir dos arquivos .safetensors
    pub fn load(model_dir: &Path, device: &Device, mcfg: &crate::ModelConfig) -> Result<Self> {
        let hidden_size = mcfg.hidden_size;
        let intermediate_size = mcfg.intermediate_size.unwrap_or(12288);
        let num_layers = mcfg.num_hidden_layers;
        let vocab_size = mcfg.vocab_size;

        log::info!("⏳ Carregando Qwen 8B: {} camadas, hidden={}, intermediate={}, vocab={} ({:?})", 
            num_layers, hidden_size, intermediate_size, vocab_size, device);
        
        let mut filenames = vec![];
        for entry in std::fs::read_dir(model_dir).expect("Failed to read model dir") {
            let path = entry.expect("Failed to read entry").path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                filenames.push(path);
            }
        }
        filenames.sort();
        
        if filenames.is_empty() {
            return Err(candle_core::Error::msg("Nenhum arquivo .safetensors encontrado no diretório").into());
        }

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::BF16, device)? };
        let embed_tokens = vb.get((vocab_size, hidden_size), "model.embed_tokens.weight")?;
        
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let vb_l = vb.pp(format!("model.layers.{}", i));
            
            let attention_wq = candle_nn::linear_no_bias(hidden_size, hidden_size, vb_l.pp("self_attn.q_proj"))?;
            let k_weight = vb_l.pp("self_attn.k_proj").get((1024, hidden_size), "weight")?;
            let v_weight = vb_l.pp("self_attn.v_proj").get((1024, hidden_size), "weight")?;
            
            let attention_wk = Linear::new(k_weight, None);
            let attention_wv = Linear::new(v_weight, None);
            let attention_wo = candle_nn::linear_no_bias(hidden_size, hidden_size, vb_l.pp("self_attn.o_proj"))?;
            
            let mlp_gate = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb_l.pp("mlp.gate_proj"))?;
            let mlp_up = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb_l.pp("mlp.up_proj"))?;
            let mlp_down = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb_l.pp("mlp.down_proj"))?;
            
            let input_layernorm = candle_nn::rms_norm(hidden_size, 1e-6, vb_l.pp("input_layernorm"))?;
            let post_attention_layernorm = candle_nn::rms_norm(hidden_size, 1e-6, vb_l.pp("post_attention_layernorm"))?;
            
            layers.push(QwenLayer {
                attention_wq, attention_wk, attention_wv, attention_wo,
                mlp_gate, mlp_up, mlp_down,
                input_layernorm, post_attention_layernorm,
                kv_cache: None,
            });
        }

        let norm = candle_nn::rms_norm(hidden_size, 1e-6, vb.pp("model.norm"))?;
        let lm_head = candle_nn::linear_no_bias(hidden_size, vocab_size, vb.pp("lm_head"))?;

        log::info!("🔥 Malha KAN/Qwen carregada com sucesso!");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
        })
    }

    /// Reseta o cache de memória (chamar antes de um novo prompt)
    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
        }
    }

    /// Forward pass real com KV-Cache: transforma tokens em probabilidades
    pub fn forward(&mut self, tokens: &Tensor, _pos: usize) -> Result<Tensor> {
        let mut x = self.embed(&tokens)?;
        
        // Atravessa as 36 camadas (Coluna Cortical)
        for layer in &mut self.layers {
            // 1. Atenção Real
            let residual = x.clone();
            let x_norm = layer.input_layernorm.forward(&x)?;
            
            let q = layer.attention_wq.forward(&x_norm)?;
            let k = layer.attention_wk.forward(&x_norm)?;
            let v = layer.attention_wv.forward(&x_norm)?;
            
            // Aplicar KV-Cache
            let (k, v) = match &layer.kv_cache {
                None => {
                    layer.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
                Some((prev_k, prev_v)) => {
                    // Concatena o novo token com o cache anterior
                    let k = Tensor::cat(&[prev_k, &k], 1)?;
                    let v = Tensor::cat(&[prev_v, &v], 1)?;
                    layer.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            };

            // Cálculo de Atenção: softmax(Q @ K.T / sqrt(dk)) @ V
            let dk = (q.dims()[q.dims().len() - 1] as f64).sqrt();
            let attn_weights = (q.matmul(&k.transpose(1, 2)?)? / dk)?;
            let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
            let attn_out = attn_probs.matmul(&v)?;
            
            let attn_final = layer.attention_wo.forward(&attn_out)?;
            x = (attn_final + residual)?;

            // 2. MLP Real (SwiGLU)
            let residual = x.clone();
            let x_norm = layer.post_attention_layernorm.forward(&x)?;
            let gate = layer.mlp_gate.forward(&x_norm)?;
            let up = layer.mlp_up.forward(&x_norm)?;
            
            let mlp_out = (gate.silu()? * up)?;
            let mlp_out = layer.mlp_down.forward(&mlp_out)?;
            x = (mlp_out + residual)?;
        }

        // 3. Final Projection
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        let tokens = tokens.flatten_all()?;
        let embeddings = Tensor::embedding(&tokens, &self.embed_tokens)?;
        embeddings.unsqueeze(0)
    }
}
