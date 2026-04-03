// ============================================================================
// model/architecture.rs — Descritor de Arquitetura Universal para Modelos LLM
// ============================================================================
// Auto-detecta família do modelo a partir de config.json e determina:
//   - Tipo de normalização (RMSNorm / LayerNorm)
//   - Função de ativação (SiLU / GELU)
//   - Estrutura do MLP (Gated / Dense)
//   - Mapeamento de nomes de tensores nos safetensors
//   - Tokens EOS do generation_config.json
// ============================================================================

use std::path::Path;

// ── Enums de variantes arquiteturais ────────────────────────────────────────

/// Tipo de normalização usado nas camadas do transformer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    /// RMSNorm — Qwen, Llama, Mistral, Gemma, DeepSeek
    RMSNorm,
    /// LayerNorm — Phi, StableLM, Falcon
    LayerNorm,
}

/// Função de ativação usada no MLP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// SiLU (Swish) — Qwen, Llama, Mistral, DeepSeek
    SiLU,
    /// GELU — Phi, Gemma, Falcon, StableLM
    GELU,
}

/// Estrutura do MLP (feed-forward network).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MlpType {
    /// Gated projection: gate_proj + up_proj → activation → down_proj
    /// Usado por: Qwen, Llama, Mistral, Phi, Gemma, DeepSeek
    GatedProjection,
    /// Dense simples: dense_h_to_4h → activation → dense_4h_to_h
    /// Usado por: Falcon
    DenseHTo4H,
}

// ── Mapeamento de nomes de tensores ─────────────────────────────────────────

/// Mapeia nomes lógicos (embed, q_proj, etc.) para os nomes reais nos safetensors.
/// Cada família de modelo pode ter convenções diferentes.
#[derive(Debug, Clone)]
pub struct TensorNameMap {
    // ── Tensores globais ──
    pub embed: String,           // ex: "model.embed_tokens.weight"
    pub final_norm: String,      // ex: "model.norm.weight"
    pub lm_head: String,         // ex: "lm_head.weight"

    // ── Prefixo de camada ──
    pub layer_prefix: String,    // ex: "model.layers" → "model.layers.{i}"

    // ── Sufixos por camada (atenção) ──
    pub attn_q: String,          // ex: "self_attn.q_proj.weight"
    pub attn_k: String,
    pub attn_v: String,
    pub attn_o: String,

    // ── Sufixos por camada (MLP) ──
    pub mlp_gate: String,        // ex: "mlp.gate_proj.weight"
    pub mlp_up: String,
    pub mlp_down: String,

    // ── Sufixos por camada (normalização) ──
    pub input_norm: String,      // ex: "input_layernorm.weight"
    pub post_attn_norm: String,  // ex: "post_attention_layernorm.weight"

    // ── Sufixos opcionais (bias, QK norm) ──
    pub attn_q_bias: Option<String>,  // ex: "self_attn.q_proj.bias"
    pub attn_k_bias: Option<String>,
    pub attn_v_bias: Option<String>,
    pub q_norm: Option<String>,       // ex: "self_attn.q_norm.weight"
    pub k_norm: Option<String>,
    // ── Norm bias (LayerNorm precisa de bias, RMSNorm não) ──
    pub input_norm_bias: Option<String>,
    pub post_attn_norm_bias: Option<String>,
    pub final_norm_bias: Option<String>,
}

impl TensorNameMap {
    /// Gera o nome completo do tensor para uma camada específica.
    /// Ex: layer_tensor(5, &self.attn_q) → "model.layers.5.self_attn.q_proj.weight"
    pub fn layer_tensor(&self, layer_idx: usize, suffix: &str) -> String {
        format!("{}.{}.{}", self.layer_prefix, layer_idx, suffix)
    }

    /// Gera o nome completo para um sufixo opcional.
    pub fn layer_tensor_opt(&self, layer_idx: usize, suffix: &Option<String>) -> Option<String> {
        suffix.as_ref().map(|s| self.layer_tensor(layer_idx, s))
    }

    /// Mapeamento padrão para modelos que seguem a convenção HuggingFace.
    /// Cobre: Qwen, Llama, Mistral, Phi, Gemma, DeepSeek, StableLM
    pub fn standard() -> Self {
        Self {
            embed: "model.embed_tokens.weight".into(),
            final_norm: "model.norm.weight".into(),
            lm_head: "lm_head.weight".into(),
            layer_prefix: "model.layers".into(),
            attn_q: "self_attn.q_proj.weight".into(),
            attn_k: "self_attn.k_proj.weight".into(),
            attn_v: "self_attn.v_proj.weight".into(),
            attn_o: "self_attn.o_proj.weight".into(),
            mlp_gate: "mlp.gate_proj.weight".into(),
            mlp_up: "mlp.up_proj.weight".into(),
            mlp_down: "mlp.down_proj.weight".into(),
            input_norm: "input_layernorm.weight".into(),
            post_attn_norm: "post_attention_layernorm.weight".into(),
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            q_norm: None,
            k_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            final_norm_bias: None,
        }
    }

    /// Mapeamento para Falcon (usa prefixo `transformer.h.X` e nomes diferentes).
    pub fn falcon() -> Self {
        Self {
            embed: "transformer.word_embeddings.weight".into(),
            final_norm: "transformer.ln_f.weight".into(),
            lm_head: "lm_head.weight".into(),
            layer_prefix: "transformer.h".into(),
            attn_q: "self_attention.query_key_value.weight".into(),
            attn_k: "self_attention.query_key_value.weight".into(), // fused QKV
            attn_v: "self_attention.query_key_value.weight".into(),
            attn_o: "self_attention.dense.weight".into(),
            mlp_gate: "mlp.dense_h_to_4h.weight".into(),
            mlp_up: "mlp.dense_h_to_4h.weight".into(), // fused gate+up
            mlp_down: "mlp.dense_4h_to_h.weight".into(),
            input_norm: "input_layernorm.weight".into(),
            post_attn_norm: "post_attention_layernorm.weight".into(), // may not exist in some Falcon variants
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            q_norm: None,
            k_norm: None,
            input_norm_bias: Some("input_layernorm.bias".into()),
            post_attn_norm_bias: Some("post_attention_layernorm.bias".into()),
            final_norm_bias: Some("transformer.ln_f.bias".into()),
        }
    }
}

// ── Descritor completo da arquitetura ───────────────────────────────────────

/// Descritor completo da arquitetura do modelo, derivado de config.json.
/// Usado pelo forward pass e pelo loader para saber como tratar cada modelo.
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    /// Nome da família: "qwen3", "llama", "mistral", "phi", "gemma", "falcon", etc.
    pub family: String,
    /// Tipo de normalização (RMSNorm ou LayerNorm)
    pub norm_type: NormType,
    /// Função de ativação do MLP (SiLU ou GELU)
    pub activation: Activation,
    /// Estrutura do MLP
    pub mlp_type: MlpType,
    /// Se o modelo aplica RMSNorm per-head em Q e K antes do RoPE (Qwen3)
    pub has_qk_norm: bool,
    /// Se as projeções Q/K/V têm bias (Qwen2/2.5)
    pub has_qkv_bias: bool,
    /// Se lm_head compartilha pesos com embed_tokens
    pub tie_word_embeddings: bool,
    /// IDs dos tokens EOS (carregados de generation_config.json)
    pub eos_token_ids: Vec<u32>,
    /// Mapeamento de nomes de tensores
    pub tensor_map: TensorNameMap,
}

impl ModelArchitecture {
    /// Detecta a arquitetura a partir do config.json do modelo.
    /// Lê `architectures[0]` e campos específicos para determinar a família.
    pub fn from_config_json(config_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(config_path)?;
        let v: serde_json::Value = serde_json::from_str(&content)?;

        let arch_str = v["architectures"].as_array()
            .and_then(|a| a.first())
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        let tie_embeddings = v["tie_word_embeddings"].as_bool().unwrap_or(false);

        // Detectar se model tem QKV bias (check campo explícito ou inferir pela família)
        let has_qkv_bias_field = v["attention_bias"].as_bool();

        let (family, norm_type, activation, mlp_type, has_qk_norm, has_qkv_bias, tensor_map) =
            match arch_str {
                // ── Qwen ──
                s if s.contains("Qwen3") => (
                    "qwen3".into(),
                    NormType::RMSNorm,
                    Activation::SiLU,
                    MlpType::GatedProjection,
                    true,   // Qwen3 tem QK norm
                    has_qkv_bias_field.unwrap_or(false),
                    {
                        let mut m = TensorNameMap::standard();
                        m.q_norm = Some("self_attn.q_norm.weight".into());
                        m.k_norm = Some("self_attn.k_norm.weight".into());
                        m
                    },
                ),
                s if s.contains("Qwen2") => (
                    "qwen2".into(),
                    NormType::RMSNorm,
                    Activation::SiLU,
                    MlpType::GatedProjection,
                    false,
                    has_qkv_bias_field.unwrap_or(true), // Qwen2/2.5 tem bias por padrão
                    {
                        let mut m = TensorNameMap::standard();
                        if has_qkv_bias_field.unwrap_or(true) {
                            m.attn_q_bias = Some("self_attn.q_proj.bias".into());
                            m.attn_k_bias = Some("self_attn.k_proj.bias".into());
                            m.attn_v_bias = Some("self_attn.v_proj.bias".into());
                        }
                        m
                    },
                ),

                // ── Llama ──
                s if s.contains("Llama") => (
                    "llama".into(),
                    NormType::RMSNorm,
                    Activation::SiLU,
                    MlpType::GatedProjection,
                    false,
                    has_qkv_bias_field.unwrap_or(false),
                    TensorNameMap::standard(),
                ),

                // ── Mistral ──
                s if s.contains("Mistral") => (
                    "mistral".into(),
                    NormType::RMSNorm,
                    Activation::SiLU,
                    MlpType::GatedProjection,
                    false,
                    has_qkv_bias_field.unwrap_or(false),
                    TensorNameMap::standard(),
                ),

                // ── Phi ──
                s if s.contains("Phi") => (
                    "phi".into(),
                    NormType::LayerNorm,
                    Activation::GELU,
                    MlpType::GatedProjection,
                    false,
                    has_qkv_bias_field.unwrap_or(true), // Phi geralmente tem bias
                    {
                        let mut m = TensorNameMap::standard();
                        m.attn_q_bias = Some("self_attn.q_proj.bias".into());
                        m.attn_k_bias = Some("self_attn.k_proj.bias".into());
                        m.attn_v_bias = Some("self_attn.v_proj.bias".into());
                        m.input_norm_bias = Some("input_layernorm.bias".into());
                        m.post_attn_norm_bias = Some("post_attention_layernorm.bias".into());
                        m.final_norm_bias = Some("model.norm.bias".into());
                        m
                    },
                ),

                // ── Gemma ──
                s if s.contains("Gemma") => (
                    "gemma".into(),
                    NormType::RMSNorm,
                    Activation::GELU,
                    MlpType::GatedProjection,
                    false,
                    has_qkv_bias_field.unwrap_or(false),
                    TensorNameMap::standard(),
                ),

                // ── Falcon ──
                s if s.contains("Falcon") => (
                    "falcon".into(),
                    NormType::LayerNorm,
                    Activation::GELU,
                    MlpType::DenseHTo4H,
                    false,
                    false,
                    TensorNameMap::falcon(),
                ),

                // ── StableLM ──
                s if s.contains("StableLm") || s.contains("StableLM") => (
                    "stablelm".into(),
                    NormType::LayerNorm,
                    Activation::SiLU,
                    MlpType::GatedProjection,
                    false,
                    has_qkv_bias_field.unwrap_or(false),
                    {
                        let mut m = TensorNameMap::standard();
                        m.input_norm_bias = Some("input_layernorm.bias".into());
                        m.post_attn_norm_bias = Some("post_attention_layernorm.bias".into());
                        m.final_norm_bias = Some("model.norm.bias".into());
                        m
                    },
                ),

                // ── DeepSeek ──
                s if s.contains("DeepSeek") || s.contains("Deepseek") => (
                    "deepseek".into(),
                    NormType::RMSNorm,
                    Activation::SiLU,
                    MlpType::GatedProjection,
                    false,
                    has_qkv_bias_field.unwrap_or(false),
                    TensorNameMap::standard(),
                ),

                // ── Genérico: tentar inferir a partir dos campos do config.json ──
                other => {
                    log::warn!(
                        "Arquitetura '{}' não reconhecida — tentando detecção genérica",
                        other
                    );

                    // Inferir norm type: se tem layer_norm_eps → LayerNorm, senão RMSNorm
                    let norm = if v["layer_norm_eps"].is_number() {
                        NormType::LayerNorm
                    } else {
                        NormType::RMSNorm
                    };

                    // Inferir ativação: se config diz hidden_act
                    let act = match v["hidden_act"].as_str() {
                        Some("gelu") | Some("gelu_new") | Some("gelu_fast") => Activation::GELU,
                        _ => Activation::SiLU,
                    };

                    (
                        other.to_lowercase().replace("forcausallm", ""),
                        norm,
                        act,
                        MlpType::GatedProjection,
                        false,
                        has_qkv_bias_field.unwrap_or(false),
                        TensorNameMap::standard(),
                    )
                }
            };

        // ── Carregar EOS tokens de generation_config.json ──
        let eos_token_ids = Self::load_eos_tokens(config_path.parent().unwrap_or(Path::new(".")));

        log::info!(
            "🏗️  Arquitetura detectada: family={}, norm={:?}, activation={:?}, mlp={:?}, qk_norm={}, tied_embed={}, eos={:?}",
            family, norm_type, activation, mlp_type, has_qk_norm, tie_embeddings, eos_token_ids
        );

        Ok(Self {
            family,
            norm_type,
            activation,
            mlp_type,
            has_qk_norm,
            has_qkv_bias,
            tie_word_embeddings: tie_embeddings,
            eos_token_ids,
            tensor_map,
        })
    }

    /// Carrega tokens EOS de generation_config.json.
    /// Suporta tanto valor único quanto array de IDs.
    /// Fallback: [2] (padrão de muitos modelos HuggingFace).
    fn load_eos_tokens(model_dir: &Path) -> Vec<u32> {
        let gen_config_path = model_dir.join("generation_config.json");
        if !gen_config_path.exists() {
            log::warn!("generation_config.json não encontrado, usando EOS fallback [2]");
            return vec![2];
        }

        let content = match std::fs::read_to_string(&gen_config_path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("Erro ao ler generation_config.json: {}, usando EOS fallback", e);
                return vec![2];
            }
        };

        let v: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("Erro ao parsear generation_config.json: {}, usando EOS fallback", e);
                return vec![2];
            }
        };

        // Campo eos_token_id pode ser:
        // - Um inteiro: "eos_token_id": 151645
        // - Um array: "eos_token_id": [151645, 151643]
        match &v["eos_token_id"] {
            serde_json::Value::Number(n) => {
                vec![n.as_u64().unwrap_or(2) as u32]
            }
            serde_json::Value::Array(arr) => {
                let ids: Vec<u32> = arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect();
                if ids.is_empty() { vec![2] } else { ids }
            }
            _ => {
                log::warn!("eos_token_id ausente ou inválido em generation_config.json");
                vec![2]
            }
        }
    }

    /// Verifica se um token é EOS para este modelo.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }
}
