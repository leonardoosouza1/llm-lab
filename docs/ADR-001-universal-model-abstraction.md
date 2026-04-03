# ADR-001: Universal Model Abstraction for Multi-Architecture Support

**Status:** Proposed
**Date:** 2026-04-03
**Deciders:** Leonardo (project owner)

## Context

IARA Engine currently supports only Qwen2/2.5/3 model families. The codebase has 44 hardcoded Qwen-specific assumptions across struct names, tensor paths, activation functions, normalization types, EOS tokens, and the architecture whitelist. Leonardo wants the engine to support N open-source model types dynamically — similar to LM Studio — including Llama, Mistral, Phi, Gemma, DeepSeek, Falcon, and StableLM.

### Current Hardcoded Assumptions (Critical)

| Category | Count | Severity | Examples |
|----------|-------|----------|----------|
| Tensor name paths | 16 | CRITICAL | `model.layers.{i}.self_attn.q_proj.weight` |
| Struct names | 8 | HIGH | `QwenModel`, `QwenLayer`, `qwen_model` |
| EOS token IDs | 7 | HIGH | `151643`, `151645` hardcoded |
| Activation function | 3 | MEDIUM | `silu_mul_inplace` hardcoded (SwiGLU) |
| Normalization type | 2 | MEDIUM | `rms_norm` only, no LayerNorm |
| Architecture whitelist | 1 | MEDIUM | Only Qwen/Llama/Mistral in `from_config_json` |
| MLP structure | 3 | MEDIUM | `gate_proj + up_proj + down_proj` hardcoded |

### Model Family Differences

| Feature | Qwen | Llama | Mistral | Phi | Gemma | Falcon | DeepSeek |
|---------|------|-------|---------|-----|-------|--------|----------|
| Tensor prefix | `model.layers.X` | `model.layers.X` | `model.layers.X` | `model.layers.X` | `model.layers.X` | `transformer.h.X` | `model.layers.X` |
| Norm type | RMSNorm | RMSNorm | RMSNorm | LayerNorm | RMSNorm | LayerNorm | RMSNorm |
| Activation | SiLU | SiLU | SiLU | GELU | GELU | GELU | SiLU |
| MLP | gate+up+down | gate+up+down | gate+up+down | gate+up+down | gate+up+down | dense_h_to_4h/dense_4h_to_h | gate+up+down |
| Attention | GQA + QK norm | GQA | GQA + sliding | GQA | GQA | MQA | MLA + MoE |
| QK norm | Qwen3 only | No | No | No | No | No | No |
| Tied embeddings | Qwen3 | No | No | Phi-4 | Yes | No | No |

## Decision

**Config-driven architecture with auto-detection from HuggingFace `config.json`.**

Rather than trait-based polymorphism per model family (which would require N impl blocks), we use a single `ModelArchitecture` descriptor struct that captures all architectural differences as data. The forward pass reads from this config to dispatch normalization, activation, and tensor loading dynamically.

### Why config-driven over trait-based:

1. **Most models share 95% of the forward pass** — only norm type, activation, and tensor names differ
2. **Zero code per new model** — just map `config.json` fields to our descriptor
3. **No dynamic dispatch overhead** — enum match is branch prediction-friendly
4. **Easier to extend** — adding a new model is adding a match arm, not a new file

## Architecture

### Core Types

```rust
/// Normalization variant used by the model.
pub enum NormType {
    RMSNorm,    // Qwen, Llama, Mistral, Gemma
    LayerNorm,  // Phi, StableLM, Falcon
}

/// Activation function used in the MLP.
pub enum Activation {
    SiLU,   // Qwen, Llama, Mistral
    GELU,   // Phi, Gemma, Falcon
}

/// MLP structure pattern.
pub enum MlpType {
    /// gate_proj + up_proj → activation → down_proj (most models)
    GatedProjection,
    /// dense_h_to_4h → activation → dense_4h_to_h (Falcon)
    DenseHTo4H,
}

/// Full model architecture descriptor — derived from config.json.
pub struct ModelArchitecture {
    pub family: String,           // "qwen3", "llama", "mistral", etc.
    pub norm_type: NormType,
    pub activation: Activation,
    pub mlp_type: MlpType,
    pub has_qk_norm: bool,
    pub has_qkv_bias: bool,
    pub tie_word_embeddings: bool,
    pub eos_token_ids: Vec<u32>,
    pub tensor_map: TensorNameMap,
}

/// Maps logical tensor names to actual safetensor paths.
pub struct TensorNameMap {
    pub embed: String,                // "model.embed_tokens.weight"
    pub final_norm: String,           // "model.norm.weight"
    pub lm_head: String,              // "lm_head.weight"
    pub layer_prefix: String,         // "model.layers"
    // Per-layer suffixes:
    pub attn_q: String,               // "self_attn.q_proj.weight"
    pub attn_k: String,               // "self_attn.k_proj.weight"
    pub attn_v: String,               // "self_attn.v_proj.weight"
    pub attn_o: String,               // "self_attn.o_proj.weight"
    pub mlp_gate: String,             // "mlp.gate_proj.weight"
    pub mlp_up: String,               // "mlp.up_proj.weight"
    pub mlp_down: String,             // "mlp.down_proj.weight"
    pub input_norm: String,           // "input_layernorm.weight"
    pub post_attn_norm: String,       // "post_attention_layernorm.weight"
    // Optional suffixes (bias, qk_norm):
    pub attn_q_bias: Option<String>,
    pub attn_k_bias: Option<String>,
    pub attn_v_bias: Option<String>,
    pub q_norm: Option<String>,       // "self_attn.q_norm.weight"
    pub k_norm: Option<String>,       // "self_attn.k_norm.weight"
}
```

### Auto-Detection Flow

```
config.json → architectures[0] → match family:
  "Qwen3ForCausalLM"   → NormType::RMSNorm, SiLU, has_qk_norm=true
  "LlamaForCausalLM"   → NormType::RMSNorm, SiLU
  "MistralForCausalLM" → NormType::RMSNorm, SiLU
  "PhiForCausalLM"     → NormType::LayerNorm, GELU
  "GemmaForCausalLM"   → NormType::RMSNorm, GELU
  "FalconForCausalLM"  → NormType::LayerNorm, GELU, MlpType::DenseHTo4H
  _                    → attempt generic detection from config fields
```

EOS tokens come from `generation_config.json` (field `eos_token_id` — can be int or array).

### Forward Pass Changes

The forward pass stays in `model/mod.rs` but dispatches on architecture:

```rust
// Normalization dispatch:
match self.arch.norm_type {
    NormType::RMSNorm => backend.rms_norm(...),
    NormType::LayerNorm => backend.layer_norm(...),
}

// Activation dispatch:
match self.arch.activation {
    Activation::SiLU => backend.silu_mul_inplace(...),
    Activation::GELU => backend.gelu_mul_inplace(...),
}
```

### Required Backend Additions (compute-core)

| New Operation | Used By |
|---------------|---------|
| `layer_norm` | Phi, StableLM, Falcon |
| `gelu_mul_inplace` | Phi, Gemma, Falcon |

These must be added to the `ComputeBackend` trait and implemented in `rocm-backend` and `opencl-backend`.

## Consequences

### What becomes easier
- Adding any HuggingFace-compatible model: just download safetensors, engine auto-detects
- Supporting new model families: add a match arm in auto-detection
- Testing: can verify tensor name mapping without loading weights

### What becomes harder
- Forward pass has more branches (but they're cheap enum matches, not vtable calls)
- New backend operations (LayerNorm, GELU) require kernel implementations

### What we'll revisit
- DeepSeek MLA (Multi-head Latent Attention) — fundamentally different attention pattern, needs dedicated forward path
- MoE (Mixture of Experts) — router + sparse experts, significant forward pass change
- These two are deferred to ADR-002

## Implementation Plan

### Phase 1: Rename + Architecture Descriptor (no behavior change)
1. Rename `QwenModel` → `CausalLM`, `QwenLayer` → `TransformerLayer`
2. Add `ModelArchitecture` struct with `NormType`, `Activation`, `MlpType` enums
3. Add `TensorNameMap` with per-family defaults
4. Add auto-detection in `ModelCfg::from_config_json` → `ModelArchitecture`
5. Load EOS tokens from `generation_config.json`

### Phase 2: Dynamic Tensor Loading
6. Replace hardcoded `"model.layers.{i}.self_attn.q_proj.weight"` with `TensorNameMap`
7. Replace all 7 hardcoded EOS token checks with `arch.eos_token_ids.contains(&token)`

### Phase 3: Pluggable Normalization + Activation
8. Add `LayerNormParams` + `layer_norm` to `ComputeBackend` trait
9. Add `GeluMulParams` + `gelu_mul_inplace` to `ComputeBackend` trait
10. Implement HIP kernels for both in `rocm-backend`
11. Implement OpenCL kernels for both in `opencl-backend`
12. Dispatch in forward pass based on `arch.norm_type` and `arch.activation`

### Phase 4: Validation
13. Verify Qwen2.5-1.5b still produces correct output (regression)
14. Verify Qwen3-4b still produces correct output (regression)
15. Test with a Llama model (new capability)

## Action Items
1. [ ] Create `ModelArchitecture`, `TensorNameMap`, enum types
2. [ ] Auto-detection from `config.json` + `generation_config.json`
3. [ ] Rename structs (QwenModel → CausalLM, QwenLayer → TransformerLayer)
4. [ ] Dynamic tensor name loading
5. [ ] Dynamic EOS token handling
6. [ ] Add LayerNorm + GELU to compute-core trait
7. [ ] Implement HIP/OpenCL kernels for LayerNorm + GELU
8. [ ] Forward pass dispatch on norm_type + activation
9. [ ] Regression test with existing Qwen models
