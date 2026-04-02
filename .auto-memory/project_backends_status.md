---
name: IARA — Status dos Backends GPU
description: ROCm completo, OpenCL 60% (5 kernels faltando), sem CPU fallback
type: project
---

## ROCm/HIP — ✅ PRODUÇÃO
Todos os 15+ kernels implementados em `hip-ffi/kernels/llm_kernels.hip`:
embed_lookup, rms_norm, rope_inplace, softmax_rows, silu_mul_inplace, copy_to_kv_cache, add_bias_inplace, gemv_q4, gravity_score, cfc_gate_update, value_norms, kv_cache_quantize_append, add_tensors_inplace, f32_to_bf16.

Suporta: Qwen3 vanilla BF16, quantização Q4, CfC correction, Gravity scoring, quantized KV cache.

Risco: downcast com `.expect()` em vez de retornar erro — pode gerar panic se tipo errado.

## OpenCL — ⚠️ PARCIAL (60%)
Faltam 5 operações:
1. `gemv_q4` — sem suporte a pesos Q4 (**bloqueia modelos quantizados**)
2. `add_tensors_inplace` — sem residual corrections CfC
3. `gravity_score` — sem LNN
4. `cfc_forward` — sem liquid correction
5. `kv_cache_quantized_append` — sem KV compression

Qwen3 vanilla BF16 **funciona** em OpenCL. Quantização e LNN não.

## CPU — ❌ Planejado, não implementado

## Matriz de compatibilidade
| Config | ROCm | OpenCL |
|--------|------|--------|
| Qwen3 BF16 | ✅ | ✅ |
| + Q4 weights | ✅ | ❌ |
| + CfC/Gravity | ✅ | ❌ |
| + Quantized KV | ✅ | ❌ |

**How to apply:** Ao adicionar features que dependem de novos kernels, lembrar de implementar em AMBOS os backends ou adicionar feature flags para desabilitar graciosamente em OpenCL.
