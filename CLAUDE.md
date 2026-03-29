# IARA LLM Engine — Visão Geral do Sistema

## O que é isso
Motor de inferência nativo para modelos Qwen3 com arquitetura de backend dinâmica.
Zero frameworks (sem PyTorch, sem candle). Carrega safetensors direto em VRAM via backend de compute.
Expõe HTTP REST (porta 3000) + WebSocket (porta 8080) com painel Vue3 em tempo real.

Suporta múltiplos backends de hardware detectados em runtime:
- **ROCm/HIP** — AMD RX 6750 XT (GFX1032 / RDNA2, 12 GB GDDR6) via hipBLAS + kernels HIP
- **OpenCL** — Intel iGPU (Iris Plus Gen 11, i7-1065G7) e qualquer GPU OpenCL 3.0
- Detecção automática: tenta ROCm → OpenCL → erro

## Estrutura de alto nível

```
llm-lab/
├── compute-core/       # Traits e param-structs (sem impl de hardware)
│   └── src/            # ComputeBackend trait, AnyTensor, params
├── rocm-backend/       # Impl AMD ROCm: wraps hip-ffi, hipBLAS + kernels HIP
├── opencl-backend/     # Impl Intel iGPU: OpenCL C kernels (JIT) + GEMM
├── hip-ffi/            # Bindings HIP/hipBLAS de baixo nível (usados por rocm-backend)
├── iara-engine/        # Servidor Rust (tokio + axum) — usa só compute-core
│   └── src/            # Ver iara-engine/CLAUDE.md
├── frontend/           # Dashboard Vue3 (5 abas)
├── models/             # Pesos safetensors (não versionados)
│   └── qwen3-8b/       # Modelo ativo (36 camadas, 8.2B params)
└── run_iara.sh         # Script de execução com vars ROCm
```

## Fluxo de dados completo

```
POST /api/generate
      ↓
handlers.rs::generate_internal()
      ↓ (SamplingParams: temperature, top_k, top_p)
qwen/forward.rs::forward_traced()
      ↓ 36× transformer block (CPU f32, referência)
      ↓ por camada: RMSNorm → Q/K/V → RoPE → KV-cache → Attention → MLP
      ↓ telemetria: LayerTrace (energy, hot_neurons, timing)
      ↓ logits [vocab_size] → sample_token()
GenerationStep { token_id, token_str, top_logits, layer_traces }
      ↓
HTTP response JSON + broadcast WS frame_tx
      ↓
Frontend: heatmap de camadas, top tokens, timeline
```

## Endpoints HTTP

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status do servidor e modelo |
| GET | `/api/models` | Lista modelos em `models/` |
| POST | `/api/load-model` | Carrega modelo selecionado |
| POST | `/api/generate` | Inferência (temperatura, top-k, top-p) |
| GET | `/api/debug/last-trace` | Último BrainTrace completo |

## Protocolo WebSocket (porta 8080)

**Cliente → Servidor:**
```json
{"type":"stimulus","text":"olá mundo"}
{"type":"reset"}
{"type":"list_layers"}
```

**Servidor → Cliente (stream por token):**
```json
{"type":"pipeline_layer","layer_idx":5,"attention_energy":42.3,...}
{"type":"pipeline_result","step":1,"generated_so_far":"olá","token_str":" mundo",...}
{"type":"status","message":"Geração completa: 12 tokens"}
```

## Como executar

```bash
# AMD ROCm (casa)
./run_iara.sh            # debug
./run_iara.sh release    # release

# Intel iGPU (notebook) — sem ROCm
cargo run -p iara-engine --no-default-features --features opencl
# Acesse: http://localhost:3000
```

## Invariantes críticas
1. **Apenas um modelo por vez** — `Arc<Mutex<Option<QwenModel>>>`
2. **Pesos em BF16 no device** — `upload_bf16()` via backend; sem conversão intermediária para F32
3. **KV-cache limitado** — janela deslizante de 4096 tokens (cap em `ModelCfg::from_config_json`)
4. **RoPE aplicado antes do cache** — K rotacionado antes de `copy_to_kv_cache`, Q por chamada
5. **reset_cache() obrigatório** — no início de cada geração nova (`QwenModel::reset_cache()`)
6. **Backend via trait** — `iara-engine` nunca importa `hip-ffi` ou `opencl-backend` diretamente

## Gotchas
- **Offsets em params** — `GemmBf16Params.a_offset`, `SgemmParams.{a,b,c}_offset`, `SoftmaxParams.offset_elems` são necessários para atenção multi-cabeça e lm_head do último token
- **Sliding window** — ao atingir `max_seq_len`, evicta token mais antigo via kernel `copy_to_kv_cache`
- **Broadcast buffer** — 512 frames; clientes lentos perdem frames silenciosamente
- **Amostragem** — `temperature=0` → greedy; `top_k=0` → desabilitado; `top_p=1.0` → desabilitado
- **OpenCL GEMM** — implementação naive (loop K); não usa clBLAST. Correto mas lento para modelos grandes

## Hardware suportado
| Backend | Hardware | Features Cargo |
|---------|----------|----------------|
| ROCm/HIP | AMD RX 6750 XT (GFX1032) | `--features rocm` |
| OpenCL | Intel Iris Plus Gen 11 (i7-1065G7) | `--features opencl` |
| (futuro) CPU | Fallback puro Rust | `--features cpu` |
