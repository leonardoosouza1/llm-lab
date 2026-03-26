# IARA LLM Engine — Visão Geral do Sistema

## O que é isso
Motor de inferência nativo para modelos Qwen3 em AMD ROCm (RX 6750 XT, GFX1032).
Zero frameworks (sem PyTorch, sem candle). Carrega safetensors direto em VRAM via HIP FFI.
Expõe HTTP REST (porta 3000) + WebSocket (porta 8080) com painel Vue3 em tempo real.

## Estrutura de alto nível

```
llm-lab/
├── iara-engine/        # Servidor Rust (tokio + axum)
│   └── src/            # Ver iara-engine/CLAUDE.md
├── hip-ffi/            # Bindings HIP → GpuTensor
│   └── src/            # Ver hip-ffi/CLAUDE.md
├── frontend/           # Dashboard Vue3 (5 abas)
│   └── CLAUDE.md
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
./run_iara.sh            # debug (com vars ROCm)
./run_iara.sh release    # release (otimizado)
# Acesse: http://localhost:3000
```

## Invariantes críticas
1. **Apenas um modelo por vez** — `Arc<Mutex<Option<QwenModel>>>`
2. **BF16 → F32 na carga** — todos os pesos convertidos antes de subir p/ VRAM
3. **KV-cache limitado** — `BoundedKvCache` com janela deslizante (`max_seq_len` do config.json)
4. **RoPE aplicado antes do cache** — K rotacionado antes de `append()`, Q rotacionado por chamada
5. **forward_traced nunca reutiliza KV sem reset** — `reset_cache()` obrigatório no início de cada geração nova

## Gotchas
- **GPU usada só para armazenamento** — forward inteiramente em CPU f32 (GEMM ~O(n³) host)
- **rocBLAS não implementado** — próxima fase (10× speedup estimado)
- **Sliding window** — ao atingir `max_seq_len`, evicta token mais antigo via `copy_within`
- **Broadcast buffer** — 512 frames; clientes lentos perdem frames silenciosamente
- **Amostragem** — `temperature=0` → greedy; `top_k=0` → desabilitado; `top_p=1.0` → desabilitado

## Hardware alvo
- AMD RX 6750 XT (GFX1032 / RDNA2, 12 GB GDDR6)
- ROCm 6.4.4 em `/opt/rocm-6.4.4`
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` (forçado no run_iara.sh)
