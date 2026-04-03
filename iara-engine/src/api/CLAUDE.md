# api/handlers.rs — Handlers HTTP

## Endpoints

| Rota | Handler | Descrição |
|------|---------|-----------|
| `GET /health` | `health_handler` | Status: ok / loading / idle |
| `GET /api/models` | `models_handler` | Escaneia models_dir, retorna ModelInfo[] |
| `POST /api/load-model` | `load_model_handler` | Carrega modelo em background |
| `POST /api/generate` | `generate_handler` | Roda inferência, retorna texto |
| `GET /api/debug/last-trace` | `debug_trace_handler` | Retorna BrainTrace do último generate |

## Tipos principais

### GenerateRequest
```rust
{
  model: String,
  prompt: String,
  stream: bool,          // default false (streaming não implementado)
  temperature: f32,      // default 0.7 — 0.0 = greedy
  top_k: u32,            // default 40  — 0 = desabilitado
  top_p: f32,            // default 0.9 — 1.0 = desabilitado
  max_tokens: usize,     // default 100
}
```

### GenerateResponse
```rust
{
  model, created_at, response: String, done: bool,
  context: Vec<u32>,       // token IDs gerados
  total_duration: u128,    // ms total
  prompt_eval_count: u32,  // tokens do prompt
  eval_count: u32,         // tokens gerados
  eval_duration: u128,
}
```

### ModelInfo
```rust
{
  name, path,
  hidden_size, num_layers, vocab_size, num_heads, num_kv_heads, head_dim,
  intermediate_size, architecture, torch_dtype,
  model_kind: "causal_lm" | "vlm" | "unknown",
  has_vision: bool,
}
```

## generate_internal — fluxo interno

```
1. Verifica model_ready / model_loading (retorna erro se não pronto)
2. Constrói SamplingParams { temperature, top_k, top_p }
3. Tokeniza prompt
4. Loop até max_tokens:
   a. Adquire lock do qwen_model
   b. forward_traced(model, &tokens, &sampling) → GenerationStep
   c. Libera lock
   d. Appenda token_str ao generated_text
   e. Adiciona step ao BrainTrace
   f. Se EOS (via arch.eos_token_ids): break
5. Finaliza BrainTrace → last_trace
6. Retorna GenerateResponse
```

**Nota:** cada `forward_traced` é síncrono e bloqueia o handler inteiro.
Com Qwen3-8B em CPU, cada token leva ~2-5s. Múltiplas requisições simultâneas
ficam em fila atrás do Mutex.

## load_model_handler — carga em background

```
1. Verifica model_loading (retorna 409 se já carregando)
2. Detecta VLM via architectures[0].endsWith("ForConditionalGeneration")
3. Define model_loading=true, model_ready=false
4. Descarrega modelo atual (lock → None)
5. spawn task:
   a. Ticker: log a cada 5s enquanto carregando
   b. load_tokenizer(&model_path)
   c. ModelCfg::from_config_json(&config.json)
   d. CausalLM::load(&model_path, cfg, arch, backend)
   e. lock → Some(model), model_ready=true, model_loading=false
   f. frame_tx.send({"type":"model_loaded",...})
6. Retorna 202 Accepted imediatamente
```

## Detecção de VLMs

Arquitetura termina em `ForConditionalGeneration` → é VLM.
VLMs são carregados como decoder de texto apenas (vision encoder ignorado).
Exibidos com badge `🖼 VLM` no frontend.

## Gotchas
- **EOS dinâmico**: carregado de `generation_config.json` via `arch.eos_token_ids`. Suporta múltiplos tokens EOS.
- **Sem streaming HTTP real**: `stream: bool` existe mas não implementa SSE — o response sempre é retornado completo.
- **Lock durante forward**: toda a geração segura o mutex. UI fica responsiva via WS broadcast.
- **config.json aninhado**: VLMs têm params em `text_config`, não na raiz. Leitura verifica ambos.
