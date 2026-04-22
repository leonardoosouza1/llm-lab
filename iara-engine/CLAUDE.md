# iara-engine — Servidor de Inferência

## Papel deste crate
Servidor tokio que orquestra:
- HTTP API via Axum (porta 3000)
- WebSocket de telemetria (porta 8080)
- Loop de engine: processa comandos WS → forward_traced → broadcast
- Estado compartilhado do modelo (Arc<Mutex<Option<CausalLM>>>)

## Módulos principais

| Módulo | Arquivo | Papel |
|--------|---------|-------|
| `main` | `src/main.rs` | Bootstrap: hardware, estado, rotas, loop de engine |
| `api/handlers` | `src/api/handlers.rs` | Handlers HTTP (/health, /models, /load-model, /generate, /debug) |
| `acp` | `src/acp/` | Agent Communication Protocol (handlers, state, types) |
| `model` | `src/model/mod.rs` | CausalLM universal: carga + forward GPU via ComputeBackend |
| `model/architecture` | `src/model/architecture.rs` | ModelArchitecture auto-detect (Qwen, Llama, Mistral, ...) |
| `model/kv_cache` | `src/model/kv_cache.rs` | GpuKvCache: AnyTensor pré-alocado, janela deslizante |
| `model/ntc_cache` | `src/model/ntc_cache.rs` | Neural Tensor Cache (quantizado) |
| `inference/forward` | `src/inference/forward.rs` | forward_traced com LayerTrace por camada |
| `inference/sampling` | `src/inference/sampling.rs` | SamplingParams: temperatura, top-k, top-p |
| `inference/tracer` | `src/inference/tracer.rs` | BrainTrace / telemetria |
| `storage` | `src/storage/` | safetensors, gguf, quantized (Q4/Q4K/Q6K), fp8, bitnet, ptt, surgery, iara_format |
| `tokenizer` | `src/tokenizer/mod.rs` | BPE via HuggingFace tokenizers, trocável (RwLock) |
| `transport` | `src/transport/mod.rs` | Servidor WS, ClientMessage/EngineCommand |
| `hardware` | `src/hardware/mod.rs` | HardwareManager: GPU detection, VRAM info |
| `debug/brain` | `src/debug/brain.rs` | BrainTrace: timeline completa para replay |
| `lnn` | `src/lnn/` | Liquid Neural Network: cfc, esn, gravity, sae, train |
| `llama` | `src/llama/` | Integração llama.cpp via llama-sys (batch_server, engine, cfc_draft) |
| `speculative` | `src/speculative/` | Speculative decoding framework |

## Módulos exploratórios (feature `experimental`, não integrados ao forward)
`src/experimental/`: `physics`, `liquid_brain`, `liquid_network`, `orchestrator`,
`actuator`, `homeostasis`. Stubs de pesquisa. Compilados só com `--features experimental`.

## AppState (estado compartilhado entre handlers)

```rust
pub struct AppState {
    pub model:         Arc<Mutex<Option<CausalLM>>>,    // modelo (None até carregar)
    pub last_trace:    Arc<Mutex<Option<BrainTrace>>>,   // último trace para /debug
    pub model_ready:   Arc<AtomicBool>,                  // true após carga completa
    pub model_loading: Arc<AtomicBool>,                  // evita cargas concorrentes
    pub models_dir:    Arc<PathBuf>,                     // pasta com subpastas de modelos
    pub frame_tx:      broadcast::Sender<String>,        // canal WS (512 frames buffer)
}
```

## run_engine_loop — Loop WS

Consome `EngineCommand` do canal `cmd_rx`:
- `Resonate(text)` / `Stimulus(text)`: tokeniza → loop `forward_traced` → broadcast frames
- `ListLayers`: gera nomes de camadas, envia como `{"type":"layers",...}`
- `SelectLayer(name)`: placeholder UI (sem efeito no modelo)
- `Reset`: envia status (reset real é via `model.reset_cache()` no generate)

**Importante:** o loop usa `SamplingParams::default()` (temperature=0.7, top_k=40, top_p=0.9).
Para controle fino de sampling via WS, seria necessário passar params no `ClientMessage`.

## Sequência de inicialização (main.rs)

```
1. HardwareManager::new()         → detecta GPU, loga VRAM
2. Arc::new(Mutex::new(None))     → model vazio
3. broadcast::channel(512)        → frame_tx para WS
4. spawn(start_ws_server)         → escuta :8080
5. spawn(run_engine_loop)         → processa EngineCommand
6. AppState { ... }               → estado compartilhado
7. Router::new().route(...)       → HTTP com ServeDir (frontend/)
8. axum::serve(:3000)             → bloqueia aqui
```

## Dependências externas chave

```toml
tokio        = { version = "1", features = ["full"] }
axum         = "0.7"
tokio-tungstenite = "0.24"
compute-core = { path = "../compute-core" }      # trait ComputeBackend
hip-ffi      = { path = "../hip-ffi", optional = true }  # ROCm feature
rocm-backend = { path = "../rocm-backend", optional = true }
opencl-backend = { path = "../opencl-backend", optional = true }
llama-sys    = { path = "../llama-sys" }         # llama.cpp FFI
tokenizers   = "0.21"                    # BPE HuggingFace
rand         = "0.8"                     # multinomial sampling
half         = "2"                       # BF16 → F32
memmap2      = "0.9"                     # mmap safetensors
```
