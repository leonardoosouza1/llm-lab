# iara-engine — Servidor de Inferência

## Papel deste crate
Servidor tokio que orquestra:
- HTTP API via Axum (porta 3000)
- WebSocket de telemetria (porta 8080)
- Loop de engine: processa comandos WS → forward_traced → broadcast
- Estado compartilhado do modelo (Arc<Mutex<Option<QwenModel>>>)

## Módulos principais

| Módulo | Arquivo | Papel |
|--------|---------|-------|
| `main` | `src/main.rs` | Bootstrap: hardware, estado, rotas, loop de engine |
| `api/handlers` | `src/api/handlers.rs` | Handlers HTTP (/health, /models, /load-model, /generate, /debug) |
| `model` | `src/model.rs` | QwenModel: carga safetensors + forward pass CPU |
| `qwen/forward` | `src/qwen/forward.rs` | forward_traced com telemetria completa por camada |
| `qwen/sampling` | `src/qwen/sampling.rs` | sample_token: temperatura, top-k, top-p |
| `kv_cache` | `src/kv_cache.rs` | BoundedKvCache: janela deslizante, memória fixa |
| `tokenizer` | `src/tokenizer.rs` | BPE via HuggingFace tokenizers, trocável (RwLock) |
| `websocket` | `src/websocket.rs` | Servidor WS, protocolo ClientMessage/EngineCommand |
| `hardware` | `src/hardware/mod.rs` | HardwareManager: HIP device, VRAM info |
| `debug/brain` | `src/debug/brain.rs` | BrainTrace: timeline completa para replay |

## Módulos exploratorios (não integrados ao path de inferência)
`physics`, `physics_gpu`, `liquid_brain`, `liquid_network`, `simulation`,
`proprioception`, `actuator`, `homeostasis`, `nlm_orchestrator`, `prompt`,
`gpu_orchestrator`, `inference` — stubs / pesquisa. Não afetam a geração de texto.

## AppState (estado compartilhado entre handlers)

```rust
pub struct AppState {
    pub qwen_model:    Arc<Mutex<Option<QwenModel>>>,   // modelo (None até carregar)
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
2. Arc::new(Mutex::new(None))     → qwen_model vazio
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
hip-ffi      = { path = "../hip-ffi" }   # GPU tensors
tokenizers   = "0.21"                    # BPE HuggingFace
rand         = "0.8"                     # multinomial sampling
half         = "2"                       # BF16 → F32
memmap2      = "0.9"                     # mmap safetensors
```
