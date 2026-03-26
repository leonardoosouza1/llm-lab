# 🚀 IARA Ollama: Implementation Roadmap

## Status Atual

✅ Arquitetura desenhada
✅ Código inicial criado (handlers, forward.rs, brain.rs)
⏳ Próximos passos para funcionar de verdade

---

## O Que Temos

### Documentos
1. **IARA_OLLAMA_ARCHITECTURE.md** — Visão completa do fluxo
2. **Código Rust Parcial:**
   - `src/api/handlers.rs` — HTTP endpoints
   - `src/qwen/forward.rs` — Forward pass com tracing
   - `src/debug/brain.rs` — BrainTrace & análise

### O Que Falta

```
🔴 CRÍTICO (bloqueia tudo):
├─ Integração do forward.rs com modelo existente
├─ Endpoint /api/generate funcionando
└─ BrainTrace sendo populado

🟡 IMPORTANTE:
├─ IARA context async injection
├─ Streaming response (Server-Sent Events)
└─ Debug endpoints

🟢 BÔNUS:
├─ Telemetry dashboard
├─ Performance profiling
└─ Multi-GPU support
```

---

## Week 1: Core Generation Pipeline

### Task 1.1: Estender QwenModel
```rust
// No src/model.rs, adicionar:
impl QwenModel {
    pub fn layers(&self) -> &Vec<QwenLayer> { ... }
    pub fn norm(&self) -> &RmsNorm { ... }
    pub fn lm_head(&self) -> &Linear { ... }
    pub fn device(&self) -> &Device { ... }
    pub fn tokenizer(&self) -> &Tokenizer { ... }  // ← New!
}
```

**Por quê:** `forward_and_trace()` precisa disso.

**Teste:**
```bash
cargo build --features forward_tracing
```

---

### Task 1.2: Implementar forward_and_trace() Real

O arquivo `src/qwen/forward.rs` precisa ser integrado:

```rust
// src/qwen/mod.rs
pub mod forward;
pub use forward::{GenerationStep, LayerActivationTrace, forward_and_trace};
```

**Checklist:**
- [ ] `candle_nn::ops::softmax` está disponível?
- [ ] `Tensor::silu()` funciona?
- [ ] Testes com um token dummy

**Teste:**
```rust
#[tokio::test]
async fn test_forward_trace() {
    let model = load_test_model().await;
    let step = forward_and_trace(&model, 42, 1).await;
    assert!(step.is_ok());
    let gen_step = step.unwrap();
    assert!(!gen_step.layer_traces.is_empty());
    println!("Generated: {}, Layers: {}", gen_step.token_str, gen_step.layer_traces.len());
}
```

---

### Task 1.3: HTTP Server Setup (Axum)

```rust
// src/main.rs
use axum::{
    routing::post,
    Router,
};

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/api/generate", post(handlers::generate_handler))
        .route("/health", get(handlers::health_handler))
        .with_state(AppState { ... });

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

**Teste:**
```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b","prompt":"Hello","stream":false}'
```

---

### Task 1.4: BrainTrace Integration

```rust
// No generate_handler:
let mut brain_trace = BrainTrace::new(req.prompt.clone());

for step_num in 0..max_tokens {
    let gen_step = forward_and_trace(&model, last_token, pos).await?;
    brain_trace.add_step(gen_step);  // ← Popula BrainTrace
    // ...
}

brain_trace.finalize();
*state.last_trace.lock().await = Some(brain_trace);
```

**Teste:**
```bash
# Após gerar:
curl http://localhost:8080/api/debug/last-trace | jq '.summary'
```

---

## Week 2: IARA Integration (Async, Non-blocking)

### Task 2.1: IARAContext

```rust
// src/iara/context.rs (novo arquivo)
pub struct IARAContext {
    pub liquid_net: Arc<Mutex<LiquidNetwork>>,
    pub simulation: Arc<Mutex<Simulation>>,
}

pub async fn inject_and_simulate(
    iara: &IARAContext,
    traces: &[LayerActivationTrace],
) -> Result<SimulationSnapshot> {
    // Implementar conforme IARA_OLLAMA_ARCHITECTURE.md
}
```

**Teste:**
```rust
#[tokio::test]
async fn test_inject_and_simulate() {
    let iara = setup_iara().await;
    let traces = vec![/* dummy traces */];
    let snapshot = inject_and_simulate(&iara, &traces).await;
    assert!(snapshot.is_ok());
}
```

---

### Task 2.2: Non-blocking Spawn

```rust
// No generate_handler, após cada token:
let iara = state.iara.clone();
let traces = gen_step.layer_traces.clone();

tokio::spawn(async move {
    if let Ok(snapshot) = inject_and_simulate(&iara, &traces).await {
        log::info!("IARA: {} active nodes", snapshot.nodes_active);
    }
});
```

**Importante:** NÃO usar `.await` aqui!
Spawn e continue.

---

## Week 3: Streaming & Debug

### Task 3.1: Server-Sent Events (SSE)

```rust
// handlers.rs
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::{self, Stream};

pub async fn generate_handler_streaming(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::iter(/* generation steps */)
        .map(|step| {
            let event = Event::default()
                .json_data(GenerateStreamChunk { ... })?;
            Ok(event)
        });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

**Teste:**
```bash
curl -N http://localhost:8080/api/generate \
  -H 'Accept: text/event-stream' \
  -d '{"prompt":"Hello","stream":true}' | grep data
```

---

### Task 3.2: Debug Endpoints

```rust
pub async fn debug_trace_handler(...) -> Json<BrainAnalysis> {
    let trace = state.last_trace.lock().await.clone();
    let analysis = trace.analyze();
    Json(analysis)
}

pub async fn debug_export_handler(...) -> impl IntoResponse {
    let trace = state.last_trace.lock().await.clone();
    let csv = trace.to_csv();
    (
        StatusCode::OK,
        [("Content-Type", "text/csv")],
        csv,
    )
}
```

---

## Week 4: Polish & Testing

### Task 4.1: End-to-End Test

```rust
#[tokio::test]
async fn test_full_pipeline() {
    let app = setup_app().await;

    // Request
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/generate")
                .json(GenerateRequest { ... })?
                .build()?,
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    let gen_resp: GenerateResponse = serde_json::from_slice(&body).unwrap();

    assert!(!gen_resp.response.is_empty());
    assert!(gen_resp.eval_count > 0);

    println!("Generated: {}", gen_resp.response);
}
```

---

### Task 4.2: Performance Profiling

```bash
# Flamegraph
cargo install flamegraph
cargo flamegraph --bin iara-engine -- --model-dir ./models

# Resultado: flamegraph.svg
# Identifica gargalos
```

---

## Estimated Timeline

```
✅ Done (já)
├─ Architecture design
└─ Code skeleton

⏳ 2-3 weeks
├─ Week 1: Core generation (forward + HTTP)
├─ Week 2: IARA async injection
├─ Week 3: Streaming + debug
└─ Week 4: Polish + tests

🎯 Target: Funcional end-of-week 3
   (pode fazer tudo, sem otimizações)
```

---

## Success Criteria (Go/No-Go)

```
✅ After Week 1:
curl -X POST localhost:8080/api/generate \
  -d '{"prompt":"test"}' \
→ Returns valid JSON with response

✅ After Week 2:
curl localhost:8080/api/generate?stream=true \
→ SSE stream de tokens + ativações

✅ After Week 3:
curl localhost:8080/api/debug/last-trace | jq
→ {"most_active_layers": [...], "summary": "..."}

✅ After Week 4:
cargo test --release && cargo bench
→ All tests pass, performance characterized
```

---

## Critical Dependencies

```toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
futures = "0.3"
candle-core = "0.8"
candle-nn = "0.8"
log = "0.4"
env_logger = "0.11"
chrono = "0.4"
async-stream = "0.3"  # ← Para streaming
thiserror = "1.0"
anyhow = "1.0"
```

Adicionar ao `Cargo.toml` se faltarem.

---

## File Checklist

```
✅ IARA_OLLAMA_ARCHITECTURE.md       (criado)
✅ src/api/handlers.rs                (criado)
✅ src/qwen/forward.rs                (criado)
✅ src/debug/brain.rs                 (criado)

⏳ src/api/mod.rs                     (criar, pub mod handlers)
⏳ src/qwen/mod.rs                    (estender, pub mod forward)
⏳ src/debug/mod.rs                   (criar, pub mod brain)
⏳ src/iara/context.rs                (criar, async injection)
⏳ src/main.rs                         (refactor para Axum)
⏳ tests/integration_tests.rs         (criar, E2E tests)
```

---

## Quick Start Template

```bash
# 1. Clonar / atualizar
cd ~/projects/LLM/llm-lab/iara-engine

# 2. Build
cargo build --release

# 3. Run
cargo run --release -- --model-dir ./models

# 4. Test
curl -X POST http://localhost:8080/api/generate \
  -d '{"model":"qwen3-8b","prompt":"O que é IA?","stream":false}' | jq

# 5. Debug
curl http://localhost:8080/api/debug/last-trace | jq '.summary'
```

---

## Próxima Conversão: Detalhes de Implementação

Quer que eu:

1. **Revise o código existente** antes de começar?
2. **Crie as estruturas faltantes** (mod.rs files)?
3. **Detalhe a integração com QwenModel**?
4. **Setup Axum + routing**?
5. **Escreva testes end-to-end**?

Qual primeiro?

