# 🚀 IARA as Functional Ollama: Text-to-Rich-Response Pipeline

## Core Insight

**Não analisar ondas. Fazer funcionar.** Depois abstração vem naturalmente quando você vê os dados reais.

```
Entrada: "O que é um carro?"
         ↓
    [Tokenizer]
         ↓
    [Qwen Forward Pass] (real inference)
         ↓
    [Rastreamento de ativações] (quais camadas dispararam?)
         ↓
    [Simulação IARA] (contexto neurobiológico)
         ↓
Saída: "Um carro é um veículo de transporte com 4 rodas..."
       + Telemetria: {layers_active: [0,1,5,12,...], energy: 45.2, path: [...]}
```

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    IARA Ollama Server                   │
│                  (Single Rust Process)                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │ HTTP/REST API (axum)                            │ │
│  │ ───────────────────────────────────────────────  │ │
│  │ POST /api/generate                              │ │
│  │ {                                               │ │
│  │   "model": "qwen3-8b",                         │ │
│  │   "prompt": "O que é um carro?",              │ │
│  │   "stream": true                               │ │
│  │ }                                               │ │
│  └──────────────┬───────────────────────────────────┘ │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────────┐ │
│  │ Tokenizer & Input Processing                   │ │
│  │ ───────────────────────────────────────────────  │ │
│  │ • prompt → token IDs                           │ │
│  │ • Setup context tracking                       │ │
│  └──────────────┬───────────────────────────────────┘ │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────────┐ │
│  │ Qwen Model (Forward Pass Only)                  │ │
│  │ ───────────────────────────────────────────────  │ │
│  │ • Real tensor operations on GPU                 │ │
│  │ • Tracking: logits, attention patterns          │ │
│  │ • KV-cache for efficiency                       │ │
│  │ • IMPORTANT: Rastrear cada layer ativado!      │ │
│  └──────────────┬───────────────────────────────────┘ │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────────┐ │
│  │ Activation Logger                               │ │
│  │ ───────────────────────────────────────────────  │ │
│  │ • Qual camada ativou mais?                      │ │
│  │ • Padrão de attention weights                   │ │
│  │ • Energy dissipation                            │ │
│  │ • Layer activation path                         │ │
│  └──────────────┬───────────────────────────────────┘ │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────────┐ │
│  │ Token Sampling & Generation                     │ │
│  │ ───────────────────────────────────────────────  │ │
│  │ • Top-K + Temperature sampling                  │ │
│  │ • Greedy decoding option                        │ │
│  │ • Stopping conditions                           │ │
│  └──────────────┬───────────────────────────────────┘ │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────────┐ │
│  │ IARA Simulation (Async Context)                 │ │
│  │ ───────────────────────────────────────────────  │ │
│  │ • Mapear ativações Qwen → Liquid Network      │ │
│  │ • Rodar 100 ms de simulação em paralelo        │ │
│  │ • Coletar dinâmica emergente                    │ │
│  │ • NÃO bloqueia geração!                         │ │
│  └──────────────┬───────────────────────────────────┘ │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────────┐ │
│  │ Response Streaming                              │ │
│  │ ───────────────────────────────────────────────  │ │
│  │ • JSON streaming (Ollama-compatible)            │ │
│  │ • Token by token                                │ │
│  │ • Metadata: activations, timing                 │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow: From Text to Response

### Phase 1: Input & Tokenization

```rust
// user_input.rs
pub struct GenerationRequest {
    pub model: String,           // "qwen3-8b"
    pub prompt: String,          // "O que é um carro?"
    pub stream: bool,            // true para streaming
    pub temperature: f32,        // 0.7
    pub top_k: u32,             // 40
    pub max_tokens: usize,      // 100
}

pub struct TokenizationContext {
    pub token_ids: Vec<u32>,
    pub prompt_length: usize,
    pub max_gen_tokens: usize,
    pub current_position: usize,
}
```

### Phase 2: Model Forward Pass with Activation Tracking

```rust
// qwen_forward.rs
pub struct LayerActivationTrace {
    pub layer_idx: usize,
    pub attention_energy: f32,      // Sum of attention weights
    pub mlp_energy: f32,            // MLP gate output magnitude
    pub residual_norm: f32,         // Norm of residual connection
    pub max_activation: f32,        // Peak activation in layer
    pub timestamp_ms: u128,
}

pub struct GenerationStep {
    pub token_id: u32,
    pub token_str: String,
    pub logits: Vec<f32>,           // Top-K logits (sparse)
    pub attention_weights: Vec<f32>, // Last layer attention pattern
    pub layer_traces: Vec<LayerActivationTrace>,  // Per-layer activation
    pub generation_time_ms: u128,
}

pub async fn forward_and_trace(
    model: &QwenModel,
    token_id: u32,
    position: usize,
) -> Result<GenerationStep> {
    let mut traces = Vec::new();

    // Forward through each layer, collecting traces
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let start = std::time::Instant::now();

        // Attention
        let attention_out = layer.attention_wq.forward(&hidden)?;
        let attention_energy = attention_out.pow(&Tensor::new(&[2.0], &device)?)?.sum_all()?.to_scalar::<f32>()?;

        // MLP
        let mlp_gate = layer.mlp_gate.forward(&hidden)?;
        let mlp_energy = mlp_gate.pow(&Tensor::new(&[2.0], &device)?)?.sum_all()?.to_scalar::<f32>()?;

        // Residual
        let residual_norm = (layer.output.clone() + &attention_out)?.norm_all()?.to_scalar::<f32>()?;

        traces.push(LayerActivationTrace {
            layer_idx,
            attention_energy,
            mlp_energy,
            residual_norm,
            max_activation: attention_energy.max(mlp_energy),
            timestamp_ms: start.elapsed().as_millis(),
        });
    }

    let generation_time_ms = std::time::Instant::now().elapsed().as_millis();

    Ok(GenerationStep {
        token_id,
        token_str: tokenizer.decode(&[token_id])?,
        logits: top_k_logits(&final_logits, 40),
        attention_weights: last_attention.to_vec(),
        layer_traces: traces,
        generation_time_ms,
    })
}
```

### Phase 3: IARA Simulation (Async, Non-blocking)

```rust
// iara_context.rs
pub struct IARAContext {
    pub liquid_net: Arc<Mutex<LiquidNetwork>>,
    pub simulation: Arc<Mutex<Simulation>>,
}

pub async fn inject_and_simulate(
    iara: &IARAContext,
    traces: &[LayerActivationTrace],
) -> Result<SimulationSnapshot> {
    // Extrair energias das camadas
    let layer_energies: Vec<f32> = traces
        .iter()
        .map(|t| t.attention_energy + t.mlp_energy)
        .collect();

    // Injetar como estímulo na LiquidNetwork
    {
        let mut net = iara.liquid_net.lock().unwrap();
        net.inject_layer_stimulus_batch(&layer_energies)?;
    }

    // Rodar 100 ms de simulação em paralelo
    let start = std::time::Instant::now();
    while start.elapsed().as_millis() < 100 {
        let mut sim = iara.simulation.lock().unwrap();
        sim.tick(0.016); // 60 Hz tick

        let mut net = iara.liquid_net.lock().unwrap();
        let _ = net.step();

        drop(sim);
        drop(net);

        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    // Coletar snapshot
    let net = iara.liquid_net.lock().unwrap();
    let snapshot = SimulationSnapshot {
        nodes_active: net.nodes.iter().filter(|n| n.state.abs() > 0.01).count(),
        mean_energy: net.mean_energy(),
        phase_coherence: net.phase_coherence(),
        timestamp_ms: start.elapsed().as_millis(),
    };

    Ok(snapshot)
}
```

### Phase 4: Response Streaming

```rust
// streaming.rs
pub async fn stream_generation(
    model: &QwenModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    iara: &IARAContext,
) -> Result<impl Stream<Item = Result<GenerationStreamChunk>>> {
    let token_ids = tokenizer.encode(prompt)?;
    let mut current_pos = token_ids.len();
    let max_tokens = 100;

    async_stream::stream! {
        for step_num in 0..max_tokens {
            // 1. Forward pass com tracing
            let gen_step = forward_and_trace(
                model,
                token_ids[current_pos - 1],
                current_pos,
            ).await?;

            // 2. Sample next token
            let next_token = sample_token(&gen_step.logits, temperature=0.7)?;

            // 3. Trigger IARA simulation (async, no block!)
            let iara_sim = inject_and_simulate(&iara, &gen_step.layer_traces);

            // 4. Send to client immediately (don't wait for IARA)
            let chunk = GenerationStreamChunk {
                token: gen_step.token_str.clone(),
                position: current_pos,
                layer_activations: gen_step.layer_traces.clone(),
                timing: gen_step.generation_time_ms,
            };

            yield Ok(chunk);

            // 5. Await IARA simulation (background, don't block next token)
            if let Ok(snapshot) = iara_sim.await {
                log::info!(
                    "IARA: {} active nodes, energy={:.2}",
                    snapshot.nodes_active,
                    snapshot.mean_energy
                );
            }

            current_pos += 1;

            if next_token == tokenizer.eos_token() {
                break;
            }

            token_ids.push(next_token);
        }
    }
}
```

---

## 3. HTTP API: Ollama-Compatible

```rust
// handlers.rs
use axum::{
    extract::Json,
    response::sse::{Event, KeepAlive, Sse},
    Router,
};

#[derive(Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[derive(Serialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub context: Vec<u32>,
    pub total_duration: u128,
    pub load_duration: u128,
    pub prompt_eval_count: u32,
    pub eval_count: u32,
    pub eval_duration: u128,
}

#[derive(Serialize)]
pub struct StreamChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub context: Vec<u32>,
    // IARA-specific
    pub layer_activations: Option<Vec<LayerActivationTrace>>,
    pub simulation_snapshot: Option<SimulationSnapshot>,
}

pub async fn generate_handler(
    State(app_state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    if req.stream {
        // Streaming response
        let stream = stream_generation(
            &app_state.model,
            &app_state.tokenizer,
            &req.prompt,
            &app_state.iara,
        )
        .await
        .unwrap();

        Sse::new(stream).keep_alive(KeepAlive::default())
    } else {
        // Non-streaming response
        let full_response = generate_full(
            &app_state.model,
            &app_state.tokenizer,
            &req.prompt,
            &app_state.iara,
        )
        .await
        .unwrap();

        Json(full_response).into_response()
    }
}

pub async fn health_handler() -> Json<serde_json::json!({
    "status": "ok",
    "model": "qwen3-8b",
    "version": "1.0.0",
})) {}
```

---

## 4. The "Debug Brain" Feature

Aqui é onde você **entende o que realmente aconteceu**:

```rust
// debug_brain.rs
pub struct BrainTrace {
    pub prompt: String,
    pub response: String,
    pub steps: Vec<GenerationStep>,
    pub iara_snapshots: Vec<SimulationSnapshot>,
    pub layer_activation_path: Vec<Vec<usize>>,  // Which layers fired each step
    pub energy_trajectory: Vec<f32>,              // Energy over time
    pub attention_pattern_heatmap: Vec<Vec<f32>>, // Attention across all steps
}

impl BrainTrace {
    pub fn analyze(&self) -> BrainAnalysis {
        BrainAnalysis {
            most_active_layers: self.most_active_layers(),
            energy_progression: self.energy_progression(),
            attention_flow: self.attention_flow(),
            iara_resonance_score: self.iara_resonance_score(),
            semantic_clustering: self.semantic_clustering(),
        }
    }

    fn most_active_layers(&self) -> Vec<(usize, f32)> {
        // Count layer activations across all steps
        let mut layer_energy: HashMap<usize, f32> = HashMap::new();

        for step in &self.steps {
            for trace in &step.layer_traces {
                *layer_energy.entry(trace.layer_idx).or_insert(0.0)
                    += trace.max_activation;
            }
        }

        let mut sorted: Vec<_> = layer_energy.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(10);
        sorted
    }

    fn energy_progression(&self) -> Vec<f32> {
        self.steps
            .iter()
            .map(|step| {
                step.layer_traces.iter().map(|t| t.max_activation).sum::<f32>()
                    / step.layer_traces.len() as f32
            })
            .collect()
    }

    fn attention_flow(&self) -> AttentionFlow {
        // Como a atenção "flui" pelo texto gerado
        // Primeiro token atende contexto, depois se especializa
        AttentionFlow {
            early_focus: self.steps[0].attention_weights.clone(),
            late_focus: self.steps[self.steps.len() - 1].attention_weights.clone(),
            transition: self.compute_attention_shift(),
        }
    }

    fn iara_resonance_score(&self) -> f32 {
        // Quanto a simulação IARA "resonou" com a geração
        // Alta coerência = modelo funcionou bem + IARA capturou a dinâmica
        let coherence = self.iara_snapshots
            .iter()
            .map(|s| s.phase_coherence)
            .sum::<f32>()
            / self.iara_snapshots.len() as f32;

        coherence
    }
}

pub struct BrainAnalysis {
    pub most_active_layers: Vec<(usize, f32)>,
    pub energy_progression: Vec<f32>,
    pub attention_flow: AttentionFlow,
    pub iara_resonance_score: f32,
    pub semantic_clustering: Vec<TokenCluster>,
}

// Endpoint: GET /api/debug/last-trace
pub async fn debug_trace_handler(
    State(app_state): State<AppState>,
) -> Json<BrainAnalysis> {
    let trace = app_state.brain_trace.lock().unwrap().clone();
    Json(trace.analyze())
}
```

---

## 5. Key Architecture Points

### Why This Works

```
✅ Fluxo simples:
   Entrada → Qwen (rastreando) → IARA (async) → Saída

✅ Não bloqueia:
   IARA roda em background, próximo token gerado enquanto IARA simula

✅ Entendível:
   Cada GenerationStep tem todas as ativações
   → Pode reconstuir exatamente o que o modelo fez

✅ Escalável:
   Qwen forward em GPU
   IARA simulation em CPU cores paralelos
   Ambos em async, sem contenção

✅ Debugável:
   BrainTrace salva tudo
   BrainAnalysis extrai insights
   JSON export para análise
```

### What's Different from Current IARA

```
❌ Antes:
   • Simulação era o foco
   • Qwen era "oráculo" para gerar telemetria
   • WebSocket streaming de heatmaps
   • Frontend era central

✅ Agora:
   • Qwen é o foco (inference real)
   • Rastreamento de ativações como subproduto
   • IARA rodando em background (contextual)
   • API é central (Ollama-compatible)
   • Debug brain revela a dinâmica
```

---

## 6. Implementation Roadmap

### Week 1: Core Generation Pipeline
```
[ ] Refactor QwenModel para adicionar LayerActivationTrace
[ ] Implementar forward_and_trace() funcional
[ ] Streaming response basics
[ ] HTTP endpoints
```

### Week 2: IARA Integration
```
[ ] Async IARA injection
[ ] Non-blocking simulation spawning
[ ] Snapshot collection
[ ] Response enrichment com IARA data
```

### Week 3: Debug Brain
```
[ ] BrainTrace collection
[ ] Analysis functions
[ ] JSON export
[ ] Debug endpoints
```

### Week 4: Polish & Optimization
```
[ ] Profiling (onde está o bottleneck?)
[ ] Caching estratégico
[ ] Error handling robusto
[ ] Tests end-to-end
```

---

## 7. Code Structure

```
iara-engine/
├── src/
│   ├── main.rs                    # HTTP server + main loop
│   ├── api/
│   │   ├── handlers.rs            # /api/generate, /api/debug
│   │   └── types.rs               # Request/Response types
│   ├── qwen/
│   │   ├── model.rs               # QwenModel (existing)
│   │   ├── forward.rs             # ✨ NEW: forward_and_trace()
│   │   └── sampling.rs            # Token sampling
│   ├── iara/
│   │   ├── sim.rs                 # Simulation (existing)
│   │   ├── liquid.rs              # LiquidNetwork (existing)
│   │   └── context.rs             # ✨ NEW: IARAContext, injection
│   ├── debug/
│   │   ├── brain.rs               # ✨ NEW: BrainTrace, analysis
│   │   └── export.rs              # JSON export
│   └── util/
│       ├── logger.rs              # Structured logging
│       └── metrics.rs             # Telemetry
├── Cargo.toml
└── tests/
    └── integration_tests.rs       # End-to-end tests
```

---

## 8. Example Usage

### Client Side (Python)

```python
import requests
import json

BASE_URL = "http://localhost:8080"

# Streaming generation
response = requests.post(
    f"{BASE_URL}/api/generate",
    json={
        "model": "qwen3-8b",
        "prompt": "O que é um carro?",
        "stream": True,
        "temperature": 0.7,
    },
    stream=True,
)

full_response = ""
for chunk in response.iter_lines():
    if chunk:
        data = json.loads(chunk)
        print(data["response"], end="", flush=True)

        # Opcionalmente coletar ativações
        if "layer_activations" in data:
            activations = data["layer_activations"]
            print(f"\n[Layer {activations[-1]['layer_idx']} fired]", end="")

print()

# Debug brain
debug = requests.get(f"{BASE_URL}/api/debug/last-trace").json()
print("\nBrain Analysis:")
print(f"Most active layers: {debug['most_active_layers'][:5]}")
print(f"IARA resonance: {debug['iara_resonance_score']:.2f}")
print(f"Attention flow: {debug['attention_flow']}")
```

### Shell Usage (Like Ollama)

```bash
# Simple request
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "prompt": "Explique machine learning em 3 frases",
    "stream": false
  }'

# Streaming
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "prompt": "O que é inteligência artificial?",
    "stream": true
  }' | jq -r '.response'

# Debug brain (ver qual foi o caminho de ativação)
curl http://localhost:8080/api/debug/last-trace | jq '.most_active_layers'
```

---

## 9. Why This Approach is Superior

```
Antes (Atual IARA):
❌ Foco em visualização de dinâmica (ondas)
❌ Qwen era secundário ("oráculo")
❌ Difícil entender o que realmente aconteceu
❌ Acoplado a WebSocket/frontend

Depois (IARA Ollama):
✅ Foco em funcionamento real (texto coeso)
✅ Qwen é primário, rastreado completamente
✅ BrainTrace + Analysis = ENTENDER tudo
✅ Desacoplado, API agnóstica
✅ Pode integrar em qualquer sistema
✅ Dados reais para treino (gradientes later!)

Analogia:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Antes: Assistir carro se mover em simulação
Depois: DIRIGIR o carro de verdade, depois analisar a dirigida

Antes tinha sentido para R&D.
Depois faz sentido para PRODUTO.
```

---

## 10. Success Criteria

✅ `curl http://localhost:8080/api/generate` retorna texto coeso
✅ Resposta é **exatamente** como Qwen faria (sem IARA corromper)
✅ IARA roda em background, não afeta latência
✅ `BrainTrace` salva tudo
✅ Analisando `BrainAnalysis`, você entende o caminho

Quando conseguir tudo isso:
→ Você tem um **Ollama funcional em Rust**
→ Você tem dados reais para debugar
→ Você tem base para treino (Python bridge depois)
→ Você pode começar **abstrações reais**

