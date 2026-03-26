# 🧬 IARA Resonance Engine — Análise Arquitetural Completa

## 📋 Resumo Executivo

O **IARA** é um motor de simulação neuromórfica inovador que combina três camadas de computação:

1. **Camada de Física de Rede (CfC Physics)** — Neurônios Liquid Time Constant (LTC) simulando dinâmica de propagação de sinal em uma malha acoplada
2. **Camada de Rede Líquida (Liquid Brain)** — Rede de 256 osciladores acoplados com STDP + Homeostase + Ritmos Circadianos
3. **Camada Funcional (Qwen Model)** — Modelo LLM real (Qwen 8B em Rust/Candle) para geração e reasoning

O sistema expõe tudo via **WebSocket** para visualização e interação em tempo real.

---

## 1. Contexto & Propósito

### O que é IARA?
Um **motor de simulação neuromórfica** que:
- Carrega tensores (pesos) de um modelo Qwen 8B como condições iniciais
- Simula dinâmica neuronal em tempo real (60 Hz)
- Mantém um "cérebro líquido" paralelo de 256 osciladores com memória adaptativa (STDP)
- Responde a estímulos textuais e gera respostas via o modelo real na GPU

### Casos de Uso
- **Visualização neuronal**: Entender como os pesos do modelo distribuem energia durante inferência
- **Pesquisa neurodinâmica**: Estudar padrões de acoplamento, sincronização e entropia em redes líquidas
- **Interface sensoriomotora**: Conectar entrada de texto → simulação de dinâmica → saída de resposta generativa

---

## 2. Arquitetura de Alto Nível

```
┌──────────────────────────────────────────────────────────────────┐
│                    IARA Resonance Engine                         │
└──────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼─────┐  ┌────▼──────┐  ┌───▼────────┐
        │   Fauna     │  │  Liquid   │  │   Qwen     │
        │  (CfC Net)  │  │  Brain    │  │   Model    │
        │  60 Hz      │  │  100 Hz   │  │  (GPU)     │
        └───────┬─────┘  └────┬──────┘  └───┬────────┘
                │              │              │
                └──────────────┼──────────────┘
                               │
                    ┌──────────▼────────────┐
                    │  WebSocket Broadcast  │
                    │  (20 Hz Telemetry)    │
                    └──────────┬────────────┘
                               │
                        ┌──────▼──────┐
                        │  Frontend    │
                        │  (Browser)   │
                        └──────────────┘
```

### Componentes Principais

| Componente | Frequência | Responsabilidade |
|-----------|-----------|------------------|
| **Simulation (Fauna)** | 60 Hz | Dinâmica de neurônios CfC + Acoplamento de Kuramoto com inibição lateral |
| **LiquidNetwork (Brain)** | 100 Hz | Osciladores adaptativos com STDP + Homeostase + Ritmos circadianos |
| **QwenModel** | On-demand | Carregamento de pesos na GPU + Forward pass com KV-cache |
| **WebSocket Server** | 20 Hz | Streaming de telemetria para visualização em tempo real |
| **ActuatorBus** | 1 Hz | Comando emergentes (Research, Focus, Inhibition) |
| **HomeostaticRegulator** | 1 Hz | Manutenção de ritmos circadianos e homeostase energética |

---

## 3. Camada 1: Física de Rede (Simulation/Fauna)

### Arquivo: `simulation.rs`

#### Propósito
Gerenciar um **array paralelo de neurônios CfC** que simulam dinâmica de membrana com propagação acoplada em uma malha 2D.

#### Estrutura de Dados

```rust
pub struct Simulation {
    pub nodes: Vec<NeuronioCfC>,      // Array de N neurônios CfC
    pub layer_name: String,            // Qual tensor estamos visualizando
    pub tick_count: u64,               // Número total de iterações
    pub grid_size: usize,              // Dimensão do grid NxN para heatmap 2D
}

pub struct NeuronioCfC {
    pub x: f32,           // Estado atual (potencial de membrana)
    pub weight: f32,      // Peso do tensor (usado para inicializar τ e K_ij)
    pub tau: f32,         // Constante de tempo adaptativa
    pub threshold: f32,   // Limite de disparo
}
```

#### Dinâmica Física

**Passo 1: Evolução individual (Closed-form LTC)**

Cada neurônio evolui segundo a equação diferencial:
```
dx_i/dt = -γ·x_i + tanh(I_in)
```

Onde:
- `γ` = coeficiente de decay (amortecimento)
- `I_in` = corrente de entrada (acoplamento sináptico + estímulo externo)
- Integração numericamente via **RK4** (4ª ordem Runge-Kutta) em `physics.rs`

**Passo 2: Acoplamento de Kuramoto com Inibição Lateral (4-vizinhança)**

Após atualizar estados individuais, vizinhos (conectados em grid 2D) influenciam-se:

```
Δx_i = Σ K_ij · (x_j - x_i)   onde j ∈ vizinhos
```

Coupling coefficient:
```
K_ij = base_k · (1 - 2.8 · distance)
distance = min(|w_j - w_i|, 2.0) / 2.0
```

**Interpretação Neurobiológica:**
- Se `w_j ≈ w_i` (pesos similares) → `K_ij > 0` (atração, **sincronização**)
- Se `w_j ≠ w_i` (pesos diferentes) → `K_ij < 0` (repulsão, **inibição lateral**)
- Resultado: **selectividade semântica** — neurônios similares ressoam juntos

#### Métodos Chave

```rust
impl Simulation {
    /// Cria N neurônios CfC a partir de array de pesos f32
    pub fn from_weights(weights: &[f32], layer_name: &str) -> Self

    /// Injeta pulso de energia externo nos neurônios
    pub fn apply_stimulus(&mut self, energy: &[f32])

    /// Avança 1 time-step: atualiza estados + acoplamento Kuramoto
    pub fn tick(&mut self, dt: f32)

    /// Extrai ativação para heatmap 2D (grid_size x grid_size)
    pub fn snapshot_grid(&self) -> Vec<f32>

    /// Calcula energia total: Σ |x_i|
    pub fn total_energy(&self) -> f32

    /// Conta neurônios "ativos" (|x_i| > threshold)
    pub fn active_count(&self, threshold: f32) -> usize
}
```

---

## 4. Camada 2: Rede Líquida (Liquid Brain)

### Arquivo: `liquid_network.rs`

#### Propósito
Manter um **"sistema nervoso central" adaptativo e independente** que:
- Recebe telemetria em tempo real do modelo LLM
- Evolui com **STDP (Spike-Timing-Dependent Plasticity)**
- Regula homeostase via **ritmos circadianos** (sleep-wake cycles)
- Modula viscosidade com base em stress de hardware

#### Estrutura de Dados

```rust
pub struct LiquidNetwork {
    pub nodes: Vec<LiquidNode>,
    pub weights: Vec<Vec<f64>>,       // Matriz sináptica W[i][j] (adaptativa)
    pub num_layers: usize,             // Mapeado para 36 camadas do Qwen
    pub viscosity: f64,                // μ (aumenta sob stress → propag. mais lenta)
}

pub struct LiquidNode {
    pub id: NodeId,
    pub state: f64,          // x(t) — ativação instantânea
    pub phase: f64,          // φ ∈ [0, 2π] — fase do oscilador
    pub frequency: f64,      // ω — frequência natural (rad/s)
    pub amplitude: f64,      // A — envelope (modulado por entropia)
}

pub struct LlmTelemetry {
    pub token: String,
    pub entropy_chaos: f64,           // Shannon H(p) — incerteza (0=certo, 1=aleatório)
    pub context_energy: f64,          // Grounding energy — quanto o modelo "sabe"
    pub hardware_stress: f64,         // Proxy de GPU temp (0=cool, 1=hot)
}
```

#### Dinâmica (RK4 @ 100 Hz)

**Equação diferencial de cada nó:**
```
dx_i/dt = -γ·x_i + tanh(Σ_j W[i][j]·x_j + s_i(t))
```

Onde:
- `γ` = amortecimento (inversamente proporcional à viscosidade μ)
- `W[i][j]` = peso sináptico (adaptativo via STDP)
- `s_i(t)` = estímulo externo (injetado da telemetria do LLM)

**Integração numérica:**
- RK4 com step h = 0.01 s (100 Hz)
- Viscosidade μ reduz efetivamente a taxa de propagação
- `dx_i/dt_eff = (dx_i/dt) / μ` sob stress

#### Learning Rule: STDP (Spike-Timing-Dependent Plasticity)

```
ΔW[i][j] = η · x_i(t) · x_j(t - Δt)  [pair-based Hebbian]
```

**Interpretação:**
- Se neurônio j dispara ANTES de i por Δt ≈ 10 ms → STDP potencia (LTP)
- Se j dispara DEPOIS de i → STDP enfraquece (LTD)
- Resultado: Aprendizado causal da dinâmica

**Implementação:**
```rust
fn slow_update(&mut self, normalized_energy: f64) {
    // a cada slow_tick (1 Hz)
    for i in 0..N {
        for j in 0..N {
            delta_w = learning_rate * state[i] * state[j];
            weights[i][j] += delta_w;
            // Soft bounds para evitar explosion
            weights[i][j] = weights[i][j].clamp(-5.0, 5.0);
        }
    }
}
```

#### Homeostase Circadiana

**Regulação @ 1 Hz:**
```rust
pub fn circadian_phase(&self) -> CircadianPhase {
    match (time_of_day % 24.0) {
        0.0..=6.0   => CircadianPhase::DeepSleep,
        6.0..=8.0   => CircadianPhase::Waking,
        8.0..=22.0  => CircadianPhase::Awake,
        22.0..=24.0 => CircadianPhase::Drowsy,
        _ => CircadianPhase::Awake,
    }
}
```

**Efeito:**
- Deep Sleep: γ aumenta (atividade diminui), taxa STDP aumenta (consolidação)
- Awake: γ normal, STDP baseado em entropia do LLM
- Stress alto: μ aumenta (propagação mais lenta, economia de energia)

---

## 5. Camada 3: Modelo Funcional (Qwen 8B on GPU)

### Arquivo: `model.rs` & `generate.rs`

#### Propósito
Carregar e executar o **Qwen 8B real em Candle** (Rust ML framework) para:
- Responder perguntas via geração de texto (modo "Resonate")
- Fornecer telemetria em tempo real (tokens, logits, entropia)
- Usar como "oráculo" para validar dinâmica da simulação

#### Arquitetura Qwen 8B

```
┌──────────────────────────────────────────┐
│  Token Embedding (152K → 2048D)          │
├──────────────────────────────────────────┤
│  36 Transformer Layers                   │
│  ├─ Input LayerNorm (RMS)                │
│  ├─ MultiHeadAttention (GQA)             │  Grouped Query Attention
│  │   ├─ Q projection: hidden → 32 heads  │  32 query heads
│  │   ├─ K/V projection: hidden → 8 heads │  8 key/value heads (grouped)
│  │   └─ KV-Cache (otimizado)             │  Skip recompute K,V
│  ├─ Post-Attention LayerNorm             │
│  ├─ MLP (Feed-forward)                   │
│  │   ├─ gate_proj: hidden → intermediate │
│  │   ├─ up_proj: hidden → intermediate   │
│  │   ├─ activation (SiLU/GELU)           │
│  │   └─ down_proj: intermediate → hidden │
│  └─ Residual connections (x + f(x))     │
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  Final RMS Norm                          │
├──────────────────────────────────────────┤
│  Output Projection (2048 → 152K)         │  Logits
└──────────────────────────────────────────┘
```

#### Carregamento na GPU

```rust
pub fn load(model_dir: &Path, device: &Device, mcfg: &ModelConfig) -> Result<Self> {
    // 1. Escanear diretório para arquivos .safetensors
    let filenames = fs::read_dir(model_dir)?
        .filter_map(|e| {
            let p = e?.path();
            if p.extension() == Some("safetensors") { Some(p) } else { None }
        })
        .collect::<Vec<_>>();

    // 2. Usar VarBuilder com mmap (memory-mapped I/O)
    //    Zero-copy: não carrega tudo na RAM/VRAM, acessa sob demanda
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&filenames, DType::BF16, device)?
    };

    // 3. Construir layers com candle_nn
    for i in 0..num_layers {
        let vb_l = vb.pp(format!("model.layers.{}", i));

        let attention_wq = candle_nn::linear_no_bias(hidden_size, hidden_size, ...)?;
        let attention_wk = ...
        // ... etc para 36 layers

        layers.push(QwenLayer { attention_wq, attention_wk, ..., kv_cache: None });
    }

    // 4. Alocar na GPU via Candle (automático)
    //    Se CUDA/HIP disponível: GPU
    //    Fallback: CPU
    Ok(Self { embed_tokens, layers, norm, lm_head, device })
}
```

#### Forward Pass (Streaming)

```rust
pub async fn forward(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
    // 1. Emissão do token em hidden space (embedding)
    let hidden = embed_tokens[input_ids[0]];  // [2048]

    // 2. Loop por 36 camadas
    for layer in &mut layers {
        let hidden_norm = layer.input_layernorm.forward(&hidden)?;

        // Attention com KV-cache
        // Sem cache: Q@K^T @ V tem custo O(n²) em comprimento
        // Com cache: reusa K,V do passo anterior → O(n) amortizado
        let attn_out = layer.attention_wq.forward(&hidden_norm)?;
        let attn_out = attn_out @ layer.attention_wo.weight;

        // MLP (Feed-forward)
        let mlp_norm = layer.post_attention_layernorm.forward(&hidden)?;
        let gate = layer.mlp_gate.forward(&mlp_norm)?;
        let up = layer.mlp_up.forward(&mlp_norm)?;
        let mlp_out = layer.mlp_down.forward(&(gate * up).silu())?;

        // Residual connection
        hidden = hidden + attn_out + mlp_out;
    }

    // 3. Final projection
    let norm_out = norm.forward(&hidden)?;
    let logits = lm_head.forward(&norm_out)?;  // [152K]

    Ok(logits)
}
```

#### Sampling & Generation

```rust
pub async fn generate_response(prompt: &str, model: &mut QwenModel,
                               ftx: &broadcast::Sender<String>,
                               max_tokens: usize) -> Result<String> {
    let mut response = String::new();

    for step in 0..max_tokens {
        let logits = model.forward(&tokenizer.encode(prompt))?;

        // Top-K + temperature sampling
        let token = sample_top_k(&logits, k=40, temperature=0.7);
        let token_str = tokenizer.decode(&[token]);

        response.push_str(&token_str);

        // Enviar streaming via WebSocket
        let msg = json!({
            "type": "generation_step",
            "token": token_str,
            "step": step,
            "total_steps": max_tokens,
        });
        ftx.send(msg.to_string())?;
    }

    Ok(response)
}
```

---

## 6. Integração: Loop Principal (main.rs)

### Fluxo de Execução Completo

```
┌────────────────────────────────────────────────────────┐
│ INICIALIZAÇÃO                                          │
├────────────────────────────────────────────────────────┤
│ 1. env_logger setup → logging com timestamps           │
│ 2. parse_args() → EngineConfig (port, layer, etc)     │
│ 3. config.json → ModelConfig (hidden_size, vocab, etc)│
│ 4. safetensors index → lista de N tensores            │
│ 5. Selecionar tensor inicial (default: layer.0)       │
│ 6. read_tensor via mmap → pesos f32 do tensor         │
│ 7. Simulation::from_weights(pesos) → N neurônios CfC  │
│ 8. LiquidNetwork::new(256, num_layers) → 256 oscil.  │
│ 9. QwenModel::load(device) → GPU/CPU                 │
│ 10. Inicializar HomeostaticRegulator → salvar estado  │
└────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ WS Listener  │  │ Physics Loop │  │ Visualization│
│ :8080        │  │ 100 Hz       │  │ 20 Hz        │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                    ┌────▼────┐
                    │ Main Loop│ (60 Hz)
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    Processar       Tick Simulation  Enviar Telemetria
    Comandos        apply_stimulus   snapshot_grid()
```

### Arquitetura Concorrente (Tokio)

**Task 1: WebSocket Server**
- Escuta em TCP:8080
- Recebe comandos JSON do frontend
- Envia frames de telemetria via broadcast

**Task 2: LiquidNetwork Physics (100 Hz)**
- Loop: tick() + RK4 integrator
- Injetar telemetria do LLM
- Slow updates @ 100th tick (homeostase)

**Task 3: HomeostaticRegulator (1 Hz)**
- Observa energia média da LiquidNetwork
- Ajusta γ para manter homeostase
- Modula ritmo circadiano

**Task 4: ActuatorBus (1 Hz)**
- Observa estado de LiquidNetwork
- Dispara comandos emergentes (Research, Focus, Inhibition)
- Envia ações para o frontend

**Task 5: Brain Map Visualization (20 Hz)**
- Snapshot de state/phase da LiquidNetwork
- Encoda em JSON (heatmap/vector field)
- Envia via broadcast

**Task 6: Main Loop (60 Hz)**
```rust
loop {
    // 1. Processar comandos do WS (non-blocking try_recv)
    while let Ok(cmd) = cmd_rx.try_recv() {
        match cmd {
            EngineCommand::Stimulus(text) => {
                let energy = prompt::text_to_energy(&text);
                sim.apply_stimulus(&energy);
            }
            EngineCommand::Resonate(text) => {
                // Spawn async task para geração
                tokio::spawn(async { generate_response(...) });
            }
            EngineCommand::SelectLayer(name) => {
                // Carregar novo tensor via mmap
                sim = Simulation::from_weights(...);
            }
            EngineCommand::ListLayers => {
                // Enviar lista via WebSocket
            }
            EngineCommand::Reset => {
                // sim.reset_all()
            }
        }
    }

    // 2. Aplicar estímulos do pipeline (layer-wise)
    while let Ok((layer_idx, energy)) = think_rx.try_recv() {
        sim.apply_stimulus(&energy);
        liquid_net.inject_layer_stimulus(layer_idx, &energy);
    }

    // 3. Tick da simulação Fauna
    sim.tick(dt);

    // 4. Enviar frame de telemetria (a cada 2 ticks)
    if sim.tick_count % 2 == 0 {
        let grid = sim.snapshot_grid();
        let frame = TelemetryFrame {
            msg_type: "frame",
            grid_size: 128,
            data: grid,
            tick: sim.tick_count,
            layer: sim.layer_name.clone(),
            energy: sim.total_energy(),
            active_nodes: sim.active_count(0.01),
            total_nodes: sim.nodes.len(),
        };
        frame_tx.send(serde_json::to_string(&frame)?)?;
    }

    // 5. Sleepando até próximo tick
    let elapsed = loop_start.elapsed();
    if elapsed < tick_duration {
        tokio::time::sleep(tick_duration - elapsed).await;
    }
}
```

### Canais de Comunicação

| Canal | Tipo | Frequência | Conteúdo | Sender | Receiver |
|-------|------|-----------|----------|--------|----------|
| `cmd_rx` | MPSC | On-demand | EngineCommand (stimulus, reset, etc) | WS handler | Main loop |
| `think_rx` | MPSC | On-demand | (layer_idx, energy) pulsos | Pipeline | Sim + LNN |
| `lnn_tel_tx` | Broadcast | 100 Hz | LlmTelemetry | NLM orchestrator | LNN physics |
| `frame_tx` | Broadcast | 20 Hz | TelemetryFrame JSON | Main loop | WS clients |

---

## 7. Protocolo WebSocket

### Client → Server (JSON)

```json
// Estimular com texto
{"type": "stimulus", "text": "carro"}

// Gerar resposta (modo generativo)
{"type": "resonate", "text": "O que é um carro?"}

// Trocar tensor visualizado
{"type": "select_layer", "name": "model.layers.5.mlp.down_proj"}

// Resetar simulação
{"type": "reset"}

// Listar tensores disponíveis
{"type": "list_layers"}
```

### Server → Client (JSON)

```json
// Frame de telemetria (60 fps)
{
  "type": "frame",
  "grid_size": 128,
  "data": [0.1, 0.05, 0.02, ...],
  "tick": 12345,
  "layer": "model.layers.0.input_layernorm",
  "energy": 45.23,
  "active_nodes": 1024,
  "total_nodes": 2048
}

// Ação de atuador
{
  "type": "actuator",
  "actuator": "ResearchActuator",
  "action": "Pesquisando sobre: neuroplasticidade"
}

// Status/Log
{
  "type": "status",
  "message": "✅ Tensor 'model.layers.5.mlp' carregado com 4M nós"
}

// Progresso de geração
{
  "type": "generation_step",
  "token": "carro",
  "step": 5,
  "total_steps": 100
}

// Lista de camadas
{
  "type": "layers",
  "names": ["model.embed_tokens", "model.layers.0.input_layernorm", ...]
}
```

---

## 8. Decision Points & Trade-offs Arquiteturais

### ✅ Decisão 1: CfC Networks ao invés de Spiking Neurons
**Alternativas consideradas:**
- Spiking (Izhikevich, LIF): Mais biológico, mas requer dt → ms-scale
- LIF com aprox.: Mais rápido, menos preciso
- CfC (Closed-form): Tem solução analítica

**Escolha: CfC**
- Estabilidade numérica garantida
- RK4 mantém precisão com dt = 16ms
- 60 Hz viável em CPU/GPU consumer
- Trade-off: Realismo biológico vs. performance

### ✅ Decisão 2: LiquidBrain Paralelo à Simulação
**Alternativas consideradas:**
- Single net: Mais simples, menos overhead
- Dual layer: Separação de responsabilidades

**Escolha: Dual (Simulation + LiquidBrain)**
- Simulação = fenômeno (visualização, análise)
- LiquidBrain = comportamento (aprendizado, homeostase)
- Permite evolução independente
- Trade-off: Overhead de sincronização vs. modularidade

### ✅ Decisão 3: Qwen Real na GPU vs. Apenas Análise Estática de Pesos
**Alternativas consideradas:**
- Apenas tensores: Análise rápida, sem dinâmica
- Qwen completo: Lento, mas funcional
- Lightweight proxy (DistilBERT): Mais rápido

**Escolha: Qwen Completo**
- Respostas reais (modo Resonate)
- Telemetria dinâmica (entropia, logits)
- Validação: Como o modelo "pensa" durante geração
- Trade-off: Latência de ~5-10s por resposta vs. realismo

### ✅ Decisão 4: Acoplamento de Kuramoto (4-vizinhança 2D)
**Alternativas consideradas:**
- Sem acoplamento: Independente, simples
- Acoplamento completo (all-to-all): Lindo, O(N²) caro
- Kuramoto 4-vizinhança: Eficiente, interpretável

**Escolha: Kuramoto 4-vizinhança**
- O(N) performance
- Inibição lateral natural (selectividade)
- Heatmap 2D fácil de visualizar
- Trade-off: Alcance limitado vs. performance

### ✅ Decisão 5: ROCm/HIP Nativo vs. CUDA-only
**Alternativas consideradas:**
- CUDA-only: Simples, mas proprietary (NVIDIA)
- HIP + fallback CPU: Complexo, mas agnóstico

**Escolha: ROCm com fallback**
- Suporte a AMD GPUS (Navi 22, RDNA)
- CPU fallback automático se GPU falharem
- Trade-off: Build complexity vs. hardware flexibility

### ✅ Decisão 6: Memory-Mapped I/O (mmap) para Pesos
**Alternativas consideradas:**
- Carregar tudo na RAM: Rápido, mas muita memória
- Carregar sob demanda (lazy): Mais lento, pouca memória
- mmap: Zero-copy, on-demand paging

**Escolha: mmap via memmap2 + VarBuilder**
- Eficiência: 8B modelo → ~16 GB arquivo, mas só paga pelo acesso
- Transparente: OS gerencia paging automaticamente
- Trade-off: Page faults ocasionais vs. escalabilidade

---

## 9. Padrões Arquiteturais Utilizados

| Padrão | Implementação | Benefício |
|--------|--------------|----------|
| **Layered** | Fauna / LiquidBrain / Model em módulos separados | Independência, testabilidade |
| **Pipeline** | Estímulo → Sim → Brain → Atuadores | Fluxo claro de dados |
| **Broadcast** | tokio::sync::broadcast para telemetria | Múltiplos subscribers, eficiente |
| **Command** | EngineCommand enum + match | Extensibilidade, type-safe |
| **Observer** | ActuatorBus observa LiquidNetwork | Desacoplamento, reatividade |
| **Memory-mapped** | VarBuilder + memmap2 | Escalabilidade sem trade-off RAM |
| **Async/Await** | tokio::spawn para tasks | Concorrência sem threads pesadas |

---

## 10. Constraints & Limitações Atuais

### Performance
- **Max Nodes**: 1,000,000 (limite por RAM/VRAM)
- **Tick Rate**: 60 Hz recomendado (ajustável via CLI)
- **Telemetry**: Enviada a cada 2 ticks (~30 FPS ao cliente)
- **GPU Memory**: 8B modelo Qwen ≈ 16 GB (Bf16)

### Funcionalidade
- **Hardware Agnóstico**: Candle tenta CUDA → ROCm → CPU automaticamente
- **Sincronização**: Mutex ao invés de lock-free (aceitável em 60 Hz)
- **Escalabilidade**: Rayon paraleliza neurônios, não distribuições
- **Sem Persistência**: Estado não salvo entre sessões

### Estabilidade Numérica
- RK4 integrador previne blow-up
- Viscosidade μ amortece sob stress
- Homeostase circadiana previne convergência para morte/explosão
- Bounds em STDP weights (`[-5, 5]`) previne saturação

---

## 11. Pontos Fortes da Arquitetura

✅ **Modularidade Científica**: Três camadas bem separadas (Physics, Neuro, LLM)

✅ **Real-time Interaction**: WebSocket streaming permite visualização e controle ao vivo

✅ **Fundamentado Biologicamente**: STDP, Kuramoto, homeostase são modelos reais

✅ **Escalável**: Rayon + Candle suportam múltiplos nós, GPUs

✅ **Interativo**: Frontend pode mudar tensores, estímulos, ver efeitos imediatos

✅ **Eficiente**: mmap reduz overhead de memória, KV-cache reduz latência de geração

✅ **Agnóstico a Hardware**: CPU/CUDA/ROCm fallback automático

---

## 12. Pontos Fracos & Oportunidades de Melhoria

⚠️ **Falta de Persistência**
- Estado da LiquidNetwork não é salvo
- *Solução*: RocksDB para snapshots, restauração em boot

⚠️ **Acoplamento CfC Apenas Local**
- Sem conexões de longo alcance (long-range projections)
- *Solução*: Adicionar "hubs" globais (WTA, attention-like)

⚠️ **Sem Backprop**
- Gradientes não fluem das simulações para o modelo
- *Solução*: PyTorch binding para co-training (IARA como loss)

⚠️ **Visualização Limitada**
- Apenas heatmap 2D
- *Solução*: 3D WebGL, grafo de conexões, phase space exploration

⚠️ **Sem Controle de Versão de Modelos**
- Sempre carrega versão local
- *Solução*: HuggingFace Hub integration para swaps dinâmicos

⚠️ **Sem Metrics/Monitoring**
- Logging manual, sem dashboards
- *Solução*: Prometheus/Grafana para telemetria

---

## 13. Roadmap Sugerido

### Phase 1 (Curto Prazo: Próximos 2 Meses)
- [ ] Persistência de estado (RocksDB snapshots)
- [ ] Visualização 3D (Three.js WebGL)
- [ ] Logging estruturado com eventos
- [ ] Tests unitários para physics/STDP
- [ ] Documentação de API WebSocket

### Phase 2 (Médio Prazo: 3-6 Meses)
- [ ] Multi-GPU support (distribuir osciladores)
- [ ] Backprop-enabled training (PyTorch bridge)
- [ ] Neurotransmitter dynamics (dopamine, GABA modulation)
- [ ] CLI para batch analysis
- [ ] Benchmarks comparativos

### Phase 3 (Longo Prazo: 6+ Meses)
- [ ] Recurrent state para context windows > 32K
- [ ] Embodied interaction (robótica simulada, feedback)
- [ ] Meta-learning (IARA aprendendo hiperparâmetros)
- [ ] Sparse coupling (reduzir O(N²) complexidade)
- [ ] Hierarchical architecture (múltiplas escalas)

---

## 14. Diagrama de Dependências de Módulos

```
┌────────────────────────────────────────────────┐
│              main.rs (Orquestrador)            │
│  - parse_args, init, spawn tasks, main loop   │
└────────────────────────────────────────────────┘
    │    │        │        │        │        │
    ├────┼────┬───┼────┬───┼───┬────┼─────┬─┤
    │    │    │   │    │   │   │    │     │ │
    ▼    ▼    ▼   ▼    ▼   ▼   ▼    ▼     ▼ ▼
 websocket model physics generation homeostasis actuator
 (WS)      (Qwen (CfC)  (streaming) (circadian)  (React)
            8B)

    │ dependências ↓
    │
 simulation ◄─── physics ◄─── NeuronioCfC
    │
    ├──── liquid_network ◄─── STDP, ODE solver
    │
    └──── safetensors_reader ◄─── memmap2, serde
         └──── tokenizer ◄─── BPE encoding
```

---

## 15. Métricas de Sucesso

### Performance
- Latência de tick: < 20ms (target 16ms para 60Hz)
- Memória: < 32 GB (8B model + 1M nodes)
- GPU utilization: 80-95% durante forward pass

### Científico
- Entropia vs. ativação: correlação negativa (aumento stress → diminui sincronização)
- STDP learning: convergência em < 1 hora
- Homeostase: energia média ±10% sobre 24h simulado

### UX
- WebSocket latência: < 100ms (telemetria)
- Responsividade: Click-to-visualization < 500ms
- Browser: Chrome/Firefox, sem freezes

---

## 16. Conclusão

**IARA é uma arquitetura inovadora que funde três disciplinas:**

- 🧠 **Neurodinâmica**: CfC networks, Kuramoto coupling, STDP, homeostase circadiana
- 🤖 **IA Generativa**: Qwen 8B com streaming real-time
- 📡 **Real-time Visualization**: WebSocket, telemetria dinâmica

A modularidade permite que cada camada evolua independentemente, enquanto o sistema integrado oferece insights únicos sobre como modelos de linguagem podem ser mapeados em dinâmica neuronal mensurável e adaptativa.

**Próximos passos ideais:**
1. Persistência de estado (salvar aprendizado)
2. Visualização 3D (exploração melhor)
3. Backprop (co-training com Qwen)
4. Multi-GPU (escalabilidade)

---

**Documento gerado em:** 2026-03-25
**Versão:** 1.0
**Autor:** Leonardo (via Claude Architecture Analysis)

---

## Apêndice A: Glossário de Termos

- **CfC**: Closed-form Continuous — neurônios com solução analítica
- **LTC**: Liquid Time Constant — dinâmica adaptativa de membrana
- **STDP**: Spike-Timing-Dependent Plasticity — aprendizado causal
- **Kuramoto**: Modelo de sincronização de osciladores acoplados
- **GQA**: Grouped Query Attention — otimização de KV-cache
- **mmap**: Memory-mapped I/O — acesso eficiente a arquivos grandes
- **RK4**: Runge-Kutta 4ª ordem — integração numérica precisa
- **μ (Viscosidade)**: Fator que amortece propagação sob stress
- **Homeostase**: Manutenção de estado equilibrado energético

---

## Apêndice B: Referências Técnicas

- Bellec et al., "Biologically plausible learning" (CfC networks)
- Kuramoto, "Chemical Oscillations, Waves, and Turbulence"
- Bi & Poo, "Synaptic Modifications in Cultured Hippocampal Neurons" (STDP)
- Vaswani et al., "Attention is All You Need" (Transformer)
- Hu et al., "Scaling Vision Transformers" (GQA concept)
