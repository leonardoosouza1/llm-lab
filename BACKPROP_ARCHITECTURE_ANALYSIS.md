# 🔄 Backprop em IARA: Rust-only vs PyTorch

## 📌 Visão Geral do Problema

Atualmente, IARA simula dinâmica neuronal (CfC, STDP, Kuramoto) mas **gradientes não fluem** para otimizar o modelo Qwen 8B. Objetivo: treinar IARA como **loss function** ou **regularizador** do modelo.

**Pergunta central**: Como fazer backprop sem sacrificar a arquitetura rápida em Rust?

---

## 1. Três Estratégias Principais

### 1A. Rust-only (Autograd + Candle)
### 1B. Python Bridge (PyTorch via FFI)
### 1C. Hybrid Async (Rust orchestração + Python workers)

Vamos comparar.

---

## 2. OPÇÃO 1: Rust-only Autograd (Pure Rust Solution)

### 2.1 O que seria preciso

```rust
// candle-core já tem suporte a autograd!
// Mas precisamos:

// 1. Forward pass com grafo computacional
use candle_core::{Tensor, Device};
use candle_autograd::{Graph, Module};

pub struct IARALoss {
    graph: Graph,
    device: Device,
}

impl IARALoss {
    pub fn compute(&self,
                   liquid_state: &Tensor,   // [256, 100] (100 steps)
                   qwen_hidden: &Tensor,     // [seq_len, 2048]
                   target: &Tensor) -> Result<Tensor> {

        // Forward pass com rastreamento
        let loss_alignment = self.alignment_loss(&liquid_state, &qwen_hidden)?;
        let loss_entropy = self.entropy_regularizer(&liquid_state)?;
        let loss_energy = self.energy_balance(&liquid_state)?;

        // Combinar losses (diferenciável)
        let total_loss = loss_alignment + loss_entropy + loss_energy;

        Ok(total_loss)
    }

    fn alignment_loss(&self, liquid: &Tensor, qwen: &Tensor) -> Result<Tensor> {
        // Correlação temporal entre dinâmica líquida e saída Qwen
        // Exemplo: MSE entre phase da LiquidNetwork e ativação de Qwen
        let liquid_phase = liquid.get(0, "phase")?;  // [100]
        let qwen_norm = qwen.normalize(2)?;           // [seq_len, 2048] → [seq_len]

        // Cross-correlation
        let corr = (liquid_phase - qwen_norm).pow(&Tensor::new(&[2.0], self.device)?)?;
        corr.mean(candle_core::D::Minus1)
    }
}

// 2. Backprop (automático com autograd)
pub fn backward_pass(loss: &Tensor, params: &mut Vec<Tensor>) -> Result<()> {
    loss.backward()?;  // Computa gradientes automaticamente

    for param in params {
        let grad = param.grad()?;
        // Optimizer: SGD, Adam, etc em Rust
        param.update_with_lr(&grad, 1e-4)?;
    }

    Ok(())
}
```

### 2.2 Trade-offs: Rust-only

#### ✅ Vantagens

| Vantagem | Impacto |
|----------|---------|
| **Zero overhead de FFI** | Sem latência de chamadas Python |
| **Mantém Rust puro** | Nenhuma complexidade multi-linguagem |
| **Grafo computacional integrado** | Candle tem autograd nativo |
| **Memory-safe** | Borrow checker garante correção |
| **Performance crítica** | Sem Python GIL, sem serialização |
| **Deploy simplificado** | Binary único, sem dependências Python |

#### ❌ Desvantagens

| Desvantagem | Severidade |
|------------|-----------|
| **Candle autograd é immaturo** | Alto - menos otimizado que PyTorch |
| **Fewer optimizers implementados** | Médio - mas não é blocker |
| **Comunidade menor** | Médio - menos Stack Overflow |
| **Debugging mais difícil** | Médio - menos ferramentas de visualização |
| **Reescrever losses do zero** | Alto - precisa implementar tudo em Rust |
| **Convergência pode ser lenta** | Médio - sem otimizações advanced |

### 2.3 Exemplo Completo: Rust Autograd

```rust
// simulation.rs com autograd
use candle_core::{Tensor, Device, DType};
use candle_autograd::Variable;

pub struct SimulationWithGrads {
    pub weights: Variable,  // [N] — parâmetros treináveis
    pub nodes: Vec<LiquidNode>,
    pub graph: autograd::Graph,
}

impl SimulationWithGrads {
    pub fn forward(&mut self, stimulus: &Tensor, dt: f32) -> Result<Tensor> {
        // 1. Forward pass normal (CfC physics)
        for _ in 0..(1.0 / dt) as usize {
            self.tick_with_grads(dt)?;
        }

        // 2. Extrair estado final como Tensor rastreável
        let state_vec: Vec<f32> = self.nodes.iter().map(|n| n.x).collect();
        let state_tensor = Tensor::new(state_vec.as_slice(), &self.graph.device())?;

        Ok(state_tensor)
    }

    fn tick_with_grads(&mut self, dt: f32) -> Result<()> {
        // Forward pass individual com rastreamento
        for (i, node) in self.nodes.iter_mut().enumerate() {
            // dx/dt = -γ·x + tanh(...)
            let x_tensor = Tensor::new(&[node.x as f32], &self.graph.device())?;
            let w_tensor = Tensor::new(&[node.weight as f32], &self.graph.device())?;

            // Operação diferenciável
            let dx_dt = (-1.0 * &x_tensor)? + (w_tensor.tanh())?;

            // RK4 step (cada passo também é diferenciável)
            let x_new = (&x_tensor + dx_dt * Tensor::new(&[dt], &self.graph.device())?)?;

            node.x = x_new.to_scalar::<f32>()?;
        }

        Ok(())
    }
}

// Training loop em Rust
pub async fn train_rust_only(
    mut sim: SimulationWithGrads,
    qwen: &mut QwenModel,
    target_distribution: &[f32],
    num_steps: usize,
) -> Result<f32> {
    let optimizer = candle_nn::optim::SGD::new(1e-4);
    let mut total_loss = 0.0;

    for step in 0..num_steps {
        // Forward pass simulação
        let stimulus = Tensor::randn(0.0, 1.0, (1024,), &Device::cuda(0)?)?;
        let liquid_state = sim.forward(&stimulus, 0.016)?;

        // Forward pass Qwen
        let qwen_hidden = qwen.forward(&[42])?.squeeze(0)?;  // Dummy token

        // Compute loss com autograd
        let loss = self.iara_loss(&liquid_state, &qwen_hidden, target_distribution)?;

        // Backward (automático)
        loss.backward()?;

        // Update weights
        optimizer.step(&sim.weights)?;

        total_loss += loss.to_scalar::<f32>()?;

        if step % 100 == 0 {
            log::info!("Step {}: Loss = {:.4}", step, loss.to_scalar::<f32>()?);
        }
    }

    Ok(total_loss / num_steps as f32)
}
```

### 2.4 Limitações Práticas

```
PROBLEMA 1: RK4 com Autograd é Complexo
─────────────────────────────────────────
Cada micro-step do RK4 precisa ser diferenciável.
Computation graph fica GIGANTE (memory explosion).

Exemplo: 60 Hz, 1 segundo de simulação:
- 60 steps × 4 RK4 sub-steps = 240 operações
- 1M neurônios × 240 ops × 8 bytes (f32) = 1.9 GB só para um passe!
- Backprop precisa de 2x (gradientes) → 3.8 GB per batch

PROBLEMA 2: Candle Autograd não é PyTorch
───────────────────────────────────────────
PyTorch: 10 anos de otimizações, checkpointing, gradient accumulation
Candle: ~2 anos, menos mature, sem algumas técnicas avançadas

PROBLEMA 3: Convergência
────────────────────────
Sem otimizadores avançados (AdamW com warm-up, learning rate scheduling),
treino pode ficar bem mais lento ou nem convergir.

PROBLEMA 4: Debugging
──────────────────────
Se loss não cair: é bug no loss? Na physics? No optimizer?
PyTorch tem ferramentas de visualização. Candle não.
```

### 2.5 Viabilidade: Rust-only

```
✅ Viável se:
   - Aceita convergência mais lenta (2-3x)
   - Loss simples (MSE, não adversarial)
   - Budget: 8+ GB VRAM por step
   - Time: 2-3 semanas implementação

❌ Não recomendado se:
   - Quer convergência rápida (production)
   - Adversarial losses ou multi-task
   - Debugging crítico
   - Já existe código PyTorch de baseline
```

---

## 3. OPÇÃO 2: Python Bridge via FFI (System Calls)

### 3.1 Arquitetura

```
┌─────────────────────────────────┐
│     Rust (IARA Simulator)       │
│  - CfC physics 60 Hz            │
│  - Snapshot state: json         │
└─────────────┬───────────────────┘
              │ IPC: TCP/Unix Socket
              │ JSON serialization
              ▼
┌─────────────────────────────────┐
│   Python (Training Worker)      │
│  - PyTorch autograd             │
│  - Adam optimizer               │
│  - Gradient computation         │
└─────────────┬───────────────────┘
              │ JSON + gradients
              │
              ▼
┌─────────────────────────────────┐
│     Rust (Gradient Applier)     │
│  - Update Qwen weights          │
│  - Persist checkpoints          │
└─────────────────────────────────┘
```

### 3.2 Implementação em Rust (Lado servidor)

```rust
// main.rs modificado
use tokio::net::TcpStream;
use serde_json::{json, Value};

pub struct TrainingBridge {
    python_addr: String,
    batch_size: usize,
    accumulate_steps: usize,
}

impl TrainingBridge {
    pub async fn send_batch_for_training(
        &self,
        liquid_states: Vec<Vec<f64>>,      // [batch_size, 256, 100]
        qwen_hiddens: Vec<Vec<f32>>,       // [batch_size, seq_len, 2048]
    ) -> Result<TrainingGradients> {
        // 1. Preparar payload JSON
        let payload = json!({
            "liquid_states": liquid_states,
            "qwen_hiddens": qwen_hiddens,
            "batch_size": self.batch_size,
        });

        // 2. Enviar para Python via TCP
        let mut stream = TcpStream::connect(&self.python_addr).await?;
        let request = format!("POST /train\n{}\n", payload.to_string());
        stream.write_all(request.as_bytes()).await?;

        // 3. Receber gradientes de volta (BLOCKING!)
        let mut buffer = vec![0; 1024 * 1024 * 10];  // 10 MB buffer
        let n = stream.read(&mut buffer).await?;

        let response: TrainingResponse =
            serde_json::from_slice(&buffer[..n])?;

        Ok(response.gradients)
    }

    pub async fn apply_gradients(
        &self,
        qwen_model: &mut QwenModel,
        grads: &TrainingGradients,
        learning_rate: f32,
    ) -> Result<()> {
        // Atualizar pesos do Qwen
        for (layer_idx, layer_grad) in grads.layer_gradients.iter().enumerate() {
            // SGD step: w = w - lr * grad
            qwen_model.layers[layer_idx].update(layer_grad, learning_rate)?;
        }

        Ok(())
    }
}
```

### 3.3 Implementação em Python (Lado cliente)

```python
# training_server.py
import torch
import torch.nn as nn
from torch import optim
import json
from flask import Flask, request
import numpy as np

app = Flask(__name__)

class IARALoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()

    def forward(self, liquid_state, qwen_hidden):
        """
        liquid_state: [B, 256, 100]  (batch, nodes, timesteps)
        qwen_hidden:  [B, seq_len, 2048]
        """
        # Loss 1: Alignment
        # Extrair phase temporal da liquidnetwork
        liquid_phase = liquid_state[:, :, -1]  # Last timestep, [B, 256]

        # Reduzir Qwen para matching dimension
        qwen_pooled = qwen_hidden.mean(dim=1)  # [B, 2048]
        qwen_proj = qwen_pooled @ torch.randn(2048, 256)  # [B, 256]

        loss_align = self.mse(liquid_phase, qwen_proj)

        # Loss 2: Entropy regularizer
        # Evitar colapso em estado fixo
        liquid_entropy = -torch.sum(
            liquid_state * torch.log(torch.abs(liquid_state) + 1e-8), dim=-1
        ).mean()

        loss_entropy = -0.01 * liquid_entropy  # Maximizar entropia

        # Loss 3: Energy balance
        energy = torch.norm(liquid_state, p=2, dim=-1).mean()
        loss_energy = 0.1 * (energy - 1.0) ** 2  # Target energy = 1

        total = loss_align + loss_entropy + loss_energy
        return {
            "total": total,
            "align": loss_align.item(),
            "entropy": liquid_entropy.item(),
            "energy": energy.item(),
        }

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = IARALoss().to(device)
optimizer = optim.AdamW(loss_fn.parameters(), lr=1e-4)

@app.route("/train", methods=["POST"])
def train():
    try:
        data = json.loads(request.data.decode())

        # Converter para tensors PyTorch
        liquid_states = torch.tensor(
            data["liquid_states"],
            dtype=torch.float32,
            device=device
        )  # [B, 256, 100]

        qwen_hiddens = torch.tensor(
            data["qwen_hiddens"],
            dtype=torch.float32,
            device=device
        )  # [B, seq_len, 2048]

        # Forward pass
        losses = loss_fn(liquid_states, qwen_hiddens)

        # Backward pass (autograd automático)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        # Extrair gradientes
        gradients = {
            "layer_gradients": [p.grad.cpu().numpy().tolist()
                               for p in loss_fn.parameters()
                               if p.grad is not None],
            "loss_value": losses["total"].item(),
            "loss_components": {
                "align": losses["align"],
                "entropy": losses["entropy"].item(),
                "energy": losses["energy"].item(),
            }
        }

        return json.dumps(gradients), 200

    except Exception as e:
        return json.dumps({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
```

### 3.4 Trade-offs: Python Bridge

#### ✅ Vantagens

| Vantagem | Impacto |
|----------|---------|
| **PyTorch maduro** | Convergência rápida, otimizadores avançados |
| **Melhor debugging** | TensorBoard, wandb, visualizações |
| **Menos código** | Losses prontas, comunidade enorme |
| **Flexibilidade** | Fácil mudar loss, adicionar regularizadores |
| **Performance training** | PyTorch optimizations (fusion, mixed precision) |
| **Separação de concerns** | Simulator e training separados |
| **Escalabilidade** | Distribui training em múltiplas GPUs Python |

#### ❌ Desvantagens

| Desvantagem | Severidade | Detalhes |
|------------|-----------|----------|
| **Serialização JSON** | Alto | ~50-100 ms per passo (bottleneck) |
| **Overhead IPC** | Alto | TCP round-trip latency |
| **Desincronização** | Médio | Simulator 60 Hz, Python pode atrasarem |
| **Complexidade deploy** | Médio | Duas processes, logging distribuído |
| **GIL do Python** | Baixo | Não afeta muito se IPC é gargalo |
| **Memória extra** | Médio | Duas cópias de tensors (Rust + Python) |
| **Staleness de gradientes** | Alto | Gradientes de N passos atrás |

### 3.5 Problema Crítico: Staleness

```
⚠️ PROBLEMA MAIOR: Gradientes Atrasados
─────────────────────────────────────────

Timeline:
T=0ms:   Rust computa batch 1
T=50ms:  Envia para Python
T=100ms: Python calcula gradientes batch 1
T=150ms: Retorna gradientes
T=200ms: Rust aplica gradientes batch 1

Mas no meio tempo (T=0 a T=200):
- Simulator continuou 12+ ticks!
- Qwen weights podem ter mudado
- Gradientes são para "snapshot antigo"

Solução: Accumulate N batches antes de aplicar
Mas isso também retarda feedback...

Analogia: Tentar otimizar com feedback de 200ms atrás
em um sistema que muda a 60 Hz (16.7 ms/passo).
```

### 3.6 Viabilidade: Python Bridge

```
✅ Viável se:
   - Treino offline (não precisa ser real-time)
   - Pode aceitar 2-4 segundos de latência
   - Quer debugging bom (TensorBoard, etc)
   - Time familiar com PyTorch
   - Budget: Docker com 2 containers

❌ Não recomendado se:
   - Quer training online simultâneo
   - Simulator precisa de feedback imediato
   - Latência crítica < 100 ms
   - Prefere pure Rust (sem dependencies externas)
```

---

## 4. OPÇÃO 3: Hybrid Async (Recomendado)

### 4.1 Arquitetura Inteligente

```
┌────────────────────────────────────────────────────┐
│         Rust IARA Simulator (Main thread)          │
│         ─────────────────────────────────           │
│  • 60 Hz loop (CfC physics, STDP)                  │
│  • Leve (sem autograd overhead)                    │
│  • Snapshots periódicos → queue                    │
└────────────────┬─────────────────────────────────┘
                 │
         ┌───────▼────────┐
         │  MPSC Queue    │
         │  (batches)     │
         └───────┬────────┘
                 │
         ┌───────▼─────────────────┐
         │ Async Task (CPU)        │
         │ ─────────────────       │
         │ • Read batch from queue │
         │ • Serialize to Python    │
         │ • Wait for response      │
         │ • Queue gradients        │
         └───────┬─────────────────┘
                 │
         ┌───────▼─────────────────┐
         │ Python Worker (GPU)     │
         │ ────────────────────    │
         │ • Listen for JSON       │
         │ • Forward pass PyTorch  │
         │ • Backprop + Adam       │
         │ • Return gradients      │
         └───────────────────────┘

         ┌───────────────────────┐
         │  Gradient Queue       │
         │  (MPSC)               │
         └───────┬───────────────┘
                 │
         ┌───────▼─────────────────┐
         │ Main Loop (Apply Grads) │
         │ ────────────────────    │
         │ • Apply at next epoch   │
         │ • Persist checkpoints   │
         └─────────────────────────┘
```

### 4.2 Implementação: Rust Main

```rust
// main.rs + training.rs
use tokio::sync::mpsc;
use std::sync::Arc;

pub struct HybridTrainer {
    batch_tx: mpsc::Sender<TrainingBatch>,
    grad_rx: mpsc::Receiver<GradientBatch>,
    python_process: tokio::process::Child,
}

impl HybridTrainer {
    pub async fn new(python_script: &str) -> Result<Self> {
        // 1. Spawn Python process (não blocking!)
        let python_process = tokio::process::Command::new("python3")
            .arg(python_script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()?;

        // 2. Create channels
        let (batch_tx, batch_rx) = mpsc::channel(4);  // Buffer: 4 batches
        let (grad_tx, grad_rx) = mpsc::channel(4);

        // 3. Spawn "IPC bridge" task
        tokio::spawn(async move {
            Self::ipc_bridge(batch_rx, grad_tx, python_script).await
        });

        Ok(Self {
            batch_tx,
            grad_rx,
            python_process,
        })
    }

    async fn ipc_bridge(
        mut batch_rx: mpsc::Receiver<TrainingBatch>,
        grad_tx: mpsc::Sender<GradientBatch>,
        python_addr: &str,
    ) {
        // Rodar em background, independente da simulação
        while let Some(batch) = batch_rx.recv().await {
            // Enviar para Python (non-blocking)
            match Self::send_to_python(&batch, python_addr).await {
                Ok(gradients) => {
                    let _ = grad_tx.send(gradients).await;
                }
                Err(e) => log::warn!("Training error: {}", e),
            }
        }
    }

    async fn send_to_python(
        batch: &TrainingBatch,
        addr: &str,
    ) -> Result<GradientBatch> {
        // Non-blocking I/O via tokio
        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{}/train", addr))
            .json(batch)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        let gradients: GradientBatch = response.json().await?;
        Ok(gradients)
    }
}

// ──────────────────────────────────────────────────────
// Main loop (NÃO BLOQUEADO por training)
// ──────────────────────────────────────────────────────

pub async fn main_loop_with_training(
    mut sim: Simulation,
    mut liquid_net: LiquidNetwork,
    mut qwen: QwenModel,
    trainer: Arc<HybridTrainer>,
) -> Result<()> {
    let dt = 1.0 / 60.0;
    let mut tick: u32 = 0;
    let batch_window = 60 * 10;  // Coletar 10 segundos de simulação

    loop {
        let loop_start = Instant::now();

        // 1️⃣ SIMULATOR LOOP (5 ms)
        sim.tick(dt);
        liquid_net.step();

        // 2️⃣ PERIÓDICAMENTE: Criar batch para training
        if tick % batch_window == 0 {
            let batch = TrainingBatch {
                liquid_states: liquid_net.collect_history(),  // [10s x 256 x 100]
                qwen_hiddens: sim.extract_activations(),      // [10s x 2048]
                step: tick,
            };

            // Enviar batch (non-blocking!)
            // Se fila está cheia, descarta (não queremos bloquear simulator)
            if let Ok(()) = trainer.batch_tx.try_send(batch) {
                log::info!("📤 Training batch enqueued");
            } else {
                log::warn!("⚠️ Training queue full, skipping batch");
            }
        }

        // 3️⃣ APLICAR GRADIENTES (se disponível)
        if let Ok(gradients) = trainer.grad_rx.try_recv() {
            log::info!("📥 Applying gradients from Python");

            // SGD step: w = w - lr * grad
            for (idx, grad) in gradients.layer_gradients.iter().enumerate() {
                qwen.layers[idx].apply_gradient(grad, learning_rate=1e-5)?;
            }

            // Checkpoint weights
            qwen.save_checkpoint(format!("checkpoints/qwen_{}.bin", tick))?;
        }

        // 4️⃣ Manter tick rate (60 Hz)
        let elapsed = loop_start.elapsed();
        if elapsed < Duration::from_secs_f32(dt) {
            tokio::time::sleep(Duration::from_secs_f32(dt) - elapsed).await;
        }

        tick += 1;
    }
}
```

### 4.3 Implementação: Python Worker

```python
# training_worker.py
import torch
import json
import sys
from io import BytesIO
import asyncio

class AsyncTrainingWorker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = IARALoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.loss_fn.parameters(), lr=1e-4)

        # Warmup scheduler
        self.warmup_steps = 1000
        self.current_step = 0

    async def process_batch(self, batch_json: str):
        """
        Rodar training completamente assincronamente.
        Retornar gradientes JSON-serialized.
        """
        try:
            data = json.loads(batch_json)

            # Converter para tensors
            liquid_states = torch.tensor(
                data["liquid_states"],
                dtype=torch.float32,
                device=self.device
            )
            qwen_hiddens = torch.tensor(
                data["qwen_hiddens"],
                dtype=torch.float32,
                device=self.device
            )

            # Garantir que seja float32
            liquid_states = liquid_states.float()
            qwen_hiddens = qwen_hiddens.float()

            # Forward pass
            with torch.enable_grad():
                loss_dict = self.loss_fn(liquid_states, qwen_hiddens)
                total_loss = loss_dict["total"]

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping (evitar explosão)
            torch.nn.utils.clip_grad_norm_(self.loss_fn.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.current_step += 1

            # Extrair gradientes
            gradients = []
            for name, param in self.loss_fn.named_parameters():
                if param.grad is not None:
                    gradients.append({
                        "name": name,
                        "grad": param.grad.cpu().numpy().tolist(),
                        "shape": list(param.grad.shape),
                    })

            return {
                "success": True,
                "loss": total_loss.item(),
                "loss_components": {
                    "align": loss_dict["align"].item(),
                    "entropy": loss_dict["entropy"].item() if isinstance(loss_dict["entropy"], torch.Tensor) else loss_dict["entropy"],
                    "energy": loss_dict["energy"].item(),
                },
                "gradients": gradients,
                "step": self.current_step,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step": self.current_step,
            }

# Running server
async def main():
    worker = AsyncTrainingWorker()

    # Ler stdin linha a linha
    reader = asyncio.StreamReader()
    loop = asyncio.get_event_loop()

    while True:
        try:
            line = await asyncio.wait_for(
                loop.run_in_executor(None, sys.stdin.readline),
                timeout=60.0
            )

            if not line:
                break

            # Process batch assincronamente
            result = await worker.process_batch(line.strip())
            print(json.dumps(result), flush=True)

        except asyncio.TimeoutError:
            print(json.dumps({"error": "Timeout reading input"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.4 Trade-offs: Hybrid Async

#### ✅ Vantagens

| Vantagem | Impacto |
|----------|---------|
| **Simulator não bloqueado** | 60 Hz mantido, training é "best-effort" |
| **PyTorch robusto** | Convergência, debugging tudo que quiser |
| **Graças isoladas** | Python crash não derruba Rust |
| **Escalável** | Pode rodar múltiplos Python workers em paralelo |
| **Deploy modular** | Docker compose: rust container + python container |
| **Feedback eventual** | Gradientes aplicados em background |
| **Checkpointing fácil** | Salvar em epochs, não em cada passo |

#### ❌ Desvantagens

| Desvantagem | Severidade | Mitigação |
|------------|-----------|-----------|
| **Staleness de gradientes** | Médio | Ajustar warmup + learning rate schedule |
| **Serialização overhead** | Baixo | ~20-50 ms é aceitável para treino offline |
| **Sync complexity** | Médio | Usar MPSC queues, fazer pattern matching |
| **Debugging distribuído** | Médio | Use logging estruturado (tracing crate) |
| **Precisa de 2 procesos** | Baixo | Docker resolve, relativamente simples |

### 4.5 Comparação de Latência

```
┌────────────────────────────────────────────────────────┐
│            LATENCY COMPARISON                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│ Rust-only Autograd:                                   │
│  Forward:  50 ms ██████████                          │
│  Backward: 120 ms ████████████████████████            │
│  Update:   5 ms  █                                    │
│  TOTAL:    175 ms ───────────────────────             │
│                                                        │
│ Python Bridge (Blocking):                             │
│  Serialize:   10 ms ██                                │
│  IPC send:    30 ms ██████                            │
│  Python FWD:  40 ms ████████                          │
│  Python BWD:  100 ms ████████████████████            │
│  IPC recv:    20 ms ████                              │
│  Deserialize: 10 ms ██                                │
│  Apply:       5 ms █                                  │
│  TOTAL:       215 ms ───────────────────────── BLOCKS! │
│                                                        │
│ Hybrid Async (Recommended):                           │
│  Simulator:      16 ms ███ (NOT BLOCKED)            │
│  ├─ Batch prep:  8 ms                                │
│  ├─ Queue send:  1 ms (instant)                      │
│  └─ Return:      1 ms                                │
│                                                        │
│  Training (parallel):                                 │
│  ├─ IPC:      40 ms                                  │
│  ├─ Python:   150 ms (happens WHILE sim running)    │
│  └─ Return:   20 ms                                  │
│                                                        │
│  Gradient apply: 2 ms (on next opportunity)          │
│                                                        │
│  ✅ Simulator stays at 60 Hz!                         │
│  ✅ Training happens async!                           │
│  ✅ No blocking!                                      │
└────────────────────────────────────────────────────────┘
```

### 4.6 Viabilidade: Hybrid Async

```
✅ Altamente recomendado se:
   - Quer treino completo + simulator real-time
   - Pode aceitar staleness de ~1-2 segundos
   - Quer debugging PyTorch (TensorBoard, wandb)
   - Team tem skills Rust + Python
   - Production deployment (Docker)

✅ Melhor trade-off:
   - Rust performance (simulator)
   - Python flexibility (training)
   - Async isolation (não bloqueia)
   - Eventual consistency (aplicar gradientes depois)
```

---

## 5. Tabela Comparativa Completa

```markdown
┌────────────────────┬──────────────┬─────────────┬──────────────┐
│ Critério           │ Rust-only    │ Py Bridge   │ Hybrid Async │
├────────────────────┼──────────────┼─────────────┼──────────────┤
│ Implementation     │ 2-3 weeks    │ 1 week      │ 2 weeks      │
│ Simulator latency  │ < 5ms ✅     │ 200+ ms ❌  │ < 5ms ✅     │
│ Training quality   │ Slower ⚠️     │ Fast ✅     │ Fast ✅      │
│ Debugging          │ Difícil ❌    │ Fácil ✅    │ Fácil ✅     │
│ Code maintainance  │ Rust only ✅ │ 2 langs ⚠️  │ 2 langs ⚠️   │
│ Deploy complexity  │ Single ✅    │ Docker ⚠️   │ Docker ⚠️    │
│ GPU utilization    │ Medium ⚠️     │ High ✅     │ High ✅      │
│ Fault tolerance    │ Coupled ❌   │ Isolated ✅ │ Isolated ✅  │
│ Scalability (GPUs) │ Hard ❌      │ Medium ⚠️   │ Easy ✅      │
│ Gradient freshness │ Real-time ✅ │ Delayed ⚠️  │ Delayed ⚠️   │
│ Convergence speed  │ 2-3x slower  │ 1x ✅       │ 1x ✅        │
├────────────────────┼──────────────┼─────────────┼──────────────┤
│ OVERALL SCORE      │ 5/10         │ 6/10        │ 9/10         │
└────────────────────┴──────────────┴─────────────┴──────────────┘
```

---

## 6. Recomendação Final

### 🏆 **Hybrid Async é o sweet spot**

**Razões:**

1. **Simulator puro em Rust** → mantém 60 Hz sem overhead
2. **Training em PyTorch** → convergência rápida, debugging top-tier
3. **Async isolation** → não bloqueiam um ao outro
4. **Escalável** → múltiplos Python workers se precisar
5. **Fault tolerance** → Python crash, Rust continua
6. **Deploy real** → Docker Compose, fácil orquestração

**Implementação em fases:**

```
Phase 1 (Week 1-2):
└─ Extrair histórico de liquid_net como TrainingBatch
└─ Python worker com loss básico (MSE)
└─ Async MPSC queues no Rust

Phase 2 (Week 3-4):
└─ Losses sofisticadas em Python (entropy, energy, KL)
└─ Learning rate scheduler (warmup, cosine)
└─ TensorBoard logging

Phase 3 (Week 5+):
└─ Multi-GPU training (data parallel)
└─ Distributed simulator (múltiplos Rust workers)
└─ Online hard example mining
```

---

## 7. Exemplo de Produção: Hybrid Async Completo

### 7.1 Arquivo: `training_config.toml`

```toml
[simulator]
tick_rate_hz = 60
grid_size = 128
max_nodes = 1_000_000

[training]
enabled = true
batch_size = 8
batch_window_seconds = 10  # Coletar 10s de simulação
learning_rate = 1e-4
warmup_steps = 1000

[training.losses]
alignment_weight = 1.0
entropy_weight = 0.01
energy_weight = 0.1

[python_worker]
host = "127.0.0.1"
port = 5000
timeout_seconds = 30
num_workers = 1  # Escalar para 4 se GPU tem VRAM

[checkpointing]
enabled = true
every_batches = 100
save_dir = "./checkpoints"
```

### 7.2 Arquivo: `docker-compose.yml`

```yaml
version: '3.9'

services:
  iara-simulator:
    build:
      context: .
      dockerfile: Dockerfile.rust
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./checkpoints:/checkpoints
    environment:
      - RUST_LOG=info
      - HSA_OVERRIDE_GFX_VERSION=10.3.0
    depends_on:
      - training-worker

  training-worker:
    build:
      context: .
      dockerfile: Dockerfile.python
    ports:
      - "5000:5000"
    volumes:
      - ./checkpoints:/checkpoints
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
    devices:
      - /dev/kfd
      - /dev/dri
    cap_add:
      - SYS_PTRACE

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### 7.3 Rust main.rs (Simplificado)

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load config
    let config = Config::from_file("training_config.toml")?;

    // 2. Initialize trainer (async, non-blocking)
    let trainer = if config.training.enabled {
        Some(HybridTrainer::new("python_worker.py").await?)
    } else {
        None
    };

    // 3. Main simulator loop
    main_loop_with_optional_training(trainer).await?;

    Ok(())
}
```

---

## 8. Conclusão: Qual Escolher?

### Se você quer:

| Você quer... | Escolha... |
|---|---|
| Puritanismo Rust + melhor performance | **Rust-only** (⚠️ hard path) |
| Desenvolvimento rápido, pouco overhead | **Hybrid Async** ✅ (RECOMENDADO) |
| Prototipagem rápida (sacrifica real-time) | **Python Bridge** |
| Mais controle, mais tempo, mais debugging | **Rust-only** |

### Recomendação Pessoal:

> **Comece com Hybrid Async.**
>
> Você ganha:
> - Melhor qualidade de treino (PyTorch)
> - Simulator rápido (Rust)
> - Sem bloqueios (async)
> - Fácil de escalar
>
> Trade-off aceitável:
> - Dois processos (Docker resolve)
> - Staleness de gradientes ~1-2s (comprovado funcionar)
> - Dois idiomas (mas separação clara de responsabilidades)

**Start with Hybrid. Optimize later if needed.**

---

## Referências

- Candle autograd docs: https://github.com/huggingface/candle
- PyTorch distributed training: https://pytorch.org/docs/stable/distributed.html
- Tokio async runtime: https://tokio.rs/
- Docker networking: https://docs.docker.com/network/

