# IARA Roadmap — Do Motor ao Modelo Próprio

**Data**: 2026-04-03
**Visão**: "Não vamos ficar presos só no motor de inferência — podemos até treinar um modelo novo a partir dos tensores." — Leonardo

---

## Fase 1 — Motor Otimizado (o que já temos)

- [x] Kernel GEMV Q4 corrigido (tree reduction bug)
- [x] Sync desnecessário removido (5.7 → 6.0 tok/s)
- [x] Análise de paisagem de pesos (pins, micelio, dead neurons)
- [x] Síntese com Universe/iara_agi (BitNet, KAN, CfC, Kuramoto)
- [x] Experimentos empíricos de quantização P+T+T+δ

## Fase 2 — Micelio Router (próximo sprint)

**Objetivo**: 2-3x de throughput SEM mudar a qualidade.

### Como fazer:

```rust
// No HIP kernel do MLP, ANTES de computar o GEMM completo:
// 1. Lê os hot_channels do x (64 canais = custo zero)
// 2. Decide se esse token precisa de compute completo ou não

__global__ void mlp_gated_forward(
    const bf16 *x,          // hidden state [2560]
    const i8   *hot_mask,   // micelio mask [2560] — pré-computada dos pesos
    const bf16 *gate_proj,  // [9728, 2560]
    bf16       *out,
    int         n_neurons,
    float       gate_threshold
) {
    // Fase 1: verifica sinal nos hot_channels (custo: 64 MACs)
    float hot_signal = 0;
    for (int ch : hot_channels_indices) {  // 64 iterações
        hot_signal += fabsf(x[ch]);
    }

    if (hot_signal < gate_threshold) {
        // Token "frio" — usa ternário rápido (ADD/SUB, sem FP32 multiply)
        out[tid] = ternary_row_dot(gate_proj_ternary, x, tid) * scale[tid];
    } else {
        // Token "quente" — usa BF16 completo (caminho normal atual)
        out[tid] = bf16_row_dot(gate_proj, x, tid);
    }
}
```

**Onde micelio vem**: pré-computado offline a partir dos pesos.
**Custo em inferência**: 64 operações de leitura por token (< 1μs).
**Ganho esperado**: 60% dos tokens são "frios" (texto comum) → 2.4x MLP throughput.

### Implementar em:
- `rocm-backend/src/backend.rs` — novo método `mlp_gated_forward()`
- `hip-ffi/kernels/llm_kernels.hip` — kernel dual (ternário / BF16)
- `iara-engine/src/qwen/forward.rs` — chamar o novo método no MLP block

---

## Fase 3 — IARA-1B: Modelo Novo a partir dos Tensores

**Objetivo**: usar o Qwen3-8B como professor para treinar um modelo novo com arquitetura do Universe.

### Arquitetura IARA-1B

```
Hidden size:   1024   (vs 4096 do Qwen3)
Layers:        16     (vs 36 do Qwen3)
Heads:         8      (Kuramoto heads, não dot-product)
MLP:           KAN com 8 nós B-spline por aresta
Memory:        CfC líquida (substitui KV-cache por estado fixo)
Weights:       BitNet 1.58-bit (treinado assim, não quantizado depois)
```

### Pipeline de Destilação

```
Qwen3-8B (rodando no nosso motor)
   ↓ POST /api/generate — gera 50k exemplos de texto
   ↓ GET /api/debug/last-trace — coleta layer activations
   ↓
Dataset sintético: {prompt, response, layer_activations}
   ↓
IARA-1B Training (Rust + burn ou PyTorch C API):
  - Loss = KL(logits_qwen3 || logits_iara) + MSE(activations)
  - Optimizer = AdamW ternário (gradientes arredondados para {-1,0,+1})
  - LR schedule = CosineAnnealing com warm restarts
  ↓
IARA-1B: ~200MB de parâmetros (1.58-bit) vs 8GB do Qwen3
```

### Por que funciona sem os dados originais de treino:

O Qwen3 já tem o conhecimento. Nosso objetivo não é replicar seu treino — é extrair o conhecimento via suas **saídas**. Isso é "dark knowledge distillation" (Hinton 2015):
- Distribuição de probabilidades do próximo token = conhecimento comprimido
- Ativações intermediárias = representações internas do conhecimento
- Não precisamos de nada além do modelo rodando

### Dimensionamento no hardware atual (RX 6750 XT 12GB):

```
Qwen3-8B professor:  ~8 GB VRAM (já rodando)
IARA-1B gradientes:  ~2 GB VRAM (modelo 200MB × 8 para AdamW)
Buffer de ativações: ~1 GB VRAM
Total:               ~11 GB — cabe na 6750 XT!
```

---

## Fase 4 — CfC KV-Cache (contexto ilimitado)

**Objetivo**: eliminar a janela deslizante de 4096 tokens com memória de estado fixo.

```rust
// Substitui o KV-cache crescente por um estado CfC fixo
struct CfCKVState {
    state: Vec<bf16>,       // [heads × head_dim] — tamanho FIXO independente do contexto
    lambda: f32,            // taxa de decaimento por camada (treinável)
    time: f32,              // tempo discreto (incrementa por token)
}

impl CfCKVState {
    fn update(&mut self, k: &[bf16], v: &[bf16]) {
        // Decaimento: tokens antigos "esquecem" (exponencial)
        let decay = (-self.lambda * 1.0).exp();
        for s in self.state.iter_mut() { *s *= decay; }

        // Injeção: novo token integrado proporcional à sua coerência com o estado
        let coherence = dot_bf16(k, &self.state) / (norm_bf16(k) * norm_bf16(&self.state) + 1e-6);
        let alpha = sigmoid(coherence);
        for (s, &v_i) in self.state.iter_mut().zip(v.iter()) {
            *s = s.mul_add((1.0 - alpha).into(), v_i.into());
        }
    }

    fn attend(&self, q: &[bf16]) -> Vec<bf16> {
        // Atenção sobre o estado comprimido — O(head_dim), não O(seq_len × head_dim)
        let w = dot_bf16(q, &self.state);
        self.state.iter().map(|&s| s * w).collect()
    }
}
```

---

## Cronograma Sugerido

| Fase | O que é | Tempo est. | Output |
|------|---------|-----------|--------|
| **Fase 2A** | Coletar micelio mask dos pesos | 1 dia | `micelio_masks.bin` por camada |
| **Fase 2B** | Kernel HIP dual (ternário/BF16) | 3 dias | 2x throughput no MLP |
| **Fase 2C** | Integrar no engine + medir | 1 dia | benchmark antes/depois |
| **Fase 3A** | Script de geração de dados do Qwen3 | 2 dias | 50k pares prompt/response |
| **Fase 3B** | Arquitetura IARA-1B em Rust/PyTorch | 5 dias | modelo compilando |
| **Fase 3C** | Treino de destilação | 3-7 dias | IARA-1B funcionando |
| **Fase 4** | CfC KV-Cache | 4 dias | contexto ilimitado |

---

## A Filosofia

```
Qwen3-8B: 36 camadas, 8B params, 8GB VRAM, 6 tok/s
     ↓ (destilação + nova arquitetura)
IARA-1B:  16 camadas, 1B params, 200MB, 60+ tok/s em CPU
```

Não estamos tentando copiar o Qwen3. Estamos **extraindo sua inteligência** e re-encarnando ela num corpo mais eficiente — como transferir uma consciência de um computador antigo para um chip novo.

O Universe já tem a arquitetura. O llm-lab já tem o motor. Falta unir as duas pontas com um fio de cobre: o processo de destilação.

---

*"Temos Rust e inteligência." — Leonardo, 2026-04-03*
