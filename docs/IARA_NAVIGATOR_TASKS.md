# IARA Ternary Navigator — Implementation Plan

## Objetivo
Construir um sistema de inferência onde o modelo **navega** ao invés de calcular.
Combina: ternário (mapa), coordenadas angulares (LSH), merge de layers, busca heurística (A*).

## Arquitetura Alvo
```
INPUT → Ternary Scan (375MB) → Graph Router (A*) → Super Layers (4x) → OUTPUT
         0.1ms                   0.05ms               1-3ms
         (vs 6ms baseline full 28 layers)
```

## Fases e Dependências

```
Fase 1: BI Analysis          (base — sem dependências)
  └─> Fase 3: Super Layers   (depende dos BI scores)
        └─> Fase 4: Graph Router UCS  (depende dos super layers)
              └─> Fase 5: A* Router   (depende do grafo)
                    └─> Fase 6: E2E Benchmark (depende de tudo)

Fase 2: LSH Lookup           (paralela — só depende do ternário existente)
  └─> Fase 5: A* Router      (LSH alimenta h_similarity da heurística)
```

---

## Fase 1 — Block Influence Analysis
**Arquivo**: `iara-train/examples/bi_analysis.rs`
**Objetivo**: Identificar quais layers são redundantes
**Método**: BI[i] = 1 − cos_sim(layer_i_input, layer_i_output)
**Mudança no modelo**: push embed inicial em `all_hidden_states`

**Output esperado**:
```
Layer  BI Score  Recommendation
  0    0.341     KEEP   (high influence, embedding transition)
  1    0.287     KEEP
  ...
 14    0.031     MERGE  (low influence → candidate)
  ...
 27    0.198     KEEP   (high influence, final projection)
```

**Critério de sucesso**: Identificar 2-3 grupos de layers com BI < 0.05 (candidatos a merge).

---

## Fase 2 — LSH Coordinate Lookup
**Arquivo**: `iara-train/examples/lsh_lookup.rs`
**Objetivo**: Provar que coordenadas angulares + LSH encontram neurônios ativos sem matmul completo
**Método**: SimHash (random projection) sobre pesos ternários

**Output esperado**:
```
Layer gate_proj L14 (1536×8960 neurons)
LSH Tables: 8,  Buckets: 256
Input → hash → top-k bucket
Recall@32:   91.3%  (LSH encontrou 91% dos top-32 neurônios ativos)
Recall@64:   96.7%
Lookup time: 0.08ms  vs  Full matmul: 1.2ms  → 15x speedup
```

**Critério de sucesso**: Recall@64 > 90% com speedup > 5x.

---

## Fase 3 — Super Layer Construction
**Arquivo**: `iara-train/examples/super_layer.rs`
**Dependência**: scores BI da Fase 1
**Objetivo**: Fundir 28 layers → 4 super layers

**Grupos propostos** (ajustar após BI real):
- SL0: layers 0-6   (compreensão/embedding)
- SL1: layers 7-13  (raciocínio)
- SL2: layers 14-20 (conhecimento)
- SL3: layers 21-27 (geração)

**Método**:
1. Average weights → base super layer
2. SVD(original_i - base) → low-rank delta (rank=64)
3. Reconstrução: super_layer + delta_i ≈ original_layer_i

**Output esperado**:
```
SL0 (layers 0-6):  base_size=1.2GB → delta rank-64=48MB  CosSim=0.982
SL1 (layers 7-13): base_size=1.2GB → delta rank-64=48MB  CosSim=0.979
...
Total: 4.8GB base (shared) + 4×48MB deltas = 5.0GB vs 28×0.43GB=12GB original
```

**Critério de sucesso**: CosSim > 0.95 entre saída do grupo original vs super layer.

---

## Fase 4 — Graph Router (UCS)
**Arquivo**: `iara-train/examples/graph_router.rs`
**Dependência**: Fase 3 (super layers)
**Objetivo**: Rotear tokens pelo caminho de mínimo custo

**Grafo**:
- Nós: {SL0, SL1, SL2, SL3, OUT}
- Arestas: custo = FLOPs estimados (baseado em esparsidade ternária)
- Roteamento: UCS (Dijkstra sem heurística)

**Output esperado**:
```
Prompt: "def fibonacci(n):"
  Ternary scan → sparsity per SL: [0.34, 0.28, 0.61, 0.41]
  UCS path: SL0 → SL2 → SL3  (skip SL1, economy=25% FLOPs)
  Quality: CosSim vs baseline = 0.991

Prompt: "What is the capital of France?"
  Ternary scan → sparsity per SL: [0.31, 0.52, 0.44, 0.38]
  UCS path: SL0 → SL1 → SL3  (skip SL2, economy=25%)
  Quality: CosSim vs baseline = 0.988
```

**Critério de sucesso**: >20% FLOPs economy com CosSim > 0.98 vs baseline.

---

## Fase 5 — A* Router com Heurísticas
**Arquivo**: `iara-train/examples/astar_router.rs`
**Dependências**: Fase 4 + Fase 2
**Objetivo**: Heurísticas reduzem nodes explorados vs UCS

**Heurísticas**:
- `h_sparsity(n)`: ternary zero ratio das layers seguintes
- `h_confidence(n)`: 1 - max_softmax(partial_logits)
- `h_combined(n)`: w1*h_sparsity + w2*h_confidence

**Output esperado**:
```
UCS:  avg 3.2 nodes explored per token, 100% optimal paths
A*:   avg 1.8 nodes explored per token, 98.7% same paths (2.1x menos trabalho)

h_sparsity alone:    admissible=YES, accuracy=96.2%
h_confidence alone:  admissible=YES, accuracy=97.1%
h_combined:          admissible=YES, accuracy=98.7%
```

**Critério de sucesso**: A* explora <60% dos nós que UCS explora, mantendo >97% das rotas ótimas.

---

## Fase 6 — E2E Benchmark
**Arquivo**: `iara-train/examples/e2e_benchmark.rs`
**Dependências**: Todas as fases anteriores
**Objetivo**: Comparar IARA Navigator vs baseline 28-layers

**Prompts de teste** (5 categorias):
1. Código: "Write a bubble sort in Rust"
2. Matemática: "What is the derivative of x^3 + 2x?"
3. Linguagem: "Translate 'hello world' to French"
4. Lógica: "If A>B and B>C, what can we say about A and C?"
5. Conhecimento: "What is the capital of Brazil?"

**Métricas**:
- tokens/segundo (latência)
- Perplexidade (qualidade)
- Memória VRAM usada
- FLOPs estimados por token
- Cache hit rate (após warmup)

**Output esperado**:
```
┌─────────────────────────────────────────────────────────────────┐
│ IARA Navigator vs Baseline — E2E Results                        │
├─────────────────┬──────────────┬──────────────┬─────────────────┤
│ Metric          │ Baseline     │ IARA         │ Improvement     │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Tokens/sec      │ ~160         │ ~400-600     │ 2.5-3.7×        │
│ VRAM (weights)  │ 3.0 GB       │ 0.9 GB       │ 3.3× less       │
│ FLOPs/token     │ 100%         │ 25-40%       │ 2.5-4×          │
│ Perplexity      │ baseline     │ +2-5%        │ ±acceptable     │
│ Cache hit       │ N/A          │ ~30%         │ new capability  │
└─────────────────┴──────────────┴──────────────┴─────────────────┘
```

**Critério de sucesso**: >2x speedup com perplexidade degradando <10%.

---

## Estrutura de Arquivos

```
iara-train/
├── examples/
│   ├── ternary_test.rs      ✅ DONE — prova ternário funciona
│   ├── bi_analysis.rs       📋 Fase 1 — TODO
│   ├── lsh_lookup.rs        📋 Fase 2 — TODO
│   ├── super_layer.rs       📋 Fase 3 — TODO (depende Fase 1)
│   ├── graph_router.rs      📋 Fase 4 — TODO (depende Fase 3)
│   ├── astar_router.rs      📋 Fase 5 — TODO (depende Fases 2,4)
│   └── e2e_benchmark.rs     📋 Fase 6 — TODO (depende todas)
└── src/
    ├── model/
    │   ├── qwen2.rs         ✅ + patch: push initial embed in hidden_states
    │   └── ...
    └── navigator/           📋 Módulo novo — structs reutilizáveis
        ├── mod.rs
        ├── bi.rs            (Block Influence calculator)
        ├── lsh.rs           (SimHash tables)
        ├── super_layer.rs   (SuperLayer struct + merge logic)
        └── router.rs        (Graph + UCS + A*)
```

---

## Ordem de Execução

```bash
# Fase 1 (roda em ~2 min, requer modelo carregado)
cargo run -p iara-train --example bi_analysis --features wgpu

# Fase 2 (roda em ~1 min, usa ternary_test infra existente)
cargo run -p iara-train --example lsh_lookup --features wgpu

# Fase 3 (roda em ~5 min, requer Fase 1 completa)
cargo run -p iara-train --example super_layer --features wgpu

# Fase 4 (roda em ~3 min, requer Fase 3)
cargo run -p iara-train --example graph_router --features wgpu

# Fase 5 (roda em ~3 min, requer Fases 2+4)
cargo run -p iara-train --example astar_router --features wgpu

# Fase 6 (roda em ~10 min, requer todas)
cargo run -p iara-train --example e2e_benchmark --features wgpu
```
