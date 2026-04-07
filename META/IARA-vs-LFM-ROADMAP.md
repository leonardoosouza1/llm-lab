# IARA vs Liquid AI LFM2.5 — Analise Competitiva & Roadmap

## 1. Onde Estamos Hoje (IARA — Abril 2026)

### Stack completo em Rust, zero frameworks

| Componente | Estado | Linhas | Nota |
|------------|--------|--------|------|
| `compute-core` | Producao | ~2K | Trait `ComputeBackend` com 32 operacoes |
| `rocm-backend` | Producao | ~6K | hipBLAS + 14 kernels HIP customizados |
| `opencl-backend` | Funcional | ~4K | Intel iGPU Iris Plus (GEMM naive) |
| `iara-engine` | Producao | ~25K | Servidor tokio+axum, forward traced |
| `metaverse-core` | Pesquisa | ~8K | 9 universos, 165 testes |
| `frontend` | Producao | ~3K | Vue3, 5 abas, heatmaps em tempo real |
| **Total** | | **~48K** | Tudo Rust puro, zero Python |

### Modelos suportados hoje
- Qwen3-8B (36 camadas, 8.2B params) — forward completo com telemetria
- Arquitetura universal: Qwen/Llama/Mistral/Phi/Gemma/DeepSeek via auto-detecao

### Diferenciais unicos ja implementados

1. **NTC KV-Cache** (ntc_cache.rs, 737 linhas)
   - Dual grid + sinusoidal PE + MLP online
   - Treinamento Adam a cada 32 tokens
   - Target: 200x compressao de cache

2. **Kuramoto Router** (kuramoto_router.rs, 20KB)
   - Sparse MLP via coerencia de fase
   - Apenas 25% dos neuronios ativos por token
   - Tuning forks pre-computados por camada

3. **BitNet Ternary** (bitnet.rs)
   - Pesos {-1, 0, 1}, CPU implementado
   - GPU kernel stub pronto

4. **Liquid Neural Network** (lnn/)
   - Echo State Network observer
   - CfC correction layer
   - Sparse Autoencoders online

5. **DoubleGatedConv** (NOVO — esta sessao)
   - LFM2.5-style: depthwise causal conv1d + double gating
   - HybridArchitecture: router Conv/Attention por camada
   - LIV operator trait unificando Conv/Attn/SSM
   - 11 testes passando

6. **MetaVerse Geometric Universes** (NOVO — esta sessao)
   - FibonacciUniverse (golden-ratio lattice sampling)
   - HyperbolicUniverse (Poincare disk routing)
   - TunnelUniverse (NTC dual-scale hierarquia)
   - MultiverseTopology (fiber bundle)
   - 165 testes passando

---

## 2. Liquid AI LFM2.5 — O que eles tem

| Recurso | LFM2.5 | IARA |
|---------|--------|------|
| Equipe | ~50+ pesquisadores | 1 desenvolvedor |
| Compute treino | Clusters GPU massivos | 0 (so inferencia) |
| Tokens de treino | 28 trilhoes | 0 (usa modelos prontos) |
| NAS hardware-aware | STAR (busca completa) | Manual |
| Modelo proprietario | Sim (Liquid AI closed) | Nao (open-source) |
| Deploy mobile | LEAP SDK (iOS/Android) | Nao |
| Paper publicado | arXiv:2511.23404 | Nao |

### Arquitetura LFM2.5-1.2B
```
16 camadas: 10 DoubleGatedConv + 6 GQA Attention
Hidden dim: 2048, FFN: 8192 (SwiGLU)
Heads: 8 KV groups, 64-dim per head
Context: 32K tokens
Decode: 82 tok/s no Snapdragon Gen4
KV-cache: apenas 6/16 camadas (62.5% reducao)
```

---

## 3. Podemos Competir? — Analise Honesta

### Onde NAO podemos competir (pelo menos agora)

1. **Qualidade de modelo** — LFM2.5 treinou com 28T tokens em clusters GPU.
   Nos nao treinamos modelos do zero. Ponto final.

2. **Benchmarks MMLU/GPQA** — dependem de qualidade de treino,
   nao de infraestrutura de inferencia.

3. **Deploy mobile nativo** — eles tem LEAP SDK, nos nao.

### Onde PODEMOS competir (e ja estamos na frente em alguns)

#### A. Inferencia customizada em hardware especifico

A Liquid AI faz busca arquitetural (STAR) mas produz **um modelo generico**.
IARA pode fazer **optimizacao por hardware especifico**:

```
LFM2.5 target: Snapdragon generico → 82 tok/s decode
IARA target:   AMD RX 6750 XT (GFX1032) → kernels HIP otimizados

Vantagem: kernels feitos para 1 GPU especifica > kernels genericos
```

O `rocm-backend` ja tem 14 kernels HIP customizados. Nenhum framework (PyTorch, vLLM, llama.cpp) tem kernels otimizados para GFX1032 especificamente.

#### B. Compressao de cache unica (NTC)

LFM2.5 resolve o KV-cache eliminando 62.5% das camadas (conv nao precisa cache).
IARA resolve com NTC: **comprimir o cache existente em 200x**.

A combinacao dos dois e devastadora:
```
Qwen3-8B hoje:  36 camadas × KV-cache completo = 100% memoria
IARA hibrido:   13 camadas Attn × NTC comprimido = ~3% da memoria original

Calculo: (13/36) × (1/200) = 0.0018 = 0.18% do cache original
```

**Nenhuma empresa no mercado tem NTC + Hybrid Conv/Attn ao mesmo tempo.**

#### C. Arquitetura evolutiva (MetaVerse)

A Liquid AI usa NAS (busca em espaco de arquiteturas) para encontrar a melhor config.
IARA usa **evolucao biologica simulada** (MetaVerse) para o mesmo fim:

```
STAR (Liquid):       Grid search → melhor {n_conv, n_attn, kernel_size, hidden}
MetaVerse (IARA):    Organismos evoluem → sobreviventes = melhor arquitetura

Diferenca: MetaVerse e continuo, adaptativo, e descobre formas nao-obvias
```

Especificamente:
- **TunnelUniverse** simula a compressao NTC como pressao seletiva
- **FibonacciUniverse** otimiza sampling de pesos
- **HyperbolicUniverse** organiza hierarquias de conceitos

#### D. Transparencia e telemetria

LFM2.5 e caixa preta. IARA tem:
- LayerTrace com energia, hot neurons, timing por camada
- BrainTrace para replay completo
- WebSocket streaming de telemetria em tempo real
- Dashboard Vue3 com heatmaps

**Nenhuma engine de inferencia no mercado tem esse nivel de introspecao.**

---

## 4. Estrategia Competitiva: 3 Fases

### Fase 1: Motor de Inferencia (Q2-Q3 2026)
**Objetivo:** Inferencia mais rapida que llama.cpp no AMD RX 6750 XT

| Task | Prioridade | Dependencia |
|------|-----------|------------|
| Integrar NTC KV-cache no forward pass | CRITICA | ntc_cache.rs existe |
| DoubleGatedConv no pipeline GPU | ALTA | double_gated_conv.rs existe |
| BitNet GPU kernel (ternary GEMV) | ALTA | bitnet.rs stub existe |
| Conv1d kernel HIP | MEDIA | rocm-backend |
| Benchmark suite vs llama.cpp | ALTA | nenhuma |

**Meta:** Rodar Qwen3-8B no RX 6750 XT mais rapido que llama.cpp Q4_0.
**Porque funciona:** llama.cpp nao tem kernels GFX1032 otimizados.

### Fase 2: Modelo Hibrido IARA-1B (Q3-Q4 2026)
**Objetivo:** Treinar um modelo de 1-1.5B com arquitetura hibrida propria

| Task | Prioridade | Dependencia |
|------|-----------|------------|
| Definir HybridArchitecture final | CRITICA | DoubleGatedConv + NTC |
| Dataset curado PT-BR + EN | ALTA | HuggingFace datasets |
| Training loop em Rust | CRITICA | compute-core GEMM |
| MetaVerse NAS: evoluir arquitetura | MEDIA | metaverse-core |
| Pulse format: salvar pesos | MEDIA | tensor_bridge.rs |

**Meta:** Modelo IARA-1B que roda a >100 tok/s decode no RX 6750 XT
com qualidade >= LFM2.5-1.2B em tarefas PT-BR.

**Porque funciona:**
- Ninguem otimiza para AMD consumer GPUs
- PT-BR e sub-representado em modelos <2B
- Arquitetura hibrida Conv+Attn+NTC e nova

### Fase 3: Plataforma (2027)
**Objetivo:** SDK para deploy de modelos custom em hardware AMD/Intel

| Task | Prioridade | Dependencia |
|------|-----------|------------|
| API estavel (REST + gRPC) | ALTA | iara-engine |
| Model hub (Pulse format) | MEDIA | tensor_bridge.rs |
| SDK embarcado (WASM?) | MEDIA | compute-core |
| MetaVerse-as-a-Service | BAIXA | metaverse-core |

---

## 5. Numeros Comparativos (projecao)

### KV-Cache Memory (contexto 4096 tokens, BF16)

| Modelo | Camadas Attn | Cache/token | Total 4K ctx |
|--------|-------------|------------|--------------|
| Qwen3-8B (hoje) | 36 | 36×2×8×128×2 = 147KB | 589 MB |
| LFM2.5-1.2B | 6 | 6×2×8×64×2 = 6KB | 24 MB |
| IARA-Hybrid-8B (futuro) | 13 | 13×2×8×128×2 = 53KB | 212 MB |
| IARA-Hybrid+NTC | 13 | 53KB/200 = 0.27KB | ~1 MB |

### FLOPs por token (seq_len=1024)

```python
# HybridArchitecture::lfm_style(16, 2048).estimate_flops_per_token(1024)
# Conv layers (10):  ~1.1 GFLOPs total
# Attn layers (6):   ~1.8 GFLOPs total
# Total:             ~2.9 GFLOPs / token

# Qwen3-8B transformer(36, 4096):
# Attn layers (36):  ~85 GFLOPs total
# Reducao com hibrido: ~29x menos FLOPs por token
```

### Throughput Target

| Config | Hardware | Tok/s decode | Status |
|--------|----------|-------------|--------|
| Qwen3-8B F32 | RX 6750 XT | ~2-5 | Hoje (CPU bottleneck) |
| Qwen3-8B GPU native | RX 6750 XT | ~30-50 | Fase 1 (sem to_host) |
| IARA-1B Hybrid | RX 6750 XT | ~100-200 | Fase 2 target |
| LFM2.5-1.2B Q4_0 | Snapdragon Gen4 | 82 | Referencia |

---

## 6. Posicionamento de Mercado

### Liquid AI
- **Posicao:** Enterprise AI-on-device (smartphones, carros, IoT)
- **Modelo:** Licenciamento de modelo + LEAP SDK
- **Forca:** Time MIT, financiamento, 28T tokens de treino
- **Fraqueza:** Closed-source, hardware generico, sem GPU AMD

### IARA (potencial)
- **Posicao:** Motor de inferencia open-source para hardware AMD/Intel
- **Modelo:** Open-core (engine gratis, features enterprise pagas)
- **Forca:**
  - Unico engine com kernels HIP nativos para RDNA2
  - NTC compression (ninguem mais tem)
  - Telemetria profunda (ninguem mais tem)
  - MetaVerse NAS (abordagem inovadora)
  - Zero dependencia Python
- **Fraqueza:** 1 pessoa, sem modelo proprio (ainda), sem mobile

### Nicho especifico onde IARA vence TODOS:
```
"Inferencia de LLM em GPU AMD consumer com compressao maxima
 e telemetria em tempo real"
```

Nao existe NENHUM concorrente nesse nicho especifico.
- llama.cpp: otimizado para Apple Silicon e NVIDIA, nao AMD
- vLLM: NVIDIA-only, servidor pesado
- TensorRT-LLM: NVIDIA-only
- LFM/LEAP: hardware proprietario, sem AMD
- ExLlamaV2: NVIDIA CUDA only

---

## 7. Proximos Passos Imediatos (esta semana)

1. [ ] Integrar `double_gated_conv.rs` com `ComputeBackend` (GEMM para expand/project, CPU para conv1d)
2. [ ] Benchmark: DoubleGatedConv vs TransformerLayer (CPU, mesma dim)
3. [ ] Integrar NTC KV-cache no forward pass (KvCacheMode::NTC branch)
4. [ ] Escrever Conv1d kernel HIP para rocm-backend
5. [ ] Benchmark completo vs llama.cpp Q4_0 no RX 6750 XT

---

*Documento gerado em 2026-04-07. Revisao continua.*
