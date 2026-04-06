# SPEC: IARA Pulse Architecture — Inferência Neural Biológica

**Autores**: Leonardo + Claude (co-cientistas)
**Data**: 2026-04-06
**Status**: Implementação ativa

---

## Visão

Transformar a IARA Engine de um motor de inferência sequencial (todas as layers, todos os neurônios, todos os tokens) em um **sistema neural biológico** com:

- **Pulso semântico**: cada token percorre apenas as layers que precisa
- **3 velocidades**: rápida (ternário), média (P+T), lenta (BF16 completo)
- **Consciência contínua**: liquid network + homeostasis + actuators observam e modulam
- **Cirurgia de tensores**: instrumentar, podar, fundir, reexportar

## Gap Analysis

### O que EXISTE e FUNCIONA
| Módulo | Arquivo | Status |
|--------|---------|--------|
| Forward pass completo | model/mod.rs:940-1452 | ✅ Funcional |
| ESN Observer (256 neurons) | lnn/esn.rs | ✅ Funcional + treina online |
| CfC Correction Layer | lnn/cfc.rs | ✅ Forward implementado |
| Gravity System | lnn/gravity.rs | ✅ Compute implementado |
| Liquid Network (256 CfC) | experimental/liquid_network.rs | ✅ 95% completo |
| Homeostasis FSM | experimental/homeostasis.rs | ✅ 100% funcional |
| Actuator Bus (3 bandas) | experimental/actuator.rs | ✅ 100% funcional |
| CfC Simulation | experimental/simulation.rs | ✅ 100% funcional |
| Pipeline c/ stimulus | experimental/pipeline.rs | ✅ 90% completo |
| .iara v2 format | storage/iara_format.rs | ✅ Reader + Writer |
| P+T+T quantização | storage/bitnet.rs + RESEARCH.md | ✅ Validado empiricamente |

### O que FALTA (gaps)
| Gap | Impacto | Onde implementar |
|-----|---------|-----------------|
| **Loader .iara v2 → inference** | Bloqueante — sem isso nada roda | model/mod.rs (novo path de loading) |
| **ESN layer_budget output** | Core do Pulse Router | lnn/esn.rs (novo output) |
| **Early exit por coerência** | Skip de layers | model/mod.rs (no layer loop) |
| **Orchestrator** | Cola tudo junto | experimental/orchestrator.rs (novo) |
| **Stimulus wiring** | LiquidNet recebe ativações | model/mod.rs → channel → liquid_network |
| **Adaptive quant por depth** | Fast/medium/slow layers | model/mod.rs + storage/iara_format.rs |
| **Tensor importance** | Base para surgery | inference/importance.rs (novo) |
| **Testes end-to-end reais** | Validação | tests/ (novo) |

## Arquitetura: 3 Camadas de Velocidade

```
Token → ESN analisa → layer_budget = {fast, medium, slow}
                ↓
    ┌── FAST (layers 0-N/4): ternário, 25% neurônios ──┐
    │   Custo: ~0.1ms/layer. Token trivial sai aqui.    │
    └────────────────────────────────────────────────────┘
                ↓ (se coerência < θ)
    ┌── MEDIUM (layers N/4-3N/4): P+T, CfC corrige ────┐
    │   Custo: ~0.5ms/layer. Maioria dos tokens.        │
    └────────────────────────────────────────────────────┘
                ↓ (se coerência ainda < θ)
    ┌── SLOW (layers 3N/4-N): BF16, todos neurônios ───┐
    │   Custo: ~1ms/layer. Tokens complexos.            │
    └────────────────────────────────────────────────────┘
```

## Implementação: 4 Streams Paralelos

### Stream 1: .iara v2 Loader + Validação
- Criar `load_from_iara()` em model/mod.rs
- Mapear tensores .iara → buffers GPU (mmap zero-copy)
- Rodar Qwen2.5-1.5B .iara, gerar texto, medir tok/s

### Stream 2: Pulse Router (ESN + Early Exit)
- Adicionar `layer_budget()` ao ESN (output: quantas layers usar)
- Implementar `phase_coherence()` no layer loop
- Skip de layers quando coerência > threshold

### Stream 3: Orchestrator + Wiring
- Criar ExperimentalOrchestrator (liquid_net + homeostasis + actuators)
- Wiring de stimulus: forward() → channel → liquid_network
- Ticker de 1 Hz (homeostasis) e 100 Hz (physics)

### Stream 4: Adaptive Quant + Tensor Importance
- Quant por profundidade no .iara v2 converter
- Instrumentação de importância por tensor durante inferência
- Exportar relatório de tensores vivos/mortos/dormindo
