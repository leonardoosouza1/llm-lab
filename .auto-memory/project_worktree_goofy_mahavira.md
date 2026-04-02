---
name: IARA — Worktree Acidental goofy-mahavira
description: Subprojeto criado por Claude Code anterior contém módulos LNN, ACP e storage quantizado não integrados ao main
type: project
---

Em `.claude/worktrees/goofy-mahavira/` existe um branch divergente com código Rust substancial NÃO presente no projeto principal:

## Módulos Novos (prontos para integrar)

### 1. `src/lnn/` — Liquid Neural Networks
- `esn.rs` — Echo State Network: observador de entropia, modula temperature de sampling
- `cfc.rs` — Closed-form Continuous Liquid: corrige erros de quantização em hidden states
- `gravity.rs` — Semantic Gravity: identifica tokens de alta "massa semântica", roteia para CfC

### 2. `src/storage/` — Quantização Avançada
- `quantized.rs` — BF16 → INT4 group-wise (group_size=128), nibble packing
- `decoder.rs` — zero-copy memmap de safetensors
- Formato `.iara` para cache pré-quantizado (skip quantização CPU no segundo load)

### 3. `src/acp/` — Agent Communication Protocol
- REST API em `/acp/v1/*` para coordenação multi-agente
- Endpoints: /ping, /agents, /runs, /runs/:id/events, etc.
- Permite que o sistema .agent/ orquestre runs no engine

### 4. `src/experimental/` — Research Modules
- gpu_orchestrator, homeostasis, physics_gpu, simulation, liquid_brain, nlm_orchestrator

## Recomendação
Integrar lnn/ + storage/ + acp/ ao projeto principal em branches separadas. Experimental/ deixar como está por ora.

**Why:** Claude Code anterior foi instruído a trabalhar nessas features mas usou worktree em vez de commitar ao main.

**How to apply:** Antes de reimplementar qualquer feature de LNN, quantização ou ACP — verificar se já existe código funcional no worktree.
