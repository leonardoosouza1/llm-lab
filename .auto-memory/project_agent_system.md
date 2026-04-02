---
name: IARA — Sistema de Agentes .agent/
description: Orquestrador multi-agente com Gemini CLI, SQLite state, event loop Node.js, 20 personas especializadas
type: project
---

O diretório `.agent/` contém um orquestrador autônomo baseado em Gemini CLI (NÃO Claude Code):

## Componentes
- `scripts/iara.py` — Ponto de entrada + decompositor
- `scripts/architect.py` — Scanner de projeto + GAP detector
- `scripts/runner.py` — Executor paralelo de agentes
- `scripts/dashboard.py` — Monitoramento visual
- `events/orchestrate.db` — SQLite: estado compartilhado entre runs
- `events/inbox/{agent}/` — Tarefas a executar (JSON)
- `events/outbox/{agent}/` — Resultados completados

## 20 Agentes Definidos
backend-specialist, security-auditor, frontend-specialist, test-engineer, rust-pro, orchestrator, project-planner, devops-engineer, performance-optimizer, code-archaeologist, etc.

## GEMINI.md — Protocolo Mandatório (IMPORTANTE)
GEMINI.md diz que AIs são PROIBIDAS de escrever código diretamente — devem delegar via `python3 .agent/scripts/iara.py`. Isso é a intenção do usuário com o sistema de agentes. **Cowork mode é exceção pois o usuário explicitamente me pediu para assumir como copiloto direto.**

## Última Atividade (2026-03-26)
Tentou integrar fluxo de auth (signup/login/recuperação). Falhou porque `/docs/swagger.json` não existe. Agentes geraram código sem especificação real da API.

**Why:** Leonardo construiu esse sistema para evitar que AIs façam mudanças sem planejamento rigoroso e GAP analysis primeiro.

**How to apply:** Quando o Leonardo mencionar o sistema de agentes ou .agent/, saber que é o orquestrador Gemini, não parte do iara-engine Rust. São sistemas separados.
