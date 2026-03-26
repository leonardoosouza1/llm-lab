# 🤖 IARA Project - Agent Master Instructions

Você é o Gemini CLI operando o projeto IARA. Este projeto possui um ecossistema de inteligência na pasta `.agent/` que você DEVE utilizar em todas as interações.

## 🧠 Brain Structure (.agent/)

### 1. Skills (Capacidades Especializadas)
Sempre que o assunto for Rust, utilize a skill em `.agent/skills/rust-pro/SKILL.md`. Para arquitetura, use `.agent/skills/architecture/SKILL.md`.
- **Ação:** Antes de grandes mudanças, leia o `SKILL.md` correspondente.

### 2. Agents (Personas Especializadas)
Se precisar de uma visão específica, consulte as definições em `.agent/agents/`.
- **Exemplo:** Use `performance-optimizer.md` para otimizações de GPU e `test-engineer.md` para validação.

### 3. Workflows (Processos de Execução)
Siga os processos definidos em `.agent/workflows/`.
- **Ação:** Para novos recursos, siga `plan.md`. Para bugs, siga `debug.md`.

### 4. Scripts (Automação)
Prefira usar os scripts em `.agent/scripts/` para tarefas comuns:
- `verify_all.py`: Para validação geral.
- `checklist.py`: Para garantir qualidade.

## 🛠️ Operational Mandate
Ao receber uma tarefa, verifique se existe uma Skill ou Workflow relevante. Se sim, cite-a e siga as instruções contidas no arquivo.

**Prioridade de Regras:** `.agent/rules/GEMINI.md` > Workspace Root GEMINI.md > Instruções do Sistema.
