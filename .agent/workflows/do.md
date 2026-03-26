---
description: Orquestração inteligente de tarefas baseada em domínio e complexidade.
---

# /do - Execução Inteligente com Decomposição por Agentes

Este workflow recebe um prompt livre do usuário, decompõe em tasks atômicas e despacha cada uma para o agente especialista correto via Gemini CLI.

## Arquitetura

```
/do "prompt" → Decomposer (Gemini CLI) → Tasks na Inbox → Event Loop → Gemini CLI por agente → Outbox
```

## Processo de Execução (v2 Unified)

O comando `/do` agora utiliza o motor IARA v2, que é centralizado no script `iara.py`.

### 1. EXECUÇÃO UNIFICADA

# 🚀 Workflow: /do (O Protocolo do Arquiteto)

> [!CAUTION]
> **INSTRUÇÃO MANDATÓRIA PARA IA:**
> 1. **MAPEIE:** Antes de tudo, ache os arquivos e linhas. Use `grep` e `list_dir`.
> 2. **QUESTIONE:** Não assuma frameworks ou URLs. Pergunte os furos (GAPs).
> 3. **ESPECIFIQUE:** Crie um prompt técnico rico.
> 4. **DELEGUE:** `python3 .agent/scripts/iara.py "<prompt técnico>"`
Basta rodar o comando mestre. Ele cuidará da Descoberta, Decomposição, Execução Paralela e Dashboard de uma vez.

```bash
// turbo
python3 .agent/scripts/iara.py "$ARGUMENTS"
```

### 2. O QUE ACONTECE POR BAIXO

1.  **Arquitetura (architect.py)**: Faz o scanner do projeto e identifica "furos" (Gaps). Se houver dúvidas, ele parará e perguntará.
2.  **Motor (runner.py)**: Se a spec estiver pronta, inicia o paralelismo em segundo plano.
3.  **Dashboard (dashboard.py)**: Abre o Mission Control automaticamente para você acompanhar o progresso.

### 3. MONITORAMENTO

Acompanhe o progresso:

```bash
# Ver log em tempo real
tail -f .agent/events/mcp_orchestrator.log

# Listar tasks
ls -la .agent/events/inbox/*/
ls -la .agent/events/outbox/*/

# Ver resultado de uma task
cat .agent/events/outbox/{agent}/{task_file}.json | python -m json.tool
```

### 4. COLETA DE RESULTADOS

Após conclusão, os resultados ficam em:
- `✅ .agent/events/outbox/{agent}/task_{id}.json`
- `❌ .agent/events/failed/{agent}/task_{id}.json`

## Exemplo de Uso

```
/do Preciso integrar API do swagger na pasta docs/ e criar fluxo completo de signup, login e recuperar senha
```

**Resultado da decomposição:**

| #  | Agente              | Tarefa                                |
|----|---------------------|---------------------------------------|
| 1  | backend-specialist  | Criar serviços de integração com API  |
| 2  | security-auditor    | Definir estratégia de auth (JWT/OAuth)|
| 3  | backend-specialist  | Implementar endpoints de auth         |
| 4  | frontend-specialist | Criar telas de signup/login/recovery  |
| 5  | test-engineer       | Testes dos fluxos de autenticação     |

## Pré-requisitos

1. **Gemini CLI** instalado e autenticado (`gemini --version`)
2. **Event Loop rodando** em terminal separado
3. **Python 3.10+** com módulos padrão

## Uso via MCP (alternativa)

Se o MCP Server estiver configurado:
- `decompose_and_dispatch("prompt")` → decompõe e despacha
- `dispatch_agent("agent", "task")` → despacho direto
- `check_task_status("id")` → verifica status
- `list_all_tasks()` → lista todas as tasks

## Scripts

| Script | Função |
|--------|--------|
| `task_decomposer.py` | Analisa prompt → cria tasks na inbox |
| `event_loop.py` | Watcher: inbox → Gemini CLI → outbox |
| `orchestrator_mcp.py` | MCP Server com tools de orquestração |