# 🤖 IARA v2 - Agent Guide

Este diretório contém o **IARA v2 Orchestration Engine**, um sistema de execução paralela para agentes de IA que utiliza o motor SQLite para estado compartilhado.

## 📁 Estrutura do Motor
- `.agent/scripts/iara.py`: Ponto central de execução.
- `.agent/scripts/architect.py`: Scanner de projeto e decompositor de GAP.
- `.agent/scripts/runner.py`: Motor paralelo de execução.
- `.agent/scripts/dashboard.py`: Monitoramento visual.
- `.agent/events/orchestrate.db`: Registro central de estado.

## 🚀 Instalação e Verificação

Para preparar um novo projeto para o IARA v2:
1.  Copie a pasta `.agent/` para a raiz do novo projeto.
2.  Rode o instalador:
    `python3 .agent/scripts/install_iara.py`
    *Este script injeta automaticamente as regras mandatórias no GEMINI.md da raiz.*

## 🧪 Verificação Manual
Para garantir que o ambiente está pronto em um novo projeto, rode:
`python3 .agent/scripts/test_engine.py`

## 🧠 Princípios Chave
- **Hole Check**: Nunca inicie uma execução com dúvidas técnicas.
- **Pato**: Seus agentes são patos. Dê a eles caminhos exatos, versões e dados reais de memória compartilhada.
- **Shared Memory**: Use o SQLite `shared_state` para passar o bastão entre tarefas.
