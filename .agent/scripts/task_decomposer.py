#!/usr/bin/env python3
"""
IARA Task Decomposer - Analisa um prompt e decompõe em tasks por agente.

Usa o Gemini CLI para analisar o prompt do usuário e retorna uma lista
de tasks atômicas, cada uma com o agente responsável.

Uso:
  python task_decomposer.py "Seu prompt aqui"
  python task_decomposer.py --file prompt.txt
"""

import os
import sys
import json
import uuid
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVENTS_DIR = PROJECT_ROOT / ".agent" / "events"
INBOX_DIR = EVENTS_DIR / "inbox"
AGENTS_DIR = PROJECT_ROOT / ".agent" / "agents"

GEMINI_BIN = shutil.which("gemini") or "/home/leonardo/.nvm/versions/node/v22.16.0/bin/gemini"

# Agentes disponíveis no ecossistema
AVAILABLE_AGENTS = [
    "backend-specialist",
    "frontend-specialist", 
    "database-architect",
    "security-auditor",
    "test-engineer",
    "devops-engineer",
    "mobile-developer",
    "debugger",
    "performance-optimizer",
    "seo-specialist",
    "documentation-writer",
    "explorer-agent",
]

DECOMPOSE_PROMPT = """Você é o IARA Orchestrator. Sua função é analisar um pedido do usuário e decompor em tarefas atômicas, atribuindo cada uma ao agente especialista correto.

## Agentes Disponíveis:
{agents}

## Regras:
1. Cada task deve ser ATÔMICA (uma responsabilidade clara)
2. Tasks devem ter ORDEM DE EXECUÇÃO (dependências respeitadas)  
3. Agrupar por agente quando possível (um agente pode ter múltiplas tasks)
4. Se o pedido menciona "API", "swagger", "endpoints" → backend-specialist
5. Se menciona "tela", "UI", "componente", "página" → frontend-specialist
6. Se menciona "login", "signup", "auth", "senha" → security-auditor + backend-specialist + frontend-specialist
7. Se menciona "banco", "tabela", "migration" → database-architect
8. Se menciona "teste", "test" → test-engineer
9. Se menciona "deploy", "docker", "CI/CD" → devops-engineer

## Pedido do Usuário:
{user_prompt}

## RESPONDA APENAS com JSON válido:
{{
  "analysis": "Análise breve do pedido (1-2 frases)",
  "domains_detected": ["lista de domínios detectados"],
  "tasks": [
    {{
      "order": 1,
      "agent": "nome-do-agente",
      "title": "Título curto da task",
      "description": "Descrição detalhada do que o agente deve fazer",
      "depends_on": [],
      "files_hint": ["arquivos que provavelmente serão afetados"]
    }}
  ],
  "execution_plan": "Resumo do plano de execução em 1-2 frases"
}}"""


def decompose(user_prompt: str) -> dict:
    """Envia prompt ao Gemini CLI para decomposição em tasks."""
    
    agents_list = "\n".join(f"- {a}" for a in AVAILABLE_AGENTS)
    full_prompt = DECOMPOSE_PROMPT.format(
        agents=agents_list,
        user_prompt=user_prompt
    )
    
    env = os.environ.copy()
    env["TERM"] = "dumb"
    env["CI"] = "true"
    
    print("🧠 Analisando pedido com Gemini CLI...")
    
    result = subprocess.run(
        [
            GEMINI_BIN, "-p", full_prompt,
            "-m", "gemini-2.5-flash",
            "-o", "json",
            "--accept-raw-output-risk"
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode != 0:
        err = result.stderr or result.stdout
        raise Exception(f"Gemini CLI falhou: {err[:300]}")
    
    output = json.loads(result.stdout)
    raw_response = output.get("response", "")
    
    # Extrai JSON da resposta
    import re
    
    # Tenta direto
    try:
        return json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # ```json ... ```
    match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    
    # { ... }
    start = raw_response.find('{')
    end = raw_response.rfind('}')
    if start != -1 and end > start:
        return json.loads(raw_response[start:end + 1])
    
    raise ValueError(f"Não foi possível extrair JSON da resposta: {raw_response[:200]}")


def dispatch_tasks(decomposition: dict, original_prompt: str) -> list:
    """Cria arquivos de task na inbox para cada agente."""
    
    tasks = decomposition.get("tasks", [])
    dispatched = []
    
    for task in tasks:
        agent = task["agent"]
        task_id = str(uuid.uuid4())[:8]
        
        # Cria a inbox do agente
        agent_inbox = INBOX_DIR / agent
        agent_inbox.mkdir(parents=True, exist_ok=True)
        
        task_data = {
            "task_id": task_id,
            "agent": agent,
            "task": task["description"],
            "context": {
                "original_request": original_prompt,
                "task_title": task["title"],
                "order": task["order"],
                "depends_on": task.get("depends_on", []),
                "files_hint": task.get("files_hint", []),
                "project_path": str(PROJECT_ROOT),
                "analysis": decomposition.get("analysis", "")
            },
            "created_at": datetime.now().isoformat()
        }
        
        filename = f"task_{task_id}.json"
        filepath = agent_inbox / filename
        
        with open(filepath, "w") as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        
        dispatched.append({
            "task_id": task_id,
            "agent": agent,
            "title": task["title"],
            "order": task["order"],
            "file": str(filepath)
        })
        
        print(f"  📤 Task #{task['order']}: @{agent} → {task['title']} (ID: {task_id})")
    
    return dispatched


def main():
    if len(sys.argv) < 2:
        print("Uso: python task_decomposer.py 'seu prompt aqui'")
        print("     python task_decomposer.py --file prompt.txt")
        sys.exit(1)
    
    if sys.argv[1] == "--file":
        with open(sys.argv[2], "r") as f:
            user_prompt = f.read().strip()
    else:
        user_prompt = " ".join(sys.argv[1:])
    
    print("=" * 60)
    print("🤖 IARA Task Decomposer")
    print(f"📝 Prompt: {user_prompt[:100]}...")
    print("=" * 60)
    
    # 1. Decompõe
    try:
        decomposition = decompose(user_prompt)
    except Exception as e:
        print(f"❌ Erro na decomposição: {e}")
        sys.exit(1)
    
    # 2. Mostra análise
    print(f"\n📊 Análise: {decomposition.get('analysis', 'N/A')}")
    print(f"🎯 Domínios: {', '.join(decomposition.get('domains_detected', []))}")
    print(f"📋 Tasks: {len(decomposition.get('tasks', []))}")
    print(f"📝 Plano: {decomposition.get('execution_plan', 'N/A')}")
    print()
    
    # 3. Mostra tasks antes de despachar
    tasks = decomposition.get("tasks", [])
    print("┌─────────────────────────────────────────────────────────┐")
    print("│  # │ Agente                │ Tarefa                     │")
    print("├─────────────────────────────────────────────────────────┤")
    for t in tasks:
        agent_short = t["agent"][:20].ljust(20)
        title_short = t["title"][:25].ljust(25)
        print(f"│ {t['order']:>2} │ {agent_short} │ {title_short} │")
    print("└─────────────────────────────────────────────────────────┘")
    
    # 4. Despacha
    print("\n📤 Despachando tasks para inbox...")
    dispatched = dispatch_tasks(decomposition, user_prompt)
    
    # 5. Salva o plano
    plan_file = EVENTS_DIR / "last_decomposition.json"
    with open(plan_file, "w") as f:
        json.dump({
            "prompt": user_prompt,
            "decomposition": decomposition,
            "dispatched": dispatched,
            "created_at": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ {len(dispatched)} tasks despachadas!")
    print(f"📄 Plano salvo: {plan_file}")
    print(f"\n💡 Certifique-se que o Event Loop está rodando:")
    print(f"   python {PROJECT_ROOT / '.agent/scripts/event_loop.py'}")


if __name__ == "__main__":
    main()
