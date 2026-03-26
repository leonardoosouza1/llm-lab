#!/usr/bin/env python3
"""
IARA Orchestrator MCP Server - Expõe tools de orquestração via FastMCP.

Tools disponíveis:
  - decompose_and_dispatch: Analisa prompt e despacha tasks por agente
  - dispatch_agent: Envia task direta para um agente
  - check_task_status: Verifica status de uma task
  - get_agent_result: Recupera resultado de task concluída
  - list_all_tasks: Lista todas as tasks no sistema
"""

import os
import json
import uuid
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("⚠️ FastMCP não disponível. Instale com: pip install mcp")
    raise

# === CONFIGURAÇÃO ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / ".agent" / "events"
INBOX_DIR = BASE_DIR / "inbox"
PROCESSING_DIR = BASE_DIR / "processing"
OUTBOX_DIR = BASE_DIR / "outbox"
FAILED_DIR = BASE_DIR / "failed"

GEMINI_BIN = shutil.which("gemini") or "/home/leonardo/.nvm/versions/node/v22.16.0/bin/gemini"

AVAILABLE_AGENTS = [
    "backend-specialist", "frontend-specialist", "database-architect",
    "security-auditor", "test-engineer", "devops-engineer",
    "mobile-developer", "debugger", "performance-optimizer",
    "seo-specialist", "documentation-writer", "explorer-agent",
]

mcp = FastMCP("IARA Orchestrator")


def _ensure_dirs(agent_role: str = None):
    for d in [INBOX_DIR, PROCESSING_DIR, OUTBOX_DIR, FAILED_DIR]:
        target = d / agent_role if agent_role else d
        target.mkdir(parents=True, exist_ok=True)


@mcp.tool()
def decompose_and_dispatch(user_prompt: str) -> str:
    """
    Analisa um prompt do usuário, decompõe em tasks atômicas e despacha 
    cada uma para o agente especialista correto.
    
    Args:
        user_prompt: O pedido completo do usuário (ex: "Integrar API de auth com signup e login")
    
    Returns:
        Relatório com a decomposição e IDs das tasks despachadas.
    """
    decompose_prompt = f"""Você é o IARA Orchestrator. Analise o pedido e decomponha em tasks atômicas.

Agentes disponíveis: {', '.join(AVAILABLE_AGENTS)}

Regras:
- Cada task = uma responsabilidade atômica
- Respeitar dependências (order)
- Agente deve ser exatamente um dos disponíveis acima

Pedido: {user_prompt}

RESPONDA APENAS JSON:
{{
  "analysis": "análise breve",
  "domains_detected": ["domínios"],
  "tasks": [
    {{
      "order": 1,
      "agent": "nome-do-agente",
      "title": "Título curto",
      "description": "O que o agente deve fazer em detalhes"
    }}
  ]
}}"""

    env = os.environ.copy()
    env["TERM"] = "dumb"
    env["CI"] = "true"
    
    try:
        result = subprocess.run(
            [GEMINI_BIN, "-p", decompose_prompt, "-m", "gemini-2.5-flash",
             "-o", "json", "--accept-raw-output-risk"],
            env=env, cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            return f"❌ Erro ao decompor: {(result.stderr or result.stdout)[:300]}"
        
        output = json.loads(result.stdout)
        raw = output.get("response", "")
        
        # Extrai JSON
        import re
        decomp = None
        for extractor in [
            lambda: json.loads(raw),
            lambda: json.loads(re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL).group(1)),
            lambda: json.loads(raw[raw.find('{'):raw.rfind('}') + 1]),
        ]:
            try:
                decomp = extractor()
                break
            except Exception:
                continue
        
        if not decomp:
            return f"❌ Não foi possível extrair JSON: {raw[:200]}"
        
        # Despacha tasks
        dispatched = []
        for task in decomp.get("tasks", []):
            task_id = str(uuid.uuid4())[:8]
            agent = task["agent"]
            
            _ensure_dirs(agent)
            task_data = {
                "task_id": task_id,
                "agent": agent,
                "task": task["description"],
                "context": {
                    "original_request": user_prompt,
                    "task_title": task["title"],
                    "order": task["order"],
                    "project_path": str(PROJECT_ROOT)
                },
                "created_at": datetime.now().isoformat()
            }
            
            filepath = INBOX_DIR / agent / f"task_{task_id}.json"
            with open(filepath, "w") as f:
                json.dump(task_data, f, indent=2, ensure_ascii=False)
            
            dispatched.append(f"  #{task['order']} @{agent}: {task['title']} (ID: {task_id})")
        
        report = f"""🤖 DECOMPOSIÇÃO COMPLETA

📊 Análise: {decomp.get('analysis', 'N/A')}
🎯 Domínios: {', '.join(decomp.get('domains_detected', []))}
📋 Tasks despachadas: {len(dispatched)}

{'chr(10)'.join(dispatched)}

💡 O Event Loop processará as tasks automaticamente.
   Verifique status com check_task_status(task_id)."""
        
        return report
        
    except Exception as e:
        return f"❌ Erro: {str(e)[:300]}"


@mcp.tool()
def dispatch_agent(agent_role: str, task_text: str, context: str = "") -> str:
    """
    Envia uma tarefa direta para um agente especialista.
    
    Args:
        agent_role: O papel do agente (ex: 'frontend-specialist')
        task_text: A instrução detalhada para o agente
        context: Contexto adicional opcional (JSON string)
    
    Returns:
        O ID da tarefa para acompanhamento.
    """
    if agent_role not in AVAILABLE_AGENTS:
        return f"❌ Agente '{agent_role}' não encontrado. Disponíveis: {', '.join(AVAILABLE_AGENTS)}"
    
    task_id = str(uuid.uuid4())[:8]
    _ensure_dirs(agent_role)
    
    ctx = {}
    if context:
        try:
            ctx = json.loads(context)
        except json.JSONDecodeError:
            ctx = {"raw_context": context}
    
    task_data = {
        "task_id": task_id,
        "agent": agent_role,
        "task": task_text,
        "context": ctx,
        "created_at": datetime.now().isoformat()
    }
    
    filepath = INBOX_DIR / agent_role / f"task_{task_id}.json"
    with open(filepath, "w") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)
    
    return f"✅ Task despachada! ID: {task_id} → @{agent_role}"


@mcp.tool()
def check_task_status(task_id: str) -> str:
    """
    Verifica o status de uma tarefa pelo ID.
    
    Args:
        task_id: O ID retornado pelo dispatch_agent ou decompose_and_dispatch
    """
    patterns = [
        (INBOX_DIR, "⏳ PENDENTE (aguardando Event Loop)"),
        (PROCESSING_DIR, "🔄 PROCESSANDO (Gemini CLI executando)"),
        (OUTBOX_DIR, "✅ CONCLUÍDO"),
        (FAILED_DIR, "❌ FALHOU"),
    ]
    
    for base_dir, status in patterns:
        matches = list(base_dir.glob(f"**/task_{task_id}.json"))
        if not matches:
            matches = list(base_dir.glob(f"**/*{task_id}*.json"))
        
        if matches:
            extra = ""
            if "CONCLUÍDO" in status:
                try:
                    with open(matches[0], "r") as f:
                        data = json.load(f)
                    resp = data.get("agent_response", {})
                    if isinstance(resp, dict):
                        extra = f"\n  Resumo: {resp.get('summary', resp.get('result', 'N/A'))[:200]}"
                except Exception:
                    pass
            
            if "FALHOU" in status:
                error_file = matches[0].with_suffix(".error")
                if error_file.exists():
                    extra = f"\n  Erro: {error_file.read_text()[:200]}"
            
            return f"Status: {status}{extra}"
    
    return "❓ NÃO ENCONTRADO (verifique o ID)"


@mcp.tool()
def get_agent_result(task_id: str) -> str:
    """
    Recupera o resultado completo de uma tarefa concluída.
    
    Args:
        task_id: O ID da tarefa
    """
    matches = list(OUTBOX_DIR.glob(f"**/task_{task_id}.json"))
    if not matches:
        matches = list(OUTBOX_DIR.glob(f"**/*{task_id}*.json"))
    
    if not matches:
        return "❌ Resultado não disponível. Task pode estar pendente ou ter falhado."
    
    with open(matches[0], "r") as f:
        data = json.load(f)
    
    resp = data.get("agent_response", data.get("response", "Sem resposta"))
    model = data.get("model_used", "unknown")
    meta = data.get("_meta", {})
    
    if isinstance(resp, dict):
        formatted = json.dumps(resp, indent=2, ensure_ascii=False)
    else:
        formatted = str(resp)
    
    return f"""--- RESULTADO @{meta.get('agent', 'unknown')} ---
Modelo: {model}
Task: {meta.get('original_task', 'N/A')[:100]}
Completado: {meta.get('completed_at', 'N/A')}

{formatted}"""


@mcp.tool()
def list_all_tasks() -> str:
    """Lista todas as tasks no sistema (inbox, processing, outbox, failed)."""
    
    report = []
    for label, base_dir, icon in [
        ("PENDENTES", INBOX_DIR, "⏳"),
        ("PROCESSANDO", PROCESSING_DIR, "🔄"),
        ("CONCLUÍDAS", OUTBOX_DIR, "✅"),
        ("FALHARAM", FAILED_DIR, "❌"),
    ]:
        tasks = list(base_dir.glob("**/*.json")) if base_dir.exists() else []
        # Filter out .error files
        tasks = [t for t in tasks if not t.name.endswith(".error")]
        if tasks:
            report.append(f"\n{icon} {label} ({len(tasks)}):")
            for t in tasks:
                agent = t.parent.name
                report.append(f"  - @{agent}: {t.name}")
    
    if not report:
        return "📭 Nenhuma task no sistema."
    
    return "📋 TASKS NO SISTEMA" + "\n".join(report)


if __name__ == "__main__":
    mcp.run()
