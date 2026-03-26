#!/usr/bin/env python3
import os
import sqlite3
import subprocess
import json
from pathlib import Path
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("❌ FastMCP não disponível. Instale com: pip install mcp")
    raise

# === CONFIGURAÇÃO ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / ".agent" / "events" / "orchestrate.db"
SCRIPTS_DIR = PROJECT_ROOT / ".agent" / "scripts"

mcp = FastMCP("IARA v2 Engine")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@mcp.tool()
def run_iara(prompt: str) -> str:
    """
    Inicia o fluxo completo do IARA v2: Descoberta, Planejamento e Execução.
    
    Args:
        prompt: O pedido do usuário.
    """
    iara_path = PROJECT_ROOT / "iara.py"
    try:
        # Run unified iara command
        # Note: In MCP context, we might not want to launch the interactive dashboard 
        # unless it's supported by the client. We'll run the scripts directly.
        
        result = subprocess.run(
            ["python3", str(iara_path), prompt],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        return result.stdout + (result.stderr if result.stderr else "")
    except Exception as e:
        return f"❌ Erro ao executar IARA: {e}"

@mcp.tool()
def check_job_status(run_id: Optional[int] = None) -> str:
    """
    Verifica o status do último Job ou de um ID específico.
    """
    try:
        conn = get_db()
        if run_id:
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        else:
            run = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        
        if not run:
            return "📭 Nenhum Job encontrado."
        
        tasks = conn.execute("SELECT * FROM tasks WHERE run_id = ?", (run['id'],)).fetchall()
        
        report = [f"📊 JOB #{run['id']}: {run['status']}"]
        report.append(f"Prompt: {run['prompt'][:100]}...")
        report.append("-" * 20)
        
        for t in tasks:
            report.append(f"[{t['status']:<10}] @{t['agent_role']:<20} | {t['title']}")
            
        return "\n".join(report)
    except Exception as e:
        return f"❌ Erro ao consultar banco: {e}"

@mcp.tool()
def get_shared_memory(run_id: int) -> str:
    """
    Recupera a memória compartilhada (Shared State) de um Job.
    """
    try:
        conn = get_db()
        states = conn.execute("SELECT * FROM shared_state WHERE run_id = ?", (run_id,)).fetchall()
        if not states:
            return "🧠 Memória vazia para este Job."
        
        report = [f"🧠 MEMÓRIA JOB #{run_id}:"]
        for s in states:
            report.append(f"  • {s['key']}: {str(s['value'])[:100]}...")
        return "\n".join(report)
    except Exception as e:
        return f"❌ Erro: {e}"

if __name__ == "__main__":
    mcp.run()
