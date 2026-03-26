#!/usr/bin/env python3
"""
IARA Event Loop - Processa tarefas na inbox e executa via Gemini CLI.

Arquitetura:
  inbox/{agent}/*.json → processing/{agent}/ → outbox/{agent}/ (ou failed/{agent}/)

Cada task.json contém:
  { "task_id": "uuid", "agent": "role", "task": "instrução", "context": {} }
"""

import os
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# === CONFIGURAÇÃO ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # llm-lab/
AGENT_DIR = PROJECT_ROOT / ".agent" / "events"
INBOX_DIR = AGENT_DIR / "inbox"
PROCESSING_DIR = AGENT_DIR / "processing"
OUTBOX_DIR = AGENT_DIR / "outbox"
FAILED_DIR = AGENT_DIR / "failed"
LOG_FILE = AGENT_DIR / "mcp_orchestrator.log"

# Gemini CLI config
GEMINI_BIN = shutil.which("gemini") or "/home/leonardo/.nvm/versions/node/v22.16.0/bin/gemini"
POLL_INTERVAL = 2  # seconds
TASK_TIMEOUT = 300  # 5 min max per task

MODEL_LADDER = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def log_event(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")


def ensure_dirs(agent_role: str = None):
    for d in [INBOX_DIR, PROCESSING_DIR, OUTBOX_DIR, FAILED_DIR]:
        target = d / agent_role if agent_role else d
        target.mkdir(parents=True, exist_ok=True)


def build_prompt(agent_role: str, task_instruction: str, context: dict = None) -> str:
    """Monta o prompt para o Gemini CLI com contexto do agente."""
    
    context_block = ""
    if context:
        if context.get("original_request"):
            context_block += f"\nPedido original do usuário: {context['original_request']}"
        if context.get("previous_results"):
            context_block += f"\nResultados anteriores de outros agentes:\n{json.dumps(context['previous_results'], indent=2, ensure_ascii=False)}"
        if context.get("project_path"):
            context_block += f"\nCaminho do projeto: {context['project_path']}"
        if context.get("files"):
            context_block += f"\nArquivos relevantes: {', '.join(context['files'])}"

    return f"""Você é o agente especialista: @{agent_role}.
{context_block}

Tarefa: {task_instruction}

CRÍTICO: Sua resposta FINAL deve ser APENAS um objeto JSON válido:
{{
    "status": "success" | "error",
    "summary": "resumo conciso do que foi feito",
    "result": "detalhes da resolução",
    "files_modified": ["lista de arquivos modificados, se houver"],
    "recommendations": ["recomendações para próximos passos"]
}}
Não inclua texto explicativo fora do bloco JSON."""


def extract_json_from_response(raw: str) -> dict:
    """Extrai JSON limpo da resposta do LLM (que pode vir em markdown)."""
    import re
    
    # Tenta direto
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    
    if not isinstance(raw, str):
        return {"status": "success", "result": str(raw)}
    
    # Busca ```json ... ```
    match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Busca ``` ... ```
    match = re.search(r'```\s*(.*?)\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Busca primeiro { até último }
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    return {"status": "success", "result": raw}


def run_gemini(prompt: str, agent_role: str) -> dict:
    """Executa Gemini CLI com model ladder e retorna o resultado."""
    
    env = os.environ.copy()
    # CRUCIAL: manter HOME real para OAuth funcionar
    env["TERM"] = "dumb"
    env["CI"] = "true"
    
    last_error = ""
    
    for model in MODEL_LADDER:
        try:
            log_event(f"  🚀 Tentando @{agent_role} com modelo: {model}")
            
            result = subprocess.run(
                [
                    GEMINI_BIN, "-p", prompt,
                    "-m", model,
                    "-o", "json",
                    "--accept-raw-output-risk"
                ],
                env=env,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=TASK_TIMEOUT
            )
            
            if result.returncode != 0:
                err_msg = result.stderr or result.stdout
                # Rate limit → próximo modelo
                if any(kw in err_msg for kw in ["429", "RESOURCE_EXHAUSTED", "No capacity", "overloaded"]):
                    log_event(f"  ⚠️ Modelo {model} exaurido (429). Tentando próximo...")
                    last_error = err_msg[:200]
                    continue
                else:
                    raise Exception(f"Gemini CLI falhou (rc={result.returncode}): {err_msg[:300]}")
            
            # Parse o JSON de output do CLI
            output = json.loads(result.stdout)
            
            # Extrai a resposta do agente
            raw_response = output.get("response", "")
            agent_json = extract_json_from_response(raw_response)
            output["agent_response"] = agent_json
            output["model_used"] = model
            
            return output
            
        except subprocess.TimeoutExpired:
            log_event(f"  ⏰ Timeout ({TASK_TIMEOUT}s) com modelo {model}. Próximo...")
            last_error = f"Timeout after {TASK_TIMEOUT}s"
            continue
        except json.JSONDecodeError as e:
            log_event(f"  ⚠️ JSON inválido do CLI com {model}: {e}")
            # Se o CLI retornou algo, tenta usar mesmo assim
            if result and result.stdout:
                return {
                    "response": result.stdout[:2000],
                    "agent_response": extract_json_from_response(result.stdout),
                    "model_used": model,
                    "parse_warning": "Output was not valid JSON from CLI"
                }
            last_error = str(e)
            continue
        except Exception as e:
            if model == MODEL_LADDER[-1]:
                raise
            log_event(f"  ❌ Erro com {model}: {str(e)[:100]}. Próximo...")
            last_error = str(e)[:200]
            continue
    
    raise Exception(f"Todos os modelos falharam. Último erro: {last_error}")


def process_task(task_file: Path, agent_role: str):
    """Processa uma tarefa: inbox → processing → outbox/failed."""
    log_event(f"🔄 Processando tarefa: {task_file.name} (Agent: @{agent_role})")
    
    # Move para processing
    ensure_dirs(agent_role)
    processing_file = PROCESSING_DIR / agent_role / task_file.name
    
    try:
        task_file.rename(processing_file)
    except FileNotFoundError:
        log_event(f"  ⚠️ Tarefa {task_file.name} já foi movida. Skipping.")
        return
    
    try:
        with open(processing_file, "r") as f:
            task_data = json.load(f)
        
        task_instruction = task_data.get("task")
        if not task_instruction:
            raise ValueError("Campo 'task' ausente no JSON")
        
        context = task_data.get("context", {})
        prompt = build_prompt(agent_role, task_instruction, context)
        
        log_event(f"  🚀 Executando Gemini CLI → Agent: @{agent_role}")
        output_data = run_gemini(prompt, agent_role)
        
        # Salva resultado no outbox
        outbox_file = OUTBOX_DIR / agent_role / processing_file.name
        
        # Adiciona metadata
        output_data["_meta"] = {
            "task_id": task_data.get("task_id", "unknown"),
            "agent": agent_role,
            "completed_at": datetime.now().isoformat(),
            "original_task": task_instruction[:200]
        }
        
        with open(outbox_file, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        log_event(f"✅ Tarefa concluída: {task_file.name} → outbox/{agent_role}/")
        
        # Limpa processing
        processing_file.unlink(missing_ok=True)
        
    except Exception as e:
        log_event(f"❌ Erro processando {task_file.name}: {str(e)[:300]}")
        
        # Move para failed
        failed_file = FAILED_DIR / agent_role / processing_file.name
        try:
            processing_file.rename(failed_file)
        except Exception:
            pass
        
        # Salva detalhes do erro
        error_file = FAILED_DIR / agent_role / f"{processing_file.stem}.error"
        with open(error_file, "w") as f:
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            f.write(f"agent: {agent_role}\n")
            f.write(f"error: {str(e)}\n")


def recover_stuck_tasks():
    """Recupera tarefas presas em processing/ por mais de 10 min."""
    if not PROCESSING_DIR.exists():
        return
    
    for agent_dir in PROCESSING_DIR.iterdir():
        if not agent_dir.is_dir():
            continue
        for task_file in agent_dir.glob("*.json"):
            age = time.time() - task_file.stat().st_mtime
            if age > 600:  # 10 min
                log_event(f"🔁 Recuperando tarefa stuck: {task_file.name} (age={int(age)}s)")
                # Move de volta para inbox
                inbox_dest = INBOX_DIR / agent_dir.name / task_file.name
                inbox_dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    task_file.rename(inbox_dest)
                except Exception as e:
                    log_event(f"  ⚠️ Falha ao recuperar: {e}")


def watch_inbox():
    """Loop principal: observa a inbox e processa tarefas."""
    ensure_dirs()
    log_event("=" * 60)
    log_event("👁️  IARA Event Loop iniciado")
    log_event(f"   Project: {PROJECT_ROOT}")
    log_event(f"   Gemini: {GEMINI_BIN}")
    log_event(f"   Poll: {POLL_INTERVAL}s | Timeout: {TASK_TIMEOUT}s")
    log_event(f"   Models: {' → '.join(MODEL_LADDER)}")
    log_event("=" * 60)
    
    cycle = 0
    
    try:
        while True:
            cycle += 1
            
            # A cada 30 ciclos (1 min), verifica stuck tasks
            if cycle % 30 == 0:
                recover_stuck_tasks()
            
            if INBOX_DIR.exists():
                for agent_dir in INBOX_DIR.iterdir():
                    if not agent_dir.is_dir():
                        continue
                    agent_role = agent_dir.name
                    for task_file in sorted(agent_dir.glob("*.json")):
                        process_task(task_file, agent_role)
            
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        log_event("🛑 Event Loop encerrado pelo usuário.")


if __name__ == "__main__":
    watch_inbox()
