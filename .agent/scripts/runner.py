import os
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from state_manager import get_next_ready_tasks, update_task_status, set_shared_state, get_shared_state

# Configuration
GEMINI_BIN = "/home/leonardo/.nvm/versions/node/v22.16.0/bin/gemini"
PROJECT_ROOT = "/home/leonardo/projects/LLM/llm-lab"
MAX_CONCURRENCY = 4
POLL_INTERVAL = 5 # seconds

def execute_agent_task(task):
    """Executes a single agent task using the Gemini CLI."""
    task_id = task['id']
    agent_role = task['agent_role']
    prompt_spec = task['description']
    run_id = task['run_id']
    
    print(f"🚀 [Task #{task_id}] Despachando @{agent_role}...")
    update_task_status(task_id, 'RUNNING')
    
    # Load shared state for this run
    current_state = get_shared_state(run_id)
    state_context = f"\nSHARED CONTEXT:\n{json.dumps(current_state, indent=2)}" if current_state else ""
    
    full_prompt = f"""
    Você é o agente especialista: @{agent_role}.
    Sua tarefa é parte de um projeto maior coordenado pelo IARA v2 na máquina do Leonardo.

    REGRAS DO PROJETO:
    - Siga estritamente o princípio do Pato: seja exato, forneça caminhos e versões.
    - COMUNICAÇÃO ENTRE AGENTES: Se sua tarefa gera conteúdo para o próximo agente (ex: um texto de arquivo, um endpoint descoberto), você **DEVE** colocá-lo no campo "shared_state_updates" do JSON final.

    {state_context}

    TAREFA ESPECÍFICA:
    {prompt_spec}

    CRÍTICO: Sua resposta FINAL deve ser APENAS um objeto JSON válido.
    Não use ferramentas externas como 'write_file' a menos que seja explicitamente pedido na tarefa.
    Se sua função é APENAS gerar conteúdo para outro agente, use o JSON abaixo:

    {{
        "status": "success" | "error",
        "summary": "resumo conciso do que foi feito",
        "shared_state_updates": {{"NOME_DA_CHAVE": "CONTEUDO_PARA_O_PROXIMO_AGENTE"}},
        "result": "detalhes técnicos aqui"
    }}
    """
    
    try:
        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["CI"] = "true"
        # Ensure HOME is preserved for OAuth
        if "HOME" not in env: env["HOME"] = "/home/leonardo"
        
        result = subprocess.run(
            [GEMINI_BIN, "--approval-mode", "yolo", "-p", full_prompt, "-m", "gemini-2.5-flash", "-o", "json", "--accept-raw-output-risk"],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
            timeout=300 # 5 minute timeout per agent
        )
        
        if result.returncode != 0:
            error_data = {"status": "error", "result": result.stderr}
            update_task_status(task_id, 'FAILED', error_data)
            print(f"❌ [Task #{task_id}] Falhou: {result.stderr[:100]}...")
            return
            
        # Parse JSON output
        try:
            envelope = json.loads(result.stdout.strip())
            inner_content = envelope.get("response", "")
            
            # Strip markdown backticks if present
            if "```json" in inner_content:
                inner_content = inner_content.split("```json")[1].split("```")[0].strip()
            elif "```" in inner_content:
                inner_content = inner_content.split("```")[1].split("```")[0].strip()
            
            output_data = json.loads(inner_content)
            
            # Update shared state if agent provided any
            updates = output_data.get("shared_state_updates", {})
            for k, v in updates.items():
                set_shared_state(run_id, k, v, agent_role)
            
            status = 'COMPLETED' if output_data.get("status") == "success" else 'FAILED'
            update_task_status(task_id, status, output_data)
            print(f"✅ [Task #{task_id}] Concluída: {output_data.get('summary', 'Sem resumo')}")
            
        except json.JSONDecodeError:
            print(f"⚠️  [Task #{task_id}] Erro de parsing JSON. Resposta bruta salva.")
            update_task_status(task_id, 'FAILED', {"status": "error", "result": "JSON Decode Error", "raw": result.stdout})

    except subprocess.TimeoutExpired:
        print(f"⏰ [Task #{task_id}] Timeout atingido (300s).")
        update_task_status(task_id, 'FAILED', {"status": "error", "result": "Timeout"})
    except Exception as e:
        print(f"❌ [Task #{task_id}] Erro inesperado: {e}")
        update_task_status(task_id, 'FAILED', {"status": "error", "result": str(e)})

def run_loop():
    print(f"👁️  IARA Parallel Runner iniciado (Concurrency: {MAX_CONCURRENCY})...")
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        futures = {}
        
        while True:
            # Check for new ready tasks
            ready_tasks = get_next_ready_tasks(max_tasks=MAX_CONCURRENCY - len(futures))
            
            for task in ready_tasks:
                if task['id'] not in futures:
                    future = executor.submit(execute_agent_task, task)
                    futures[task['id']] = future
            
            # Clean up finished futures
            done_ids = []
            for tid, fut in futures.items():
                if fut.done():
                    done_ids.append(tid)
            
            for tid in done_ids:
                del futures[tid]
            
            # If no tasks and no futures, we could exit or wait. 
            # In IARA v2, it's a long-running service loop.
            if not ready_tasks and not futures:
                # print(".", end="", flush=True) # Heartbeat
                pass
                
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\n👋 Runner finalizado pelo usuário.")
