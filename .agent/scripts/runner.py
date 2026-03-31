import os
import json
import asyncio
import subprocess
import logging
from state_manager import get_next_ready_tasks, update_task_status, set_shared_state, get_shared_state

# Basic configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants from the user request
CONFIG_GEMINI_BIN = "/home/leonardo/.nvm/versions/node/v22.16.0/bin/gemini"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
AGENTS_DIR = os.path.join(PROJECT_ROOT, ".agent", "agents")
MAX_CONCURRENCY = 10
POLL_INTERVAL = 3  # seconds

# 2. Intelligent Model Mapping
MODEL_MAP = {
    "orchestrator": "gemini-3.1-pro",
    "project-planner": "gemini-3.1-pro",
    "backend-specialist": "gemini-3.1-pro",
    "default": "gemini-3.1-flash"
}

def get_model_for_agent(agent_role):
    """Returns the correct Gemini model based on the agent's role."""
    clean_role = agent_role.replace("@", "").lower()
    if "pro" in clean_role or "archaeologist" in clean_role or "auditor" in clean_role:
        return "gemini-3.1-pro"
    return MODEL_MAP.get(agent_role, MODEL_MAP["default"])

async def read_agent_context_natively(agent_role):
    """
    3. Dynamic Quality Injection: Reads the agent's markdown file to be injected into the prompt.
    This is an async function to avoid blocking the main event loop.
    """
    clean_role = agent_role.replace("@", "")
    file_path = os.path.join(AGENTS_DIR, f"{clean_role}.md")
    skill_path1 = os.path.join(PROJECT_ROOT, ".agent", "skills", clean_role, "SKILL.md")
    skill_path2 = os.path.join(PROJECT_ROOT, ".agent", "skills", f"{clean_role}.md")
    
    target_path = None
    if os.path.exists(file_path):
        target_path = file_path
    elif os.path.exists(skill_path1):
        target_path = skill_path1
    elif os.path.exists(skill_path2):
        target_path = skill_path2
        
    if not target_path:
        logging.warning(f"Agent context file not found for role {agent_role}")
        return ""

    try:
        # Running file I/O in a separate thread for non-blocking operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: open(target_path, "r", encoding="utf-8").read())
    except Exception as e:
        logging.error(f"Error reading agent context for {agent_role}: {e}")
        return ""

async def execute_agent_task(task, semaphore):
    """
    Core function to execute a single agent task using asyncio.create_subprocess_exec.
    """
    task_id = task['id']
    agent_role = task['agent_role'].replace("@", "")
    prompt_spec = task['description']
    run_id = task['run_id']
    
    model_name = get_model_for_agent(agent_role)

    # 1. Using asyncio.Semaphore to limit concurrency
    async with semaphore:
        logging.info(f"[Task #{task_id}] Starting task for agent @{agent_role} with model {model_name}.")
        
        # 4. Thread-safe database calls
        await asyncio.to_thread(update_task_status, task_id, 'RUNNING')
        
        agent_rules = await read_agent_context_natively(agent_role)
        
        full_prompt = f"""
        Você é o agente especialista: @{agent_role}.
        Sua tarefa é parte de um projeto maior coordenado pelo IARA v2 na máquina do Leonardo.

        REGRAS DO PROJETO:
        - Siga estritamente o princípio do Pato: seja exato, forneça caminhos e versões.
        - COMUNICAÇÃO ENTRE AGENTES: Se sua tarefa gera conteúdo para o próximo agente (ex: um texto de arquivo, um endpoint descoberto), você **DEVE** colocá-lo no campo "shared_state_updates" do JSON final.
        
        REGRAS DO AGENTE:
        {agent_rules}

        TAREFA ESPECÍFICA:
        {prompt_spec}

        CRÍTICO: Sua resposta FINAL deve ser APENAS um objeto JSON válido.
        Se sua função é APENAS gerar conteúdo para outro agente, use o JSON abaixo:

        {{
            "status": "success" | "error",
            "summary": "resumo conciso do que foi feito",
            "shared_state_updates": {{"NOME_DA_CHAVE": "CONTEUDO_PARA_O_PROXIMO_AGENTE"}},
            "result": "detalhes técnicos aqui"
        }}
        """

        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["CI"] = "true" # Ensure non-interactive mode for CLI

        try:
            # 1. Replace ThreadPoolExecutor with asyncio.create_subprocess_exec
            proc = await asyncio.create_subprocess_exec(
                CONFIG_GEMINI_BIN,
                "--approval-mode", "yolo", # Non-interactive
                "-p", full_prompt,
                "-m", model_name,
                "-o", "json",
                "--accept-raw-output-risk", # To get the pure JSON
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=PROJECT_ROOT,
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=400)
            except asyncio.TimeoutError:
                proc.kill()
                logging.error(f"[Task #{task_id}] Timeout atingido (400s).")
                await asyncio.to_thread(update_task_status, task_id, 'FAILED', {"status": "error", "result": "Timeout atingido (400s)"})
                return

            if proc.returncode != 0:
                logging.error(f"[Task #{task_id}] Gemini CLI failed. stderr: {stderr.decode()}")
                await asyncio.to_thread(update_task_status, task_id, 'FAILED', {"status": "error", "result": stderr.decode()})
                return

            # 5. Strict JSON Payload handling
            response_text = stdout.decode().strip()
            try:
                # The response might be wrapped in a larger JSON envelope by the CLI
                cli_envelope = json.loads(response_text)
                # The actual agent's JSON is expected inside the 'response' key
                agent_json_str = cli_envelope.get("response", "{}")

                # The agent might return a string with markdown, so we clean it
                if "```json" in agent_json_str:
                    agent_json_str = agent_json_str.split("```json")[1].split("```")[0]
                
                output_data = json.loads(agent_json_str)

                # Update shared state if necessary
                if "shared_state_updates" in output_data:
                    for key, value in output_data["shared_state_updates"].items():
                        await asyncio.to_thread(set_shared_state, run_id, key, value, agent_role)

                task_status = 'COMPLETED' if output_data.get("status") == "success" else 'FAILED'
                await asyncio.to_thread(update_task_status, task_id, task_status, output_data)
                logging.info(f"[Task #{task_id}] Task completed with status: {task_status}")

            except json.JSONDecodeError as e:
                logging.error(f"[Task #{task_id}] Failed to decode JSON from Gemini CLI output: {e}\\nRaw output: {response_text}")
                await asyncio.to_thread(update_task_status, task_id, 'FAILED', {"status": "error", "result": "JSON Decode Error", "raw_output": response_text})

        except Exception as e:
            logging.critical(f"[Task #{task_id}] A critical error occurred during subprocess execution: {e}")
            await asyncio.to_thread(update_task_status, task_id, 'FAILED', {"status": "error", "result": str(e)})

async def main_loop():
    """
    Main event loop to poll for tasks and dispatch them.
    """
    logging.info(f"IARA Async Runner started with concurrency limit of {MAX_CONCURRENCY}.")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    running_tasks = set()

    while True:
        try:
            available_slots = MAX_CONCURRENCY - len(running_tasks)
            if available_slots > 0:
                # 4. Thread-safe database calls
                tasks_to_run = await asyncio.to_thread(get_next_ready_tasks, available_slots)
                
                for task in tasks_to_run:
                    if task['id'] not in running_tasks:
                        logging.info(f"Dispatching task {task['id']} for agent {task['agent_role']}")
                        
                        # Create a task in the asyncio event loop
                        async_task = asyncio.create_task(execute_agent_task(task, semaphore))
                        running_tasks.add(task['id'])

                        # Add a callback to remove the task from the set when it's done
                        async_task.add_done_callback(lambda t, task_id=task['id']: running_tasks.discard(task_id))

            await asyncio.sleep(POLL_INTERVAL)
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            # Avoid crashing the main loop
            await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("IARA Async Runner is shutting down.")
    except Exception as e:
        logging.critical(f"A critical error caused the runner to exit: {e}")
