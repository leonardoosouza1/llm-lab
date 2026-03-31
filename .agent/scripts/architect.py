import os
import sys
import json
import subprocess
from state_manager import add_run, add_task, initialize_db

# Configuration
GEMINI_BIN = "/home/leonardo/.nvm/versions/node/v22.16.0/bin/gemini"
PROJECT_ROOT = "/home/leonardo/projects/LLM/llm-lab"

def run_gemini(prompt, model="gemini-3.1-pro"):
    """Runs Gemini CLI and returns the raw output."""
    try:
        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["CI"] = "true"
        
        result = subprocess.run(
            [GEMINI_BIN, "-p", prompt, "-m", model, "-o", "json", "--accept-raw-output-risk"],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode != 0:
            print(f"❌ Gemini CLI Error: {result.stderr}")
            return None
            
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Error running Gemini: {e}")
        return None

def discovery_scan():
    """Performs a quick scan of the project to gather context."""
    context = {}
    
    # Check for package.json
    pkg_path = os.path.join(PROJECT_ROOT, "package.json")
    if os.path.exists(pkg_path):
        with open(pkg_path, "r") as f:
            context["package_json"] = json.load(f)
            
    # Check file structure (shallow)
    files = []
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        # Skip heavy folders
        if "node_modules" in dirs: dirs.remove("node_modules")
        if ".git" in dirs: dirs.remove(".git")
        
        rel_path = os.path.relpath(root, PROJECT_ROOT)
        if rel_path == ".":
            files.extend(filenames)
        else:
            files.extend([os.path.join(rel_path, f) for f in filenames])
            
        if len(files) > 100: break # Safety limit
    
    context["file_structure"] = files
    return context

def architect_pipeline(user_prompt):
    print("🏗️  IARA Architect: Inciando descoberta...")
    context = discovery_scan()
    
    # Prepare the "Gap Analysis" prompt
    gap_prompt = f"""
    Você é o Arquiteto Sênior do sistema IARA. 
    Seu objetivo é analisar o pedido do usuário e o contexto do projeto para identificar "furos" (gaps de informação) ou recomendar pacotes antes de planejar.

    CONTEXTO DO PROJETO:
    {json.dumps(context, indent=2)}

    PEDIDO DO USUÁRIO:
    {user_prompt}

    INSTRUÇÕES:
    1. Analise se faltam informações críticas (URL de API, Endpoints, Estratégia de Auth, etc).
    2. Verifique se pacotes necessários estão faltando (ex: se pediu form e não tem library de form, recomende uma).
    3. Se houver gaps, retorne um objeto JSON com "status": "GAP" e uma lista de "questions" para o usuário.
    4. Se estiver tudo claro, retorne "status": "READY" e uma lista de "tasks" para decomposição.

    FORMATO DE RESPOSTA (JSON APENAS):
    {{
      "status": "GAP" | "READY",
      "questions": ["Pergunta 1", "Pergunta 2"],
      "recommendations": ["Recomendação 1"],
      "tasks": [
        {{
          "id": 1,
          "agent": "agent-role",
          "title": "Titulo",
          "description": "Spec Sênior Completa (Princípio do Pato)",
          "depends_on": []
        }}
      ]
    }}
    """
    
    print("🧠 Analisando gaps e decompondo...")
    response_raw = run_gemini(gap_prompt, "gemini-3.1-pro")
    if not response_raw:
        print("❌ Sem resposta do Gemini.")
        return
        
    print(f"DEBUG: Resposta bruta recebida.")
    try:
        envelope = json.loads(response_raw)
        inner_content = envelope.get("response", "")
        
        # Strip markdown backticks if present
        if "```json" in inner_content:
            inner_content = inner_content.split("```json")[1].split("```")[0].strip()
        elif "```" in inner_content:
            inner_content = inner_content.split("```")[1].split("```")[0].strip()
        else:
            start = inner_content.find('{')
            end = inner_content.rfind('}')
            if start != -1 and end != -1:
                inner_content = inner_content[start:end+1]
            
        data = json.loads(inner_content)
        print(f"DEBUG: Status retornado: {data.get('status')}")
        
        if data.get("status") == "GAP":
            print("\n⚠️  GAPS IDENTIFICADOS PELO ARQUITETO:")
            for q in data.get("questions", []):
                print(f"  ❓ {q}")
            for r in data.get("recommendations", []):
                print(f"  💡 Sugestão: {r}")
            return
            
        if data.get("status") == "READY":
            print("✅ Projeto pronto para execução. Criando Job no SQLite...")
            run_id = add_run(user_prompt, context)
            
            task_mapping = {} 
            
            tasks_list = data.get("tasks", [])
            for t_info in tasks_list:
                task_id = add_task(
                    run_id=run_id,
                    agent_role=t_info["agent"],
                    title=t_info["title"],
                    description=t_info["description"]
                )
                task_mapping[t_info["id"]] = task_id
            
            from state_manager import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            for t_info in tasks_list:
                this_task_id = task_mapping[t_info["id"]]
                for dep_temp_id in t_info.get("depends_on", []):
                    dep_real_id = task_mapping.get(dep_temp_id)
                    if dep_real_id:
                        cursor.execute('INSERT OR IGNORE INTO task_dependencies (task_id, depends_on_task_id) VALUES (?, ?)', 
                                       (this_task_id, dep_real_id))
            conn.commit()
            conn.close()
            
            print(f"🚀 Job #{run_id} criado com {len(task_mapping)} tarefas!")
            print(f"💡 Use: python3 .agent/scripts/runner.py")

    except Exception as e:
        print(f"❌ Erro ao processar resposta do Arquiteto: {e}")
        print(f"Resposta bruta: {response_raw}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 architect.py 'seu prompt aqui'")
    else:
        initialize_db() # ensure db exists
        architect_pipeline(sys.argv[1])
