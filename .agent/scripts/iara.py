import sys
import os
import subprocess
import time
import json

# Configuration
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getcwd() # Consider current working directory as project root

# Add scripts dir to path for imports
sys.path.append(SCRIPTS_DIR)
from state_manager import initialize_db

def initialize_engine():
    initialize_db()
    print("✅ Engine IARA v2 Inicializada.")

def run_script(script_name, args=None, capture=False):
    path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = ["python3", path]
    if args:
        cmd.extend(args)
    
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    else:
        return subprocess.Popen(cmd, cwd=PROJECT_ROOT)

def show_status():
    from state_manager import get_status_summary
    stats = get_status_summary()
    if not stats["run_id"]:
        print("📭 Nenhum processo ativo encontrado.")
        return
    
    tasks = stats["tasks"]
    total = sum(tasks.values())
    completed = tasks.get("COMPLETED", 0)
    failed = tasks.get("FAILED", 0)
    pending = tasks.get("PENDING", 0)
    running = tasks.get("RUNNING", 0)
    
    print(f"📊 STATUS IARA [Run #{stats['run_id']}]:")
    print(f"✅ Concluído: {completed}/{total}")
    if running: print(f"⏳ Rodando: {running}")
    if pending: print(f"💤 Pendente: {pending}")
    if failed:  print(f"❌ Falhou: {failed}")
    
    prog = (completed / total * 100) if total > 0 else 0
    bar = "█" * int(prog/5) + "░" * (20 - int(prog/5))
    print(f"📈 Progresso: [{bar}] {prog:.1f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="IARA v2 - Gateway de Orquestração")
    parser.add_argument("prompt", nargs='?', help="Prompt principal ou comando")
    parser.add_argument("--register", action="store_true", help="Registra tarefa na fila assíncrona")
    parser.add_argument("--agent", help="Agente específico (ex: @backend-specialist)")
    parser.add_argument("--skills", help="Skills separadas por vírgula (ex: @rust-pro,@clean-code)")
    parser.add_argument("--status", action="store_true", help="Mostra resumo do progresso")
    
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    initialize_db()

    if args.register:
        from state_manager import add_task, get_latest_run_id, add_run
        run_id = get_latest_run_id()
        if not run_id:
            run_id = add_run("Manual Session")
        
        agent = args.agent or "@orchestrator"
        skills = args.skills or ""
        prompt = args.prompt or "Execução manual"
        
        print(f"📥 REGISTRANDO DEMANDA: {agent} [Skills: {skills}]")
        
        # Build context from prompt + skills
        context = {
            "prompt": prompt,
            "skills": skills.split(",") if skills else []
        }
        
        task_id = add_task(
            run_id=run_id,
            agent_role=agent.replace("@", ""),
            title=f"Manual: {prompt[:30]}...",
            description=prompt,
            context_json=context
        )
        print(f"🚀 Job #{run_id} | Tarefa #{task_id} registrada no banco.")
        return

    if not args.prompt:
        parser.print_help()
        return

    user_prompt = args.prompt

    print("\n--- 🏗️  FASE 1: ARQUITETURA E DESCOBERTA ---")
    # Run architect and capture output to see if it's READY or GAP
    result = run_script("architect.py", [user_prompt], capture=True)
    
    output = result.stdout or ""
    print(output)
    
    if "⚠️  GAPS IDENTIFICADOS" in output:
        print("\n🛑 IARA: Aguardando definições técnicas para prosseguir.")
        return

    if "🚀 Job #" in output:
        print("\n--- ⚙️  FASE 2: EXECUÇÃO PARALELA (MOTOR) ---")
        # Start runner in background
        print("🚀 Iniciando Motor (Runner) em segundo plano...")
        run_script("runner.py")
        
        print("\n--- 📊 FASE 3: MISSION CONTROL (DASHBOARD) ---")
        # Start dashboard in foreground
        time.sleep(2) # Wait for runner to pick up tasks
        try:
            subprocess.run(["python3", os.path.join(SCRIPTS_DIR, "dashboard.py")], cwd=PROJECT_ROOT)
        except KeyboardInterrupt:
            print("\n👋 Dashboard fechado. O Runner continuará em segundo plano.")

if __name__ == "__main__":
    main()
