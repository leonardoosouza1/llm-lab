import os
import sys
import subprocess
import sqlite3

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getcwd()
DB_PATH = os.path.join(PROJECT_ROOT, ".agent/events/orchestrate.db")

def verify_step(name, check_fn):
    print(f"🔍 Verificando {name}...", end=" ", flush=True)
    success, msg = check_fn()
    if success:
        print(f"{GREEN}OK{RESET}")
    else:
        print(f"{RED}FALHA{RESET}")
        print(f"   ❌ {msg}")
    return success

def check_structure():
    required = ["architect.py", "runner.py", "state_manager.py", "dashboard.py", "iara.py"]
    for f in required:
        if not os.path.exists(os.path.join(SCRIPTS_DIR, f)):
            return False, f"Arquivo {f} não encontrado em {SCRIPTS_DIR}"
    return True, ""

def check_gemini():
    try:
        result = subprocess.run(["gemini", "--version"], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)

def check_db():
    if not os.path.exists(DB_PATH):
        return False, "Banco de dados não encontrado (rode iara.py uma vez para inicializar)"
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)

def run_tests():
    print("🧪 Bateria de Testes IARA v2\n")
    
    steps = [
        ("Estrutura de Arquivos", check_structure),
        ("Gemini CLI (Conectividade)", check_gemini),
        ("SQLite (Estado)", check_db)
    ]
    
    all_ok = True
    for name, fn in steps:
        if not verify_step(name, fn):
            all_ok = False
            
    print("\n" + ("🎉 IARA operacional!" if all_ok else "⚠️  Problemas detectados na instalação."))
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    run_tests()
