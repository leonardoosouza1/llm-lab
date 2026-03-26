import sqlite3
import os
import time
import sys
from datetime import datetime

# Paths
PROJECT_ROOT = "/home/leonardo/projects/LLM/llm-lab"
DB_PATH = os.path.join(PROJECT_ROOT, ".agent/events/orchestrate.db")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_status(status):
    colors = {
        'PENDING': '\033[90m', # Gray
        'READY': '\033[94m',   # Blue
        'RUNNING': '\033[93m', # Yellow
        'COMPLETED': '\033[92m', # Green
        'FAILED': '\033[91m'   # Red
    }
    reset = '\033[0m'
    return f"{colors.get(status, '')}{status}{reset}"

def show_dashboard():
    if not os.path.exists(DB_PATH):
        print("❌ Arquivo de banco de dados não encontrado.")
        return

    while True:
        try:
            clear_screen()
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            print("═══ IARA v2 MISSION CONTROL Dashboard ═══")
            print(f"Update: {datetime.now().strftime('%H:%M:%S')}\n")

            # Get latest run
            cursor.execute('SELECT * FROM runs ORDER BY id DESC LIMIT 1')
            run = cursor.fetchone()
            
            if not run:
                print("Nenhum Job encontrado no banco.")
            else:
                print(f"📡 JOB #{run['id']}: {run['prompt'][:60]}...")
                print(f"Status: {format_status(run['status'])}")
                print("─" * 40)

                # Get tasks
                cursor.execute('SELECT id, agent_role, title, status, started_at, completed_at FROM tasks WHERE run_id = ? ORDER BY id', (run['id'],))
                tasks = cursor.fetchall()
                
                for t in tasks:
                    start = t['started_at'][-8:] if t['started_at'] else "--:--:--"
                    end = t['completed_at'][-8:] if t['completed_at'] else "--:--:--"
                    print(f"[{format_status(t['status']):<20}] @{t['agent_role']:<20} | {t['title']}")
                
                print("\n─" * 40)
                # Show shared state
                cursor.execute('SELECT key, value FROM shared_state WHERE run_id = ?', (run['id'],))
                states = cursor.fetchall()
                if states:
                    print("🧠 MEMÓRIA COMPARTILHADA (Shared State):")
                    for s in states:
                        val_preview = str(s['value'])[:40] + "..." if len(str(s['value'])) > 40 else s['value']
                        print(f"  • {s['key']}: {val_preview}")

            conn.close()
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n👋 Dashboard encerrado.")
            break
        except Exception as e:
            print(f"Erro no Dashboard: {e}")
            time.sleep(2)

if __name__ == "__main__":
    show_dashboard()
