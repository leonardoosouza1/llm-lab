import sqlite3
import json
import os
from datetime import datetime

# Paths
PROJECT_ROOT = "/home/leonardo/projects/LLM/llm-lab"
DB_DIR = os.path.join(PROJECT_ROOT, ".agent/events")
DB_PATH = os.path.join(DB_DIR, "orchestrate.db")

def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initializes the SQLite schema for IARA v2."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table: runs (high-level jobs)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT NOT NULL,
        findings TEXT, -- JSON string of discovery phase results
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    )
    ''')
    
    # Table: tasks (individual agent assignments)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        agent_role TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        context_json TEXT, -- Senior spec passed to the agent
        status TEXT DEFAULT 'PENDING', -- PENDING, READY, RUNNING, COMPLETED, FAILED
        priority INTEGER DEFAULT 1,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        result TEXT, -- JSON string of agent output
        FOREIGN KEY (run_id) REFERENCES runs (id)
    )
    ''')
    
    # Table: task_dependencies (execution graph)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS task_dependencies (
        task_id INTEGER NOT NULL,
        depends_on_task_id INTEGER NOT NULL,
        PRIMARY KEY (task_id, depends_on_task_id),
        FOREIGN KEY (task_id) REFERENCES tasks (id),
        FOREIGN KEY (depends_on_task_id) REFERENCES tasks (id)
    )
    ''')
    
    # Table: shared_state (inter-agent memory)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS shared_state (
        run_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        written_by_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, key),
        FOREIGN KEY (run_id) REFERENCES runs (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")

def add_run(prompt, findings=None):
    conn = get_connection()
    cursor = conn.cursor()
    findings_json = json.dumps(findings) if findings else None
    cursor.execute('INSERT INTO runs (prompt, findings) VALUES (?, ?)', (prompt, findings_json))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def add_task(run_id, agent_role, title, description, context_json=None, dependencies=None):
    """
    Adds a task and its dependencies.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # Normalize context: it could be a string (prompt) or a complex dict
    ctx_str = json.dumps(context_json) if isinstance(context_json, (dict, list)) else str(context_json)
    
    cursor.execute('''
        INSERT INTO tasks (run_id, agent_role, title, description, context_json)
        VALUES (?, ?, ?, ?, ?)
    ''', (run_id, agent_role, title, description, ctx_str))
    task_id = cursor.lastrowid
    
    if dependencies:
        for dep_id in dependencies:
            cursor.execute('INSERT INTO task_dependencies (task_id, depends_on_task_id) VALUES (?, ?)', 
                           (task_id, dep_id))
    
    conn.commit()
    conn.close()
    return task_id

def get_latest_run_id():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def update_task_status(task_id, status, result=None):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    if status == 'RUNNING':
        cursor.execute('UPDATE tasks SET status = ?, started_at = ? WHERE id = ?', (status, now, task_id))
    elif status in ['COMPLETED', 'FAILED']:
        res_str = json.dumps(result) if result else None
        cursor.execute('UPDATE tasks SET status = ?, completed_at = ?, result = ? WHERE id = ?', 
                       (status, now, res_str, task_id))
    else:
        cursor.execute('UPDATE tasks SET status = ? WHERE id = ?', (status, task_id))
        
    conn.commit()
    conn.close()

def set_shared_state(run_id, key, value, agent_role=None):
    conn = get_connection()
    cursor = conn.cursor()
    val_str = json.dumps(value) if not isinstance(value, str) else value
    cursor.execute('''
        INSERT OR REPLACE INTO shared_state (run_id, key, value, written_by_agent)
        VALUES (?, ?, ?, ?)
    ''', (run_id, key, val_str, agent_role))
    conn.commit()
    conn.close()

def get_shared_state(run_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT key, value FROM shared_state WHERE run_id = ?', (run_id,))
    rows = cursor.fetchall()
    conn.close()
    return {row['key']: row['value'] for row in rows}

def get_next_ready_tasks(max_tasks=4):
    """
    Retrieves tasks that are PENDING and have all their dependencies COMPLETED.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Complex query to find tasks with satisfied dependencies
    query = '''
    SELECT t.* FROM tasks t
    WHERE t.status = 'PENDING'
    AND NOT EXISTS (
        SELECT 1 FROM task_dependencies td
        JOIN tasks dep ON td.depends_on_task_id = dep.id
        WHERE td.task_id = t.id
        AND dep.status != 'COMPLETED'
    )
    LIMIT ?
    '''
    cursor.execute(query, (max_tasks,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_status_summary():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get total tasks per status
    cursor.execute("""
        SELECT status, COUNT(*) 
        FROM tasks 
        GROUP BY status
    """)
    rows = cursor.fetchall()
    summary = {row[0]: row[1] for row in rows}
    
    # Get latest active run id
    cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
    run_id = cursor.fetchone()
    
    conn.close()
    return {
        "run_id": run_id[0] if run_id else None,
        "tasks": summary
    }

if __name__ == "__main__":
    initialize_db()
