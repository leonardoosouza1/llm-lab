#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Protocolo Mandatório P0
PROTOCOL = """# 🤖 IARA v2 - MANDATORY PROTOCOL (READ FIRST)

> [!IMPORTANT]
> **CRITICAL RULE (P0):** AI ASSISTANTS ARE FORBIDDEN FROM WRITING PROJECT CODE DIRECTLY.
> 
> Any task (Feature, Bug, Refactor, Design) MUST be delegated to the IARA v2 Engine:
> 1. **MANDATORY TOOL:** `python3 .agent/scripts/iara.py "<prompt>"`
> 2. **NO MANUAL PLANNING:** Do not create `implementation_plan.md` or `task.md` manually. Let the `iara.py` (Architect) do it.
> 3. **NO BRAINSTORMING:** Do not use Socratic questions unless prompted by the IARA Architect's "GAP" output.
> 4. **ROLE:** You are a DISPATCHER for IARA v2. Launch the script and monitor output.
"""

GLOBAL_MANDATE = """# 🛑 IARA v2 GLOBAL MANDATE (P0 - HIGHEST PRIORITY)

> [!CAUTION]
> **ABSOLUTE RULE:** ALL AUTOMATIC MODES (PLAN, EDIT, ORCHESTRATE) ARE OVERRIDDEN.
> 
> 1. **FORBIDDEN:** Direct use of `replace_file_content` or `write_file` for project code.
> 2. **MANDATORY:** Execute `python3 .agent/scripts/iara.py "<prompt>"` for every user request.
> 3. **ZERO-TOUCH:** You do not plan. You do not code. You only TRIGGER and MONITOR the IARA v2 Orchestrator.
"""

def inject_protocol(file_path, protocol_text):
    path = Path(file_path)
    if path.exists():
        content = path.read_text()
        if "IARA v2 - MANDATORY PROTOCOL" in content or "IARA v2 GLOBAL MANDATE" in content:
            print(f"✅ Protocolo já presente em {file_path}")
            return
        print(f"💉 Injetando protocolo em {file_path}...")
        path.write_text(protocol_text + "\n" + content)
    else:
        print(f"📝 Criando {file_path} com protocolo...")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(protocol_text)

def main():
    print("🛠️  Instalador IARA v2 - Preparando Ambiente...")
    
    # 1. Detectar Root
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent.parent
    
    # 2. Injetar Protocolos
    inject_protocol(project_root / "GEMINI.md", PROTOCOL)
    inject_protocol(project_root / ".agent" / "rules" / "GEMINI.md", GLOBAL_MANDATE)
    
    # 3. Inicializar DB
    print("🗄️  Inicializando Banco de Dados...")
    sys.path.append(str(scripts_dir))
    from iara import initialize_engine
    initialize_engine()
    
    # 4. Rodar Testes
    print("🧪 Executando bateria de testes...")
    from test_engine import run_tests
    try:
        run_tests()
    except SystemExit as e:
        if e.code == 0:
            print("\n🎉 IARA v2 instalado e pronto para o combate! 🚀🦆")
        else:
            print("\n⚠️  Instalação concluída com alertas. Verifique os erros acima.")

if __name__ == "__main__":
    main()
