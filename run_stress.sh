#!/bin/bash
# =============================================================
# run_stress.sh — Stress Test + Ollama Comparison
# =============================================================
# Requer: Terminal 1 rodando ./run_iara.sh release
# =============================================================

set -e

# 1. Caminhos ROCm (idêntico ao run_iara.sh)
export ROCM_PATH=/opt/rocm-7.2.1
export ROCM_LLVM_PATH=$ROCM_PATH/llvm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_LLVM_PATH/lib:$LD_LIBRARY_PATH
export LIBCLANG_PATH=$ROCM_LLVM_PATH/lib
export CLANG_PATH=$ROCM_LLVM_PATH/bin/clang
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $ROCM_LLVM_PATH/lib/clang/22/include -I$ROCM_PATH/include"
export HSA_OVERRIDE_GFX_VERSION=10.3.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CARGO_BIN="${HOME}/.cargo/bin/cargo"
MODEL="${1:-qwen2.5-1.5b}"

echo ""
echo "================================================================"
echo "  IARA STRESS TEST — Terminal 2"
echo "================================================================"

# 2. Verificar servidor
echo ""
echo "[1/5] Verificando servidor IARA..."
if ! curl -s localhost:3000/health > /dev/null 2>&1; then
    echo "❌ Servidor não está rodando em localhost:3000"
    echo "   Rode primeiro:  ./run_iara.sh release"
    exit 1
fi
echo "✅ Servidor online"

# 3. Carregar modelo (pula se já carregado)
echo ""
echo "[2/5] Verificando modelo: $MODEL"
HEALTH_STATUS=$(curl -s localhost:3000/health 2>/dev/null)
IS_OK=$(echo "$HEALTH_STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('status')=='ok' else '')" 2>/dev/null || echo "")

if [ -n "$IS_OK" ]; then
    CURRENT=$(echo "$HEALTH_STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model','?'))" 2>/dev/null || echo "?")
    echo "✅ Modelo já carregado: $CURRENT (pulando load)"
else
    echo "📦 Carregando modelo: $MODEL"
    curl -s -X POST localhost:3000/api/load-model \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$MODEL\"}" 2>/dev/null || true

    echo ""
    echo "⏳ Aguardando modelo carregar..."
    for i in $(seq 1 10); do
        IS_READY=$(curl -s localhost:3000/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('status')=='ok' else '')" 2>/dev/null || echo "")
        if [ -n "$IS_READY" ]; then
            echo ""
            echo "✅ Modelo carregado!"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo ""
            echo "❌ Timeout (120s). Verificar Terminal 1."
            exit 1
        fi
        printf "."
        sleep 1
    done
fi

# 4. Health check completo
echo ""
echo "[3/5] Status do sistema:"
curl -s localhost:3000/health | python3 -m json.tool 2>/dev/null || curl -s localhost:3000/health

# 5. Teste rápido de sanidade
echo ""
echo "[4/5] Teste de sanidade (1 token)..."
SANITY=$(curl -s -X POST localhost:3000/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Hello\",\"max_tokens\":1,\"temperature\":0.7}" \
    --max-time 60 2>/dev/null)

if [ -z "$SANITY" ]; then
    echo "❌ Geração falhou — modelo pode não estar pronto"
    echo "   Verifique o Terminal 1 para erros"
    exit 1
fi
echo "✅ Geração OK:"
echo "$SANITY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'   Token: {d.get(\"response\",\"?\")!r}  Tempo: {d.get(\"total_duration\",0)}ms')" 2>/dev/null || echo "$SANITY"

# 6. Stress test completo
echo ""
echo "[5/5] Rodando stress test completo..."
echo "================================================================"
$CARGO_BIN run --release -p iara-engine --bin stress_test --features rocm

# 7. Ollama comparison (opcional)
echo ""
echo "================================================================"
echo ""
read -p "Rodar comparação com Ollama? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./compare_ollama.sh
fi

echo ""
echo "🏁 Stress test completo!"
