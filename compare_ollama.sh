#!/bin/bash
# ============================================================================
# IARA vs Ollama — Head-to-Head Benchmark (Sequential, GPU-safe)
# ============================================================================
# Testa IARA e Ollama SEPARADAMENTE para não competir por VRAM.
#
# Fluxo:
#   1. Roda benchmarks da IARA (servidor já deve estar rodando com modelo)
#   2. Para o servidor IARA / descarrega modelo → libera VRAM
#   3. Inicia Ollama com HSA_OVERRIDE para usar a GPU AMD
#   4. Roda benchmarks do Ollama
#   5. Para Ollama → libera VRAM
#   6. Mostra comparação
#
# Prerequisites:
#   1. IARA server running:   ./run_iara.sh release
#   2. Model loaded (run_stress.sh faz isso)
#   3. Ollama installed:      which ollama
# ============================================================================

set -euo pipefail

# ── ROCm env (necessário para Ollama usar a GPU AMD) ──────────────────────
export ROCM_PATH=/opt/rocm-6.4.4
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}

IARA_URL="http://localhost:3000"
OLLAMA_URL="http://localhost:11434"
MODEL_OLLAMA="qwen2.5:1.5b"
MODEL_IARA="qwen2.5-1.5b"
NUM_RUNS=3
MAX_TOKENS=50

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}================================================================${NC}"
echo -e "${BOLD}    IARA vs Ollama — Sequential GPU Benchmark${NC}"
echo -e "${BOLD}    (cada engine usa a GPU sozinha — sem competição)${NC}"
echo -e "${BOLD}================================================================${NC}"

# ── Prompts ────────────────────────────────────────────────────────────────
PROMPTS=(
    "What is artificial intelligence?"
    "Explain how transformers work in neural networks."
    "Write a short poem about the ocean and stars."
)

# ── Benchmark functions ───────────────────────────────────────────────────
run_iara() {
    local prompt="$1"
    local max_tok="$2"
    local result
    result=$(curl -s -X POST "$IARA_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_IARA\",\"prompt\":\"$prompt\",\"max_tokens\":$max_tok,\"temperature\":0.7}" \
        --max-time 120 2>/dev/null)

    if [ -z "$result" ]; then
        echo "0 0 0"
        return
    fi

    python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    eval_count = d.get('eval_count', 0)
    eval_dur_ms = d.get('eval_duration', 0)
    total_ms = d.get('total_duration', 0)
    tok_s = eval_count / (eval_dur_ms / 1000.0) if eval_dur_ms > 0 else 0
    print(f'{eval_count} {total_ms} {tok_s:.1f}')
except:
    print('0 0 0')
" <<< "$result" 2>/dev/null || echo "0 0 0"
}

run_ollama() {
    local prompt="$1"
    local result
    result=$(curl -s -X POST "$OLLAMA_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_OLLAMA\",\"prompt\":\"$prompt\",\"stream\":false,\"options\":{\"num_predict\":$MAX_TOKENS,\"temperature\":0.7}}" \
        --max-time 120 2>/dev/null)

    if [ -z "$result" ]; then
        echo "0 0 0"
        return
    fi

    python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    eval_count = d.get('eval_count', 0)
    eval_dur_ns = d.get('eval_duration', 0)
    total_ns = d.get('total_duration', 0)
    tok_s = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns > 0 else 0
    total_ms = total_ns / 1e6
    print(f'{eval_count} {total_ms:.0f} {tok_s:.1f}')
except:
    print('0 0 0')
" <<< "$result" 2>/dev/null || echo "0 0 0"
}

# ══════════════════════════════════════════════════════════════════════════
# FASE 1: BENCHMARK IARA (GPU exclusiva)
# ══════════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}━━━ FASE 1: IARA Benchmark ━━━${NC}"
echo -e "${CYAN}[1/6] Verificando servidor IARA...${NC}"

IARA_HEALTH=$(curl -s "$IARA_URL/health" 2>/dev/null || echo '{"status":"offline"}')
IARA_STATUS=$(echo "$IARA_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','offline'))" 2>/dev/null || echo "offline")
IARA_MODEL=$(echo "$IARA_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model','none'))" 2>/dev/null || echo "none")

if [ "$IARA_STATUS" != "ok" ]; then
    echo -e "${RED}❌ IARA server não está rodando ou modelo não carregado${NC}"
    echo -e "   Status: $IARA_STATUS"
    echo -e "   Rode primeiro: ./run_iara.sh release  +  ./run_stress.sh"
    exit 1
fi
echo -e "  Status: ${GREEN}$IARA_STATUS${NC}  Model: ${GREEN}$IARA_MODEL${NC}"

# Capturar VRAM da IARA
IARA_VRAM=$(echo "$IARA_HEALTH" | python3 -c "
import json,sys
d=json.load(sys.stdin)
total = d.get('vram_total_gb') or 0
free = d.get('vram_free_gb') or 0
print(f'{total - free:.2f}')
" 2>/dev/null || echo "?")

# Warmup IARA
echo -e "\n${CYAN}[2/6] Warmup IARA...${NC}"
curl -s -X POST "$IARA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_IARA\",\"prompt\":\"hi\",\"max_tokens\":1}" > /dev/null 2>&1 || true
echo "  Done."

# Rodar benchmarks IARA
echo -e "\n${CYAN}[3/6] Rodando benchmarks IARA ($NUM_RUNS runs × ${#PROMPTS[@]} prompts)...${NC}"

declare -a IARA_TPS_ALL
declare -a IARA_RESULTS

for prompt in "${PROMPTS[@]}"; do
    short_prompt="${prompt:0:45}..."
    for run in $(seq 1 $NUM_RUNS); do
        read -r i_tok i_ms i_tps <<< $(run_iara "$prompt" $MAX_TOKENS)
        IARA_TPS_ALL+=("$i_tps")
        IARA_RESULTS+=("$short_prompt|$i_tok|$i_ms|$i_tps|$run/$NUM_RUNS")
        printf "  %-48s %6s tok  %8sms  %8s tok/s  [%s]\n" \
            "$short_prompt" "$i_tok" "$i_ms" "$i_tps" "$run/$NUM_RUNS"
    done
done

echo -e "  ${GREEN}✅ IARA benchmark completo${NC}"

# ══════════════════════════════════════════════════════════════════════════
# FASE 2: LIBERAR VRAM + PREPARAR OLLAMA
# ══════════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}━━━ FASE 2: Preparando Ollama ━━━${NC}"
echo -e "${YELLOW}[4/6] Liberando VRAM...${NC}"

# Descarregar modelo IARA da VRAM (libera ~4GB)
echo "  Descarregando modelo IARA da VRAM..."
UNLOAD_RESULT=$(curl -s -X POST "$IARA_URL/api/unload-model" 2>/dev/null || echo "")
if [ -n "$UNLOAD_RESULT" ]; then
    UNLOAD_MSG=$(echo "$UNLOAD_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message','?'))" 2>/dev/null || echo "?")
    echo -e "  ${GREEN}✅ IARA: $UNLOAD_MSG${NC}"
else
    echo -e "  ${YELLOW}⚠  Não conseguiu descarregar modelo IARA (endpoint /api/unload-model não disponível?)${NC}"
    echo -e "  ${YELLOW}   VRAM da IARA pode afetar benchmark do Ollama${NC}"
fi
sleep 2  # dar tempo para VRAM ser liberada

# Parar Ollama se estava rodando antes (limpar VRAM)
if pgrep -x ollama > /dev/null 2>&1; then
    echo "  Parando Ollama anterior..."
    ollama stop "$MODEL_OLLAMA" 2>/dev/null || true
    sleep 1
fi

# Verificar Ollama instalado
if ! command -v ollama &>/dev/null; then
    echo -e "${RED}❌ Ollama não instalado. Instale: curl -fsSL https://ollama.com/install.sh | sh${NC}"
    echo ""
    echo "Resultados parciais (só IARA):"
    IARA_AVG=$(python3 -c "
vals = [float(x) for x in '${IARA_TPS_ALL[*]}'.split() if float(x) > 0]
print(f'{sum(vals)/len(vals):.1f}' if vals else '0.0')
" 2>/dev/null || echo "0.0")
    echo "  IARA avg: $IARA_AVG tok/s"
    exit 1
fi

# Iniciar Ollama com env ROCm (GPU AMD)
echo "  Iniciando Ollama com HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION..."
if ! curl -s "$OLLAMA_URL/api/tags" &>/dev/null; then
    HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION \
    ROCM_PATH=$ROCM_PATH \
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    echo "  Ollama PID: $OLLAMA_PID"

    # Esperar Ollama ficar pronto
    for i in $(seq 1 15); do
        if curl -s "$OLLAMA_URL/api/tags" &>/dev/null; then
            break
        fi
        if [ "$i" -eq 15 ]; then
            echo -e "${RED}❌ Ollama não iniciou em 15s${NC}"
            exit 1
        fi
        sleep 1
    done
else
    OLLAMA_PID=""
    echo "  Ollama já estava rodando."
fi

# Pull modelo se necessário
HAS_MODEL=$(ollama list 2>/dev/null | grep -c "$MODEL_OLLAMA" || true)
if [ "$HAS_MODEL" -eq 0 ]; then
    echo -e "  ${YELLOW}Baixando $MODEL_OLLAMA (pode demorar)...${NC}"
    HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION ollama pull "$MODEL_OLLAMA"
fi
echo -e "  Ollama: ${GREEN}ready${NC}  Model: ${GREEN}$MODEL_OLLAMA${NC}"

# Verificar se Ollama está usando GPU
echo "  Verificando aceleração GPU..."
OLLAMA_GPU=$(ollama ps 2>/dev/null | grep -i "gpu\|rocm\|100%" || echo "")
if [ -n "$OLLAMA_GPU" ]; then
    echo -e "  ${GREEN}✅ Ollama usando GPU${NC}"
else
    echo -e "  ${YELLOW}⚠  Ollama pode estar em CPU (verifique 'ollama ps' após warmup)${NC}"
fi

# ══════════════════════════════════════════════════════════════════════════
# FASE 3: BENCHMARK OLLAMA (GPU exclusiva)
# ══════════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}━━━ FASE 3: Ollama Benchmark ━━━${NC}"

# Warmup Ollama (carrega modelo na GPU)
echo -e "${CYAN}[5/6] Warmup Ollama (carregando na GPU)...${NC}"
curl -s -X POST "$OLLAMA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_OLLAMA\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" > /dev/null 2>&1 || true
echo "  Done."

# Checar GPU após warmup
echo "  Ollama ps:"
ollama ps 2>/dev/null | head -5 || true

# Rodar benchmarks Ollama
echo -e "\n${CYAN}[6/6] Rodando benchmarks Ollama ($NUM_RUNS runs × ${#PROMPTS[@]} prompts)...${NC}"

declare -a OLLAMA_TPS_ALL
declare -a OLLAMA_RESULTS

for prompt in "${PROMPTS[@]}"; do
    short_prompt="${prompt:0:45}..."
    for run in $(seq 1 $NUM_RUNS); do
        read -r o_tok o_ms o_tps <<< $(run_ollama "$prompt")
        OLLAMA_TPS_ALL+=("$o_tps")
        OLLAMA_RESULTS+=("$short_prompt|$o_tok|$o_ms|$o_tps|$run/$NUM_RUNS")
        printf "  %-48s %6s tok  %8sms  %8s tok/s  [%s]\n" \
            "$short_prompt" "$o_tok" "$o_ms" "$o_tps" "$run/$NUM_RUNS"
    done
done

echo -e "  ${GREEN}✅ Ollama benchmark completo${NC}"

# ══════════════════════════════════════════════════════════════════════════
# RESULTADOS
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}================================================================${NC}"
echo -e "${BOLD}              RESULTADOS — IARA vs Ollama${NC}"
echo -e "${BOLD}================================================================${NC}"
echo ""

# Tabela lado a lado
printf "${BOLD}%-48s %8s %10s %10s %10s %6s${NC}\n" \
    "Prompt" "Engine" "Tokens" "Latency" "tok/s" "Run"
printf "%s\n" "$(printf '═%.0s' {1..96})"

# Intercalar resultados por prompt
PROMPT_IDX=0
for prompt in "${PROMPTS[@]}"; do
    for run_idx in $(seq 0 $((NUM_RUNS - 1))); do
        IDX=$((PROMPT_IDX * NUM_RUNS + run_idx))

        IFS='|' read -r ip it im itps ir <<< "${IARA_RESULTS[$IDX]}"
        printf "%-48s ${GREEN}%8s${NC} %8s %8sms %8s %6s\n" \
            "$ip" "IARA" "$it" "$im" "$itps" "$ir"

        IFS='|' read -r op ot om otps orr <<< "${OLLAMA_RESULTS[$IDX]}"
        printf "%-48s ${CYAN}%8s${NC} %8s %8sms %8s %6s\n" \
            "" "Ollama" "$ot" "$om" "$otps" "$orr"
    done
    printf "%s\n" "$(printf '─%.0s' {1..96})"
    PROMPT_IDX=$((PROMPT_IDX + 1))
done

# Médias
IARA_AVG=$(python3 -c "
vals = [float(x) for x in '${IARA_TPS_ALL[*]}'.split() if float(x) > 0]
print(f'{sum(vals)/len(vals):.1f}' if vals else '0.0')
" 2>/dev/null || echo "0.0")

OLLAMA_AVG=$(python3 -c "
vals = [float(x) for x in '${OLLAMA_TPS_ALL[*]}'.split() if float(x) > 0]
print(f'{sum(vals)/len(vals):.1f}' if vals else '0.0')
" 2>/dev/null || echo "0.0")

RATIO=$(python3 -c "
i=float('$IARA_AVG'); o=float('$OLLAMA_AVG')
if o > 0: print(f'{i/o:.2f}x')
elif i > 0: print('IARA ∞')
else: print('N/A')
" 2>/dev/null || echo "N/A")

echo ""
echo -e "${BOLD}  ┌─────────────────────────────────────────┐${NC}"
printf "${BOLD}  │  %-20s %12s %6s │${NC}\n" "Engine" "Avg tok/s" "GPU"
echo -e "${BOLD}  ├─────────────────────────────────────────┤${NC}"
printf "  │  ${GREEN}%-20s${NC} %12s %6s │\n" "IARA" "$IARA_AVG" "ROCm"
printf "  │  ${CYAN}%-20s${NC} %12s %6s │\n" "Ollama" "$OLLAMA_AVG" "ROCm"
echo -e "${BOLD}  ├─────────────────────────────────────────┤${NC}"
printf "  │  IARA / Ollama: ${BOLD}%-23s${NC} │\n" "$RATIO"
echo -e "${BOLD}  └─────────────────────────────────────────┘${NC}"

echo ""
echo -e "  ${CYAN}VRAM Usage:${NC}"
echo "    IARA:   ~${IARA_VRAM} GB (medido antes do benchmark)"
OLLAMA_PS_LINE=$(ollama ps 2>/dev/null | tail -1 || echo "?")
echo "    Ollama: $OLLAMA_PS_LINE"

# Parar Ollama se nós que iniciamos
if [ -n "${OLLAMA_PID:-}" ]; then
    echo ""
    echo "  Parando Ollama (PID $OLLAMA_PID)..."
    ollama stop "$MODEL_OLLAMA" 2>/dev/null || true
    kill "$OLLAMA_PID" 2>/dev/null || true
fi

# Recarregar modelo IARA (para não deixar servidor vazio)
echo ""
echo "  Recarregando modelo IARA..."
curl -s -X POST "$IARA_URL/api/load-model" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$MODEL_IARA\"}" > /dev/null 2>&1 || true
echo -e "  ${GREEN}✅ Modelo IARA sendo recarregado em background${NC}"

echo ""
echo -e "${GREEN}🏁 Benchmark completo!${NC}"
echo -e "   Modelo: $MODEL_OLLAMA  |  Runs: $NUM_RUNS × ${#PROMPTS[@]} prompts  |  max_tokens: $MAX_TOKENS"
echo ""
