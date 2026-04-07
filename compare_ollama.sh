#!/bin/bash
# ============================================================================
# IARA vs Ollama — Head-to-Head Benchmark
# ============================================================================
# Prerequisites:
#   1. IARA server running:   ./run_iara.sh release
#   2. Model loaded:          curl -X POST localhost:3000/api/load-model \
#                               -H 'Content-Type: application/json' \
#                               -d '{"name":"qwen2.5-1.5b"}'
#   3. Ollama installed:      which ollama
# ============================================================================

set -euo pipefail

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
echo -e "${BOLD}          IARA vs Ollama — Head-to-Head Benchmark${NC}"
echo -e "${BOLD}================================================================${NC}"

# ── Check IARA server ──────────────────────────────────────────────────────
echo -e "\n${CYAN}[1] Checking IARA server...${NC}"
IARA_HEALTH=$(curl -s "$IARA_URL/health" 2>/dev/null || echo '{"status":"offline"}')
IARA_STATUS=$(echo "$IARA_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','offline'))" 2>/dev/null || echo "offline")
IARA_MODEL=$(echo "$IARA_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model','none'))" 2>/dev/null || echo "none")

if [ "$IARA_STATUS" != "ok" ] && [ "$IARA_STATUS" != "ready" ]; then
    echo -e "${RED}IARA server not running at $IARA_URL${NC}"
    echo "Start it: ./run_iara.sh release"
    exit 1
fi
echo -e "  Status: ${GREEN}$IARA_STATUS${NC}  Model: ${GREEN}$IARA_MODEL${NC}"

# ── Check/pull Ollama model ────────────────────────────────────────────────
echo -e "\n${CYAN}[2] Checking Ollama...${NC}"
if ! command -v ollama &>/dev/null; then
    echo -e "${RED}Ollama not installed${NC}"
    exit 1
fi

# Start ollama serve if not running
if ! curl -s "$OLLAMA_URL/api/tags" &>/dev/null; then
    echo "  Starting ollama serve..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Check if model exists
HAS_MODEL=$(ollama list 2>/dev/null | grep -c "$MODEL_OLLAMA" || true)
if [ "$HAS_MODEL" -eq 0 ]; then
    echo -e "  ${YELLOW}Pulling $MODEL_OLLAMA (this may take a few minutes)...${NC}"
    ollama pull "$MODEL_OLLAMA"
fi
echo -e "  Ollama: ${GREEN}ready${NC}  Model: ${GREEN}$MODEL_OLLAMA${NC}"

# ── Prompts ────────────────────────────────────────────────────────────────
PROMPTS=(
    "What is artificial intelligence?"
    "Explain how transformers work in neural networks."
    "Write a short poem about the ocean and stars."
)

# ── Warmup ─────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}[3] Warmup...${NC}"
curl -s -X POST "$IARA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_IARA\",\"prompt\":\"hi\",\"max_tokens\":1}" > /dev/null 2>&1 || true

curl -s -X POST "$OLLAMA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_OLLAMA\",\"prompt\":\"hi\",\"stream\":false}" > /dev/null 2>&1 || true

echo "  Done."

# ── Benchmark function ─────────────────────────────────────────────────────
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
    d = json.loads('''$result''')
    eval_count = d.get('eval_count', 0)
    eval_dur_ms = d.get('eval_duration', 0)
    total_ms = d.get('total_duration', 0)
    tok_s = eval_count / (eval_dur_ms / 1000.0) if eval_dur_ms > 0 else 0
    print(f'{eval_count} {total_ms} {tok_s:.1f}')
except:
    print('0 0 0')
" 2>/dev/null || echo "0 0 0"
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
    d = json.loads('''$result''')
    eval_count = d.get('eval_count', 0)
    eval_dur_ns = d.get('eval_duration', 0)
    total_ns = d.get('total_duration', 0)
    tok_s = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns > 0 else 0
    total_ms = total_ns / 1e6
    print(f'{eval_count} {total_ms:.0f} {tok_s:.1f}')
except:
    print('0 0 0')
" 2>/dev/null || echo "0 0 0"
}

# ── Run benchmarks ─────────────────────────────────────────────────────────
echo -e "\n${CYAN}[4] Running benchmarks ($NUM_RUNS runs per prompt, max_tokens=$MAX_TOKENS)...${NC}"
echo ""

printf "${BOLD}%-45s %8s %10s %10s %10s %10s${NC}\n" \
    "Prompt" "Engine" "Tokens" "Latency" "tok/s" "Run"
printf "%s\n" "$(printf '=%.0s' {1..100})"

declare -a IARA_TPS_ALL
declare -a OLLAMA_TPS_ALL

for prompt in "${PROMPTS[@]}"; do
    short_prompt="${prompt:0:40}..."

    for run in $(seq 1 $NUM_RUNS); do
        # IARA
        read -r i_tok i_ms i_tps <<< $(run_iara "$prompt" $MAX_TOKENS)
        IARA_TPS_ALL+=("$i_tps")
        printf "%-45s %8s %8s %8sms %8s %8s\n" \
            "$short_prompt" "IARA" "$i_tok" "$i_ms" "$i_tps" "$run/$NUM_RUNS"

        # Ollama
        read -r o_tok o_ms o_tps <<< $(run_ollama "$prompt")
        OLLAMA_TPS_ALL+=("$o_tps")
        printf "%-45s %8s %8s %8sms %8s %8s\n" \
            "" "Ollama" "$o_tok" "$o_ms" "$o_tps" "$run/$NUM_RUNS"
    done
    echo ""
done

# ── Summary ────────────────────────────────────────────────────────────────
echo -e "${BOLD}================================================================${NC}"
echo -e "${BOLD}                        SUMMARY${NC}"
echo -e "${BOLD}================================================================${NC}"

# Average tok/s
IARA_AVG=$(python3 -c "
vals = [float(x) for x in '${IARA_TPS_ALL[*]}'.split() if float(x) > 0]
print(f'{sum(vals)/len(vals):.1f}' if vals else '0.0')
" 2>/dev/null || echo "0.0")

OLLAMA_AVG=$(python3 -c "
vals = [float(x) for x in '${OLLAMA_TPS_ALL[*]}'.split() if float(x) > 0]
print(f'{sum(vals)/len(vals):.1f}' if vals else '0.0')
" 2>/dev/null || echo "0.0")

echo ""
printf "  %-20s %12s\n" "Engine" "Avg tok/s"
printf "  %s\n" "$(printf '-%.0s' {1..35})"
printf "  %-20s %12s\n" "IARA" "$IARA_AVG"
printf "  %-20s %12s\n" "Ollama" "$OLLAMA_AVG"

RATIO=$(python3 -c "
i=float('$IARA_AVG'); o=float('$OLLAMA_AVG')
if o > 0: print(f'{i/o:.2f}x')
elif i > 0: print('IARA only')
else: print('N/A')
" 2>/dev/null || echo "N/A")

echo ""
echo -e "  IARA / Ollama ratio: ${BOLD}$RATIO${NC}"

# VRAM comparison
echo ""
echo -e "  ${CYAN}VRAM Usage:${NC}"
IARA_VRAM=$(echo "$IARA_HEALTH" | python3 -c "
import json,sys
d=json.load(sys.stdin)
used=d.get('vram_total_gb',0)-d.get('vram_free_gb',0)
print(f'{used:.2f}')
" 2>/dev/null || echo "?")
echo "    IARA:   ${IARA_VRAM} GB"

OLLAMA_PS=$(ollama ps 2>/dev/null || echo "")
echo "    Ollama: $(echo "$OLLAMA_PS" | tail -1)"

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
