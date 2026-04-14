#!/usr/bin/env bash
# =============================================================================
# bench.sh — IARA Inference Engine Benchmark
# Measures tok/s for different model configs on AMD RX 6750 XT
# =============================================================================
# Usage:
#   ./scripts/bench.sh                      # all configs
#   ./scripts/bench.sh bf16                 # BF16 only
#   ./scripts/bench.sh w8                   # W8 (INT8 weights) only
#   ./scripts/bench.sh q4k                  # GGUF Q4K only
#   IARA_BENCH_TOKENS=100 ./scripts/bench.sh # more tokens for better avg
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

export ROCM_PATH=/opt/rocm-7.2.1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=$ROOT/vendor/llama-cpp/lib:$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH

MODELS_DIR="${ROOT}/models"
BENCH_TOKENS="${IARA_BENCH_TOKENS:-50}"
PROMPT="${IARA_BENCH_PROMPT:-Tell me a short story about a robot learning to paint.}"

# IARA binary
BINARY="${ROOT}/target/release/iara-engine"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: $BINARY not found. Run: cargo build --release -p iara-engine --features rocm"
    exit 1
fi

# Section profiling helper — captures tok/s from log output
run_bench() {
    local model="$1"
    local label="$2"
    local extra_env="$3"

    if [ ! -d "$MODELS_DIR/$model" ] && [ ! -f "$MODELS_DIR/$model" ]; then
        echo "  SKIP: model not found: $model"
        return
    fi

    echo ""
    echo "--- [$label] model=$model ---"

    # Run IARA with IARA_SECTION_PROF=1 to get timing breakdown
    local tmplog=$(mktemp)
    env $extra_env \
        RUST_LOG=warn \
        IARA_SECTION_PROF=1 \
        "$BINARY" "$MODELS_DIR" 2>"$tmplog" &
    local pid=$!

    # Wait for server to start
    for i in $(seq 1 30); do
        sleep 1
        if curl -sf http://localhost:3000/health > /dev/null 2>&1; then
            break
        fi
    done

    # Load model
    if [ -d "$MODELS_DIR/$model" ]; then
        curl -sf -X POST http://localhost:3000/api/load-model \
            -H 'Content-Type: application/json' \
            -d "{\"model_name\": \"$model\"}" > /dev/null
    fi

    # Wait for model load
    for i in $(seq 1 60); do
        sleep 1
        local ready=$(curl -sf http://localhost:3000/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_ready','false'))" 2>/dev/null || echo "false")
        if [ "$ready" = "True" ] || [ "$ready" = "true" ]; then
            break
        fi
    done

    # Generate tokens and measure
    local t_start=$(date +%s%N)
    curl -sf -X POST http://localhost:3000/api/generate \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $BENCH_TOKENS, \"temperature\": 0}" > /dev/null
    local t_end=$(date +%s%N)

    local elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    local toks_per_s=$(echo "scale=1; $BENCH_TOKENS * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "?")

    echo "  RESULT: ${BENCH_TOKENS} tokens in ${elapsed_ms}ms = ${toks_per_s} tok/s"

    # Show section profile if available
    if grep -q "SECTION PROFILE" "$tmplog" 2>/dev/null; then
        echo "  Section profile:"
        grep -E "NORM|QKV|ROPE|ATTN|WO|MLP|TOTAL" "$tmplog" | head -10
    fi

    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    rm -f "$tmplog"
}

CONFIG="${1:-all}"

echo "=== IARA Benchmark === ($(date))"
echo "Hardware: AMD RX 6750 XT (GFX1032, RDNA2, 12GB, ~250 GB/s)"
echo "Physics:"
echo "  BF16 (Qwen2.5-1.5B, ~3GB): theoretical max = 83 tok/s"
echo "  W8   (INT8 quant, ~1.5GB): theoretical max = 167 tok/s"
echo "  Q4K  (GGUF Q4_K_M, ~1GB):  theoretical max = 250+ tok/s"
echo "  llama.cpp Q4_K_M reference: 163 tok/s (target: 200+)"
echo ""

case "$CONFIG" in
    bf16|all)
        run_bench "qwen2.5-1.5b" "BF16 native" ""
        ;&
    w8)
        if [ "$CONFIG" = "w8" ] || [ "$CONFIG" = "all" ]; then
            run_bench "qwen2.5-1.5b" "W8 INT8 (IARA_W8=1)" "IARA_W8=1"
        fi
        ;&
    q4k)
        if [ "$CONFIG" = "q4k" ] || [ "$CONFIG" = "all" ]; then
            run_bench "qwen2.5-1.5b-q4km.gguf" "Q4K GGUF" ""
        fi
        ;;
esac

echo ""
echo "=== Benchmark complete ==="
echo "Target: 200 tok/s (requires Q4K native path or W8 near-theoretical)"
