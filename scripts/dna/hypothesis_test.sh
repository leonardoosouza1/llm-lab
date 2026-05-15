#!/bin/bash
# Run a hypothesis test on Qwen3-4B DNA: compress with given flags, run inference,
# emit short summary of V drift L2 + token throughput + first 60 chars output.
# Usage: ./hypothesis_test.sh <label> <dna_compress_flags...>
set -e
source "$(dirname "$0")/env.sh"
LABEL="$1"
shift
LOG="/tmp/hyp_${LABEL}.log"
OUT_DNA="$MODELS_DIR/qwen3-4b.dna"

pkill -9 -f iara-engine 2>/dev/null || true
sleep 2
[ -f "$OUT_DNA" ] && mv "$OUT_DNA" "${OUT_DNA}.before_${LABEL}"

echo "=== HYPOTHESIS: $LABEL ==="
echo "flags: $@"
"$DNA_COMPRESS_BIN" --input "$MODELS_DIR/qwen3-4b" --output "$OUT_DNA" "$@" 2>&1 | tail -2
SIZE=$(ls -l "$OUT_DNA" | awk '{print $5}')
echo "size: $(numfmt --to=iec $SIZE 2>/dev/null || echo $SIZE)"

"$(dirname "$0")/engine_start.sh" "$LOG" "IARA_DBG_HIDDEN=1" >/dev/null
curl -s -X POST "$HOST/api/load-model" -H 'Content-Type: application/json' -d '{"name":"qwen3-4b"}' > /dev/null
wait_loaded

# Quick chat-template test
RESP=$(curl -s -X POST "$HOST/api/generate" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","prompt":"<|im_start|>user\nQual a capital do Brasil?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","max_tokens":15,"temperature":0.0}' --max-time 60)
TOKSEC=$(echo "$RESP" | jq -r '.tokens_per_second')
TXT=$(echo "$RESP" | jq -r '.response' | head -c 80)
echo "throughput: $TOKSEC tok/s"
echo "response: $TXT"

# Extract V_drift
V_L2=$(grep 'DBGH \[L0.v_raw_postproj\]' "$LOG" | head -1 | grep -oE 'L2=[0-9.]+' | head -1)
ATTN_L2=$(grep 'DBGH \[L0.attn_out\]' "$LOG" | head -1 | grep -oE 'L2=[0-9.]+' | head -1)
PRE_FINAL_L2=$(grep 'DBGH \[pre_final_norm' "$LOG" | head -1 | grep -oE 'L2=[0-9.]+' | head -1)
echo "L0.v_raw: $V_L2 (safetensors=L2=0.9070)"
echo "L0.attn_out: $ATTN_L2 (safetensors=L2=1.8144)"
echo "pre_final_norm: $PRE_FINAL_L2 (safetensors=L2=2116)"
echo "--- end $LABEL ---"
