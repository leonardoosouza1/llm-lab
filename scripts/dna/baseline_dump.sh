#!/bin/bash
# Dump baseline DBGH for a model. Run twice (safetensors vs DNA) to compare.
# Usage: ./baseline_dump.sh <model_name> <variant_suffix>
#   First run captures whatever model is currently in models/ for that name.
set -e
source "$(dirname "$0")/env.sh"
MODEL="$1"
SUFFIX="$2"
LOG="/tmp/baseline_${MODEL}_${SUFFIX}.log"

"$(dirname "$0")/engine_start.sh" "$LOG" "IARA_DBG_HIDDEN=1"

curl -s -X POST "$HOST/api/load-model" -H 'Content-Type: application/json' \
  -d "{\"name\":\"$MODEL\"}" > /dev/null
wait_loaded
echo "loaded $MODEL: $(health)"

# Single token generation triggers all dumps.
curl -s -X POST "$HOST/api/generate" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"oi\",\"max_tokens\":1,\"temperature\":0.0}" --max-time 120 \
  | jq -r '.response, .tokens_per_second'

echo "=== DBGH dumps ($MODEL $SUFFIX):"
"$(dirname "$0")/dump_dbg.sh" "$LOG"
