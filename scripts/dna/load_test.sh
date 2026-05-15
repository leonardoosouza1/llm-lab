#!/bin/bash
# Load a model and run test prompt(s). Reports speed + first 200 chars.
# Usage: ./load_test.sh <model_name> [prompt="oi"]
set -e
source "$(dirname "$0")/env.sh"
MODEL="${1:-qwen3-4b}"
PROMPT="${2:-oi}"

curl -s -X POST "$HOST/api/load-model" -H 'Content-Type: application/json' \
  -d "{\"name\":\"$MODEL\"}" | head -c 100; echo
wait_loaded
echo "loaded: $(health)"

echo "=== plain prompt ==="
time curl -s -X POST "$HOST/api/generate" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":15,\"temperature\":0.0}" --max-time 60 \
  | jq -r '.response, .tokens_per_second'

echo "=== chat template (Qwen3 think-off) ==="
CHAT="<|im_start|>user\n${PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
time curl -s -X POST "$HOST/api/generate" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"$CHAT\",\"max_tokens\":30,\"temperature\":0.1}" --max-time 60 \
  | jq -r '.response, .tokens_per_second'
