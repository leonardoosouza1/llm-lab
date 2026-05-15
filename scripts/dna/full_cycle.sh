#!/bin/bash
# Full debug cycle: compress → restart engine with DBG → load → test → dump.
# Usage: ./full_cycle.sh <variant_label> [compress_flags...]
#   variant_label becomes filename suffix for log
set -e
source "$(dirname "$0")/env.sh"
LABEL="$1"
shift
LOG="/tmp/engine_${LABEL}.log"
OUT_DNA="$MODELS_DIR/qwen3-4b.dna"

# Backup current DNA if exists
[ -f "$OUT_DNA" ] && mv "$OUT_DNA" "$OUT_DNA.prev_${LABEL}"

# Compress
"$(dirname "$0")/compress.sh" "$MODELS_DIR/qwen3-4b" "$OUT_DNA" "$@"

# Start engine with debug
"$(dirname "$0")/engine_start.sh" "$LOG" "IARA_DBG_HIDDEN=1"

# Load + test
"$(dirname "$0")/load_test.sh" qwen3-4b "Qual a capital do Brasil?"

# Dumps
echo "=== DBGH dumps:"
"$(dirname "$0")/dump_dbg.sh" "$LOG"
