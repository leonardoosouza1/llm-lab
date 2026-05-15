#!/bin/bash
# Start iara-engine in background with DNA debug env. Kills existing first.
# Usage: ./engine_start.sh [LOG_PATH] [extra_env=val ...]
set -e
source "$(dirname "$0")/env.sh"
LOG="${1:-$ENGINE_LOG}"
shift || true
# Optional extra env vars passed as KEY=VAL args (e.g. IARA_DBG_HIDDEN=1)
EXTRA_ENV=""
for arg in "$@"; do EXTRA_ENV="$EXTRA_ENV $arg"; done

pkill -9 -f iara-engine 2>/dev/null || true
sleep 3
ss -lntp 2>/dev/null | grep -qE ":(3000|8080)" && { echo "ports still busy"; sleep 3; }

nohup env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES RUST_LOG=$RUST_LOG $EXTRA_ENV "$ENGINE_BIN" "$MODELS_DIR" > "$LOG" 2>&1 &
disown
PID=$!
echo "engine PID=$PID log=$LOG"
wait_engine
echo "ready: $(health)"
