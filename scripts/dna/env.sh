#!/bin/bash
# Source-only env vars for DNA scripts. Use: source scripts/dna/env.sh
LLM_LAB_DIR="${LLM_LAB_DIR:-/home/leonardo/projects/LLM/llm-lab}"
export LD_LIBRARY_PATH="$LLM_LAB_DIR/vendor/llama-cpp/lib:/opt/rocm-7.2.1/lib:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCR_VISIBLE_DEVICES=0
export RUST_LOG="${RUST_LOG:-info}"
ENGINE_BIN="$LLM_LAB_DIR/target/release/iara-engine"
DNA_COMPRESS_BIN="$LLM_LAB_DIR/target/release/dna_compress"
DNA_INSPECT_BIN="$LLM_LAB_DIR/target/release/dna_inspect"
MODELS_DIR="$LLM_LAB_DIR/models"
ENGINE_LOG="${ENGINE_LOG:-/tmp/engine_dna.log}"
HOST="http://localhost:3000"

# Wait until /health responds.
wait_engine() { until curl -s "$HOST/health" 2>/dev/null | grep -q '"status"'; do sleep 2; done; }
# Wait until model_loading=false.
wait_loaded() { until [ "$(curl -s "$HOST/health" 2>/dev/null | jq -r '.model_loading')" = "false" ]; do sleep 3; done; }
# Health summary.
health() { curl -s "$HOST/health" | jq -r '.model, .model_vram_gb, .vram_free_gb' | paste -sd' | ' -; }
