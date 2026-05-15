#!/bin/bash
# Quantize a HuggingFace model to GGUF Q4_K_M via llama.cpp pipeline.
# Runs convert_hf_to_gguf.py + llama-quantize. Uses vendored llama.cpp-turboquant-hip.
#
# Usage: ./quantize_q4km.sh <hf_model_dir> [output_path]
#   hf_model_dir: e.g. models/qwen3-4b
#   output_path:  defaults to <hf_dir>-q4km.gguf next to input
set -euo pipefail

LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_CPP="${LLAMA_CPP_DIR:-$LAB_DIR/../llama.cpp-turboquant-hip}"
QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
CONVERT_SCRIPT="$LLAMA_CPP/convert_hf_to_gguf.py"

# Build quantize if missing.
if [ ! -x "$QUANTIZE_BIN" ]; then
    echo "Building llama-quantize from $LLAMA_CPP..."
    mkdir -p "$LLAMA_CPP/build"
    (cd "$LLAMA_CPP/build" && \
        cmake -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
              -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
              -DCMAKE_BUILD_TYPE=Release .. >/dev/null 2>&1 && \
        make llama-quantize -j8 >/dev/null 2>&1)
fi

INPUT="${1:?usage: $0 <hf_model_dir> [output.gguf]}"
INPUT_ABS="$(cd "$INPUT" && pwd)"
BASE="$(basename "$INPUT_ABS")"
OUTPUT="${2:-$LAB_DIR/models/${BASE}-q4km.gguf}"
TMP_F16="$LAB_DIR/models/${BASE}-f16-tmp.gguf"

echo "[1/3] HF → fp16 GGUF: $INPUT_ABS → $TMP_F16"
python3 "$CONVERT_SCRIPT" --outtype f16 --outfile "$TMP_F16" "$INPUT_ABS" 2>&1 | tail -3

echo "[2/3] fp16 → Q4_K_M GGUF: $TMP_F16 → $OUTPUT"
"$QUANTIZE_BIN" "$TMP_F16" "$OUTPUT" Q4_K_M 2>&1 | tail -5

echo "[3/3] cleanup fp16 intermediate"
rm "$TMP_F16"

ls -lh "$OUTPUT"
echo "done. Engine loads via name '$BASE-q4km' (matches filename minus .gguf)."
