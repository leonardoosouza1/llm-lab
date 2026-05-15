#!/bin/bash
# Compress safetensors → DNA. Kills engine first to free VRAM.
# Usage: ./compress.sh <input_dir_or_file> <output.dna> [flags...]
set -e
source "$(dirname "$0")/env.sh"
INPUT="$1"
OUTPUT="$2"
shift 2

pkill -9 -f iara-engine 2>/dev/null || true
sleep 2

echo "compressing $INPUT → $OUTPUT $*"
"$DNA_COMPRESS_BIN" --input "$INPUT" --output "$OUTPUT" "$@" 2>&1 | tail -5
ls -lh "$OUTPUT"
