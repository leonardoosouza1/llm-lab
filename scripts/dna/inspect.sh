#!/bin/bash
# Wrap dna_inspect with quick filters.
# Usage: ./inspect.sh <model.dna> [filter]
source "$(dirname "$0")/env.sh"
DNA="$1"
FILTER="${2:-}"
"$DNA_INSPECT_BIN" --input "$DNA" --filter "$FILTER" 2>&1 | tail -30
