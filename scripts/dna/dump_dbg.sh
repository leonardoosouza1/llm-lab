#!/bin/bash
# Show IARA_DBG_HIDDEN dumps from engine log.
# Usage: ./dump_dbg.sh [log_path]
LOG="${1:-/tmp/engine_dna.log}"
grep DBGH "$LOG" | sed -E 's/.*DBGH /DBGH /'
