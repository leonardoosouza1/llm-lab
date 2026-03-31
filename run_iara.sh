#!/bin/bash
# =============================================================
# run_iara.sh — V5: ROCm-Native LLVM/Clang (AMD RX 6750 XT)
# =============================================================

set -e

# 1. Limpeza de Travas
rm -f iara-engine/target/.package-cache

# 2. Caminhos Oficiais ROCm 6.4.4
export ROCM_PATH=/opt/rocm-6.4.4
export ROCM_LLVM_PATH=$ROCM_PATH/llvm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_LLVM_PATH/lib:$LD_LIBRARY_PATH

# 3. Forçar Bindgen a usar o LLVM do ROCm
export LIBCLANG_PATH=$ROCM_LLVM_PATH/lib
export CLANG_PATH=$ROCM_LLVM_PATH/bin/clang
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $ROCM_LLVM_PATH/lib/clang/19/include -I$ROCM_PATH/include"

# 4. Overrides de Hardware
export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "🎯 Usando LLVM do ROCm: $ROCM_LLVM_PATH"
echo "🔧 GFX_OVERRIDE: $HSA_OVERRIDE_GFX_VERSION"

# 5. Execução (workspace root)
# Argumento opcional: diretório dos modelos (default: ./models)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
MODELS_DIR="${2:-$SCRIPT_DIR/models}"

CARGO_BIN="${HOME}/.cargo/bin/cargo"

if [ "$1" == "release" ]; then
    RUST_LOG=info "$CARGO_BIN" run --release -p iara-engine --features rocm -- "$MODELS_DIR" 
else
    RUST_LOG=info "$CARGO_BIN" run -p iara-engine --features rocm -- "$MODELS_DIR"
fi
