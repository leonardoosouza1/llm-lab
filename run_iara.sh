#!/bin/bash
# =============================================================
# run_iara.sh — V4: ROCm-Native LLVM/Clang (AMD RX 6750 XT)
# =============================================================

# 1. Limpeza de Travas
rm -f iara-engine/target/.package-cache

# 2. Caminhos Oficiais ROCm 6.4.4
export ROCM_PATH=/opt/rocm-6.4.4
export ROCM_LLVM_PATH=$ROCM_PATH/llvm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_LLVM_PATH/lib:$LD_LIBRARY_PATH

# 3. Forçar Bindgen a usar o LLVM do ROCm
# Isso evita o conflito entre Clang do sistema e Headers do ROCm
export LIBCLANG_PATH=$ROCM_LLVM_PATH/lib
export CLANG_PATH=$ROCM_LLVM_PATH/bin/clang
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $ROCM_LLVM_PATH/lib/clang/19/include -I$ROCM_PATH/include"

# 4. Overrides de Hardware
export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "🎯 Usando LLVM do ROCm: $ROCM_LLVM_PATH"
echo "🔧 GFX_OVERRIDE: $HSA_OVERRIDE_GFX_VERSION"

# 5. Execução (workspace root — passa diretório do modelo como argumento)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${2:-$SCRIPT_DIR/models/qwen3-8b}"
if [ "$1" == "release" ]; then
    cargo run --release -p iara-engine -- "$MODEL_DIR"
else
    cargo run -p iara-engine -- "$MODEL_DIR"
fi
