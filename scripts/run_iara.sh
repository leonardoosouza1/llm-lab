#!/bin/bash
# =============================================================
# run_iara.sh — V5: ROCm-Native LLVM/Clang (AMD RX 6750 XT)
# =============================================================

set -e

# 1. Limpeza de Travas
rm -f iara-engine/target/.package-cache

# 2. ROCm paths — override ROCM_PATH para o seu install (default: /opt/rocm)
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_LLVM_PATH="${ROCM_LLVM_PATH:-$ROCM_PATH/llvm}"
# llama.cpp vendored libs + ROCm
# Script mora em scripts/ — ROOT_DIR é o workspace raiz
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH=$ROOT_DIR/vendor/llama-cpp/lib:$ROCM_PATH/lib:$ROCM_LLVM_PATH/lib:$LD_LIBRARY_PATH

# 3. Forçar Bindgen a usar o LLVM do ROCm
export LIBCLANG_PATH=$ROCM_LLVM_PATH/lib
export CLANG_PATH=$ROCM_LLVM_PATH/bin/clang
# clang resource-dir version varia por release do ROCm — auto-detecta, fallback 22
ROCM_CLANG_INC="$(ls -d "$ROCM_LLVM_PATH"/lib/clang/*/include 2>/dev/null | head -1)"
export BINDGEN_EXTRA_CLANG_ARGS="-isystem ${ROCM_CLANG_INC:-$ROCM_LLVM_PATH/lib/clang/22/include} -I$ROCM_PATH/include"

# 4. Overrides de Hardware
export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "🎯 Usando LLVM do ROCm: $ROCM_LLVM_PATH"
echo "🔧 GFX_OVERRIDE: $HSA_OVERRIDE_GFX_VERSION"

# 5. Execução (workspace root)
# Argumentos:
#   $1 = modo: release | debug | q4 | q4-release
#   $2 = diretório dos modelos (default: ./models)
SCRIPT_DIR="$ROOT_DIR"
cd "$SCRIPT_DIR"

CARGO_BIN="${HOME}/.cargo/bin/cargo"
MODE="${1:-debug}"

# Parse mode
case "$MODE" in
    release)
        MODELS_DIR="${2:-$SCRIPT_DIR/models}"
        RUST_LOG=info "$CARGO_BIN" run --release -p iara-engine --bin iara-engine --features rocm -- "$MODELS_DIR"
        ;;
    native|native-release)
        export IARA_NO_LLAMA=1
        MODELS_DIR="${2:-$SCRIPT_DIR/models}"
        echo "🔬 Backend nativo forçado (IARA_NO_LLAMA=1) — telemetria completa"
        RUST_LOG=info "$CARGO_BIN" run --release -p iara-engine --bin iara-engine --features rocm -- "$MODELS_DIR"
        ;;
    native-prof)
        export IARA_NO_LLAMA=1
        export IARA_SECTION_PROF=1
        MODELS_DIR="${2:-$SCRIPT_DIR/models}"
        echo "🔬 Backend nativo + section profiling (IARA_SECTION_PROF=1) — lento mas detalhado"
        RUST_LOG=info "$CARGO_BIN" run --release -p iara-engine --bin iara-engine --features rocm -- "$MODELS_DIR"
        ;;
    q4|q4-release)
        export IARA_QUANTIZE=1
        MODELS_DIR="${2:-$SCRIPT_DIR/models}"
        echo "🔧 Quantização Q4 habilitada (IARA_QUANTIZE=1)"
        echo "   Modelo será quantizado BF16→Q4 no carregamento (~4x menos VRAM)"
        RUST_LOG=info "$CARGO_BIN" run --release -p iara-engine --bin iara-engine --features rocm -- "$MODELS_DIR"
        ;;
    debug)
        MODELS_DIR="${2:-$SCRIPT_DIR/models}"
        RUST_LOG=info "$CARGO_BIN" run -p iara-engine --bin iara-engine --features rocm -- "$MODELS_DIR"
        ;;
    *)
        echo "Uso: $0 [release|debug|q4|native|native-prof] [models_dir]"
        echo "  release     — build otimizado, llama.cpp backend (~163 tok/s)"
        echo "  native      — backend nativo forçado (IARA_NO_LLAMA=1), telemetria completa"
        echo "  native-prof — native + section profiling por camada (mais lento, para diagnóstico)"
        echo "  q4          — build otimizado, pesos quantizados Q4"
        echo "  debug       — build debug"
        exit 1
        ;;
esac
