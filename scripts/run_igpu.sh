#!/bin/bash
# =============================================================
# run_igpu.sh — Intel iGPU (Iris Plus Gen11, i7-1065G7, UMA)
# =============================================================
# Pré-requisito:
#   sudo apt install -y intel-opencl-icd ocl-icd-libopencl1
#
# Uso:
#   ./run_igpu.sh           # debug
#   ./run_igpu.sh release   # release (mais lento pra compilar, mais rápido pra rodar)
#   ./run_igpu.sh release /caminho/para/models
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Verifica driver OpenCL Intel
if [ ! -f /etc/OpenCL/vendors/intel.icd ]; then
    echo "[ERRO] Driver OpenCL Intel não instalado."
    echo "       Execute: sudo apt install -y intel-opencl-icd"
    exit 1
fi

# Diretório de modelos (segundo argumento, senão ./models)
MODELS_DIR="${2:-$SCRIPT_DIR/models}"

if [ ! -d "$MODELS_DIR" ]; then
    echo "[AVISO] models_dir não existe: $MODELS_DIR"
    echo "        Crie a pasta e adicione um modelo (ex: qwen2.5-0.5b/)"
fi

CARGO_BIN="${HOME}/.cargo/bin/cargo"

# cl3 usa dlopen("libOpenCL.so") por padrão — mas Ubuntu instala libOpenCL.so.1
# sem o symlink sem versão. A var OPENCL_DYLIB_PATH redireciona o dlopen.
export OPENCL_DYLIB_PATH=/lib/x86_64-linux-gnu/libOpenCL.so.1

echo "Backend: OpenCL (Intel iGPU UMA — zero-copy)"
echo "OPENCL_DYLIB_PATH: $OPENCL_DYLIB_PATH"
echo "models:  $MODELS_DIR"

if [ "$1" == "release" ]; then
    RUST_LOG=info "$CARGO_BIN" run --release \
        -p iara-engine \
        --no-default-features \
        --features opencl \
        -- "$MODELS_DIR"
else
    RUST_LOG=info "$CARGO_BIN" run \
        -p iara-engine \
        --no-default-features \
        --features opencl \
        -- "$MODELS_DIR"
fi
