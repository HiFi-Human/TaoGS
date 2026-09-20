#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc was not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

# setup.py imports torch while constructing CUDAExtension. Build against the
# active environment and never resolve a different torch during compilation.
"${PYTHON_BIN}" -c 'import torch; assert torch.version.cuda is not None, "Install CUDA-enabled PyTorch first (see README.md)"'
export MAX_JOBS="${MAX_JOBS:-4}"
for extension in simple-knn diff-gaussian-rasterization-taming fused-ssim; do
  "${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps \
    "${REPO_ROOT}/third_party/${extension}"
done
# TaoGS imports the bundled RoMa directly from third_party/RoMa. Its upstream
# setup.py also requests unrelated training/benchmark dependencies, so do not
# install that package or its requirements for TaoGS inference.
