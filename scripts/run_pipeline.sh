#!/usr/bin/env bash
# Portable motion + appearance recipe with EDGS -> FPS initialization.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash scripts/run_pipeline.sh DATASET FLOW_NPZ NEW_RUN_ROOT [FRAME_ST=0] [FRAME_ED=300]"
  echo "Environment: GPU_ID=0, PYTHON_BIN=python, CUDA_HOME=/usr/local/cuda-11.8"
  exit 0
fi
if (( $# < 3 || $# > 5 )); then
  echo "Usage: bash scripts/run_pipeline.sh DATASET FLOW_NPZ NEW_RUN_ROOT [FRAME_ST] [FRAME_ED]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$1"
FLOW_PATH="$2"
RUN_ROOT="$3"
FRAME_ST="${4:-0}"
FRAME_ED="${5:-300}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1
for value in "${FRAME_ST}" "${FRAME_ED}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid integer: ${value}" >&2; exit 2; }
done
if [[ -e "${RUN_ROOT}" || -L "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite existing run root: ${RUN_ROOT}" >&2
  exit 2
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_inputs.py" \
  "${DATASET_ROOT}" "${FLOW_PATH}" --init-mode edgs --frame-st "${FRAME_ST}" --frame-ed "${FRAME_ED}"
mkdir -p "${RUN_ROOT}/logs"
echo "GPU=${CUDA_VISIBLE_DEVICES}; frames=[${FRAME_ST},${FRAME_ED}); output=${RUN_ROOT}"
TAOGS_RUN_ROOT_PRECHECKED=1 "${PYTHON_BIN}" -u "${REPO_ROOT}/train.py" \
  --stage all -s "${DATASET_ROOT}" -m "${RUN_ROOT}" \
  --frame_st "${FRAME_ST}" --frame_ed "${FRAME_ED}" \
  --flow_path "${FLOW_PATH}" --parallel_load \
  2>&1 | tee "${RUN_ROOT}/logs/train.log"
