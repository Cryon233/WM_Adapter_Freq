#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate wm-a100
source ./env_jepa.sh
if [[ -f ./env_libero.sh ]]; then
    source ./env_libero.sh
fi

unset CUDA_VISIBLE_DEVICES
export GPUS="${GPUS:-0,1,2,3}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export DOWNLOAD_ASSETS=0
export FORCE_ASSETS=0

exec python scripts/launch_cross_benchmark_suite.py "$@"
