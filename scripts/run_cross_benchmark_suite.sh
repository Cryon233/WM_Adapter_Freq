#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda activation script not found: $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "${CONDA_ENV:-wm-a100}"
if [[ ! -f ./env_jepa.sh ]]; then
    echo "Required JEPA-WM environment file not found: $ROOT/env_jepa.sh" >&2
    exit 1
fi
source ./env_jepa.sh
if [[ -f ./env_libero.sh ]]; then
    source ./env_libero.sh
fi

if [[ -n "${GPUS:-}" ]]; then
    export GPUS
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export GPUS="$CUDA_VISIBLE_DEVICES"
else
    export GPUS="0,1,2,3"
fi
unset CUDA_VISIBLE_DEVICES
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export DOWNLOAD_ASSETS=0
export FORCE_ASSETS=0

exec python scripts/launch_cross_benchmark_suite.py "$@"
