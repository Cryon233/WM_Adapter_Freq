#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/data/users/zhaoyanghe/control-frequency-wm}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-wm-a100}"
cd "$PROJECT_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export GPUS="${GPUS:-0,1,2,3}"
export DOWNLOAD_ASSETS=0
export FORCE_ASSETS=0
export PYTHONUNBUFFERED=1
python scripts/run_all_paper_experiments.py
