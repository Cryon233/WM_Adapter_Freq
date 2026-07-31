#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/control-frequency-wm}"
CONFIG="${CONFIG:-configs/experiment/robocasa_pilot.yaml}"
CACHE="$PROJECT_ROOT/storage/feature_cache/jepa_wm_droid_robocasa_pilot.h5"
LOG_DIR="$PROJECT_ROOT/logs/first_batch_training"

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env_jepa.sh"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$LOG_DIR"

echo "Project: $PROJECT_ROOT"
echo "Config:  $CONFIG"
echo "GPU:     $CUDA_VISIBLE_DEVICES"

if [[ ! -s "$CACHE" ]]; then
    echo "=== Building shared feature cache ==="
    python scripts/build_feature_cache.py \
        --config "$CONFIG" \
        2>&1 | tee "$LOG_DIR/build_feature_cache.log"
else
    echo "=== Reusing existing cache: $CACHE ==="
fi

METHODS=(dct_adapter token_mlp lora)

for method in "${METHODS[@]}"; do
    checkpoint="$PROJECT_ROOT/checkpoints/jepa_wm_droid/robocasa/${method}_final.pt"

    if [[ -s "$checkpoint" && "${FORCE:-0}" != "1" ]]; then
        echo "=== Skipping $method; checkpoint exists: $checkpoint ==="
        continue
    fi

    echo "=== Training $method ==="
    python scripts/train_adapter.py \
        --config "$CONFIG" \
        method="$method" \
        2>&1 | tee "$LOG_DIR/train_${method}.log"

    echo "=== Finished $method ==="
done

echo
echo "=== First training batch complete ==="
ls -lh "$PROJECT_ROOT/checkpoints/jepa_wm_droid/robocasa/"*_final.pt
