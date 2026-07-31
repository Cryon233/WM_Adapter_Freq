#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Full A100 experiment runner for WM_Adapter_Freq
#
# Pipeline:
#   1. Activate conda env
#   2. Patch/check feature-cache precision
#   3. Download RoboCasa kitchen assets (optional, enabled by default)
#   4. Build one shared feature cache on GPU 3
#   5. Train DCT / Token-MLP / LoRA concurrently on GPUs 0 / 1 / 2
#   6. Run 8 planning evaluations in two 4-GPU waves
#   7. Write a compact result summary
#
# Current fixed protocol:
#   RoboCasa PnPCounterTop / place, seed 42, protocol_v2
# ============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/control-frequency-wm}"
CONDA_ENV="${CONDA_ENV:-wm-a100}"
CONFIG="${CONFIG:-configs/experiment/robocasa_pilot.yaml}"

CACHE_BATCH="${CACHE_BATCH:-8}"
CACHE_WORKERS="${CACHE_WORKERS:-8}"

TRAIN_BATCH="${TRAIN_BATCH:-16}"
TRAIN_ACCUM="${TRAIN_ACCUM:-2}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"

PLAN_CHUNK="${PLAN_CHUNK:-32}"

DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-1}"
FORCE_CACHE="${FORCE_CACHE:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_PLAN="${FORCE_PLAN:-0}"

TASK_SLUG="${TASK_SLUG:-place}"
EVAL_SEED="${EVAL_SEED:-42}"
PROTOCOL_DIR="${PROTOCOL_DIR:-protocol_v2}"

CACHE="$PROJECT_ROOT/storage/feature_cache/jepa_wm_droid_robocasa_pilot.h5"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/jepa_wm_droid/robocasa"
RESULT_ROOT="$PROJECT_ROOT/outputs/jepa_wm_droid/robocasa/$PROTOCOL_DIR/$TASK_SLUG/seed_$EVAL_SEED"
LOG_ROOT="$PROJECT_ROOT/logs/full_a100_experiment"

mkdir -p "$LOG_ROOT" "$CHECKPOINT_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

die() {
    log "ERROR: $*"
    exit 1
}

cleanup_children() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        log "A command failed; terminating remaining child jobs."
        jobs -pr | xargs -r kill 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup_children EXIT INT TERM

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || die "conda is not available in PATH"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

cd "$PROJECT_ROOT"
[[ -f "$PROJECT_ROOT/env_jepa.sh" ]] || die "Missing env_jepa.sh"
source "$PROJECT_ROOT/env_jepa.sh"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

log "Project: $PROJECT_ROOT"
log "Conda env: $CONDA_ENV"
log "Config: $CONFIG"

python - <<'PY'
import os
from pathlib import Path
import torch

required = {
    "JEPA-WM checkpoint": os.environ.get("JEPA_WM_DROID_CKPT"),
    "DINOv3 checkpoint": os.environ.get("DINOV3_VITL16_CKPT"),
    "RoboCasa HDF5": os.environ.get("JEPAWM_ROBOCASA_HDF5"),
}
missing = []
for name, value in required.items():
    if not value or not Path(value).is_file() or Path(value).stat().st_size == 0:
        missing.append(f"{name}: {value}")
if missing:
    raise SystemExit("Missing required assets:\n" + "\n".join(missing))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() < 4:
    raise SystemExit(f"Need at least 4 visible GPUs, found {torch.cuda.device_count()}")

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
for i in range(4):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY

python - <<'PY'
from pathlib import Path

path = Path("src/wm_adapter/data/feature_cache.py")
text = path.read_text(encoding="utf-8")

patched_marker = 'if key in {"clean_prefix_tokens", "ood_prefix_tokens"}:'
if patched_marker not in text:
    old = '''    @staticmethod
    def _dtype_for(key: str) -> np.dtype[Any]:
        if key in FEATURE_KEYS:
            return np.dtype(np.float16)
        if key == "actions":
            return np.dtype(np.float32)
'''
    new = '''    @staticmethod
    def _dtype_for(key: str) -> np.dtype[Any]:
        # Prefix activations are before the final DINOv3 block/norm and can
        # exceed the finite fp16 range. Store them in float32.
        if key in {"clean_prefix_tokens", "ood_prefix_tokens"}:
            return np.dtype(np.float32)
        if key in {"clean_context_final_latent", "clean_future_latent"}:
            return np.dtype(np.float16)
        if key == "actions":
            return np.dtype(np.float32)
'''
    if old not in text:
        raise SystemExit(
            "Cannot apply feature-cache precision patch automatically: "
            "expected source block was not found."
        )
    text = text.replace(old, new)

text = text.replace(
    'CACHE_SCHEMA_VERSION = "jepa_wm_robocasa_feature_cache_v1"',
    'CACHE_SCHEMA_VERSION = "jepa_wm_robocasa_feature_cache_v2"',
)
path.write_text(text, encoding="utf-8")
print("Feature-cache precision patch is present.")
PY

if [[ -s "$CACHE" && "$FORCE_CACHE" != "1" ]]; then
    if ! python - "$CACHE" <<'PY'
import sys
import h5py
import numpy as np

path = sys.argv[1]
with h5py.File(path, "r") as f:
    ok = (
        bool(f.attrs.get("finalized", False))
        and f.attrs.get("schema_version") == "jepa_wm_robocasa_feature_cache_v2"
        and f["clean_prefix_tokens"].dtype == np.dtype(np.float32)
        and f["ood_prefix_tokens"].dtype == np.dtype(np.float32)
    )
raise SystemExit(0 if ok else 1)
PY
    then
        log "Existing cache is old or incompatible; deleting it."
        rm -f "$CACHE"
    fi
fi

if [[ "$FORCE_CACHE" == "1" ]]; then
    log "FORCE_CACHE=1: deleting existing cache."
    rm -f "$CACHE"
fi

if [[ ! -f third_party/robosuite/robosuite/macros.py ]]; then
    log "Creating robosuite macros.py"
    python third_party/robosuite/robosuite/scripts/setup_macros.py \
        2>&1 | tee "$LOG_ROOT/setup_robosuite_macros.log"
fi

if [[ ! -f third_party/robocasa/robocasa/macros.py ]]; then
    log "Creating robocasa macros.py"
    python third_party/robocasa/robocasa/scripts/setup_macros.py \
        2>&1 | tee "$LOG_ROOT/setup_robocasa_macros.log"
fi

if [[ "$DOWNLOAD_ASSETS" == "1" ]]; then
    ASSET_SCRIPT="third_party/robocasa/robocasa/scripts/download_kitchen_assets.py"
    [[ -f "$ASSET_SCRIPT" ]] || die "Missing RoboCasa asset downloader: $ASSET_SCRIPT"
    log "Ensuring RoboCasa kitchen assets are available."
    python "$ASSET_SCRIPT" 2>&1 | tee "$LOG_ROOT/download_kitchen_assets.log"
else
    log "DOWNLOAD_ASSETS=0: skipping kitchen asset download."
fi

if [[ ! -s "$CACHE" ]]; then
    log "GPU 3: building shared feature cache."
    CUDA_VISIBLE_DEVICES=3 \
    python scripts/build_feature_cache.py \
        --config "$CONFIG" \
        cache.encoder_batch_size="$CACHE_BATCH" \
        cache.num_workers="$CACHE_WORKERS" \
        2>&1 | tee "$LOG_ROOT/build_feature_cache.log"
else
    log "Reusing compatible feature cache: $CACHE"
fi

train_one() {
    local method="$1"
    local gpu="$2"
    local checkpoint="$CHECKPOINT_DIR/${method}_final.pt"
    local logfile="$LOG_ROOT/train_${method}.log"

    if [[ -s "$checkpoint" && "$FORCE_TRAIN" != "1" ]]; then
        log "Skipping training $method: checkpoint exists."
        return 0
    fi

    log "GPU $gpu: training $method."
    CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/train_adapter.py \
        --config "$CONFIG" \
        method="$method" \
        training.batch_size="$TRAIN_BATCH" \
        training.gradient_accumulation="$TRAIN_ACCUM" \
        training.num_workers="$TRAIN_WORKERS" \
        training.precision=bf16 \
        >"$logfile" 2>&1
}

declare -a TRAIN_PIDS=()
declare -a TRAIN_NAMES=()

for spec in "dct_adapter:0" "token_mlp:1" "lora:2"; do
    method="${spec%%:*}"
    gpu="${spec##*:}"
    train_one "$method" "$gpu" &
    TRAIN_PIDS+=("$!")
    TRAIN_NAMES+=("$method")
done

train_status=0
for i in "${!TRAIN_PIDS[@]}"; do
    if wait "${TRAIN_PIDS[$i]}"; then
        log "Training finished: ${TRAIN_NAMES[$i]}"
    else
        log "Training failed: ${TRAIN_NAMES[$i]}; see $LOG_ROOT/train_${TRAIN_NAMES[$i]}.log"
        train_status=1
    fi
done
[[ "$train_status" == "0" ]] || die "At least one training job failed."

for method in dct_adapter token_mlp lora; do
    [[ -s "$CHECKPOINT_DIR/${method}_final.pt" ]] \
        || die "Missing trained checkpoint: $CHECKPOINT_DIR/${method}_final.pt"
done

plan_one() {
    local method="$1"
    local domain="$2"
    local gpu="$3"
    local result="$RESULT_ROOT/$method/$domain/results.json"
    local logfile="$LOG_ROOT/plan_${method}_${domain}.log"

    if [[ -s "$result" && "$FORCE_PLAN" != "1" ]]; then
        log "Skipping planning $method/$domain: result exists."
        return 0
    fi

    log "GPU $gpu: planning method=$method domain=$domain"
    CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/plan.py \
        --config "$CONFIG" \
        method="$method" \
        domain="$domain" \
        planning.candidate_chunk_size="$PLAN_CHUNK" \
        >"$logfile" 2>&1
}

run_plan_wave() {
    local wave_name="$1"
    shift
    local specs=("$@")
    local -a pids=()
    local -a names=()
    local status=0

    log "Starting planning wave: $wave_name"

    for spec in "${specs[@]}"; do
        IFS=':' read -r method domain gpu <<<"$spec"
        plan_one "$method" "$domain" "$gpu" &
        pids+=("$!")
        names+=("${method}/${domain}")
    done

    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            log "Planning finished: ${names[$i]}"
        else
            log "Planning failed: ${names[$i]}"
            status=1
        fi
    done

    [[ "$status" == "0" ]] || die "Planning wave $wave_name failed."
}

run_plan_wave "1/2" \
    "base:clean:0" \
    "base:ood:1" \
    "dct_adapter:clean:2" \
    "dct_adapter:ood:3"

run_plan_wave "2/2" \
    "token_mlp:clean:0" \
    "token_mlp:ood:1" \
    "lora:clean:2" \
    "lora:ood:3"

python - "$RESULT_ROOT" "$LOG_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
log_root = Path(sys.argv[2])
methods = ("base", "dct_adapter", "token_mlp", "lora")
domains = ("clean", "ood")

rows = []
missing = []
for method in methods:
    for domain in domains:
        path = root / method / domain / "results.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": method,
                "domain": domain,
                "success_count": int(result["success_count"]),
                "total_episodes": int(result["total_episodes"]),
                "success_rate": float(result["success_rate"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "peak_cuda_memory_bytes": int(result["peak_cuda_memory_bytes"]),
            }
        )

if missing:
    raise SystemExit("Missing result files:\n" + "\n".join(missing))

lines = [
    "# Full A100 Experiment Summary",
    "",
    "| Method | Domain | Success | Rate | Time (h) | Peak CUDA (GiB) |",
    "|---|---|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['method']} | {row['domain']} | "
        f"{row['success_count']}/{row['total_episodes']} | "
        f"{100.0 * row['success_rate']:.1f}% | "
        f"{row['elapsed_seconds'] / 3600.0:.2f} | "
        f"{row['peak_cuda_memory_bytes'] / (1024 ** 3):.2f} |"
    )

summary = "\n".join(lines) + "\n"
destination = log_root / "summary.md"
destination.write_text(summary, encoding="utf-8")
print(summary)
print(f"Summary written: {destination}")
PY

log "All training and planning evaluations completed successfully."
log "Results: $RESULT_ROOT"
log "Summary: $LOG_ROOT/summary.md"

trap - EXIT INT TERM
