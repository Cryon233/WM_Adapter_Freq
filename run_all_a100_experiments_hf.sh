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
FORCE_ASSETS="${FORCE_ASSETS:-0}"
KEEP_ASSET_ZIPS="${KEEP_ASSET_ZIPS:-1}"
HF_ENDPOINT_PRIMARY="${HF_ENDPOINT_PRIMARY:-https://hf-mirror.com}"
HF_ENDPOINT_FALLBACK="${HF_ENDPOINT_FALLBACK:-https://huggingface.co}"
HF_REPO_ID="${HF_REPO_ID:-robocasa/robocasa-assets}"

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

ROBOCASA_ASSET_ROOT="$PROJECT_ROOT/third_party/robocasa/robocasa/models/assets"
ROBOCASA_ASSET_ZIPS="$PROJECT_ROOT/storage/robocasa-assets-zips"
ROBOCASA_ASSET_MARKER="$ROBOCASA_ASSET_ROOT/.hf_robocasa_assets_complete"

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

# env_jepa.sh may be configured for a single-GPU workstation. The launcher
# itself must see all four GPUs; child processes are pinned individually.
unset CUDA_VISIBLE_DEVICES
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

validate_robocasa_assets() {
    python - "$ROBOCASA_ASSET_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = [
    root / "textures",
    root / "fixtures",
    root / "objects" / "objaverse",
    root / "objects" / "aigen_objs",
    root / "generative_textures",
]

bad = []
for path in required:
    if not path.is_dir():
        bad.append(f"missing directory: {path}")
        continue
    try:
        next(item for item in path.rglob("*") if item.is_file())
    except StopIteration:
        bad.append(f"empty directory: {path}")

if bad:
    raise SystemExit("\n".join(bad))

print("RoboCasa asset directories are present and non-empty.")
PY
}

ensure_huggingface_hub() {
    if python -c 'import huggingface_hub' >/dev/null 2>&1; then
        return 0
    fi

    log "Installing huggingface_hub from the Aliyun PyPI mirror."
    python -m pip install \
        "huggingface_hub>=0.25" \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --timeout 300
}

download_hf_assets_from_endpoint() {
    local endpoint="$1"

    HF_ENDPOINT="$endpoint" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ETAG_TIMEOUT=60 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    python - \
        "$ROBOCASA_ASSET_ZIPS" \
        "$HF_REPO_ID" \
        "$FORCE_ASSETS" \
        "$endpoint" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

download_root = Path(sys.argv[1]).expanduser().resolve()
repo_id = sys.argv[2]
force = sys.argv[3] == "1"
endpoint = sys.argv[4]

# SHA-256 and byte sizes from the Hugging Face LFS pointers.
expected = {
    "textures.zip": (
        543_180_421,
        "ba3a4dcef96e199cbd601c7cdb7eb46077fb6672611b7ac74c25a191bf0be392",
    ),
    "fixtures.zip": (
        227_181_940,
        "e7bb86e9e0ef130de78200a0a7465dfb0161ed3956c65d8b3f36bdff3746c7a8",
    ),
    "objaverse.zip": (
        2_163_884_721,
        "66a7eebef3bac855301964d35d905fc2cb44d691f007f97af08d96e83db5cc08",
    ),
    "aigen_objs.zip": (
        5_773_223_039,
        "1f88792322ccce79f775e88482626c324c6e07d002adf23320a124b72764dd19",
    ),
    "generative_textures.zip": (
        1_184_675_536,
        "bc05095af306222d9a799a9b0b5b16662aa546998622ee8fd3072566ea4c31ec",
    ),
}

download_root.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def valid(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    print(f"Verifying SHA-256: {path.name}", flush=True)
    return sha256_file(path) == digest

from huggingface_hub import hf_hub_download

print(f"Hugging Face endpoint: {endpoint}", flush=True)
print(f"Dataset repository: {repo_id}", flush=True)

for filename, (size, digest) in expected.items():
    destination = download_root / filename

    if not force and valid(destination, size, digest):
        print(f"Already complete: {destination}", flush=True)
        continue

    if destination.exists() and destination.stat().st_size == 0:
        destination.unlink()

    print(f"\nDownloading {filename} ({size / 1e9:.2f} GB)", flush=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(download_root),
            force_download=force,
        )
    ).resolve()

    if downloaded != destination.resolve():
        destination = downloaded

    if not valid(destination, size, digest):
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(f"Integrity verification failed: {filename}")

    print(f"Verified: {destination}", flush=True)

print("\nAll five RoboCasa archives downloaded and verified.", flush=True)
PY
}

extract_hf_assets() {
    python - \
        "$ROBOCASA_ASSET_ZIPS" \
        "$ROBOCASA_ASSET_ROOT" \
        "$ROBOCASA_ASSET_MARKER" \
        "$FORCE_ASSETS" <<'PY'
import json
import shutil
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

zip_root = Path(sys.argv[1]).resolve()
asset_root = Path(sys.argv[2]).resolve()
marker = Path(sys.argv[3]).resolve()
force = sys.argv[4] == "1"

archives = {
    "textures.zip": asset_root,
    "fixtures.zip": asset_root,
    "objaverse.zip": asset_root / "objects",
    "aigen_objs.zip": asset_root / "objects",
    "generative_textures.zip": asset_root,
}

targets = [
    asset_root / "textures",
    asset_root / "fixtures",
    asset_root / "objects" / "objaverse",
    asset_root / "objects" / "aigen_objs",
    asset_root / "generative_textures",
]

if force:
    for target in targets:
        if target.exists():
            print(f"Removing existing asset directory: {target}", flush=True)
            shutil.rmtree(target)
    marker.unlink(missing_ok=True)

asset_root.mkdir(parents=True, exist_ok=True)

def safe_extract(archive: ZipFile, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.infolist():
        candidate = (destination / member.filename).resolve()
        if candidate != root and root not in candidate.parents:
            raise RuntimeError(
                f"Unsafe archive member in {archive.filename}: {member.filename}"
            )
    archive.extractall(destination)

for filename, destination in archives.items():
    archive_path = zip_root / filename
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing archive: {archive_path}")

    print(f"Extracting {filename} -> {destination}", flush=True)
    try:
        with ZipFile(archive_path, "r") as archive:
            safe_extract(archive, destination)
    except BadZipFile as error:
        raise RuntimeError(f"Invalid zip archive: {archive_path}") from error

counts = {}
for target in targets:
    if not target.is_dir():
        raise RuntimeError(f"Expected asset directory was not created: {target}")
    count = sum(1 for item in target.rglob("*") if item.is_file())
    if count == 0:
        raise RuntimeError(f"Expected asset directory is empty: {target}")
    counts[str(target.relative_to(asset_root))] = count
    print(f"Validated {target}: {count} files", flush=True)

marker.write_text(
    json.dumps(
        {
            "source": "huggingface",
            "repo_id": "robocasa/robocasa-assets",
            "archives": sorted(archives),
            "file_counts": counts,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Asset completion marker written: {marker}", flush=True)
PY
}

if [[ "$DOWNLOAD_ASSETS" == "1" ]]; then
    log "Ensuring RoboCasa assets are available through Hugging Face."

    if [[ "$FORCE_ASSETS" != "1" && -s "$ROBOCASA_ASSET_MARKER" ]]; then
        if validate_robocasa_assets; then
            log "Reusing previously extracted RoboCasa assets."
        else
            log "Asset marker exists but validation failed; rebuilding assets."
            rm -f "$ROBOCASA_ASSET_MARKER"
        fi
    fi

    if [[ "$FORCE_ASSETS" == "1" || ! -s "$ROBOCASA_ASSET_MARKER" ]]; then
        ensure_huggingface_hub
        mkdir -p "$ROBOCASA_ASSET_ZIPS" "$ROBOCASA_ASSET_ROOT"

        # Clean up incomplete files left by the original Box downloader.
        rm -f \
            "$ROBOCASA_ASSET_ROOT/textures.zip" \
            "$ROBOCASA_ASSET_ROOT/fixtures.zip" \
            "$ROBOCASA_ASSET_ROOT/objects/objaverse.zip" \
            "$ROBOCASA_ASSET_ROOT/objects/aigen_objs.zip" \
            "$ROBOCASA_ASSET_ROOT/generative_textures.zip"

        log "Downloading from primary endpoint: $HF_ENDPOINT_PRIMARY"
        if ! download_hf_assets_from_endpoint "$HF_ENDPOINT_PRIMARY" \
            2>&1 | tee "$LOG_ROOT/download_hf_assets.log"; then
            log "Primary Hugging Face endpoint failed; retrying with $HF_ENDPOINT_FALLBACK"
            download_hf_assets_from_endpoint "$HF_ENDPOINT_FALLBACK" \
                2>&1 | tee -a "$LOG_ROOT/download_hf_assets.log"
        fi

        extract_hf_assets 2>&1 | tee "$LOG_ROOT/extract_hf_assets.log"
        validate_robocasa_assets

        if [[ "$KEEP_ASSET_ZIPS" != "1" ]]; then
            log "KEEP_ASSET_ZIPS=0: deleting downloaded archives."
            rm -f "$ROBOCASA_ASSET_ZIPS"/*.zip
        fi
    fi
else
    log "DOWNLOAD_ASSETS=0: validating existing RoboCasa assets."
    validate_robocasa_assets \
        || die "RoboCasa assets are missing; rerun with DOWNLOAD_ASSETS=1."
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
