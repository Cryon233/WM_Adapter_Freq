#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HOST="${DEPLOY_HOST:-zhaoyanghe@172.28.11.129}"
DEST="${DEPLOY_ROOT:-/data/users/zhaoyanghe/control-frequency-wm}"

ssh "$HOST" 'pkill -f "[s]cripts/plan.py" 2>/dev/null || true; pkill -f "[r]un_all_a100_experiments.sh" 2>/dev/null || true'

rsync -az --delete --info=progress2 \
  --exclude='.git/' \
  --exclude='*/.git/' \
  --exclude='storage/' \
  --exclude='outputs/' \
  --exclude='checkpoints/' \
  --exclude='logs/' \
  --exclude='data/' \
  --exclude='wandb/' \
  --exclude='.wandb/' \
  --exclude='runs/' \
  --exclude='lightning_logs/' \
  --exclude='third_party/robocasa/robocasa/models/assets/' \
  --exclude='third_party/jepa-wms/datasets/' \
  --exclude='third_party/jepa-wms/checkpoints/' \
  --exclude='third_party/dinov3/checkpoints/' \
  --exclude='third_party/dinov3/weights/' \
  --exclude='*.zip' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.tgz' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.pth' \
  --exclude='*.pth.tar' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  ./ "$HOST:$DEST/"

echo "Synchronized to $HOST:$DEST"
