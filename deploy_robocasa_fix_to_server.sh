#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="/home/zhaoyang/control-frequency-wm"
REMOTE_ROOT="/data/users/zhaoyanghe/control-frequency-wm"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <server-host> [ssh-user]" >&2
  echo "Example: $0 172.28.11.129 zhaoyanghe" >&2
  exit 2
fi

SERVER_HOST="$1"
SSH_USER="${2:-zhaoyanghe}"
REMOTE="${SSH_USER}@${SERVER_HOST}"

FILES=(
  "third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py"
  "third_party/jepa-wms/evals/simu_env_planning/planning/utils.py"
  "src/wm_adapter/planning/jepa_wm_planner.py"
  "src/wm_adapter/benchmarks/robocasa.py"
  "scripts/run_cross_backend_adapter_suite.py"
  "scripts/monitor_all_paper_experiments.py"
)

echo "[1/5] Checking local repository..."
[[ -d "${LOCAL_ROOT}/.git" ]] || {
  echo "Local Git repository not found: ${LOCAL_ROOT}" >&2
  exit 1
}
for file in "${FILES[@]}"; do
  [[ -f "${LOCAL_ROOT}/${file}" ]] || {
    echo "Missing local file: ${LOCAL_ROOT}/${file}" >&2
    exit 1
  }
done

cd "${LOCAL_ROOT}"
git diff --check
python3 -m py_compile "${FILES[@]}"

echo "[2/5] Checking that remote planning processes are stopped..."
# Bracketed characters prevent pgrep from matching this check command itself.
RUNNING="$(
  ssh "${REMOTE}" \
    "pgrep -af '[r]un_cross_backend_adapter_suite\.py|scripts/[p]lan\.py' || true"
)"
if [[ -n "${RUNNING}" ]]; then
  echo "Remote planning processes are still running:" >&2
  printf '%s\n' "${RUNNING}" >&2
  echo "Stop the suite before deploying." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${REMOTE_ROOT}/.deploy_backups/robocasa_planning_fix_${STAMP}"

echo "[3/5] Backing up the remote files and current Git diff..."
ssh "${REMOTE}" "bash -s" -- "${REMOTE_ROOT}" "${BACKUP_DIR}" "${FILES[@]}" <<'REMOTE_BACKUP'
set -euo pipefail
REMOTE_ROOT="$1"
BACKUP_DIR="$2"
shift 2
FILES=("$@")

[[ -d "${REMOTE_ROOT}/.git" ]] || {
  echo "Remote Git repository not found: ${REMOTE_ROOT}" >&2
  exit 1
}

mkdir -p "${BACKUP_DIR}/original"
cd "${REMOTE_ROOT}"

git status --short > "${BACKUP_DIR}/git-status-before.txt"
git diff --binary > "${BACKUP_DIR}/working-tree-before.patch"
git diff --cached --binary > "${BACKUP_DIR}/index-before.patch"

for file in "${FILES[@]}"; do
  [[ -f "${file}" ]] || {
    echo "Missing remote file: ${REMOTE_ROOT}/${file}" >&2
    exit 1
  }
  mkdir -p "${BACKUP_DIR}/original/$(dirname "${file}")"
  cp -a "${file}" "${BACKUP_DIR}/original/${file}"
done
REMOTE_BACKUP

echo "[4/5] Uploading the six patched files..."
cd "${LOCAL_ROOT}"
rsync -av --checksum --relative \
  "${FILES[@]/#/.\/}" \
  "${REMOTE}:${REMOTE_ROOT}/"

echo "[5/5] Validating the remote copy..."
ssh "${REMOTE}" "bash -s" -- "${REMOTE_ROOT}" "${BACKUP_DIR}" "${FILES[@]}" <<'REMOTE_VALIDATE'
set -euo pipefail
REMOTE_ROOT="$1"
BACKUP_DIR="$2"
shift 2
FILES=("$@")

cd "${REMOTE_ROOT}"
python3 -m py_compile "${FILES[@]}"
git diff --check
git status --short > "${BACKUP_DIR}/git-status-after.txt"
git diff --binary > "${BACKUP_DIR}/working-tree-after.patch"

echo
echo "Deployment complete."
echo "Remote backup: ${BACKUP_DIR}"
echo
echo "Changed files:"
git status --short -- "${FILES[@]}"
REMOTE_VALIDATE
