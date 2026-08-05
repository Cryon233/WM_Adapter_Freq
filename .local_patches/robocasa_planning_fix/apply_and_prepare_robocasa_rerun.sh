#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/data/users/zhaoyanghe/control-frequency-wm}"
PATCHER="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/apply_robocasa_planning_fixes.py}"

cd "$REPO"

PID_FILE="logs/cross_backend_adapter_v1/runner.pid"
if [[ -f "$PID_FILE" ]]; then
  PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: suite runner is still active (pid=$PID)." >&2
    echo "Attach to the dashboard and press uppercase X, then rerun this script." >&2
    exit 2
  fi
fi

python3 "$PATCHER" --repo "$REPO" --check
python3 "$PATCHER" --repo "$REPO" --apply

STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="archive/robocasa_planning_protocol_v2_buggy_${STAMP}"
mkdir -p "$ARCHIVE/outputs" "$ARCHIVE/manifests" "$ARCHIVE/state"

for name in main ablations; do
  src="outputs/cross_backend_adapter_v1/$name"
  if [[ -e "$src" ]]; then
    mv "$src" "$ARCHIVE/outputs/$name"
  fi
done

for task in robocasa_reach robocasa_place; do
  src="outputs/cross_backend_adapter_v1/manifests/evaluation/$task"
  if [[ -e "$src" ]]; then
    mkdir -p "$ARCHIVE/manifests/evaluation"
    mv "$src" "$ARCHIVE/manifests/evaluation/$task"
  fi
done

for file in \
  logs/cross_backend_adapter_v1/state.json \
  logs/cross_backend_adapter_v1/runner.pid; do
  if [[ -e "$file" ]]; then
    mv "$file" "$ARCHIVE/state/$(basename "$file")"
  fi
done

python3 -m py_compile \
  third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py \
  third_party/jepa-wms/evals/simu_env_planning/planning/utils.py \
  src/wm_adapter/planning/jepa_wm_planner.py \
  src/wm_adapter/benchmarks/robocasa.py \
  scripts/run_cross_backend_adapter_suite.py \
  scripts/monitor_all_paper_experiments.py

git diff --check

echo
echo "Source fixes applied and old Planning artifacts archived."
echo "Archive: $REPO/$ARCHIVE"
echo
echo "Review changes:"
echo "  git diff -- \"${REPO}/third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py\""
echo "  git diff --stat"
echo
echo "Recommended next command (4-job smoke test first, not the full 198-job suite):"
echo "  See README_robocasa_planning_fixes_zh.md in the patch bundle."
