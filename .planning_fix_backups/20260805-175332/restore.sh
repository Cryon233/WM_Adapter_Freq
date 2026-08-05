#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-/home/zhaoyang/control-frequency-wm}"
install -D -m 0644 "$(dirname "$0")/original/third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py" "$REPO/third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py"
install -D -m 0644 "$(dirname "$0")/original/third_party/jepa-wms/evals/simu_env_planning/planning/utils.py" "$REPO/third_party/jepa-wms/evals/simu_env_planning/planning/utils.py"
install -D -m 0644 "$(dirname "$0")/original/src/wm_adapter/planning/jepa_wm_planner.py" "$REPO/src/wm_adapter/planning/jepa_wm_planner.py"
install -D -m 0644 "$(dirname "$0")/original/src/wm_adapter/benchmarks/robocasa.py" "$REPO/src/wm_adapter/benchmarks/robocasa.py"
install -D -m 0644 "$(dirname "$0")/original/scripts/run_cross_backend_adapter_suite.py" "$REPO/scripts/run_cross_backend_adapter_suite.py"
install -D -m 0644 "$(dirname "$0")/original/scripts/monitor_all_paper_experiments.py" "$REPO/scripts/monitor_all_paper_experiments.py"
echo "Restored planning source files."
