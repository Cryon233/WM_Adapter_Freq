#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/run_cross_benchmark_suite.sh" \
    --config configs/experiment/cross_benchmark_v2.yaml "$@"
