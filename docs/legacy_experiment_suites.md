# Legacy experiment suites

`cross_backend_adapter_v1` is the active experiment protocol. The earlier
`cross_benchmark_v1` and `cross_benchmark_v2` suites, including the
`robocasa_articulated` condition, are retained only for provenance and result
inspection. Their caches, checkpoints, logs, state files, and results must not
be overwritten or deleted.

The active suite does not read or mutate either legacy artifact tree. A legacy
artifact may only be reused by future code after an explicit full-contract
validation; no such reuse is configured for `cross_backend_adapter_v1`.

The legacy launchers remain available for reproducibility, but are not part of
the active runbook. Use `scripts/run_cross_backend_adapter_v1.sh` for all new
experiments.
