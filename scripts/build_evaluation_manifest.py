from __future__ import annotations

from wm_adapter.benchmarks.factory import build_benchmark
from wm_adapter.utils.reproducibility import load_experiment_config


def main() -> None:
    cfg = load_experiment_config()
    benchmark = build_benchmark(cfg)
    task = benchmark.resolve_task(strict=True)
    task_manifest = benchmark.write_task_manifest(task)
    evaluation = benchmark.build_evaluation_manifest(
        task,
        count=int(cfg.evaluation.num_episodes),
        seed=int(cfg.evaluation.eval_seed),
        appearance_seed=int(cfg.appearance.seed),
    )
    benchmark.write_evaluation_manifest(evaluation)
    print(
        "EVALUATION_MANIFEST_COMPLETE "
        f"task={task.task_key} seed={cfg.evaluation.eval_seed} "
        f"instances={len(evaluation['instances'])} "
        f"task_manifest_sha256={task_manifest['task_manifest_sha256']} "
        f"evaluation_manifest_sha256={evaluation['evaluation_manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
