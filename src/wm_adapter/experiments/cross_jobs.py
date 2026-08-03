from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from wm_adapter.experiments.cross_benchmark import JobSpec
from wm_adapter.utils.reproducibility import resolve_path


def _task_identity(config_path: str) -> tuple[str, str]:
    cfg = OmegaConf.load(resolve_path(config_path))
    return str(cfg.benchmark.name), str(cfg.benchmark.task_key)


def _log(suite: Any, name: str) -> str:
    return str(resolve_path(str(suite.log_root)) / f"{name}.log")


def _main_cache(suite: Any, benchmark: str, task: str) -> Path:
    return resolve_path(str(suite.storage_root)) / benchmark / f"{task}.h5"


def _main_checkpoint(suite: Any, benchmark: str, task: str, method: str) -> Path:
    return resolve_path(str(suite.checkpoint_root)) / benchmark / task / f"{method}_final.pt"


def _main_result(
    suite: Any, benchmark: str, task: str, method: str, domain: str
) -> Path:
    return (
        resolve_path(str(suite.output_root))
        / "main"
        / benchmark
        / task
        / f"seed_{int(suite.main.eval_seed)}"
        / method
        / domain
        / "results.json"
    )


def build_job_graph(suite: Any, *, self_test: bool = False) -> list[JobSpec]:
    is_v2 = str(suite.suite_name) == "cross_benchmark_v2"
    ablation_phase = "HFRA ablations" if is_v2 else "DCT ablations"
    suite_config_path = str(
        suite.get("_suite_config_path", "configs/experiment/cross_benchmark_v2.yaml" if is_v2 else "configs/experiment/cross_benchmark_v1.yaml")
    )
    task_items = list(suite.tasks.items())
    methods = [str(value) for value in suite.methods]
    domains = [str(value) for value in suite.domains]
    if self_test:
        selected = {str(value) for value in suite.self_test.task_keys}
        task_items = [(key, value) for key, value in task_items if str(key) in selected]
        methods = [str(value) for value in suite.self_test.methods]
        domains = [str(value) for value in suite.self_test.domains]
    self_test_v2_training_overrides = (
        [
            f"training.max_optimizer_steps={int(suite.self_test.optimizer_steps)}",
            "training.warmup_steps=0",
            "training.num_workers=0",
            "training.microbatch_windows=1",
            "training.views_per_window=2",
            "training.gradient_accumulation=1",
        ]
        if self_test and is_v2
        else []
    )
    jobs: list[JobSpec] = []
    for task_key, config_value in task_items:
        config = str(config_value)
        benchmark, task = _task_identity(config)
        manifest = resolve_path(str(suite.manifest_root)) / "tasks" / f"{task}.json"
        evaluation_manifest = (
            resolve_path(str(suite.manifest_root)) / "evaluation" / f"{task}.json"
        )
        manifest_overrides = (
            [
                f"paths.task_manifest={manifest}",
                f"paths.evaluation_manifest={evaluation_manifest}",
            ]
            if self_test
            else []
        )
        jobs.append(
            JobSpec(
                job_id=f"preflight/{task}", phase="Task and resource preflight",
                benchmark=benchmark, task=task, command=(),
                log_path=_log(suite, f"preflight-{task}"), artifact_path=str(manifest),
                kind="task_manifest",
            )
        )
        cache = _main_cache(suite, benchmark, task)
        reuse_cache: tuple[str, ...] = ()
        if task in suite.reuse_sources and suite.reuse_sources[task].get("cache"):
            reuse_cache = (str(resolve_path(str(suite.reuse_sources[task].cache))),)
        cache_windows = int(suite.self_test.windows if self_test else suite.data.num_train_windows)
        cache_command = [
            sys.executable, "scripts/build_feature_cache.py", "--config", config,
            f"data.num_train_windows={cache_windows}", f"paths.feature_cache={cache}",
        ]
        if self_test:
            cache_command.extend(
                ["cache.num_workers=0", "cache.encoder_batch_size=1", *manifest_overrides]
            )
        jobs.append(
            JobSpec(
                job_id=f"cache/{task}", phase="Feature caches", benchmark=benchmark,
                task=task, command=tuple(cache_command),
                log_path=_log(suite, f"cache-{task}"), artifact_path=str(cache),
                kind="cache", required_count=cache_windows, reuse_sources=reuse_cache,
                dependencies=(f"preflight/{task}",),
            )
        )
        train_methods = [method for method in methods if method != "base"]
        for method in train_methods:
            checkpoint = _main_checkpoint(suite, benchmark, task, method)
            source = ()
            if task in suite.reuse_sources:
                configured = suite.reuse_sources[task].get("checkpoints", {}).get(method)
                if configured:
                    source = (str(resolve_path(str(configured))),)
            command = [
                sys.executable, "scripts/train_adapter.py", "--config", config,
                f"method={method}", f"paths.feature_cache={cache}",
                f"paths.method_checkpoint={checkpoint}",
            ]
            if self_test:
                if is_v2:
                    command.extend(
                        [
                            "suite_mode=self_test",
                            *self_test_v2_training_overrides,
                            *manifest_overrides,
                        ]
                    )
                else:
                    command.extend([
                        f"training.epochs={int(suite.self_test.epochs)}",
                        "training.num_workers=0", "training.batch_size=1",
                        "training.gradient_accumulation=1", *manifest_overrides,
                    ])
            jobs.append(
                JobSpec(
                    job_id=f"train/main/{task}/{method}",
                    phase="Main adapter training", benchmark=benchmark, task=task,
                    method=method, seed=42, command=tuple(command),
                    log_path=_log(suite, f"train-main-{task}-{method}"),
                    artifact_path=str(checkpoint), kind="checkpoint",
                    reuse_sources=source,
                    dependencies=(f"cache/{task}",),
                )
            )
        offline_windows = int(
            suite.self_test.offline_windows if self_test else suite.offline.windows
        )
        for method in methods:
            metrics = (
                resolve_path(str(suite.output_root)) / "offline" / "main" /
                benchmark / task / method / "metrics.json"
            )
            command = [
                sys.executable, "scripts/evaluate_offline_dynamics.py", "--config", config,
                f"method={method}", "domain=both", f"paths.feature_cache={cache}",
                f"offline.num_windows={offline_windows}",
                f"offline.seed={int(suite.offline.seed)}",
                f"offline.shuffle_seed={int(suite.offline.shuffle_seed)}",
                f"offline.output_directory={metrics.parent}",
            ]
            if method != "base":
                command.append(
                    f"paths.method_checkpoint={_main_checkpoint(suite, benchmark, task, method)}"
                )
            if self_test:
                command.extend(
                    [
                        "offline.num_workers=0",
                        "offline.batch_size=1",
                        "suite_mode=self_test",
                        *self_test_v2_training_overrides,
                        *manifest_overrides,
                    ]
                )
            jobs.append(
                JobSpec(
                    job_id=f"offline/main/{task}/{method}",
                    phase="Main offline evaluation", benchmark=benchmark, task=task,
                    method=method, command=tuple(command), log_path=_log(
                        suite, f"offline-main-{task}-{method}"
                    ), artifact_path=str(metrics), kind="offline",
                    required_count=offline_windows,
                    dependencies=(
                        (f"cache/{task}",)
                        if method == "base"
                        else (f"train/main/{task}/{method}",)
                    ),
                )
            )
        main_episodes = int(suite.self_test.episodes if self_test else suite.main.episodes)
        for method in methods:
            for domain in domains:
                result = _main_result(suite, benchmark, task, method, domain)
                source = ()
                if task in suite.reuse_sources and suite.reuse_sources[task].get("planning_root"):
                    source = (
                        str(
                            resolve_path(str(suite.reuse_sources[task].planning_root))
                            / method / domain / "results.json"
                        ),
                    )
                command = [
                    sys.executable, "scripts/plan.py", "--config", config,
                    f"method={method}", f"domain={domain}",
                    f"evaluation.num_episodes={main_episodes}",
                    f"evaluation.eval_seed={int(suite.main.eval_seed)}",
                    f"appearance.severity={float(suite.appearance.main_severity)}",
                    f"output.run_directory={result.parent}",
                    f"suite_mode={'self_test' if self_test else 'formal'}",
                    "suite.family=main", "suite.variant=full",
                ]
                if method != "base":
                    command.append(
                        f"paths.method_checkpoint={_main_checkpoint(suite, benchmark, task, method)}"
                    )
                if self_test:
                    command.extend(
                        [*self_test_v2_training_overrides, *manifest_overrides]
                    )
                jobs.append(
                    JobSpec(
                        job_id=f"planning/main/{task}/{method}/{domain}",
                        phase="Main closed-loop planning", benchmark=benchmark, task=task,
                        method=method, domain=domain, seed=int(suite.main.eval_seed),
                        severity=float(suite.appearance.main_severity), command=tuple(command),
                        log_path=_log(suite, f"plan-main-{task}-{method}-{domain}"),
                        artifact_path=str(result), kind="planning", required_count=main_episodes,
                        reuse_sources=source,
                        dependencies=(
                            (f"cache/{task}",)
                            if method == "base"
                            else (f"train/main/{task}/{method}",)
                        ),
                    )
                )

    if self_test:
        analysis = resolve_path(str(suite.output_root)) / "self_test_report.md"
        jobs.append(
            JobSpec(
                job_id="analysis/self_test", phase="Final analysis", benchmark="all",
                task="all", command=(
                    sys.executable, "scripts/analyze_cross_benchmark_suite.py",
                    "--suite-config", suite_config_path,
                    "--state", str(suite.state_path), "--output", str(suite.analysis_root),
                    "--self-test",
                ), log_path=_log(suite, "analysis-self-test"),
                artifact_path=str(analysis), kind="analysis",
            )
        )
        return jobs

    for task_key in suite.stability.tasks:
        config = str(suite.tasks[str(task_key)])
        benchmark, task = _task_identity(config)
        for method in suite.stability.methods:
            for train_seed in suite.stability.seeds:
                method = str(method)
                train_seed = int(train_seed)
                checkpoint = (
                    resolve_path(str(suite.checkpoint_root)) / "stability" / benchmark /
                    task / method / f"seed_{train_seed}_final.pt"
                )
                main_checkpoint = _main_checkpoint(suite, benchmark, task, method)
                reuse = (str(main_checkpoint),) if train_seed == 42 else ()
                jobs.append(
                    JobSpec(
                        job_id=f"train/stability/{task}/{method}/seed_{train_seed}",
                        phase="Training-seed stability", benchmark=benchmark, task=task,
                        method=method, seed=train_seed, command=(
                            sys.executable, "scripts/train_adapter.py", "--config", config,
                            f"method={method}", f"training.seed={train_seed}",
                            f"paths.feature_cache={_main_cache(suite, benchmark, task)}",
                            f"paths.method_checkpoint={checkpoint}",
                        ), log_path=_log(suite, f"train-stability-{task}-{method}-seed{train_seed}"),
                        artifact_path=str(checkpoint), kind="checkpoint", reuse_sources=reuse,
                        dependencies=(f"cache/{task}",),
                    )
                )
                result = (
                    resolve_path(str(suite.output_root)) / "stability" / benchmark / task /
                    method / f"train_seed_{train_seed}" / "ood" / "results.json"
                )
                main_result = _main_result(suite, benchmark, task, method, "ood")
                result_reuse = (str(main_result),) if train_seed == 42 else ()
                jobs.append(
                    JobSpec(
                        job_id=f"planning/stability/{task}/{method}/seed_{train_seed}",
                        phase="Training-seed stability", benchmark=benchmark, task=task,
                        method=method, domain="ood", seed=train_seed, severity=1.0,
                        command=(
                            sys.executable, "scripts/plan.py", "--config", config,
                            f"method={method}", "domain=ood",
                            f"paths.method_checkpoint={checkpoint}",
                            f"training.seed={train_seed}",
                            f"evaluation.num_episodes={int(suite.stability.episodes)}",
                            f"evaluation.eval_seed={int(suite.main.eval_seed)}",
                            "appearance.severity=1.0", f"output.run_directory={result.parent}",
                            "suite.family=stability", f"suite.variant=train_seed_{train_seed}",
                        ), log_path=_log(suite, f"plan-stability-{task}-{method}-seed{train_seed}"),
                        artifact_path=str(result), kind="planning",
                        required_count=int(suite.stability.episodes), reuse_sources=result_reuse,
                        dependencies=(
                            f"train/stability/{task}/{method}/seed_{train_seed}",
                        ),
                    )
                )

    for task_key in suite.severity.tasks:
        config = str(suite.tasks[str(task_key)])
        benchmark, task = _task_identity(config)
        for method_value in suite.severity.methods:
            method = str(method_value)
            for severity_value in suite.severity["values"]:
                severity = float(severity_value)
                severity_name = str(severity).replace(".", "p")
                result = (
                    resolve_path(str(suite.output_root)) / "severity" / benchmark / task /
                    method / f"severity_{severity_name}" / "results.json"
                )
                reuse = (
                    (str(_main_result(suite, benchmark, task, method, "ood")),)
                    if severity == 1.0 else ()
                )
                command = [
                    sys.executable, "scripts/plan.py", "--config", config,
                    f"method={method}", "domain=ood",
                    f"evaluation.num_episodes={int(suite.severity.episodes)}",
                    f"evaluation.eval_seed={int(suite.severity.seed)}",
                    f"appearance.severity={severity}", f"output.run_directory={result.parent}",
                    "suite.family=severity", f"suite.variant=severity_{severity_name}",
                ]
                if method != "base":
                    command.append(
                        f"paths.method_checkpoint={_main_checkpoint(suite, benchmark, task, method)}"
                    )
                jobs.append(
                    JobSpec(
                        job_id=f"planning/severity/{task}/{method}/{severity_name}",
                        phase="OOD severity", benchmark=benchmark, task=task, method=method,
                        domain="ood", seed=int(suite.severity.seed), severity=severity,
                        command=tuple(command),
                        log_path=_log(suite, f"plan-severity-{task}-{method}-{severity_name}"),
                        artifact_path=str(result), kind="planning",
                        required_count=int(suite.severity.episodes), reuse_sources=reuse,
                        dependencies=(
                            (f"cache/{task}",)
                            if method == "base"
                            else (f"train/main/{task}/{method}",)
                        ),
                    )
                )

    for task_key in suite.ablations.tasks:
        config = str(suite.tasks[str(task_key)])
        benchmark, task = _task_identity(config)
        for variant_value, spec in suite.ablations.variants.items():
            variant = str(variant_value)
            ablation_method = str(spec.get("method", "dct_adapter"))
            checkpoint = (
                resolve_path(str(suite.checkpoint_root)) / "ablations" / benchmark /
                task / f"{variant}_final.pt"
            )
            main_checkpoint = _main_checkpoint(suite, benchmark, task, ablation_method)
            checkpoint_reuse = (str(main_checkpoint),) if variant == "full" else ()
            training_overrides = [
                f"method={ablation_method}",
                f"method_configs.{ablation_method}={spec.method_config}",
            ]
            if not is_v2:
                training_overrides.extend(
                    (
                        f"training.canonical_weight={float(spec.canonical_weight)}",
                        f"training.dynamics_weight={float(spec.dynamics_weight)}",
                    )
                )
            jobs.append(
                JobSpec(
                    job_id=f"train/ablation/{task}/{variant}", phase=ablation_phase,
                    benchmark=benchmark, task=task, method=ablation_method, variant=variant,
                    seed=42, command=tuple((
                        sys.executable, "scripts/train_adapter.py", "--config", config,
                        *training_overrides,
                        f"paths.feature_cache={_main_cache(suite, benchmark, task)}",
                        f"paths.method_checkpoint={checkpoint}",
                    )), log_path=_log(suite, f"train-ablation-{task}-{variant}"),
                    artifact_path=str(checkpoint), kind="checkpoint",
                    reuse_sources=checkpoint_reuse,
                    dependencies=(f"cache/{task}",),
                )
            )
            metrics = (
                resolve_path(str(suite.output_root)) / "offline" / "ablations" /
                benchmark / task / variant / "metrics.json"
            )
            main_metrics = (
                resolve_path(str(suite.output_root)) / "offline" / "main" /
                benchmark / task / ablation_method / "metrics.json"
            )
            offline_overrides = [
                f"method={ablation_method}",
                "domain=both",
                f"method_configs.{ablation_method}={spec.method_config}",
            ]
            if not is_v2:
                offline_overrides.extend(
                    (
                        f"training.canonical_weight={float(spec.canonical_weight)}",
                        f"training.dynamics_weight={float(spec.dynamics_weight)}",
                    )
                )
            jobs.append(
                JobSpec(
                    job_id=f"offline/ablation/{task}/{variant}", phase=ablation_phase,
                    benchmark=benchmark, task=task, method=ablation_method, variant=variant,
                    command=tuple((
                        sys.executable, "scripts/evaluate_offline_dynamics.py", "--config", config,
                        *offline_overrides,
                        f"paths.method_checkpoint={checkpoint}",
                        f"paths.feature_cache={_main_cache(suite, benchmark, task)}",
                        f"offline.num_windows={int(suite.offline.windows)}",
                        f"offline.seed={int(suite.offline.seed)}",
                        f"offline.shuffle_seed={int(suite.offline.shuffle_seed)}",
                        f"offline.output_directory={metrics.parent}",
                    )), log_path=_log(suite, f"offline-ablation-{task}-{variant}"),
                    artifact_path=str(metrics), kind="offline",
                    required_count=int(suite.offline.windows),
                    reuse_sources=(str(main_metrics),) if variant == "full" else (),
                    dependencies=(f"train/ablation/{task}/{variant}",),
                )
            )
            result = (
                resolve_path(str(suite.output_root)) / "ablations" / benchmark / task /
                variant / "ood" / "results.json"
            )
            main_result = _main_result(suite, benchmark, task, ablation_method, "ood")
            planning_overrides = [
                f"method={ablation_method}",
                "domain=ood",
                f"method_configs.{ablation_method}={spec.method_config}",
            ]
            if not is_v2:
                planning_overrides.extend(
                    (
                        f"training.canonical_weight={float(spec.canonical_weight)}",
                        f"training.dynamics_weight={float(spec.dynamics_weight)}",
                    )
                )
            jobs.append(
                JobSpec(
                    job_id=f"planning/ablation/{task}/{variant}", phase=ablation_phase,
                    benchmark=benchmark, task=task, method=ablation_method, domain="ood",
                    seed=int(suite.ablations.seed), severity=1.0, variant=variant,
                    command=tuple((
                        sys.executable, "scripts/plan.py", "--config", config,
                        *planning_overrides,
                        f"paths.method_checkpoint={checkpoint}",
                        f"evaluation.num_episodes={int(suite.ablations.episodes)}",
                        f"evaluation.eval_seed={int(suite.ablations.seed)}",
                        "appearance.severity=1.0", f"output.run_directory={result.parent}",
                        "suite.family=ablation", f"suite.variant={variant}",
                    )), log_path=_log(suite, f"plan-ablation-{task}-{variant}"),
                    artifact_path=str(result), kind="planning",
                    required_count=int(suite.ablations.episodes),
                    reuse_sources=(str(main_result),) if variant == "full" else (),
                    dependencies=(f"train/ablation/{task}/{variant}",),
                )
            )

    summary = resolve_path(str(suite.analysis_root)) / (
        "summary.md" if is_v2 else "paper_summary.md"
    )
    jobs.append(
        JobSpec(
            job_id="analysis/final", phase="Final analysis", benchmark="all", task="all",
            command=(
                sys.executable, "scripts/analyze_cross_benchmark_suite.py",
                "--suite-config", suite_config_path,
                "--state", str(suite.state_path), "--output", str(suite.analysis_root),
            ), log_path=_log(suite, "analysis-final"), artifact_path=str(summary),
            kind="analysis",
        )
    )
    return jobs


def logical_rollout_counts(jobs: list[JobSpec]) -> dict[str, int]:
    planning = [job for job in jobs if job.kind == "planning"]
    main = sum(
        int(job.required_count or 0)
        for job in planning
        if job.phase == "Main closed-loop planning"
    )
    total = sum(int(job.required_count or 0) for job in planning)
    return {"main_logical_rollouts": main, "all_logical_rollouts": total}
