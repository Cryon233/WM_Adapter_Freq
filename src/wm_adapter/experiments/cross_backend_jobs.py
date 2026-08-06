from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from wm_adapter.experiments.cross_benchmark import JobSpec
from wm_adapter.utils.reproducibility import resolve_path


SUITE_NAME = "cross_backend_adapter_v1"
EXPECTED_PLANNING_JOBS = 118
EXPECTED_PLANNING_EPISODES = 3540


def _path(root: Any, *parts: str) -> Path:
    return resolve_path(str(root)).joinpath(*parts)


def _log(suite: Any, job_id: str) -> str:
    return str(_path(suite.log_root, f"{job_id.replace('/', '-')}.log"))


def _task_config(suite: Any, task: str) -> str:
    return str(suite.tasks[task])


def _model_config(suite: Any, backend: str, task: str) -> str:
    return str(suite.backends[backend].models[task])


def _manifest(suite: Any, task: str, seed: int) -> Path:
    return _path(suite.manifest_root, "evaluation", task, f"seed_{seed}.json")


def _task_manifest(suite: Any, task: str) -> Path:
    return _path(suite.manifest_root, "tasks", f"{task}.json")


def _cache(suite: Any, backend: str, task: str) -> Path:
    return _path(suite.storage_root, backend, task, "cache.h5")


def _checkpoint(
    suite: Any, backend: str, task: str, method: str, seed: int
) -> Path:
    return _path(
        suite.checkpoint_root,
        backend,
        task,
        method,
        f"seed_{seed}_final.pt",
    )


def _offline(
    suite: Any,
    backend: str,
    task: str,
    method: str,
    seed: int | None,
) -> Path:
    seed_part = "deterministic_base" if seed is None else f"seed_{seed}"
    return _path(
        suite.output_root,
        "offline",
        backend,
        task,
        method,
        seed_part,
        "metrics.json",
    )


def _planning(
    suite: Any,
    family: str,
    backend: str,
    task: str,
    method: str,
    seed: int,
    domain: str,
) -> Path:
    return _path(
        suite.output_root,
        family,
        backend,
        task,
        f"seed_{seed}",
        method,
        domain,
        "results.json",
    )


def _common_overrides(
    suite: Any, backend: str, task: str, seed: int
) -> list[str]:
    return [
        f"model_config={_model_config(suite, backend, task)}",
        f"paths.feature_cache={_cache(suite, backend, task)}",
        f"paths.task_manifest={_task_manifest(suite, task)}",
        f"paths.evaluation_manifest={_manifest(suite, task, seed)}",
        f"training.seed={seed}",
        f"evaluation.eval_seed={seed}",
    ]


def _validate_planning_design(
    suite: Any,
    *,
    seeds: list[int],
    domains: list[str],
) -> tuple[list[str], list[int], str]:
    if seeds != [42, 7, 2026]:
        raise RuntimeError(
            "Fast planning design requires seeds [42, 7, 2026], "
            f"received {seeds}"
        )
    if domains != ["ood"]:
        raise RuntimeError(
            "The primary planning matrix must contain OOD only; "
            f"received domains={domains}"
        )
    if int(suite.main.episodes) != 30:
        raise RuntimeError(
            "Fast planning design requires main.episodes=30, "
            f"found {suite.main.episodes}"
        )
    guardrail = suite.clean_guardrail
    if not bool(guardrail.enabled):
        raise RuntimeError("Clean guardrail must remain enabled")
    guardrail_domain = str(guardrail.domain)
    guardrail_methods = [str(value) for value in guardrail.methods]
    guardrail_seeds = [int(value) for value in guardrail.seeds]
    if guardrail_domain != "clean":
        raise RuntimeError(
            f"Clean guardrail domain must be 'clean', found {guardrail_domain!r}"
        )
    if guardrail_methods != ["base", "hfra"]:
        raise RuntimeError(
            "Clean guardrail methods must be [base, hfra], "
            f"found {guardrail_methods}"
        )
    if guardrail_seeds != [42]:
        raise RuntimeError(
            "Clean guardrail must use owner seed [42], "
            f"found {guardrail_seeds}"
        )
    if int(suite.ablations.episodes) != 30:
        raise RuntimeError(
            "Fast planning design requires ablations.episodes=30, "
            f"found {suite.ablations.episodes}"
        )
    return guardrail_methods, guardrail_seeds, guardrail_domain


def _main_planning_cells(
    *,
    methods: list[str],
    seeds: list[int],
    domains: list[str],
    guardrail_methods: list[str],
    guardrail_seeds: list[int],
    guardrail_domain: str,
) -> list[tuple[str, int, str, str]]:
    missing = [method for method in guardrail_methods if method not in methods]
    if missing:
        raise RuntimeError(
            "Backend does not support required clean-guardrail methods: "
            f"methods={methods}, missing={missing}"
        )

    cells: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, int, str]] = set()

    for method in methods:
        for seed in seeds:
            for domain in domains:
                key = (method, seed, domain)
                if key in seen:
                    raise RuntimeError(f"Duplicate primary planning cell: {key}")
                seen.add(key)
                cells.append((method, seed, domain, "full"))

    for method in guardrail_methods:
        for seed in guardrail_seeds:
            key = (method, seed, guardrail_domain)
            if key in seen:
                raise RuntimeError(
                    f"Clean guardrail duplicates a primary planning cell: {key}"
                )
            seen.add(key)
            cells.append((method, seed, guardrail_domain, "clean_guardrail"))

    return cells


def build_cross_backend_job_graph(suite: Any) -> list[JobSpec]:
    if str(suite.suite_name) != SUITE_NAME:
        raise ValueError(
            f"Cross-backend graph requires suite_name={SUITE_NAME!r}, "
            f"found {suite.suite_name!r}"
        )
    tasks = [str(value) for value in suite.tasks.keys()]
    backends = [str(value) for value in suite.backends.keys()]
    seeds = [int(value) for value in suite.seeds]
    domains = [str(value) for value in suite.domains]
    (
        guardrail_methods,
        guardrail_seeds,
        guardrail_domain,
    ) = _validate_planning_design(suite, seeds=seeds, domains=domains)

    if tasks != [
        "robocasa_reach",
        "robocasa_place",
        "libero_spatial_0",
        "libero_goal_0",
    ]:
        raise RuntimeError(f"Cross-backend task order changed: {tasks}")
    if "robocasa_articulated" in tasks:
        raise RuntimeError(
            "robocasa_articulated is forbidden in cross_backend_adapter_v1"
        )
    jobs: list[JobSpec] = []

    # Evaluation manifests are backend-independent and shared by every method.
    for task in tasks:
        benchmark = "libero" if task.startswith("libero_") else "robocasa"
        owner_seed = seeds[0]
        for seed in seeds:
            manifest = _manifest(suite, task, seed)
            dependencies = (
                ()
                if seed == owner_seed
                else (f"manifest/{task}/seed_{owner_seed}",)
            )
            command = (
                sys.executable,
                "scripts/build_evaluation_manifest.py",
                "--config",
                _task_config(suite, task),
                f"model_config={_model_config(suite, 'jepa_wm_droid', task)}",
                f"paths.task_manifest={_task_manifest(suite, task)}",
                f"paths.evaluation_manifest={manifest}",
                f"evaluation.num_episodes={int(suite.main.episodes)}",
                f"evaluation.eval_seed={seed}",
            )
            jobs.append(
                JobSpec(
                    job_id=f"manifest/{task}/seed_{seed}",
                    phase="Feature caches",
                    benchmark=benchmark,
                    task=task,
                    backend=None,
                    seed=seed,
                    command=command,
                    log_path=_log(suite, f"manifest/{task}/seed_{seed}"),
                    artifact_path=str(manifest),
                    kind="evaluation_manifest",
                    required_count=int(suite.main.episodes),
                    dependencies=dependencies,
                )
            )

    for backend in backends:
        methods = [str(value) for value in suite.backends[backend].methods]
        if backend == "dino_wm_droid" and any(
            value in methods for value in ("dct_adapter", "token_mlp")
        ):
            raise RuntimeError(
                "DINO-WM matrix cannot contain dct_adapter or token_mlp"
            )

        planning_cells = _main_planning_cells(
            methods=methods,
            seeds=seeds,
            domains=domains,
            guardrail_methods=guardrail_methods,
            guardrail_seeds=guardrail_seeds,
            guardrail_domain=guardrail_domain,
        )

        for task in tasks:
            benchmark = "libero" if task.startswith("libero_") else "robocasa"
            cache = _cache(suite, backend, task)
            cache_job = f"cache/{backend}/{task}"
            common = _common_overrides(suite, backend, task, seeds[0])
            jobs.append(
                JobSpec(
                    job_id=cache_job,
                    phase="Feature caches",
                    benchmark=benchmark,
                    task=task,
                    backend=backend,
                    command=(
                        sys.executable,
                        "scripts/build_feature_cache.py",
                        "--config",
                        _task_config(suite, task),
                        *common,
                        f"data.num_train_windows={int(suite.data.num_train_windows)}",
                    ),
                    log_path=_log(suite, cache_job),
                    artifact_path=str(cache),
                    kind="cache",
                    required_count=int(suite.data.num_train_windows),
                    dependencies=(f"manifest/{task}/seed_{seeds[0]}",),
                )
            )

            for method in methods:
                if method == "base":
                    continue
                for seed in seeds:
                    checkpoint = _checkpoint(
                        suite, backend, task, method, seed
                    )
                    train_id = (
                        f"train/{backend}/{task}/{method}/seed_{seed}"
                    )
                    jobs.append(
                        JobSpec(
                            job_id=train_id,
                            phase="Main adapter training",
                            benchmark=benchmark,
                            task=task,
                            backend=backend,
                            method=method,
                            seed=seed,
                            command=(
                                sys.executable,
                                "scripts/train_adapter.py",
                                "--config",
                                _task_config(suite, task),
                                *_common_overrides(
                                    suite, backend, task, seed
                                ),
                                f"method={method}",
                                f"paths.method_checkpoint={checkpoint}",
                            ),
                            log_path=_log(suite, train_id),
                            artifact_path=str(checkpoint),
                            kind="checkpoint",
                            dependencies=(cache_job,),
                        )
                    )

            for method in methods:
                offline_seeds = [None] if method == "base" else seeds
                for seed_or_none in offline_seeds:
                    run_seed = (
                        seeds[0] if seed_or_none is None else seed_or_none
                    )
                    metrics = _offline(
                        suite, backend, task, method, seed_or_none
                    )
                    offline_id = (
                        f"offline/{backend}/{task}/{method}/"
                        + (
                            "deterministic_base"
                            if seed_or_none is None
                            else f"seed_{seed_or_none}"
                        )
                    )
                    dependencies = (
                        (cache_job,)
                        if method == "base"
                        else (
                            f"train/{backend}/{task}/{method}/"
                            f"seed_{run_seed}",
                        )
                    )
                    command = [
                        sys.executable,
                        "scripts/evaluate_offline_dynamics.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(
                            suite, backend, task, run_seed
                        ),
                        f"method={method}",
                        "domain=both",
                        f"offline.num_windows={int(suite.offline.windows)}",
                        f"offline.output_directory={metrics.parent}",
                    ]
                    if method != "base":
                        command.append(
                            "paths.method_checkpoint="
                            f"{_checkpoint(suite, backend, task, method, run_seed)}"
                        )
                    jobs.append(
                        JobSpec(
                            job_id=offline_id,
                            phase="Offline MSE",
                            benchmark=benchmark,
                            task=task,
                            backend=backend,
                            method=method,
                            seed=seed_or_none,
                            command=tuple(command),
                            log_path=_log(suite, offline_id),
                            artifact_path=str(metrics),
                            kind="offline",
                            required_count=int(suite.offline.windows),
                            dependencies=dependencies,
                        )
                    )

            for method, seed, domain, variant in planning_cells:
                result = _planning(
                    suite,
                    "main",
                    backend,
                    task,
                    method,
                    seed,
                    domain,
                )
                planning_id = (
                    f"planning/main/{backend}/{task}/{method}/"
                    f"seed_{seed}/{domain}"
                )
                dependencies = [
                    cache_job,
                    f"manifest/{task}/seed_{seed}",
                ]
                command = [
                    sys.executable,
                    "scripts/plan.py",
                    "--config",
                    _task_config(suite, task),
                    *_common_overrides(
                        suite, backend, task, seed
                    ),
                    f"method={method}",
                    f"domain={domain}",
                    f"evaluation.num_episodes={int(suite.main.episodes)}",
                    "appearance.severity="
                    f"{float(suite.appearance.main_severity)}",
                    f"output.run_directory={result.parent}",
                    "suite.family=main",
                    f"suite.variant={variant}",
                ]
                if method != "base":
                    dependencies.append(
                        f"train/{backend}/{task}/{method}/seed_{seed}"
                    )
                    command.append(
                        "paths.method_checkpoint="
                        f"{_checkpoint(suite, backend, task, method, seed)}"
                    )
                jobs.append(
                    JobSpec(
                        job_id=planning_id,
                        phase="Closed-loop planning",
                        benchmark=benchmark,
                        task=task,
                        backend=backend,
                        method=method,
                        domain=domain,
                        seed=seed,
                        severity=float(
                            suite.appearance.main_severity
                        ),
                        variant=variant,
                        command=tuple(command),
                        log_path=_log(suite, planning_id),
                        artifact_path=str(result),
                        kind="planning",
                        required_count=int(suite.main.episodes),
                        dependencies=tuple(dependencies),
                    )
                )

    ablation_backend = str(suite.ablations.backend)
    ablation_domain = str(suite.ablations.domain)
    if ablation_domain != "ood":
        raise RuntimeError(
            f"HFRA core-only ablation must remain OOD, found {ablation_domain}"
        )
    for task_value in suite.ablations.tasks:
        task = str(task_value)
        benchmark = "libero" if task.startswith("libero_") else "robocasa"
        for seed in seeds:
            method = "hfra_core_only"
            checkpoint = _checkpoint(
                suite, ablation_backend, task, method, seed
            )
            train_id = (
                f"train/ablation/{ablation_backend}/{task}/"
                f"{method}/seed_{seed}"
            )
            jobs.append(
                JobSpec(
                    job_id=train_id,
                    phase="Main adapter training",
                    benchmark=benchmark,
                    task=task,
                    backend=ablation_backend,
                    method=method,
                    seed=seed,
                    variant="core_only",
                    command=(
                        sys.executable,
                        "scripts/train_adapter.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(
                            suite, ablation_backend, task, seed
                        ),
                        f"method={method}",
                        f"paths.method_checkpoint={checkpoint}",
                    ),
                    log_path=_log(suite, train_id),
                    artifact_path=str(checkpoint),
                    kind="checkpoint",
                    dependencies=(
                        f"cache/{ablation_backend}/{task}",
                    ),
                )
            )
            metrics = _offline(
                suite, ablation_backend, task, method, seed
            )
            offline_id = (
                f"offline/ablation/{ablation_backend}/{task}/"
                f"{method}/seed_{seed}"
            )
            jobs.append(
                JobSpec(
                    job_id=offline_id,
                    phase="Offline MSE",
                    benchmark=benchmark,
                    task=task,
                    backend=ablation_backend,
                    method=method,
                    seed=seed,
                    variant="core_only",
                    command=(
                        sys.executable,
                        "scripts/evaluate_offline_dynamics.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(
                            suite, ablation_backend, task, seed
                        ),
                        f"method={method}",
                        "domain=both",
                        f"paths.method_checkpoint={checkpoint}",
                        f"offline.num_windows={int(suite.offline.windows)}",
                        f"offline.output_directory={metrics.parent}",
                    ),
                    log_path=_log(suite, offline_id),
                    artifact_path=str(metrics),
                    kind="offline",
                    required_count=int(suite.offline.windows),
                    dependencies=(train_id,),
                )
            )
            result = _planning(
                suite,
                "ablations",
                ablation_backend,
                task,
                method,
                seed,
                ablation_domain,
            )
            planning_id = (
                f"planning/ablation/{ablation_backend}/{task}/"
                f"{method}/seed_{seed}/{ablation_domain}"
            )
            jobs.append(
                JobSpec(
                    job_id=planning_id,
                    phase="Closed-loop planning",
                    benchmark=benchmark,
                    task=task,
                    backend=ablation_backend,
                    method=method,
                    domain=ablation_domain,
                    seed=seed,
                    severity=float(
                        suite.appearance.main_severity
                    ),
                    variant="core_only",
                    command=(
                        sys.executable,
                        "scripts/plan.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(
                            suite, ablation_backend, task, seed
                        ),
                        f"method={method}",
                        f"domain={ablation_domain}",
                        f"paths.method_checkpoint={checkpoint}",
                        "evaluation.num_episodes="
                        f"{int(suite.ablations.episodes)}",
                        "appearance.severity="
                        f"{float(suite.appearance.main_severity)}",
                        f"output.run_directory={result.parent}",
                        "suite.family=ablation",
                        "suite.variant=core_only",
                    ),
                    log_path=_log(suite, planning_id),
                    artifact_path=str(result),
                    kind="planning",
                    required_count=int(
                        suite.ablations.episodes
                    ),
                    dependencies=(
                        train_id,
                        f"manifest/{task}/seed_{seed}",
                    ),
                )
            )

    analysis = _path(suite.analysis_root, "main_results.md")
    analysis_dependencies = tuple(
        job.job_id
        for job in jobs
        if job.kind in {"offline", "planning"}
    )
    jobs.append(
        JobSpec(
            job_id="analysis/final",
            phase="Final analysis",
            benchmark="all",
            task="all",
            command=(
                sys.executable,
                "scripts/analyze_cross_backend_adapter.py",
                "--suite-config",
                str(suite._suite_config_path),
                "--state",
                str(suite.state_path),
                "--output",
                str(suite.analysis_root),
            ),
            log_path=_log(suite, "analysis/final"),
            artifact_path=str(analysis),
            kind="analysis",
            dependencies=analysis_dependencies,
        )
    )

    planning = [job for job in jobs if job.kind == "planning"]
    episodes = sum(
        int(job.required_count or 0) for job in planning
    )
    if (
        len(planning) != EXPECTED_PLANNING_JOBS
        or episodes != EXPECTED_PLANNING_EPISODES
    ):
        raise RuntimeError(
            "Cross-backend planning matrix changed: "
            f"jobs={len(planning)} "
            f"(expected {EXPECTED_PLANNING_JOBS}), "
            f"episodes={episodes} "
            f"(expected {EXPECTED_PLANNING_EPISODES})"
        )
    if any(
        job.task == "robocasa_articulated" for job in jobs
    ):
        raise RuntimeError(
            "robocasa_articulated leaked into cross-backend graph"
        )
    return jobs


def cross_backend_rollout_counts(
    jobs: list[JobSpec],
) -> dict[str, int]:
    planning = [job for job in jobs if job.kind == "planning"]
    return {
        "planning_jobs": len(planning),
        "closed_loop_episodes": sum(
            int(job.required_count or 0) for job in planning
        ),
        "main_planning_jobs": sum(
            str(job.job_id).startswith("planning/main/")
            for job in planning
        ),
        "ood_primary_planning_jobs": sum(
            str(job.job_id).startswith("planning/main/")
            and job.domain == "ood"
            for job in planning
        ),
        "clean_guardrail_planning_jobs": sum(
            job.variant == "clean_guardrail"
            for job in planning
        ),
        "core_only_planning_jobs": sum(
            job.variant == "core_only"
            for job in planning
        ),
    }
