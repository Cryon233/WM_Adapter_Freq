from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from wm_adapter.experiments.cross_benchmark import JobSpec
from wm_adapter.utils.reproducibility import resolve_path


SUITE_NAME = "cross_backend_adapter_v1"
EXPECTED_PLANNING_JOBS = 180
EXPECTED_PLANNING_EPISODES = 4500


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


def _episode_count(suite: Any, task: str) -> int:
    if task.startswith("libero_"):
        return int(suite.main.libero_episodes)
    return int(suite.main.robocasa_episodes)


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


def _planning_overrides(
    suite: Any,
    *,
    domain: str,
    episodes: int,
    result: Path,
    family: str,
    variant: str,
) -> list[str]:
    if domain == "clean":
        evaluation_family = "identity"
        severity = 0.0
    elif domain == "ood":
        evaluation_family = str(suite.appearance.evaluation_family)
        severity = float(suite.appearance.main_severity)
    else:
        raise ValueError(f"Unsupported domain {domain!r}")
    return [
        f"domain={domain}",
        f"evaluation.num_episodes={episodes}",
        f"appearance.evaluation_family={evaluation_family}",
        f"appearance.severity={severity}",
        f"output.run_directory={result.parent}",
        f"suite.family={family}",
        f"suite.variant={variant}",
    ]


def _validate_design(suite: Any) -> tuple[list[str], list[str], list[int], list[str]]:
    tasks = [str(value) for value in suite.tasks.keys()]
    backends = [str(value) for value in suite.backends.keys()]
    seeds = [int(value) for value in suite.seeds]
    domains = [str(value) for value in suite.domains]
    if tasks != [
        "robocasa_reach",
        "robocasa_place",
        "libero_spatial_0",
        "libero_goal_0",
    ]:
        raise RuntimeError(f"Task order changed: {tasks}")
    if backends != ["jepa_wm_droid", "dino_wm_droid"]:
        raise RuntimeError(f"Backend order changed: {backends}")
    if seeds != [42, 7, 2026]:
        raise RuntimeError(f"Formal seeds changed: {seeds}")
    if domains != ["clean", "ood"]:
        raise RuntimeError(f"Formal domains changed: {domains}")
    if int(suite.main.robocasa_episodes) != 30:
        raise RuntimeError("RoboCasa requires 30 episodes")
    if int(suite.main.libero_episodes) != 20:
        raise RuntimeError("LIBERO requires 20 episodes")
    if str(suite.training.input_domain) != "clean":
        raise RuntimeError("Training input domain must be clean")
    if str(suite.training.appearance_family) != "identity":
        raise RuntimeError("Training appearance must be identity")
    for backend in backends:
        methods = [str(value) for value in suite.backends[backend].methods]
        if "dct_adapter" in methods:
            raise RuntimeError("dct_adapter is excluded from the formal suite")
        if backend == "dino_wm_droid" and "token_mlp" in methods:
            raise RuntimeError("DINO-WM cannot use token_mlp")
    return tasks, backends, seeds, domains


def build_cross_backend_job_graph(suite: Any) -> list[JobSpec]:
    if str(suite.suite_name) != SUITE_NAME:
        raise ValueError(
            f"Expected suite_name={SUITE_NAME!r}, found {suite.suite_name!r}"
        )
    tasks, backends, seeds, domains = _validate_design(suite)
    jobs: list[JobSpec] = []

    for task in tasks:
        benchmark = "libero" if task.startswith("libero_") else "robocasa"
        episodes = _episode_count(suite, task)
        for seed in seeds:
            manifest = _manifest(suite, task, seed)
            job_id = f"manifest/{task}/seed_{seed}"
            dependencies = (
                () if seed == seeds[0]
                else (f"manifest/{task}/seed_{seeds[0]}",)
            )
            jobs.append(
                JobSpec(
                    job_id=job_id,
                    phase="Feature caches",
                    benchmark=benchmark,
                    task=task,
                    backend=None,
                    seed=seed,
                    command=(
                        sys.executable,
                        "scripts/build_evaluation_manifest.py",
                        "--config",
                        _task_config(suite, task),
                        f"model_config={_model_config(suite, 'jepa_wm_droid', task)}",
                        f"paths.task_manifest={_task_manifest(suite, task)}",
                        f"paths.evaluation_manifest={manifest}",
                        f"evaluation.num_episodes={episodes}",
                        f"evaluation.eval_seed={seed}",
                    ),
                    log_path=_log(suite, job_id),
                    artifact_path=str(manifest),
                    kind="evaluation_manifest",
                    required_count=episodes,
                    dependencies=dependencies,
                )
            )

    for backend in backends:
        methods = [str(value) for value in suite.backends[backend].methods]
        for task in tasks:
            benchmark = "libero" if task.startswith("libero_") else "robocasa"
            episodes = _episode_count(suite, task)
            cache = _cache(suite, backend, task)
            cache_id = f"cache/{backend}/{task}"
            jobs.append(
                JobSpec(
                    job_id=cache_id,
                    phase="Feature caches",
                    benchmark=benchmark,
                    task=task,
                    backend=backend,
                    command=(
                        sys.executable,
                        "scripts/build_feature_cache.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(suite, backend, task, seeds[0]),
                        f"data.num_train_windows={int(suite.data.num_train_windows)}",
                    ),
                    log_path=_log(suite, cache_id),
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
                    checkpoint = _checkpoint(suite, backend, task, method, seed)
                    train_id = f"train/{backend}/{task}/{method}/seed_{seed}"
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
                                *_common_overrides(suite, backend, task, seed),
                                f"method={method}",
                                f"paths.method_checkpoint={checkpoint}",
                            ),
                            log_path=_log(suite, train_id),
                            artifact_path=str(checkpoint),
                            kind="checkpoint",
                            dependencies=(cache_id,),
                        )
                    )

            for method in methods:
                offline_seeds = [None] if method == "base" else seeds
                for seed_or_none in offline_seeds:
                    run_seed = seeds[0] if seed_or_none is None else seed_or_none
                    metrics = _offline(suite, backend, task, method, seed_or_none)
                    seed_name = (
                        "deterministic_base"
                        if seed_or_none is None
                        else f"seed_{seed_or_none}"
                    )
                    offline_id = f"offline/{backend}/{task}/{method}/{seed_name}"
                    dependencies = (
                        (cache_id,)
                        if method == "base"
                        else (f"train/{backend}/{task}/{method}/seed_{run_seed}",)
                    )
                    command = [
                        sys.executable,
                        "scripts/evaluate_offline_dynamics.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(suite, backend, task, run_seed),
                        f"method={method}",
                        "domain=both",
                        f"offline.num_windows={int(suite.offline.windows)}",
                        f"offline.output_directory={metrics.parent}",
                    ]
                    if method != "base":
                        command.append(
                            f"paths.method_checkpoint={_checkpoint(suite, backend, task, method, run_seed)}"
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

            for method in methods:
                for seed in seeds:
                    for domain in domains:
                        result = _planning(
                            suite, "main", backend, task, method, seed, domain
                        )
                        planning_id = (
                            f"planning/main/{backend}/{task}/{method}/"
                            f"seed_{seed}/{domain}"
                        )
                        dependencies = [
                            cache_id,
                            f"manifest/{task}/seed_{seed}",
                        ]
                        command = [
                            sys.executable,
                            "scripts/plan.py",
                            "--config",
                            _task_config(suite, task),
                            *_common_overrides(suite, backend, task, seed),
                            f"method={method}",
                            *_planning_overrides(
                                suite,
                                domain=domain,
                                episodes=episodes,
                                result=result,
                                family="main",
                                variant="full",
                            ),
                        ]
                        if method != "base":
                            dependencies.append(
                                f"train/{backend}/{task}/{method}/seed_{seed}"
                            )
                            command.append(
                                f"paths.method_checkpoint={_checkpoint(suite, backend, task, method, seed)}"
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
                                severity=(
                                    0.0
                                    if domain == "clean"
                                    else float(suite.appearance.main_severity)
                                ),
                                variant="full",
                                command=tuple(command),
                                log_path=_log(suite, planning_id),
                                artifact_path=str(result),
                                kind="planning",
                                required_count=episodes,
                                dependencies=tuple(dependencies),
                            )
                        )

    ablation_backend = str(suite.ablations.backend)
    ablation_method = str(suite.ablations.method)
    ablation_domains = [str(value) for value in suite.ablations.domains]
    if ablation_domains != domains:
        raise RuntimeError("Ablation domains must match main domains")
    for task_value in suite.ablations.tasks:
        task = str(task_value)
        benchmark = "libero" if task.startswith("libero_") else "robocasa"
        episodes = _episode_count(suite, task)
        for seed in seeds:
            checkpoint = _checkpoint(
                suite, ablation_backend, task, ablation_method, seed
            )
            train_id = (
                f"train/ablation/{ablation_backend}/{task}/"
                f"{ablation_method}/seed_{seed}"
            )
            jobs.append(
                JobSpec(
                    job_id=train_id,
                    phase="Main adapter training",
                    benchmark=benchmark,
                    task=task,
                    backend=ablation_backend,
                    method=ablation_method,
                    seed=seed,
                    variant="core_only",
                    command=(
                        sys.executable,
                        "scripts/train_adapter.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(suite, ablation_backend, task, seed),
                        f"method={ablation_method}",
                        f"paths.method_checkpoint={checkpoint}",
                    ),
                    log_path=_log(suite, train_id),
                    artifact_path=str(checkpoint),
                    kind="checkpoint",
                    dependencies=(f"cache/{ablation_backend}/{task}",),
                )
            )

            metrics = _offline(
                suite, ablation_backend, task, ablation_method, seed
            )
            offline_id = (
                f"offline/ablation/{ablation_backend}/{task}/"
                f"{ablation_method}/seed_{seed}"
            )
            jobs.append(
                JobSpec(
                    job_id=offline_id,
                    phase="Offline MSE",
                    benchmark=benchmark,
                    task=task,
                    backend=ablation_backend,
                    method=ablation_method,
                    seed=seed,
                    variant="core_only",
                    command=(
                        sys.executable,
                        "scripts/evaluate_offline_dynamics.py",
                        "--config",
                        _task_config(suite, task),
                        *_common_overrides(suite, ablation_backend, task, seed),
                        f"method={ablation_method}",
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

            for domain in domains:
                result = _planning(
                    suite,
                    "ablations",
                    ablation_backend,
                    task,
                    ablation_method,
                    seed,
                    domain,
                )
                planning_id = (
                    f"planning/ablation/{ablation_backend}/{task}/"
                    f"{ablation_method}/seed_{seed}/{domain}"
                )
                jobs.append(
                    JobSpec(
                        job_id=planning_id,
                        phase="Closed-loop planning",
                        benchmark=benchmark,
                        task=task,
                        backend=ablation_backend,
                        method=ablation_method,
                        domain=domain,
                        seed=seed,
                        severity=(
                            0.0
                            if domain == "clean"
                            else float(suite.appearance.main_severity)
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
                            f"method={ablation_method}",
                            f"paths.method_checkpoint={checkpoint}",
                            *_planning_overrides(
                                suite,
                                domain=domain,
                                episodes=episodes,
                                result=result,
                                family="ablation",
                                variant="core_only",
                            ),
                        ),
                        log_path=_log(suite, planning_id),
                        artifact_path=str(result),
                        kind="planning",
                        required_count=episodes,
                        dependencies=(
                            train_id,
                            f"manifest/{task}/seed_{seed}",
                        ),
                    )
                )

    analysis = _path(suite.analysis_root, "main_results.md")
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
            dependencies=tuple(
                job.job_id
                for job in jobs
                if job.kind in {"offline", "planning"}
            ),
        )
    )

    planning = [job for job in jobs if job.kind == "planning"]
    episodes = sum(int(job.required_count or 0) for job in planning)
    if (
        len(planning) != EXPECTED_PLANNING_JOBS
        or episodes != EXPECTED_PLANNING_EPISODES
    ):
        raise RuntimeError(
            f"Planning matrix changed: jobs={len(planning)}, episodes={episodes}"
        )
    return jobs


def cross_backend_rollout_counts(jobs: list[JobSpec]) -> dict[str, int]:
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
        "clean_main_planning_jobs": sum(
            str(job.job_id).startswith("planning/main/")
            and job.domain == "clean"
            for job in planning
        ),
        "ood_primary_planning_jobs": sum(
            str(job.job_id).startswith("planning/main/")
            and job.domain == "ood"
            for job in planning
        ),
        "clean_guardrail_planning_jobs": 0,
        "core_only_planning_jobs": sum(
            job.variant == "core_only" for job in planning
        ),
    }
