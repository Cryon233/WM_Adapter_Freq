from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import h5py
import torch
from omegaconf import OmegaConf

from wm_adapter.experiments.paper_suite import (
    GPUJob,
    atomic_write_json,
    preflight_resources,
    run_gpu_jobs,
    validate_feature_cache,
    validate_method_checkpoint,
    validate_offline_result,
    validate_planning_result,
)
from wm_adapter.utils.checkpoints import sha256_file
from wm_adapter.utils.reproducibility import project_root, resolve_path


SUITE_CONFIG = "configs/experiment/icra2027_suite.yaml"


def _gpu_ids() -> list[int]:
    available = torch.cuda.device_count()
    if available <= 0:
        raise RuntimeError("No CUDA devices are visible to the paper-suite runner")
    raw = os.environ.get("GPUS")
    if raw is None:
        return list(range(min(4, available)))
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"GPUS must be a comma-separated list of integers, received {raw!r}") from error
    if not values or len(values) != len(set(values)):
        raise ValueError(f"GPUS must contain unique GPU IDs, received {raw!r}")
    invalid = [value for value in values if value < 0 or value >= available]
    if invalid:
        raise ValueError(
            f"GPUS includes unavailable IDs {invalid}; visible CUDA device count is {available}"
        )
    return values


def _checkpoint_path(suite: Any, method: str) -> Path:
    return resolve_path(str(suite.formal_checkpoints[method]))


def _cache_fingerprint(path: Path) -> str:
    with h5py.File(path, "r", libver="latest", swmr=True) as cache:
        return str(cache.attrs["cache_fingerprint"])


def _mark_reused(
    state: dict[str, Any],
    job_id: str,
    phase: str,
    artifact: dict[str, Any],
) -> None:
    state["jobs"][job_id] = {
        "status": "reused",
        "phase": phase,
        "artifact": artifact,
        "updated_at_unix": time.time(),
    }


def _archive_incomplete_result(path: Path) -> None:
    if not path.exists():
        return
    digest = sha256_file(path)[:12]
    archived = path.with_name(f"{path.stem}.incomplete-{digest}{path.suffix}")
    if archived.exists():
        raise RuntimeError(
            f"Cannot preserve incomplete artifact because archive already exists: {archived}"
        )
    path.replace(archived)


def _resume_or_job(
    *,
    state: dict[str, Any],
    jobs: list[GPUJob],
    job_id: str,
    phase: str,
    command: list[str],
    log_path: Path,
    artifact_path: Path,
    validator: Callable[[], dict[str, Any]],
    archive_incomplete: bool = False,
) -> None:
    if artifact_path.exists():
        try:
            _mark_reused(state, job_id, phase, validator())
            return
        except Exception:
            if not archive_incomplete:
                raise
            _archive_incomplete_result(artifact_path)
    jobs.append(
        GPUJob(
            job_id=job_id,
            phase=phase,
            command=tuple(command),
            log_path=str(log_path),
            artifact_path=str(artifact_path),
            validate=validator,
        )
    )


def _run_phase(
    jobs: list[GPUJob],
    gpu_ids: list[int],
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    atomic_write_json(state_path, state)
    if not jobs:
        return state
    return run_gpu_jobs(jobs, gpu_ids, state_path, initial_state=state)


def _plan_command(
    config: str,
    method: str,
    domain: str,
    checkpoint: Path | None,
    run_directory: Path,
    episodes: int,
    seed: int,
    family: str,
    variant: str,
    severity: float = 1.0,
    method_config: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/plan.py",
        "--config",
        config,
        f"method={method}",
        f"domain={domain}",
        f"evaluation.num_episodes={episodes}",
        f"evaluation.eval_seed={seed}",
        f"appearance.severity={severity}",
        f"output.run_directory={run_directory}",
        f"suite.family={family}",
        f"suite.variant={variant}",
    ]
    if checkpoint is not None:
        command.append(f"paths.method_checkpoint={checkpoint}")
    if method_config is not None:
        command.append(f"method_configs.dct_adapter={method_config}")
    return command


def _validate_plan(
    path: Path,
    episodes: int,
    task: str,
    method: str,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    return validate_planning_result(
        path,
        episodes,
        expected_metadata={
            "task": task,
            "method": method,
            "domain": domain,
            "evaluation_protocol_version": "2.0",
            "cem_seed": seed,
        },
    )


def main() -> None:
    root = project_root()
    suite = OmegaConf.load(resolve_path(SUITE_CONFIG))
    state_path = resolve_path(str(suite.state_path))
    state: dict[str, Any] = {
        "suite": "icra2027",
        "protocol": str(suite.protocol),
        "status": "running",
        "started_at_unix": time.time(),
        "expected_jobs": 105,
        "jobs": {},
    }
    if os.environ.get("DOWNLOAD_ASSETS", "0") != "0" or os.environ.get("FORCE_ASSETS", "0") != "0":
        raise RuntimeError("The paper suite never downloads or replaces assets; set DOWNLOAD_ASSETS=0 and FORCE_ASSETS=0")
    place_config = str(suite.tasks.place)
    state["preflight"] = preflight_resources(place_config)
    gpu_ids = _gpu_ids()
    state["gpu_ids"] = gpu_ids
    atomic_write_json(state_path, state)

    cache_path = resolve_path(str(suite.formal_cache))
    cache_jobs: list[GPUJob] = []
    _resume_or_job(
        state=state,
        jobs=cache_jobs,
        job_id="cache/formal",
        phase="cache",
        command=[sys.executable, "scripts/build_feature_cache.py", "--config", place_config],
        log_path=resolve_path(str(suite.log_root)) / "cache-formal.log",
        artifact_path=cache_path,
        validator=lambda: validate_feature_cache(cache_path, 2000),
    )
    state = _run_phase(cache_jobs, gpu_ids[:1], state_path, state)
    cache_info = validate_feature_cache(cache_path, 2000)
    cache_fingerprint = str(cache_info["cache_fingerprint"])

    training_jobs: list[GPUJob] = []
    for method in ("dct_adapter", "token_mlp", "lora"):
        checkpoint = _checkpoint_path(suite, method)
        _resume_or_job(
            state=state,
            jobs=training_jobs,
            job_id=f"train/{method}",
            phase="training",
            command=[
                sys.executable,
                "scripts/train_adapter.py",
                "--config",
                place_config,
                f"method={method}",
                f"paths.method_checkpoint={checkpoint}",
            ],
            log_path=resolve_path(str(suite.log_root)) / f"train-{method}.log",
            artifact_path=checkpoint,
            validator=lambda method=method, checkpoint=checkpoint: validate_method_checkpoint(
                checkpoint, method, cache_fingerprint
            ),
        )
    state = _run_phase(training_jobs, gpu_ids, state_path, state)

    ablation_checkpoints: dict[str, Path] = {}
    ablation_jobs: list[GPUJob] = []
    for variant, spec in suite.dct_ablations.items():
        if "checkpoint" in spec:
            checkpoint = resolve_path(str(spec.checkpoint))
        else:
            checkpoint = resolve_path(f"checkpoints/paper_suite/dct_ablations/{variant}_final.pt")
        ablation_checkpoints[str(variant)] = checkpoint
        method_config = str(spec.method_config)
        method_options = OmegaConf.to_container(
            OmegaConf.load(resolve_path(method_config)), resolve=True
        )
        if not isinstance(method_options, dict):
            raise TypeError(f"DCT method config must be a mapping: {method_config}")
        training_weights = (
            float(spec.canonical_weight),
            float(spec.dynamics_weight),
        )
        _resume_or_job(
            state=state,
            jobs=ablation_jobs,
            job_id=f"train/dct_ablation/{variant}",
            phase="ablation_training",
            command=[
                sys.executable,
                "scripts/train_adapter.py",
                "--config",
                place_config,
                "method=dct_adapter",
                f"method_configs.dct_adapter={method_config}",
                f"training.canonical_weight={float(spec.canonical_weight)}",
                f"training.dynamics_weight={float(spec.dynamics_weight)}",
                f"paths.method_checkpoint={checkpoint}",
            ],
            log_path=resolve_path(str(suite.log_root)) / f"train-ablation-{variant}.log",
            artifact_path=checkpoint,
            validator=lambda checkpoint=checkpoint, method_options=method_options, training_weights=training_weights: validate_method_checkpoint(
                checkpoint,
                "dct_adapter",
                cache_fingerprint,
                expected_method_options=method_options,
                expected_training_weights=training_weights,
            ),
        )
    state = _run_phase(ablation_jobs, gpu_ids, state_path, state)

    offline_jobs: list[GPUJob] = []
    offline_sources: dict[str, Path] = {}
    for method in ("base", "dct_adapter", "token_mlp", "lora"):
        output = resolve_path(f"outputs/paper_suite/offline/main/{method}")
        metrics = output / "metrics.json"
        checkpoint = None if method == "base" else _checkpoint_path(suite, method)
        command = [
            sys.executable,
            "scripts/evaluate_offline_dynamics.py",
            "--config",
            place_config,
            f"method={method}",
            "domain=both",
            f"offline.num_windows={int(suite.offline.windows)}",
            f"offline.seed={int(suite.offline.seed)}",
            f"offline.shuffle_seed={int(suite.offline.shuffle_seed)}",
            f"offline.output_directory={output}",
        ]
        if checkpoint is not None:
            command.append(f"paths.method_checkpoint={checkpoint}")
        _resume_or_job(
            state=state,
            jobs=offline_jobs,
            job_id=f"offline/main/{method}",
            phase="offline",
            command=command,
            log_path=resolve_path(str(suite.log_root)) / f"offline-main-{method}.log",
            artifact_path=metrics,
            validator=lambda metrics=metrics: validate_offline_result(
                metrics, int(suite.offline.windows)
            ),
            archive_incomplete=True,
        )
        offline_sources[method] = metrics
    state = _run_phase(offline_jobs, gpu_ids, state_path, state)
    offline_jobs = []
    for variant, spec in suite.dct_ablations.items():
        if str(variant) in {"full", "rank8"}:
            _mark_reused(
                state,
                f"offline/ablation/{variant}",
                "offline",
                {
                    "source_path": str(offline_sources["dct_adapter"]),
                    "source_sha256": sha256_file(offline_sources["dct_adapter"]),
                },
            )
            continue
        output = resolve_path(f"outputs/paper_suite/offline/ablations/{variant}")
        metrics = output / "metrics.json"
        checkpoint = ablation_checkpoints[str(variant)]
        method_config = str(spec.method_config)
        _resume_or_job(
            state=state,
            jobs=offline_jobs,
            job_id=f"offline/ablation/{variant}",
            phase="offline",
            command=[
                sys.executable,
                "scripts/evaluate_offline_dynamics.py",
                "--config",
                place_config,
                "method=dct_adapter",
                "domain=both",
                f"method_configs.dct_adapter={method_config}",
                f"paths.method_checkpoint={checkpoint}",
                f"offline.num_windows={int(suite.offline.windows)}",
                f"offline.seed={int(suite.offline.seed)}",
                f"offline.shuffle_seed={int(suite.offline.shuffle_seed)}",
                f"offline.output_directory={output}",
            ],
            log_path=resolve_path(str(suite.log_root)) / f"offline-ablation-{variant}.log",
            artifact_path=metrics,
            validator=lambda metrics=metrics: validate_offline_result(
                metrics, int(suite.offline.windows)
            ),
            archive_incomplete=True,
        )
    state = _run_phase(offline_jobs, gpu_ids, state_path, state)

    planning_jobs: list[GPUJob] = []
    main_results: dict[tuple[str, str, str], Path] = {}
    for task, config in suite.tasks.items():
        for method in suite.methods:
            for domain in suite.domains:
                result = (
                    resolve_path("outputs")
                    / "jepa_wm_droid"
                    / "robocasa"
                    / str(suite.protocol)
                    / str(task)
                    / f"seed_{int(suite.main.seed)}"
                    / str(method)
                    / str(domain)
                    / "results.json"
                )
                main_results[(str(task), str(method), str(domain))] = result
                checkpoint = None if method == "base" else _checkpoint_path(suite, str(method))
                command = _plan_command(
                    str(config), str(method), str(domain), checkpoint, result.parent,
                    int(suite.main.episodes), int(suite.main.seed), "main", "full"
                )
                _resume_or_job(
                    state=state,
                    jobs=planning_jobs,
                    job_id=f"planning/main/{task}/{method}/{domain}",
                    phase="planning_main",
                    command=command,
                    log_path=resolve_path(str(suite.log_root)) / f"plan-main-{task}-{method}-{domain}.log",
                    artifact_path=result,
                    validator=lambda result=result, task=str(task), method=str(method), domain=str(domain): _validate_plan(
                        result,
                        int(suite.main.episodes),
                        task,
                        method,
                        domain,
                        int(suite.main.seed),
                    ),
                    archive_incomplete=True,
                )
    state = _run_phase(planning_jobs, gpu_ids, state_path, state)

    extra_jobs: list[GPUJob] = []
    for task in suite.multiseed.tasks:
        config = str(suite.tasks[str(task)])
        for method in suite.multiseed.methods:
            for domain in suite.multiseed.domains:
                for seed in suite.multiseed.seeds:
                    source = main_results.get((str(task), str(method), str(domain)))
                    job_id = f"planning/multiseed/{task}/{method}/{domain}/seed_{seed}"
                    if int(seed) == int(suite.main.seed) and source is not None:
                        source_info = _validate_plan(
                            source,
                            int(suite.multiseed.episodes),
                            str(task),
                            str(method),
                            str(domain),
                            int(seed),
                        )
                        _mark_reused(state, job_id, "planning_multiseed", {"source": source_info})
                        continue
                    result = resolve_path(
                        f"outputs/paper_suite/{suite.protocol}/multiseed/{task}/seed_{seed}/{method}/{domain}/results.json"
                    )
                    checkpoint = None if method == "base" else _checkpoint_path(suite, str(method))
                    _resume_or_job(
                        state=state, jobs=extra_jobs, job_id=job_id, phase="planning_multiseed",
                        command=_plan_command(
                            config, str(method), str(domain), checkpoint, result.parent,
                            int(suite.multiseed.episodes), int(seed), "multiseed", "full"
                        ),
                        log_path=resolve_path(str(suite.log_root)) / f"plan-multiseed-{task}-{method}-{domain}-seed{seed}.log",
                        artifact_path=result,
                        validator=lambda result=result, task=str(task), method=str(method), domain=str(domain), seed=int(seed): _validate_plan(result, int(suite.multiseed.episodes), task, method, domain, seed),
                        archive_incomplete=True,
                    )
    for task in suite.severity.tasks:
        config = str(suite.tasks[str(task)])
        for method in suite.severity.methods:
            for severity in suite.severity.values:
                job_id = f"planning/severity/{task}/{method}/{severity}"
                if float(severity) == 1.0:
                    source = main_results[(str(task), str(method), "ood")]
                    source_info = _validate_plan(
                        source,
                        int(suite.severity.episodes),
                        str(task),
                        str(method),
                        "ood",
                        int(suite.severity.seed),
                    )
                    _mark_reused(state, job_id, "planning_severity", {"source": source_info})
                    continue
                severity_name = str(float(severity)).replace(".", "p")
                result = resolve_path(
                    f"outputs/paper_suite/{suite.protocol}/severity/{task}/seed_{suite.severity.seed}/{method}/ood_severity_{severity_name}/results.json"
                )
                checkpoint = None if method == "base" else _checkpoint_path(suite, str(method))
                _resume_or_job(
                    state=state, jobs=extra_jobs, job_id=job_id, phase="planning_severity",
                    command=_plan_command(
                        config, str(method), "ood", checkpoint, result.parent,
                        int(suite.severity.episodes), int(suite.severity.seed), "severity", "full", float(severity)
                    ),
                    log_path=resolve_path(str(suite.log_root)) / f"plan-severity-{task}-{method}-{severity_name}.log",
                    artifact_path=result,
                    validator=lambda result=result, task=str(task), method=str(method): _validate_plan(result, int(suite.severity.episodes), task, method, "ood", int(suite.severity.seed)),
                    archive_incomplete=True,
                )
    for variant in suite.closed_loop_ablations:
        spec = suite.dct_ablations[str(variant)]
        for task in suite.ablation_tasks:
            job_id = f"planning/ablation/{variant}/{task}"
            if str(variant) == "full":
                source = main_results[(str(task), "dct_adapter", "ood")]
                source_info = _validate_plan(
                    source,
                    int(suite.ablation_episodes),
                    str(task),
                    "dct_adapter",
                    "ood",
                    42,
                )
                _mark_reused(state, job_id, "planning_ablation", {"source": source_info})
                continue
            result = resolve_path(
                f"outputs/paper_suite/{suite.protocol}/ablations/{task}/seed_42/{variant}/ood/results.json"
            )
            _resume_or_job(
                state=state, jobs=extra_jobs, job_id=job_id, phase="planning_ablation",
                command=_plan_command(
                    str(suite.tasks[str(task)]), "dct_adapter", "ood",
                    ablation_checkpoints[str(variant)], result.parent,
                    int(suite.ablation_episodes), 42, "ablation", str(variant), 1.0,
                    str(spec.method_config),
                ),
                log_path=resolve_path(str(suite.log_root)) / f"plan-ablation-{variant}-{task}.log",
                artifact_path=result,
                validator=lambda result=result, task=str(task): _validate_plan(result, int(suite.ablation_episodes), task, "dct_adapter", "ood", 42),
                archive_incomplete=True,
            )
    state = _run_phase(extra_jobs, gpu_ids, state_path, state)

    analysis_root = resolve_path(str(suite.analysis_root))
    summary = analysis_root / "paper_summary.md"
    analysis_job = GPUJob(
        job_id="analysis/final",
        phase="analysis",
        command=(
            sys.executable,
            "scripts/analyze_paper_suite.py",
            "--suite-config",
            SUITE_CONFIG,
            "--output",
            str(analysis_root),
        ),
        log_path=str(resolve_path(str(suite.log_root)) / "analysis-final.log"),
        artifact_path=str(summary),
        validate=lambda: {"path": str(summary), "sha256": sha256_file(summary)},
    )
    state = _run_phase([analysis_job], gpu_ids[:1], state_path, state)
    state["status"] = "completed"
    state["completed_at_unix"] = time.time()
    atomic_write_json(state_path, state)
    print(f"ICRA 2027 paper suite completed: {summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure_state_path = resolve_path("logs/paper_suite/state.json")
        if failure_state_path.is_file():
            failure_state = json.loads(failure_state_path.read_text(encoding="utf-8"))
        else:
            failure_state = {"suite": "icra2027", "jobs": {}}
        failure_state["status"] = "failed"
        failure_state["failed_at_unix"] = time.time()
        failure_state["error"] = f"{type(error).__name__}: {error}"
        failure_state["traceback"] = traceback.format_exc()
        atomic_write_json(failure_state_path, failure_state)
        raise
