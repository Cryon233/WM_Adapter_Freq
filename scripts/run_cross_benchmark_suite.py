from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from wm_adapter.benchmarks.base import (
    EVALUATION_MANIFEST_SCHEMA,
    canonical_sha256,
)
from wm_adapter.benchmarks.factory import build_benchmark
from wm_adapter.experiments.cross_benchmark import (
    PHASES,
    JobSpec,
    archive_incomplete,
    benchmark_subprocess_environment,
    load_suite_config,
    load_task_config,
    phase_summary,
    run_gpu_phase,
    validate_cache,
    validate_checkpoint,
    validate_offline,
    validate_planning,
    validate_task_manifest,
)
from wm_adapter.experiments.cross_jobs import build_job_graph, logical_rollout_counts
from wm_adapter.utils.checkpoints import (
    UPSTREAM_COMMITS,
    git_commit,
    sha256_dataset_path,
    sha256_file,
)
from wm_adapter.utils.reproducibility import project_root, resolve_path


DEFAULT_SUITE_CONFIG = "configs/experiment/cross_benchmark_v1.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cross_benchmark_v1 experiment graph")
    parser.add_argument("--config", default=DEFAULT_SUITE_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight-config", help=argparse.SUPPRESS)
    parser.add_argument("--preflight-output", help=argparse.SUPPRESS)
    parser.add_argument(
        "--preflight-override", action="append", default=[], help=argparse.SUPPRESS
    )
    return parser.parse_args()


def _gpu_ids(suite: Any) -> list[int]:
    raw = os.environ.get("GPUS")
    values = (
        [int(value) for value in raw.split(",") if value.strip()]
        if raw is not None
        else [int(value) for value in suite.gpu.default_ids]
    )
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError(f"GPUS must be a non-empty list of unique non-negative IDs: {raw!r}")
    return values


def _self_test_suite(suite: Any) -> Any:
    roots = suite.self_test.roots
    cloned = OmegaConf.create(OmegaConf.to_container(suite, resolve=True))
    cloned.state_path = roots.state_path
    cloned.pid_path = roots.pid_path
    cloned.log_root = roots.log_root
    cloned.storage_root = roots.storage_root
    cloned.checkpoint_root = roots.checkpoint_root
    cloned.output_root = roots.output_root
    cloned.analysis_root = roots.analysis_root
    cloned.manifest_root = f"{roots.output_root}/manifests"
    cloned.reuse_sources = {}
    selected = {str(value) for value in cloned.self_test.task_keys}
    cloned.tasks = {
        str(key): value
        for key, value in cloned.tasks.items()
        if str(key) in selected
    }
    return cloned


def _resource_report(task_config: Any, *, strict: bool, deep: bool) -> dict[str, Any]:
    errors: list[str] = []
    resources: dict[str, Any] = {}
    for key in ("jepa_checkpoint", "dinov3_checkpoint", "official_planning_config"):
        value = str(task_config.model.get(key, "")).strip()
        path = resolve_path(value) if value else None
        resources[key] = str(path) if path is not None else ""
        if path is None or not path.is_file():
            errors.append(f"{key} is missing: {path}")
        elif strict and key in {"jepa_checkpoint", "dinov3_checkpoint"}:
            resources[f"{key}_sha256"] = sha256_file(path)
    third_party = resolve_path(str(task_config.model.third_party_root))
    commits: dict[str, str] = {}
    for name, expected in UPSTREAM_COMMITS.items():
        repo = third_party / name
        if not (repo / ".git").exists():
            errors.append(f"upstream checkout is missing or not independent: {repo}")
            continue
        try:
            actual = git_commit(repo)
        except Exception as error:
            errors.append(f"cannot read {name} commit: {error}")
            continue
        commits[name] = actual
        if actual != expected:
            errors.append(f"{name} commit mismatch: expected={expected}, actual={actual}")
    resources["upstream_commits"] = commits
    if strict and str(task_config.benchmark.name) == "libero":
        import robosuite

        robosuite_version = str(getattr(robosuite, "__version__", "unknown"))
        robosuite_path = Path(robosuite.__file__).resolve()
        resources["libero_robosuite"] = {
            "version": robosuite_version,
            "path": str(robosuite_path),
        }
        if robosuite_version != "1.4.0":
            errors.append(
                "LIBERO must run with isolated robosuite 1.4.0: "
                f"loaded version={robosuite_version}, path={robosuite_path}"
            )
    benchmark = build_benchmark(task_config)
    task = benchmark.resolve_task(strict=False if not strict else True)
    if strict:
        dataset_path = resolve_path(task.dataset_path)
        actual_dataset_sha256 = sha256_dataset_path(dataset_path)
        if actual_dataset_sha256 != task.dataset_sha256:
            raise RuntimeError(
                "Resolved task dataset fingerprint changed: "
                f"task={task.task_key}, expected={task.dataset_sha256}, "
                f"actual={actual_dataset_sha256}, path={dataset_path}"
            )
        if task.bddl_path is not None:
            actual_bddl_sha256 = sha256_file(task.bddl_path)
            if actual_bddl_sha256 != task.bddl_sha256:
                raise RuntimeError(
                    f"Resolved LIBERO BDDL fingerprint changed: {task.bddl_path}"
                )
    report: dict[str, Any] = {
        "task": task.as_dict(),
        "resources": resources,
        "errors": errors,
    }
    if strict:
        if errors:
            raise RuntimeError("Resource preflight failed:\n- " + "\n- ".join(errors))
        report["benchmark_preflight"] = benchmark.preflight(deep=deep)
        resolved = benchmark.resolve_task(strict=True)
        task_manifest = benchmark.write_task_manifest(resolved)
        evaluation = benchmark.build_evaluation_manifest(
            resolved,
            count=int(task_config.evaluation.num_episodes),
            seed=int(task_config.evaluation.eval_seed),
            appearance_seed=int(task_config.appearance.seed),
        )
        benchmark.write_evaluation_manifest(evaluation)
        report["task"] = task_manifest
        report["evaluation_manifest"] = evaluation
    return report


def _run_preflight_child(args: argparse.Namespace) -> None:
    if not args.preflight_output:
        raise ValueError("--preflight-output is required with --preflight-config")
    task_config = load_task_config(
        args.preflight_config,
        overrides=args.preflight_override,
    )
    report = _resource_report(task_config, strict=True, deep=True)
    from wm_adapter.benchmarks.base import atomic_json

    atomic_json(args.preflight_output, report)


def _preflight_overrides(suite: Any, task_key: str, *, self_test: bool) -> list[str]:
    if not self_test:
        return []
    return [
        "paths.task_manifest="
        f"{resolve_path(str(suite.manifest_root)) / 'tasks' / f'{task_key}.json'}",
        "paths.evaluation_manifest="
        f"{resolve_path(str(suite.manifest_root)) / 'evaluation' / f'{task_key}.json'}",
        f"evaluation.num_episodes={int(suite.self_test.episodes)}",
        f"data.num_train_windows={int(suite.self_test.windows)}",
    ]


def _run_isolated_preflight(
    suite: Any,
    job: JobSpec,
    config_path: str,
    *,
    self_test: bool,
) -> dict[str, Any]:
    report_path = resolve_path(job.log_path).with_suffix(".report.json")
    report_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "scripts/run_cross_benchmark_suite.py",
        "--preflight-config",
        str(config_path),
        "--preflight-output",
        str(report_path),
    ]
    for value in _preflight_overrides(suite, job.task, self_test=self_test):
        command.extend(("--preflight-override", value))
    log_path = resolve_path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=project_root(),
            env=benchmark_subprocess_environment(job.benchmark),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        report_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Isolated task preflight failed: "
            f"task={job.task}, benchmark={job.benchmark}, "
            f"return_code={completed.returncode}, log={log_path}"
        )
    if not report_path.is_file():
        raise RuntimeError(
            f"Isolated task preflight produced no report: {report_path}"
        )
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def _dry_run(suite: Any, jobs: list[JobSpec]) -> None:
    task_reports: dict[str, Any] = {}
    missing: list[str] = []
    for task_key, config_path in suite.tasks.items():
        task_config = load_task_config(str(config_path))
        report = _resource_report(task_config, strict=False, deep=False)
        task_reports[str(task_key)] = report["task"]
        missing.extend(f"{task_key}: {value}" for value in report["errors"])
        failure_map = report["task"].get("candidate_failures") or {}
        for candidate, reasons in failure_map.items():
            missing.extend(f"{task_key}/{candidate}: {reason}" for reason in reasons)
    matrix: list[dict[str, Any]] = []
    reusable = 0
    pending = 0
    reusable_rollouts = 0
    intrinsic_reuse_rollouts = 0
    for job in jobs:
        candidates = (job.artifact_path,) + job.reuse_sources
        source = next((str(resolve_path(value)) for value in candidates if resolve_path(value).is_file()), None)
        linked_job = _linked_reuse_job(job)
        status = "candidate_reuse" if source is not None else "planned_reuse" if linked_job else "pending"
        reusable += source is not None or linked_job is not None
        pending += source is None and linked_job is None
        if job.kind == "planning" and source is not None:
            reusable_rollouts += int(job.required_count or 0)
        elif job.kind == "planning" and linked_job is not None:
            intrinsic_reuse_rollouts += int(job.required_count or 0)
        matrix.append(
            {
                "job_id": job.job_id,
                "phase": job.phase,
                "benchmark": job.benchmark,
                "task": job.task,
                "method": job.method,
                "domain": job.domain,
                "seed": job.seed,
                "severity": job.severity,
                "variant": job.variant,
                "status": status,
                "reuse_candidate": source,
                "planned_reuse_job": linked_job,
                "artifact_path": job.artifact_path,
            }
        )
    stage_counts = {
        phase: sum(job.phase == phase for job in jobs) for phase in PHASES
    }
    rollouts = logical_rollout_counts(jobs)
    rollouts["intrinsic_reuse_rollouts"] = intrinsic_reuse_rollouts
    rollouts["external_candidate_reusable_rollouts"] = reusable_rollouts
    rollouts["candidate_new_rollouts"] = (
        rollouts["all_logical_rollouts"]
        - intrinsic_reuse_rollouts
        - reusable_rollouts
    )
    report = {
        "suite": str(suite.suite_name),
        "resolved_task_candidates": task_reports,
        "missing_resources": missing,
        "expected_jobs": len(jobs),
        "candidate_reusable_jobs": reusable,
        "pending_jobs": pending,
        "phase_job_counts": stage_counts,
        "rollouts": rollouts,
        "gpu_queue_plan": {
            "gpu_ids": _gpu_ids(suite),
            "one_heavy_job_per_gpu": True,
            "policy": "FIFO within each phase; at most one subprocess per GPU",
        },
        "jobs": matrix,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _effective_artifact(state: dict[str, Any], job_id: str) -> str:
    entry = state["jobs"][job_id]
    return str(entry.get("reuse_source") or entry["artifact_path"])


def _linked_reuse_job(job: JobSpec) -> str | None:
    if job.phase == "Training-seed stability" and int(job.seed or -1) == 42:
        return (
            f"train/main/{job.task}/{job.method}"
            if job.kind == "checkpoint"
            else f"planning/main/{job.task}/{job.method}/ood"
        )
    if job.phase == "OOD severity" and float(job.severity or -1.0) == 1.0:
        return f"planning/main/{job.task}/{job.method}/ood"
    if job.phase == "DCT ablations" and job.variant == "full":
        return {
            "checkpoint": f"train/main/{job.task}/dct_adapter",
            "offline": f"offline/main/{job.task}/dct_adapter",
            "planning": f"planning/main/{job.task}/dct_adapter/ood",
        }.get(job.kind)
    return None


def _dependency_job(job: JobSpec) -> str | None:
    if job.kind == "checkpoint":
        return f"cache/{job.task}"
    if job.kind not in {"offline", "planning"} or job.method in {None, "base"}:
        return None
    if job.variant is not None:
        return f"train/ablation/{job.task}/{job.variant}"
    if job.phase == "Training-seed stability":
        return f"train/stability/{job.task}/{job.method}/seed_{job.seed}"
    return f"train/main/{job.task}/{job.method}"


def _wire_job(job: JobSpec, state: dict[str, Any]) -> JobSpec:
    if not job.command:
        return job
    command = list(job.command)
    if job.kind in {"checkpoint", "offline", "planning"}:
        cache_id = f"cache/{job.task}"
        if cache_id in state["jobs"]:
            command.append(f"paths.feature_cache={_effective_artifact(state, cache_id)}")
    dependency = _dependency_job(job)
    if dependency is not None and dependency in state["jobs"]:
        command.append(f"paths.method_checkpoint={_effective_artifact(state, dependency)}")
    return replace(job, command=tuple(command))


def _manifest_paths(suite: Any, task: str) -> tuple[Path, Path]:
    root = resolve_path(str(suite.manifest_root))
    return root / "tasks" / f"{task}.json", root / "evaluation" / f"{task}.json"


def _validate_evaluation_manifest(
    path: Path,
    task_manifest: dict[str, Any],
    required_count: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EVALUATION_MANIFEST_SCHEMA:
        raise RuntimeError(f"Evaluation manifest schema mismatch: {path}")
    supplied_hash = str(payload.get("evaluation_manifest_sha256", ""))
    hash_payload = {
        key: value
        for key, value in payload.items()
        if key != "evaluation_manifest_sha256"
    }
    if supplied_hash != canonical_sha256(hash_payload):
        raise RuntimeError(f"Evaluation manifest fingerprint is invalid: {path}")
    if str(payload.get("task_key")) != str(task_manifest["task_key"]):
        raise RuntimeError(f"Evaluation manifest task identity mismatch: {path}")
    if str(payload.get("task_manifest_sha256")) != str(
        task_manifest["task_manifest_sha256"]
    ):
        raise RuntimeError(f"Evaluation manifest task fingerprint mismatch: {path}")
    instances = list(payload.get("instances", []))
    if len(instances) < required_count:
        raise RuntimeError(
            "Evaluation manifest has too few instances: "
            f"required={required_count}, actual={len(instances)}, path={path}"
        )
    instance_ids = [str(value.get("instance_id", "")) for value in instances]
    if any(not value for value in instance_ids) or len(instance_ids) != len(
        set(instance_ids)
    ):
        raise RuntimeError(f"Evaluation manifest instance IDs are invalid: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "evaluation_manifest_sha256": supplied_hash,
        "available_instances": len(instances),
    }


def _reuse_completed_preflight(
    suite: Any,
    state: dict[str, Any],
    job: JobSpec,
    *,
    self_test: bool,
) -> bool:
    entry = state.get("jobs", {}).get(job.job_id, {})
    if entry.get("status") not in {"completed", "reused"}:
        return False
    stored_validation = entry.get("artifact_validation")
    if not isinstance(stored_validation, dict):
        return False
    report = stored_validation.get("report")
    if not isinstance(report, dict) or report.get("errors"):
        return False
    benchmark_report = report.get("benchmark_preflight")
    if not isinstance(benchmark_report, dict) or not bool(
        benchmark_report.get("deep_environment_check", False)
    ):
        return False
    try:
        task_validation = _validate_job(
            suite,
            state,
            job,
            job.artifact_path,
            self_test=self_test,
        )
        task_manifest = json.loads(
            resolve_path(job.artifact_path).read_text(encoding="utf-8")
        )
        _, evaluation_path = _manifest_paths(suite, job.task)
        required_count = int(
            suite.self_test.episodes if self_test else suite.main.episodes
        )
        evaluation_validation = _validate_evaluation_manifest(
            evaluation_path,
            task_manifest,
            required_count,
        )
        reported_task = report.get("task", {})
        reported_evaluation = report.get("evaluation_manifest", {})
        resources = report.get("resources", {})
        if str(reported_task.get("task_manifest_sha256")) != str(
            task_manifest["task_manifest_sha256"]
        ):
            return False
        if str(reported_evaluation.get("evaluation_manifest_sha256")) != str(
            evaluation_validation["evaluation_manifest_sha256"]
        ):
            return False
        if dict(resources.get("upstream_commits", {})) != UPSTREAM_COMMITS:
            return False
        for key in ("jepa_checkpoint", "dinov3_checkpoint", "official_planning_config"):
            if not Path(str(resources.get(key, ""))).is_file():
                return False
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False
    refreshed = dict(entry)
    refreshed.update(
        status="reused",
        artifact_validation={
            **stored_validation,
            **task_validation,
            "report": report,
            "evaluation_manifest": evaluation_validation,
        },
        gpu=None,
        pid=None,
        error=None,
        return_code=0,
    )
    state["jobs"][job.job_id] = refreshed
    return True


def _reset_failed_state_for_resume(
    state: dict[str, Any], jobs: list[JobSpec]
) -> None:
    previous_jobs = dict(state.get("jobs", {}))
    resumed_jobs: dict[str, dict[str, Any]] = {}
    for job in jobs:
        previous = previous_jobs.get(job.job_id)
        if isinstance(previous, dict) and previous.get("status") in {
            "completed",
            "reused",
        }:
            resumed_jobs[job.job_id] = previous
        else:
            resumed_jobs[job.job_id] = job.state_fields() | {"status": "pending"}
    state["jobs"] = resumed_jobs
    for key in (
        "error",
        "traceback",
        "failed_at_unix",
        "completed_at_unix",
        "stopped_at_unix",
    ):
        state.pop(key, None)
    state["status"] = "preflight"
    state["last_started_at_unix"] = time.time()
    state["restart_count"] = int(state.get("restart_count", 0)) + 1


def _cache_info(state: dict[str, Any], task: str) -> dict[str, Any]:
    entry = state["jobs"][f"cache/{task}"]
    return dict(entry["artifact_validation"])


def _validate_job(
    suite: Any,
    state: dict[str, Any],
    job: JobSpec,
    path: str,
    *,
    self_test: bool,
) -> dict[str, Any]:
    allow_legacy = (
        job.task == "robocasa_place"
        and resolve_path(path) != resolve_path(job.artifact_path)
    )
    if job.kind == "task_manifest":
        return validate_task_manifest(
            path,
            job.task,
            allow_legacy_place=allow_legacy,
        )
    if job.kind == "cache":
        task_manifest_path, _ = _manifest_paths(suite, job.task)
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        camera_contract = {
            key: task_manifest.get(key)
            for key in (
                "camera_height",
                "camera_width",
                "camera_channel_order",
                "camera_vertical_flip",
            )
        }
        return validate_cache(
            path, int(job.required_count or 0), benchmark=job.benchmark,
            task=job.task, allow_legacy_place=allow_legacy,
            expected_task_manifest_sha256=str(
                task_manifest["task_manifest_sha256"]
            ),
            expected_action_transform=task_manifest.get("action_transform"),
            expected_camera_contract=camera_contract,
        )
    if job.kind == "checkpoint":
        cache = _cache_info(state, job.task)
        task_manifest_path, _ = _manifest_paths(suite, job.task)
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        camera_contract = {
            key: task_manifest.get(key)
            for key in (
                "camera_height",
                "camera_width",
                "camera_channel_order",
                "camera_vertical_flip",
            )
        }
        task_config_path = str(suite.tasks[job.task])
        task_config = load_task_config(task_config_path)
        if job.variant is not None:
            ablation = suite.ablations.variants[job.variant]
            method_config_path = str(ablation.method_config)
            loss_weights = (
                float(ablation.canonical_weight),
                float(ablation.dynamics_weight),
            )
        else:
            method_config_path = str(task_config.method_configs[str(job.method)])
            loss_weights = (1.0, 1.0)
        method_config = OmegaConf.to_container(
            OmegaConf.load(resolve_path(method_config_path)), resolve=True
        )
        if not isinstance(method_config, dict):
            raise TypeError(f"Method config is not a mapping: {method_config_path}")
        return validate_checkpoint(
            path, str(job.method), str(cache["cache_fingerprint"]),
            benchmark=job.benchmark, task=job.task, allow_legacy_place=allow_legacy,
            expected_training_seed=int(job.seed or 42),
            expected_method_config=method_config,
            expected_loss_weights=loss_weights,
            expected_action_transform=task_manifest.get("action_transform"),
            expected_camera_contract=camera_contract,
        )
    if job.kind == "offline":
        task_manifest_path, _ = _manifest_paths(suite, job.task)
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        return validate_offline(
            path, int(job.required_count or 0), benchmark=job.benchmark,
            task=job.task, method=str(job.method),
            expected_action_transform=task_manifest.get("action_transform"),
        )
    if job.kind == "planning":
        task_manifest_path, evaluation_manifest_path = _manifest_paths(suite, job.task)
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        preflight = state["jobs"][f"preflight/{job.task}"]["artifact_validation"]["report"]
        resources = dict(preflight["resources"])
        camera_contract = {
            key: task_manifest.get(key)
            for key in (
                "camera_height",
                "camera_width",
                "camera_channel_order",
                "camera_vertical_flip",
            )
        }
        cache = _cache_info(state, job.task)
        dependency = _dependency_job(job)
        checkpoint = (
            state["jobs"][dependency]["artifact_validation"]
            if dependency is not None and dependency in state["jobs"]
            else None
        )
        return validate_planning(
            path, int(job.required_count or 0), benchmark=job.benchmark,
            task=job.task, method=str(job.method), domain=str(job.domain),
            seed=int(suite.main.eval_seed), severity=float(job.severity or 1.0),
            evaluation_manifest=evaluation_manifest_path,
            allow_legacy_place=allow_legacy,
            expected_task_manifest_sha256=str(task_manifest["task_manifest_sha256"]),
            expected_cache_fingerprint=str(cache["cache_fingerprint"]),
            expected_checkpoint_sha256=(
                str(checkpoint["sha256"]) if checkpoint is not None else None
            ),
            expected_action_convention=dict(task_manifest["action_convention"]),
            expected_action_transform=task_manifest.get("action_transform"),
            expected_camera_contract=camera_contract,
            formal_cem=not self_test,
            expected_base_checkpoint_sha256=resources.get("jepa_checkpoint_sha256"),
            expected_dinov3_checkpoint_sha256=resources.get("dinov3_checkpoint_sha256"),
            expected_appearance_seed=int(suite.appearance.seed),
            expected_appearance_pipeline=str(suite.appearance.pipeline_version),
        )
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Analysis artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _resume_or_pending(
    suite: Any,
    state: dict[str, Any],
    job: JobSpec,
    *,
    self_test: bool,
) -> JobSpec | None:
    standard = resolve_path(job.artifact_path)
    if job.kind == "analysis":
        wired = _wire_job(job, state)
        state["jobs"][job.job_id] = wired.state_fields() | {"status": "pending"}
        return wired
    if standard.is_file():
        try:
            validation = _validate_job(
                suite, state, job, str(standard), self_test=self_test
            )
        except Exception:
            if job.kind in {"cache", "checkpoint", "planning", "offline"}:
                archived = archive_incomplete(standard)
                state.setdefault("archives", []).append(str(archived))
                if job.kind == "offline":
                    rows = standard.parent / "per_window.jsonl"
                    if rows.is_file():
                        state["archives"].append(str(archive_incomplete(rows)))
            else:
                raise
        else:
            entry = job.state_fields()
            entry.update(status="reused", artifact_validation=validation)
            state["jobs"][job.job_id] = entry
            return None
    reuse_values = list(job.reuse_sources)
    linked_job = _linked_reuse_job(job)
    if linked_job is not None and linked_job in state["jobs"]:
        reuse_values.insert(0, _effective_artifact(state, linked_job))
    for source_value in reuse_values:
        source = resolve_path(source_value)
        if not source.is_file():
            continue
        try:
            validation = _validate_job(
                suite, state, job, str(source), self_test=self_test
            )
        except Exception:
            continue
        entry = job.state_fields()
        entry.update(
            status="reused", artifact_validation=validation,
            reuse_source=str(source),
        )
        state["jobs"][job.job_id] = entry
        return None
    wired = _wire_job(job, state)
    state["jobs"][job.job_id] = wired.state_fields() | {"status": "pending"}
    return wired


def _run(suite: Any, jobs: list[JobSpec], *, self_test: bool) -> None:
    state_path = resolve_path(str(suite.state_path))
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("suite") != str(suite.suite_name):
            raise RuntimeError(f"Existing state belongs to another suite: {state_path}")
    else:
        state = {
            "suite": str(suite.suite_name), "protocol": str(suite.protocol),
            "status": "preflight", "started_at_unix": time.time(), "jobs": {},
        }
    _reset_failed_state_for_resume(state, jobs)
    state["expected_jobs"] = len(jobs)
    state["state_path"] = str(state_path)
    state["git"] = {
        "commit": git_commit(project_root()),
        "dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=project_root(),
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        ),
    }
    state["gpu_ids"] = _gpu_ids(suite)
    state["self_test"] = self_test
    state["phase_summary"] = phase_summary(state, jobs)
    from wm_adapter.benchmarks.base import atomic_json
    atomic_json(state_path, state)
    if os.environ.get("DOWNLOAD_ASSETS", "0") != "0" or os.environ.get("FORCE_ASSETS", "0") != "0":
        raise RuntimeError("Cross-benchmark suite never downloads or replaces assets")
    for task_key, config_path in suite.tasks.items():
        if self_test and str(task_key) not in {str(value) for value in suite.self_test.task_keys}:
            continue
        job_id = f"preflight/{task_key}"
        job = next(value for value in jobs if value.job_id == job_id)
        if _reuse_completed_preflight(
            suite, state, job, self_test=self_test
        ):
            state["phase_summary"] = phase_summary(state, jobs)
            atomic_json(state_path, state)
            continue
        preflight_started = time.time()
        state["jobs"][job_id].update(
            status="running", start_time=preflight_started
        )
        state["phase_summary"] = phase_summary(state, jobs)
        atomic_json(state_path, state)
        report = _run_isolated_preflight(
            suite, job, str(config_path), self_test=self_test
        )
        entry = job.state_fields()
        preflight_ended = time.time()
        entry.update(
            status="completed", start_time=preflight_started,
            end_time=preflight_ended,
            elapsed_seconds=preflight_ended - preflight_started,
            artifact_validation={
                "path": job.artifact_path,
                "sha256": sha256_file(job.artifact_path),
                "report": report,
            },
        )
        state["jobs"][job_id] = entry
        state["phase_summary"] = phase_summary(state, jobs)
        atomic_json(state_path, state)
    state["status"] = "running"
    state["phase_summary"] = phase_summary(state, jobs)
    atomic_json(state_path, state)
    for phase in PHASES[1:]:
        phase_jobs = [job for job in jobs if job.phase == phase]
        # Stability and ablation phases contain both producers and consumers.
        # Preserve phase-level GPU parallelism while never starting an evaluator
        # before its task-specific checkpoint has completed or been reused.
        kind_waves = (
            (("checkpoint",), ("offline", "planning"))
            if phase == "DCT ablations"
            else (("checkpoint",), ("planning",))
            if phase == "Training-seed stability"
            else (tuple(sorted({job.kind for job in phase_jobs})),)
        )
        for kinds in kind_waves:
            wave_jobs = [job for job in phase_jobs if job.kind in kinds]
            pending: list[JobSpec] = []
            for job in wave_jobs:
                candidate = _resume_or_pending(
                    suite, state, job, self_test=self_test
                )
                if candidate is not None:
                    pending.append(candidate)
            state["phase_summary"] = phase_summary(state, jobs)
            atomic_json(state_path, state)
            if pending:
                run_gpu_phase(
                    pending, state["gpu_ids"], state_path, state,
                    lambda job, path: _validate_job(
                        suite, state, job, path, self_test=self_test
                    ),
                )
            state["phase_summary"] = phase_summary(state, jobs)
            atomic_json(state_path, state)
    state["status"] = "completed"
    state["completed_at_unix"] = time.time()
    state["phase_summary"] = phase_summary(state, jobs)
    atomic_json(state_path, state)
    atomic_json(
        resolve_path(str(suite.analysis_root)) / "run_manifest.json",
        {
            "suite": str(suite.suite_name),
            "protocol": str(suite.protocol),
            "state_path": str(state_path),
            "git": state.get("git"),
            "jobs": state["jobs"],
        },
    )


def main() -> None:
    args = _arguments()
    if args.preflight_config:
        _run_preflight_child(args)
        return
    suite = load_suite_config(args.config)
    if args.self_test:
        suite = _self_test_suite(suite)
    jobs = build_job_graph(suite, self_test=args.self_test)
    if args.dry_run:
        _dry_run(suite, jobs)
        return
    try:
        _run(suite, jobs, self_test=args.self_test)
        if args.self_test:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/monitor_cross_benchmark_suite.py",
                    "--state",
                    str(suite.state_path),
                    "--suite-config",
                    args.config,
                    "--once",
                ],
                cwd=project_root(),
                check=True,
            )
    except Exception as error:
        state_path = resolve_path(str(suite.state_path))
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file() else {"suite": str(suite.suite_name), "jobs": {}}
        )
        state.update(
            status="failed", failed_at_unix=time.time(),
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
        for entry in state.get("jobs", {}).values():
            if entry.get("status") == "running":
                entry.update(
                    status="failed", end_time=time.time(),
                    error=f"{type(error).__name__}: {error}",
                )
        for job in jobs:
            if job.job_id not in state["jobs"]:
                entry = job.state_fields()
                entry.update(
                    status="blocked",
                    error="blocked because an earlier suite job failed",
                )
                state["jobs"][job.job_id] = entry
            elif state["jobs"][job.job_id].get("status") == "pending":
                state["jobs"][job.job_id].update(
                    status="blocked",
                    error="blocked because an earlier suite job failed",
                )
        state["phase_summary"] = phase_summary(state, jobs)
        from wm_adapter.benchmarks.base import atomic_json
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
