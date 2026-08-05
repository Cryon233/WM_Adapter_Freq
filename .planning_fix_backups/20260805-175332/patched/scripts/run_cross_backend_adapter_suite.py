from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import torch

from wm_adapter.benchmarks.base import atomic_json, canonical_sha256
from wm_adapter.data.feature_cache_v2 import (
    CACHE_SCHEMA_VERSION_V2,
    verify_cache_content_v2,
)
from wm_adapter.experiments.cross_backend_jobs import (
    EXPECTED_PLANNING_EPISODES,
    EXPECTED_PLANNING_JOBS,
    build_cross_backend_job_graph,
    cross_backend_rollout_counts,
)
from wm_adapter.experiments.cross_benchmark import (
    JobSpec,
    archive_incomplete,
    block_job_for_failed_dependencies,
    load_suite_config,
    phase_summary,
    run_gpu_phase,
)
from wm_adapter.planning.jepa_wm_planner import (
    EVALUATION_PROTOCOL_VERSION,
)
from wm_adapter.utils.checkpoints import git_commit, sha256_file
from wm_adapter.utils.reproducibility import project_root, resolve_path


DEFAULT_CONFIG = "configs/experiment/cross_backend_adapter_v1.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated cross-backend adapter experiment graph"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _gpu_ids(suite: Any) -> list[int]:
    raw = os.environ.get("GPUS")
    values = (
        [int(value) for value in raw.split(",") if value.strip()]
        if raw
        else [int(value) for value in suite.gpu.default_ids]
    )
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError(f"GPUS must contain unique non-negative IDs: {raw!r}")
    return values


def _payload(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact root is not a mapping: {resolved}")
    return value


def _validate_evaluation_manifest(job: JobSpec) -> dict[str, Any]:
    payload = _payload(job.artifact_path)
    supplied = str(payload.get("evaluation_manifest_sha256", ""))
    canonical = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "evaluation_manifest_sha256"
        }
    )
    instances = payload.get("instances")
    if supplied != canonical:
        raise RuntimeError(
            f"Evaluation manifest hash mismatch: {job.artifact_path}"
        )
    if payload.get("task_key") != job.task:
        raise RuntimeError(
            f"Evaluation manifest task mismatch: expected={job.task}, "
            f"actual={payload.get('task_key')}"
        )
    if not isinstance(instances, list) or len(instances) < int(
        job.required_count or 0
    ):
        raise RuntimeError(
            f"Evaluation manifest has too few instances: {job.artifact_path}"
        )
    fixed_goal_span_steps = payload.get("fixed_goal_span_steps")
    if job.task == "robocasa_place":
        spans = [
            int(instance["segment_end"]) - int(instance["segment_start"])
            for instance in instances[: int(job.required_count or 0)]
        ]
        if (
            set(spans) != {25}
            or fixed_goal_span_steps != 25
            or bool(payload.get("legacy_place_reuse_compatible", False))
        ):
            raise RuntimeError(
                "RoboCasa Place evaluation manifest is not protocol-2.1 fixed-span: "
                f"spans={sorted(set(spans))}, "
                f"fixed_goal_span_steps={fixed_goal_span_steps}, "
                f"legacy={payload.get('legacy_place_reuse_compatible')}, "
                f"path={job.artifact_path}"
            )
    return {
        "path": str(resolve_path(job.artifact_path)),
        "sha256": sha256_file(job.artifact_path),
        "evaluation_manifest_sha256": supplied,
        "instances": len(instances),
    }


def _validate_cache(job: JobSpec) -> dict[str, Any]:
    verified = verify_cache_content_v2(job.artifact_path, chunk_windows=8)
    with h5py.File(resolve_path(job.artifact_path), "r", libver="latest", swmr=True) as handle:
        actual = {
            "schema": str(handle.attrs.get("schema_version", "")),
            "backend": str(handle.attrs.get("backend", "")),
            "task": str(handle.attrs.get("task_key", "")),
            "windows": int(handle[V2_FIRST_KEY].shape[0]),
            "requested_windows": int(handle.attrs.get("requested_window_count", -1)),
            "unique_windows": int(handle.attrs.get("unique_window_count", -1)),
            "sampling_with_replacement": bool(
                handle.attrs.get("sampling_with_replacement", False)
            ),
        }
    expected = {
        "schema": CACHE_SCHEMA_VERSION_V2,
        "backend": str(job.backend),
        "task": job.task,
        "windows": int(job.required_count or 0),
        "requested_windows": int(job.required_count or 0),
    }
    scalar_actual = {
        key: actual[key]
        for key in (
            "schema",
            "backend",
            "task",
            "windows",
            "requested_windows",
        )
    }
    if scalar_actual != expected:
        raise RuntimeError(
            f"Cross-backend cache contract mismatch: expected={expected}, actual={scalar_actual}"
        )
    if not 0 < actual["unique_windows"] <= actual["windows"]:
        raise RuntimeError(f"Cross-backend cache unique-window count is invalid: {actual}")
    if actual["sampling_with_replacement"] != (
        actual["unique_windows"] < actual["windows"]
    ):
        raise RuntimeError(f"Cross-backend cache replacement metadata is inconsistent: {actual}")
    return {**verified, **actual, "path": str(resolve_path(job.artifact_path))}


V2_FIRST_KEY = "clean_context_middle_tokens"


def _cache_validation(state: dict[str, Any], job: JobSpec) -> dict[str, Any]:
    dependency = f"cache/{job.backend}/{job.task}"
    value = state["jobs"][dependency].get("artifact_validation")
    if not isinstance(value, dict):
        raise RuntimeError(f"Validated cache dependency is missing: {dependency}")
    return value


def _checkpoint_dependency(job: JobSpec) -> str | None:
    return next(
        (value for value in job.dependencies if value.startswith("train/")),
        None,
    )


def _validate_checkpoint(
    state: dict[str, Any], job: JobSpec
) -> dict[str, Any]:
    payload = torch.load(resolve_path(job.artifact_path), map_location="cpu", weights_only=False)
    cache = _cache_validation(state, job)
    expected = {
        "backend": str(job.backend),
        "method_name": str(job.method),
        "training_seed": int(job.seed),
        "cache_fingerprint": str(cache["cache_fingerprint"]),
        "cache_file_sha256": str(cache["cache_file_sha256"]),
        "loss_name": "unified_trajectory_mse",
        "max_optimizer_steps": 2000,
        "completed_optimizer_steps": 2000,
    }
    actual = {
        key: payload.get(key)
        for key in expected
    }
    if actual != expected:
        raise RuntimeError(
            f"Cross-backend checkpoint contract mismatch: expected={expected}, actual={actual}"
        )
    data = dict(payload.get("data_metadata", {}))
    if data.get("backend") != job.backend or data.get("task_key") != job.task:
        raise RuntimeError(
            "Cross-backend checkpoint data identity mismatch: "
            f"backend={data.get('backend')}, task={data.get('task_key')}"
        )
    return {
        "path": str(resolve_path(job.artifact_path)),
        "sha256": sha256_file(job.artifact_path),
        "cache_fingerprint": expected["cache_fingerprint"],
        "cache_file_sha256": expected["cache_file_sha256"],
        "backend": job.backend,
        "task": job.task,
        "method": job.method,
        "seed": job.seed,
    }


def _validate_offline(state: dict[str, Any], job: JobSpec) -> dict[str, Any]:
    payload = _payload(job.artifact_path)
    cache = _cache_validation(state, job)
    expected_seed = None if job.method == "base" else int(job.seed)
    expected = {
        "schema_version": "cross_backend_offline_mse_v1",
        "backend": str(job.backend),
        "task": job.task,
        "method": str(job.method),
        "training_seed": expected_seed,
        "requested_window_count": int(job.required_count or 0),
        "cache_fingerprint": str(cache["cache_fingerprint"]),
        "cache_file_sha256": str(cache["cache_file_sha256"]),
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Cross-backend offline contract mismatch: expected={expected}, actual={actual}"
        )
    actual_windows = int(payload.get("window_count", -1))
    unique_windows = int(payload.get("unique_window_count", -1))
    replacement = bool(payload.get("sampling_with_replacement", True))
    allow_fewer = job.task in {"robocasa_reach", "robocasa_place"}
    if (
        actual_windows <= 0
        or actual_windows > int(job.required_count or 0)
        or unique_windows != actual_windows
        or replacement
        or (not allow_fewer and actual_windows != int(job.required_count or 0))
    ):
        raise RuntimeError(
            "Cross-backend Offline unique-window contract mismatch: "
            f"requested={job.required_count}, actual={actual_windows}, "
            f"unique={unique_windows}, sampling_with_replacement={replacement}"
        )
    required_metrics = {
        "h1_autoregressive_latent_mse",
        "h2_autoregressive_latent_mse",
        "h3_autoregressive_latent_mse",
        "future_mean_mse",
        "terminal_mse",
        "unified_6frame_trajectory_mse",
    }
    domains = payload.get("domains")
    if not isinstance(domains, dict) or set(domains) != {"clean", "ood"}:
        raise RuntimeError(f"Offline artifact lacks clean/OOD domains: {job.artifact_path}")
    for domain, metrics in domains.items():
        if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
            raise RuntimeError(
                f"Offline {domain} metrics are incomplete: required={sorted(required_metrics)}"
            )
    return {
        "path": str(resolve_path(job.artifact_path)),
        "sha256": sha256_file(job.artifact_path),
        "window_count": actual_windows,
        "unique_window_count": unique_windows,
        "sampling_with_replacement": replacement,
        **expected,
    }


def _validate_planning(state: dict[str, Any], job: JobSpec) -> dict[str, Any]:
    payload = _payload(job.artifact_path)
    cache = _cache_validation(state, job)
    success = payload.get("per_episode_success")
    seeds = payload.get("seeds", {})
    expected = {
        "backend": str(job.backend),
        "task": job.task,
        "method": str(job.method),
        "domain": str(job.domain),
        "episodes": int(job.required_count or 0),
        "training_seed": int(job.seed),
        "evaluation_seed": int(job.seed),
        "cache_fingerprint": str(cache["cache_fingerprint"]),
        "cache_file_sha256": str(cache["cache_file_sha256"]),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
    }
    actual = {
        "backend": payload.get("backend"),
        "task": payload.get("task"),
        "method": payload.get("method"),
        "domain": payload.get("domain"),
        "episodes": len(success) if isinstance(success, list) else -1,
        "training_seed": seeds.get("training"),
        "evaluation_seed": seeds.get("evaluation"),
        "cache_fingerprint": payload.get("cache_fingerprint"),
        "cache_file_sha256": payload.get("cache_file_sha256"),
        "evaluation_protocol_version": payload.get(
            "evaluation_protocol_version"
        ),
    }
    if actual != expected:
        raise RuntimeError(
            f"Cross-backend planning contract mismatch: expected={expected}, actual={actual}"
        )
    computed_success_count = sum(bool(value) for value in success)
    reported_success_count = int(payload.get("success_count", -1))
    reported_total = int(payload.get("total_episodes", -1))
    reported_rate = float(payload.get("success_rate", -1.0))
    expected_rate = computed_success_count / len(success)
    if (
        reported_success_count != computed_success_count
        or reported_total != len(success)
        or abs(reported_rate - expected_rate) > 1.0e-12
    ):
        raise RuntimeError(
            "Planning success summary is inconsistent with per_episode_success: "
            f"reported_count={reported_success_count}, "
            f"computed_count={computed_success_count}, "
            f"reported_total={reported_total}, total={len(success)}, "
            f"reported_rate={reported_rate}, expected_rate={expected_rate}"
        )
    cem = payload.get("cem", {})
    required_cem = {
        "iterations": 15,
        "num_samples": 300,
        "num_elites": 10,
        "horizon": 3,
        "num_act_stepped": 1,
        "candidate_chunk_size": 300,
    }
    actual_cem = {key: int(cem.get(key, -1)) for key in required_cem}
    if actual_cem != required_cem:
        raise RuntimeError(
            f"Formal CEM contract mismatch: expected={required_cem}, actual={actual_cem}"
        )
    if payload.get("goal_encoder") != "frozen_base":
        raise RuntimeError("Planning goal must use the frozen Base encoder")
    return {
        "path": str(resolve_path(job.artifact_path)),
        "sha256": sha256_file(job.artifact_path),
        "success_count": sum(bool(value) for value in success),
        "episodes": len(success),
        **expected,
    }


def _validate(
    state: dict[str, Any], job: JobSpec, path: str | None = None
) -> dict[str, Any]:
    if path is not None and resolve_path(path) != resolve_path(job.artifact_path):
        raise RuntimeError("Cross-backend validator does not mutate or alias source artifacts")
    if job.kind == "evaluation_manifest":
        return _validate_evaluation_manifest(job)
    if job.kind == "cache":
        return _validate_cache(job)
    if job.kind == "checkpoint":
        return _validate_checkpoint(state, job)
    if job.kind == "offline":
        return _validate_offline(state, job)
    if job.kind == "planning":
        return _validate_planning(state, job)
    resolved = resolve_path(job.artifact_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Analysis artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _state_entry(job: JobSpec, status: str) -> dict[str, Any]:
    return job.state_fields() | {"status": status}


def _archive_invalid(job: JobSpec) -> None:
    artifact = resolve_path(job.artifact_path)
    if not artifact.is_file():
        return
    archive_incomplete(artifact)
    if job.kind == "offline":
        rows = artifact.parent / "per_window.jsonl"
        if rows.is_file():
            archive_incomplete(rows)


def _initialize_state(suite: Any, jobs: list[JobSpec]) -> dict[str, Any]:
    state_path = resolve_path(str(suite.state_path))
    old = _payload(state_path) if state_path.is_file() else {}
    state: dict[str, Any] = {
        "suite": str(suite.suite_name),
        "protocol": str(suite.protocol),
        "suite_config_path": str(resolve_path(str(suite._suite_config_path))),
        "status": "initializing",
        "started_at_unix": old.get("started_at_unix", time.time()),
        "last_started_at_unix": time.time(),
        "restart_count": int(old.get("restart_count", 0)) + bool(old),
        "jobs": {},
        "git": {
            "commit": git_commit(project_root()),
            "dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=project_root(),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
        "gpu_ids": _gpu_ids(suite),
        "self_test": False,
    }
    # Publish a fresh lifecycle immediately. Deep validation of an existing
    # cache can take minutes; keeping the previous state until it finishes
    # makes a healthy restart look stuck on stale failures.
    state["jobs"] = {
        job.job_id: _state_entry(job, "pending") for job in jobs
    }
    state["phase_summary"] = phase_summary(state, jobs)
    atomic_json(state_path, state)
    for job in jobs:
        artifact = resolve_path(job.artifact_path)
        if artifact.is_file():
            # Dependencies are rebuilt in graph order, so referenced cache and
            # checkpoint validation records are available before consumers.
            validating = _state_entry(job, "running")
            validating.update(
                pid=os.getpid(),
                gpu=None,
                start_time=time.time(),
                error="validating existing artifact for reuse",
            )
            state["jobs"][job.job_id] = validating
            state["phase_summary"] = phase_summary(state, jobs)
            atomic_json(state_path, state)
            print(
                f"RESUME_VALIDATION status=started job={job.job_id} "
                f"artifact={artifact}",
                flush=True,
            )
            try:
                validation = _validate(state, job)
            except Exception as error:
                _archive_invalid(job)
                state["jobs"][job.job_id] = _state_entry(job, "pending")
                print(
                    f"RESUME_VALIDATION status=rebuild job={job.job_id} "
                    f"error_type={type(error).__name__}",
                    flush=True,
                )
            else:
                entry = _state_entry(job, "reused")
                entry["artifact_validation"] = validation
                state["jobs"][job.job_id] = entry
                print(
                    f"RESUME_VALIDATION status=reused job={job.job_id}",
                    flush=True,
                )
        else:
            state["jobs"][job.job_id] = _state_entry(job, "pending")
        state["phase_summary"] = phase_summary(state, jobs)
        atomic_json(state_path, state)
    state["status"] = "running"
    state["phase_summary"] = phase_summary(state, jobs)
    atomic_json(state_path, state)
    return state


def _run_phase(
    suite: Any, state: dict[str, Any], jobs: list[JobSpec], phase: str
) -> None:
    state_path = resolve_path(str(suite.state_path))
    phase_jobs = [job for job in jobs if job.phase == phase]
    while True:
        pending = [
            job
            for job in phase_jobs
            if state["jobs"][job.job_id]["status"] == "pending"
        ]
        if not pending:
            return
        for job in pending:
            block_job_for_failed_dependencies(job, state)
        eligible = [
            job
            for job in pending
            if state["jobs"][job.job_id]["status"] == "pending"
            and all(
                state["jobs"][dependency]["status"] in {"completed", "reused"}
                for dependency in job.dependencies
            )
        ]
        if not eligible:
            for job in pending:
                if state["jobs"][job.job_id]["status"] == "pending":
                    entry = _state_entry(job, "blocked")
                    entry["error"] = "no schedulable dependency path remains"
                    state["jobs"][job.job_id] = entry
            state["phase_summary"] = phase_summary(state, jobs)
            atomic_json(state_path, state)
            return
        run_gpu_phase(
            eligible,
            state["gpu_ids"],
            state_path,
            state,
            lambda job, path: _validate(state, job, path),
            raise_on_failure=False,
        )
        state["phase_summary"] = phase_summary(state, jobs)
        atomic_json(state_path, state)


def _dry_run(suite: Any, jobs: list[JobSpec]) -> None:
    counts = cross_backend_rollout_counts(jobs)
    phase_counts = {
        str(phase): sum(job.phase == str(phase) for job in jobs)
        for phase in suite.phases
    }
    report = {
        "suite": str(suite.suite_name),
        "tasks": list(suite.tasks.keys()),
        "backends": {
            str(name): list(spec.methods)
            for name, spec in suite.backends.items()
        },
        "seeds": [int(value) for value in suite.seeds],
        "domains": list(suite.domains),
        "expected_jobs": len(jobs),
        "phase_job_counts": phase_counts,
        "rollouts": counts,
        "expected_planning_jobs": EXPECTED_PLANNING_JOBS,
        "expected_closed_loop_episodes": EXPECTED_PLANNING_EPISODES,
        "contains_robocasa_articulated": any(
            job.task == "robocasa_articulated" for job in jobs
        ),
        "gpu_queue": {
            "ids": _gpu_ids(suite),
            "one_heavy_job_per_gpu": True,
        },
        "required_environment": [
            "JEPA_WM_DROID_CKPT",
            "DINOV3_VITL16_CKPT",
            "DINO_WM_DROID_CKPT",
            "DINOV2_VITS14_CKPT",
            "LIBERO_ROOT",
            "LIBERO_DATA_ROOT",
            "ROBOCASA_PLACE_HDF5",
        ],
        "artifact_roots": {
            "cache": str(suite.storage_root),
            "checkpoint": str(suite.checkpoint_root),
            "output": str(suite.output_root),
            "logs": str(suite.log_root),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = _arguments()
    suite = load_suite_config(args.config)
    jobs = build_cross_backend_job_graph(suite)
    if args.dry_run:
        _dry_run(suite, jobs)
        return
    if os.environ.get("DOWNLOAD_ASSETS", "0") != "0" or os.environ.get(
        "FORCE_ASSETS", "0"
    ) != "0":
        raise RuntimeError("cross_backend_adapter_v1 never downloads or replaces assets")
    state = _initialize_state(suite, jobs)
    for phase_value in suite.phases:
        _run_phase(suite, state, jobs, str(phase_value))
    failed = any(
        value.get("status") in {"failed", "blocked"}
        for value in state["jobs"].values()
    )
    state["status"] = "completed_with_failures" if failed else "completed"
    state["completed_at_unix"] = time.time()
    state["phase_summary"] = phase_summary(state, jobs)
    atomic_json(suite.state_path, state)


if __name__ == "__main__":
    main()
