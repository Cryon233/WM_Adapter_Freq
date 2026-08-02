from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import h5py
import torch
from omegaconf import OmegaConf

from wm_adapter.data.feature_cache import ARRAY_KEYS, CACHE_SCHEMA_VERSION
from wm_adapter.utils.checkpoints import (
    UPSTREAM_COMMITS,
    git_commit,
    load_method_checkpoint,
    sha256_file,
)
from wm_adapter.utils.reproducibility import project_root, resolve_path


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _experiment_config(path: str | Path) -> Any:
    experiment = OmegaConf.load(resolve_path(path))
    model = OmegaConf.load(resolve_path(str(experiment.model_config)))
    merged = OmegaConf.merge(experiment, {"model": model})
    OmegaConf.resolve(merged)
    return merged


def preflight_resources(config_path: str | Path, *, require_cuda: bool = True) -> dict[str, Any]:
    cfg = _experiment_config(config_path)
    errors: list[str] = []
    resources: dict[str, Any] = {}
    if require_cuda and not torch.cuda.is_available():
        errors.append("CUDA is not available")
    else:
        resources["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        resources["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    path_fields = {
        "jepa_checkpoint": cfg.model.jepa_checkpoint,
        "dinov3_checkpoint": cfg.model.dinov3_checkpoint,
        "robocasa_hdf5": cfg.paths.robocasa_hdf5,
        "official_planning_config": cfg.model.official_planning_config,
    }
    for name, value in path_fields.items():
        if not str(value).strip():
            errors.append(f"{name} is empty")
            continue
        path = resolve_path(str(value))
        resources[name] = str(path)
        if not path.is_file():
            errors.append(f"{name} does not exist: {path}")
    dataset_root = resolve_path(str(cfg.paths.dataset_root)) if str(cfg.paths.dataset_root).strip() else None
    resources["dataset_root"] = str(dataset_root) if dataset_root is not None else ""
    if dataset_root is None or not dataset_root.is_dir():
        errors.append(f"dataset_root does not exist: {dataset_root}")
    third_party = resolve_path(str(cfg.model.third_party_root))
    resources["upstream_commits"] = {}
    for name, expected in UPSTREAM_COMMITS.items():
        repo = third_party / name
        if not (repo / ".git").exists():
            errors.append(
                f"Upstream checkout is not an independent Git repository: {repo}"
            )
            continue
        try:
            actual = git_commit(repo)
        except (FileNotFoundError, RuntimeError) as error:
            errors.append(str(error))
            continue
        resources["upstream_commits"][name] = actual
        if actual != expected:
            errors.append(f"{name} commit mismatch: expected {expected}, found {actual}")
    asset_root = (
        third_party
        / "robocasa"
        / "robocasa"
        / "models"
        / "assets"
        / "objects"
    )
    resources["robocasa_asset_root"] = str(asset_root)
    if not asset_root.is_dir() or not any(asset_root.iterdir()):
        errors.append(
            "Downloaded RoboCasa object assets are unavailable or empty: "
            f"{asset_root}"
        )
    report = {
        "passed": not errors,
        "errors": errors,
        "resources": resources,
        "checked_at_unix": time.time(),
    }
    if errors:
        raise RuntimeError("Paper-suite preflight failed:\n- " + "\n- ".join(errors))
    return report


def validate_feature_cache(path: str | Path, expected_windows: int) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Feature cache is missing: {resolved}")
    with h5py.File(resolved, "r", libver="latest", swmr=True) as cache:
        if not bool(cache.attrs.get("finalized", False)):
            raise RuntimeError(f"Feature cache is not finalized: {resolved}")
        if str(cache.attrs.get("schema_version")) != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Feature cache schema mismatch at {resolved}: {cache.attrs.get('schema_version')}"
            )
        missing = sorted(set(ARRAY_KEYS).difference(cache.keys()))
        if missing:
            raise RuntimeError(f"Feature cache is missing datasets {missing}: {resolved}")
        lengths = {key: int(cache[key].shape[0]) for key in ARRAY_KEYS}
        if set(lengths.values()) != {expected_windows}:
            raise RuntimeError(
                f"Feature cache window count mismatch: expected={expected_windows}, actual={lengths}"
            )
        for key in ARRAY_KEYS:
            if cache[key].size and not bool(
                torch.isfinite(torch.as_tensor(cache[key][0])).all()
            ):
                raise RuntimeError(f"Feature cache contains a non-finite value in {key}[0]: {resolved}")
        return {
            "path": str(resolved),
            "window_count": expected_windows,
            "cache_fingerprint": str(cache.attrs["cache_fingerprint"]),
            "shapes": {key: list(cache[key].shape) for key in ARRAY_KEYS},
        }


def validate_method_checkpoint(
    path: str | Path,
    method_name: str,
    cache_fingerprint: str,
    *,
    expected_method_options: dict[str, Any] | None = None,
    expected_training_weights: tuple[float, float] | None = None,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = load_method_checkpoint(resolved)
    if str(payload["method_name"]) != method_name:
        raise RuntimeError(
            f"Method checkpoint mismatch: expected={method_name}, actual={payload['method_name']}"
        )
    if str(payload["cache_fingerprint"]) != cache_fingerprint:
        raise RuntimeError(f"Method checkpoint cache fingerprint mismatch: {resolved}")
    if expected_method_options is not None:
        actual_options = dict(payload["method_config"])
        defaults = {
            "temporal_pool": "mean",
            "mask_type": "adaptive",
            "use_rms_norm": True,
        }
        mismatched_options = {
            key: {
                "expected": value,
                "actual": actual_options.get(key, defaults.get(key)),
            }
            for key, value in expected_method_options.items()
            if key != "name" and actual_options.get(key, defaults.get(key)) != value
        }
        if mismatched_options:
            raise RuntimeError(
                f"Method checkpoint options do not match {resolved}: {mismatched_options}"
            )
    if expected_training_weights is not None:
        training = payload["training_config"]
        actual_weights = (
            float(training.get("canonical_weight", 1.0)),
            float(training.get("dynamics_weight", 1.0)),
        )
        if actual_weights != expected_training_weights:
            raise RuntimeError(
                f"Method checkpoint loss weights mismatch at {resolved}: "
                f"expected={expected_training_weights}, actual={actual_weights}"
            )
    count = int(payload["trainable_parameter_count"])
    if count <= 0 or not payload["peft_state_dict"]:
        raise RuntimeError(f"Method checkpoint has no PEFT parameters: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "parameter_count": count,
    }


def validate_planning_result(
    path: str | Path,
    expected_episodes: int,
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Planning result is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if expected_metadata is not None:
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected_metadata.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Planning result metadata mismatch at {resolved}: {mismatches}"
            )
    successes = payload.get("per_episode_success")
    if not isinstance(successes, list) or len(successes) < expected_episodes:
        raise RuntimeError(
            f"Planning result has fewer than {expected_episodes} episodes: {resolved}"
        )
    selected = [bool(value) for value in successes[:expected_episodes]]
    if int(payload.get("total_episodes", -1)) < expected_episodes:
        raise RuntimeError(f"Planning result total_episodes is incomplete: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "available_episodes": len(successes),
        "used_episodes": expected_episodes,
        "success_count": sum(selected),
        "success_rate": sum(selected) / expected_episodes,
    }


def validate_offline_result(path: str | Path, expected_windows: int) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Offline metrics are missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if int(payload.get("window_count", -1)) != expected_windows:
        raise RuntimeError(f"Offline result window count mismatch: {resolved}")
    identities = payload.get("window_identities")
    if not isinstance(identities, list) or len(identities) != expected_windows:
        raise RuntimeError(f"Offline result identities are incomplete: {resolved}")
    domains = payload.get("domains", {})
    if not {"clean", "ood"}.issubset(domains):
        raise RuntimeError(f"Offline result must contain clean and ood domains: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


@dataclass(frozen=True)
class GPUJob:
    job_id: str
    phase: str
    command: tuple[str, ...]
    log_path: str
    artifact_path: str
    validate: Callable[[], dict[str, Any]]


def run_gpu_jobs(
    jobs: Sequence[GPUJob],
    gpu_ids: Sequence[int],
    state_path: str | Path,
    *,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not gpu_ids:
        raise RuntimeError("At least one GPU ID is required for paper-suite jobs")
    root = project_root()
    state: dict[str, Any] = initial_state or {"jobs": {}, "started_at_unix": time.time()}
    state.setdefault("jobs", {})
    pending = list(jobs)
    running: dict[int, tuple[GPUJob, subprocess.Popen[str], Any, float]] = {}
    failed = False

    def write_state() -> None:
        state["updated_at_unix"] = time.time()
        atomic_write_json(state_path, state)

    while pending or running:
        if not failed:
            for gpu in gpu_ids:
                if gpu in running or not pending:
                    continue
                job = pending.pop(0)
                log_path = resolve_path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    list(job.command),
                    cwd=root,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                started = time.time()
                running[gpu] = (job, process, log_handle, started)
                state["jobs"][job.job_id] = {
                    "status": "running",
                    "phase": job.phase,
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": list(job.command),
                    "log_path": str(log_path),
                    "artifact_path": str(resolve_path(job.artifact_path)),
                    "started_at_unix": started,
                }
                write_state()
        if not running and failed:
            break
        time.sleep(1.0)
        for gpu, (job, process, log_handle, started) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            entry = state["jobs"][job.job_id]
            entry["ended_at_unix"] = time.time()
            entry["elapsed_seconds"] = entry["ended_at_unix"] - started
            entry["returncode"] = code
            if code == 0:
                try:
                    entry["artifact"] = job.validate()
                    entry["status"] = "completed"
                except Exception as error:
                    entry["status"] = "failed"
                    entry["error"] = f"{type(error).__name__}: {error}"
                    failed = True
            else:
                entry["status"] = "failed"
                entry["error"] = f"process exited with code {code}"
                failed = True
            del running[gpu]
            write_state()
    if failed:
        for job in pending:
            state["jobs"][job.job_id] = {"status": "blocked", "phase": job.phase}
        write_state()
        failed_jobs = [key for key, value in state["jobs"].items() if value["status"] == "failed"]
        raise RuntimeError(f"Paper-suite jobs failed: {failed_jobs}")
    return state
