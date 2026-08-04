from __future__ import annotations

import json
import hashlib
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
from omegaconf import DictConfig, OmegaConf

from wm_adapter.benchmarks.base import (
    ActionTransform,
    TASK_MANIFEST_SCHEMA,
    atomic_json,
    canonical_sha256,
)
from wm_adapter.data.feature_cache import ARRAY_KEYS, CACHE_SCHEMA_VERSION
from wm_adapter.data.feature_cache_v2 import (
    CACHE_SCHEMA_VERSION_V2,
    V2_ARRAY_KEYS,
    cache_fingerprint_v2,
    verify_cache_content_v2,
)
from wm_adapter.training.trainer_v2 import CHECKPOINT_SCHEMA_V2
from wm_adapter.utils.checkpoints import load_method_checkpoint, sha256_file
from wm_adapter.utils.reproducibility import project_root, resolve_path


PHASES = (
    "Task and resource preflight",
    "Feature caches",
    "Main adapter training",
    "Main offline evaluation",
    "Main closed-loop planning",
    "Training-seed stability",
    "OOD severity",
    "DCT ablations",
    "Final analysis",
)

_DEFAULT_MIN_GPU_FREE_MIB = 18 * 1024
_CUDA_OOM_SIGNATURES = (
    b"cuda out of memory",
    b"torch.outofmemoryerror",
    b"cuda error: out of memory",
)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    phase: str
    benchmark: str
    task: str
    command: tuple[str, ...]
    log_path: str
    artifact_path: str
    kind: str
    backend: str | None = None
    method: str | None = None
    domain: str | None = None
    seed: int | None = None
    severity: float | None = None
    variant: str | None = None
    required_count: int | None = None
    reuse_sources: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)

    def state_fields(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("reuse_sources")
        payload["artifact_validation"] = None
        payload["reuse_source"] = None
        payload["gpu"] = None
        payload["pid"] = None
        payload["start_time"] = None
        payload["end_time"] = None
        payload["elapsed_seconds"] = None
        payload["return_code"] = None
        payload["error"] = None
        return payload


def load_suite_config(path: str | Path) -> DictConfig:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Cross-benchmark suite config does not exist: {resolved}")
    cfg = OmegaConf.load(resolved)
    OmegaConf.resolve(cfg)
    cfg["_suite_config_path"] = str(resolved)
    return cfg


def load_task_config(path: str | Path, overrides: Iterable[str] = ()) -> DictConfig:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Cross-benchmark task config does not exist: {resolved}")
    task = OmegaConf.merge(OmegaConf.load(resolved), OmegaConf.from_dotlist(list(overrides)))
    model_path = resolve_path(str(task.model_config))
    model = OmegaConf.load(model_path)
    merged = OmegaConf.merge(task, {"model": model})
    OmegaConf.resolve(merged)
    return merged


def benchmark_subprocess_environment(
    benchmark: str,
    *,
    gpu: int | None = None,
) -> dict[str, str]:
    """Build an isolated subprocess environment for one benchmark.

    The pinned LIBERO checkout requires robosuite 1.4, while the pinned
    RoboCasa checkout uses its newer compositional-robot fork.  They cannot be
    imported safely in the same interpreter, so only LIBERO child processes
    receive the separately installed official compatibility tree.
    """

    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if benchmark == "robocasa":
            environment["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    if benchmark != "libero":
        return environment
    configured = environment.get("LIBERO_ROBOSUITE_ROOT", "").strip()
    dependency_root = resolve_path(
        configured or project_root() / "storage" / "dependencies" / "robosuite_1_4"
    )
    package = dependency_root / "robosuite" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(
            "LIBERO requires an isolated robosuite 1.4 installation; "
            f"expected package at {package}. Set LIBERO_ROBOSUITE_ROOT to the "
            "directory containing the official robosuite 1.4 package."
        )
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(dependency_root)
        if not existing
        else f"{dependency_root}{os.pathsep}{existing}"
    )
    environment["LIBERO_ROBOSUITE_ROOT"] = str(dependency_root)
    return environment


def archive_incomplete(path: str | Path) -> Path:
    source = resolve_path(path)
    digest = sha256_file(source)[:12]
    destination = source.with_name(f"{source.name}.incomplete-{digest}")
    archive_index = 2
    while destination.exists():
        destination = source.with_name(
            f"{source.name}.incomplete-{digest}.{archive_index}"
        )
        archive_index += 1
    source.replace(destination)
    return destination


def validate_task_manifest(
    path: str | Path,
    task: str,
    *,
    allow_legacy_place: bool = False,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if str(payload.get("task_key")) != task:
        raise RuntimeError(
            f"Task manifest mismatch: expected={task}, actual={payload.get('task_key')}, path={resolved}"
        )
    if str(payload.get("status")) != "resolved":
        raise RuntimeError(f"Task manifest is unresolved: {resolved}")
    supplied_hash = str(payload.get("task_manifest_sha256", ""))
    hash_payload = {
        key: value for key, value in payload.items() if key != "task_manifest_sha256"
    }
    if supplied_hash != canonical_sha256(hash_payload):
        raise RuntimeError(f"Task manifest fingerprint is invalid: {resolved}")
    legacy = (
        allow_legacy_place
        and payload.get("benchmark") == "robocasa"
        and task == "robocasa_place"
    )
    if not legacy and payload.get("schema_version") != TASK_MANIFEST_SCHEMA:
        raise RuntimeError(f"Task manifest schema mismatch: {resolved}")
    camera_fields = (
        "camera_height",
        "camera_width",
        "camera_channel_order",
        "camera_vertical_flip",
    )
    missing_camera = [key for key in camera_fields if payload.get(key) is None]
    if payload.get("benchmark") == "libero":
        if missing_camera:
            raise RuntimeError(
                f"Strict LIBERO task manifest lacks camera contract {missing_camera}: {resolved}"
            )
        transform = payload.get("action_transform")
        if not isinstance(transform, dict):
            raise RuntimeError(
                f"Strict LIBERO task manifest lacks an action transform: {resolved}"
            )
        if ActionTransform.from_dict(transform).as_dict() != transform:
            raise RuntimeError(
                f"LIBERO task manifest action transform is not canonical: {resolved}"
            )
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_cache(
    path: str | Path,
    windows: int,
    *,
    benchmark: str,
    task: str,
    allow_legacy_place: bool = False,
    expected_task_manifest_sha256: str | None = None,
    expected_action_transform: dict[str, Any] | None = None,
    expected_camera_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    with h5py.File(resolved, "r", libver="latest", swmr=True) as handle:
        if not bool(handle.attrs.get("finalized", False)):
            raise RuntimeError(f"Feature cache is not finalized: {resolved}")
        if str(handle.attrs.get("schema_version")) != CACHE_SCHEMA_VERSION:
            raise RuntimeError(f"Feature cache schema mismatch: {resolved}")
        missing = sorted(set(ARRAY_KEYS).difference(handle.keys()))
        if missing:
            raise RuntimeError(f"Feature cache is missing arrays {missing}: {resolved}")
        counts = {key: int(handle[key].shape[0]) for key in ARRAY_KEYS}
        if any(value < windows for value in counts.values()):
            raise RuntimeError(
                f"Feature cache has fewer than {windows} windows: {counts}, path={resolved}"
            )
        actual_benchmark = str(handle.attrs.get("benchmark", ""))
        actual_task = str(handle.attrs.get("task_key", ""))
        if (actual_benchmark, actual_task) != (benchmark, task):
            legacy_ok = allow_legacy_place and task == "robocasa_place" and not actual_benchmark
            if not legacy_ok:
                raise RuntimeError(
                    "Feature cache benchmark identity mismatch: "
                    f"expected={(benchmark, task)}, actual={(actual_benchmark, actual_task)}"
                )
        actual_manifest = str(handle.attrs.get("task_manifest_sha256", ""))
        if expected_task_manifest_sha256 is not None and actual_manifest != expected_task_manifest_sha256:
            legacy_ok = allow_legacy_place and task == "robocasa_place" and not actual_manifest
            if not legacy_ok:
                raise RuntimeError(f"Feature cache task-manifest fingerprint mismatch: {resolved}")
        if expected_action_transform is not None:
            raw_transform = handle.attrs.get("action_transform")
            actual_transform = (
                json.loads(str(raw_transform)) if raw_transform is not None else None
            )
            if actual_transform != expected_action_transform:
                if not (allow_legacy_place and task == "robocasa_place" and actual_transform is None):
                    raise RuntimeError(f"Feature cache action-transform mismatch: {resolved}")
        if expected_camera_contract is not None:
            actual_camera = {
                key: (
                    handle.attrs[key].item()
                    if hasattr(handle.attrs.get(key), "item")
                    else handle.attrs.get(key)
                )
                for key in expected_camera_contract
            }
            if actual_camera != expected_camera_contract:
                if not (
                    allow_legacy_place
                    and task == "robocasa_place"
                    and all(value is None for value in actual_camera.values())
                ):
                    raise RuntimeError(f"Feature cache camera-contract mismatch: {resolved}")
        return {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "cache_fingerprint": str(handle.attrs["cache_fingerprint"]),
            "available_windows": min(counts.values()),
            "used_windows": windows,
            "legacy": not bool(actual_benchmark),
        }


def normalize_metadata_contract(value: Any) -> Any:
    """Normalize JSON-backed metadata without weakening its semantic content."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value[:1] in {"{", "["}:
        return json.loads(value)
    return value


def _decoded_hdf5_attr(value: Any) -> Any:
    return normalize_metadata_contract(value)


def validate_cache_v2(
    path: str | Path,
    windows: int,
    *,
    benchmark: str,
    task: str,
    expected_task_manifest_sha256: str,
    expected_dataset_sha256: str,
    expected_camera_key: str,
    expected_task_upstream_commits: dict[str, str],
    expected_dataset_format: str | None,
    expected_dataset_source_identifier: str | None,
    expected_dataset_revision: str | None,
    expected_robot: str | None,
    expected_gripper: str | None,
    expected_controller_contract: dict[str, Any] | str | None,
    expected_action_transform: dict[str, Any] | None,
    expected_camera_contract: dict[str, Any],
    expected_base_checkpoint_sha256: str,
    expected_dinov3_checkpoint_sha256: str,
    deep_verify: bool = False,
    content_verification_chunk_windows: int = 8,
    include_file_sha256: bool = False,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    with h5py.File(resolved, "r", libver="latest", swmr=True) as handle:
        if str(handle.attrs.get("schema_version", "")) != CACHE_SCHEMA_VERSION_V2:
            raise RuntimeError(f"V2 feature-cache schema mismatch: {resolved}")
        if not bool(handle.attrs.get("finalized", False)):
            raise RuntimeError(f"V2 feature cache is not finalized: {resolved}")
        required_attributes = {
            "array_content_sha256",
            "content_sha256",
            "cache_fingerprint",
            "camera_key",
            "task_upstream_commits",
            "dataset_sha256",
        }
        missing_attributes = sorted(
            key for key in required_attributes if key not in handle.attrs
        )
        if missing_attributes:
            raise RuntimeError(
                f"V2 feature cache is missing metadata {missing_attributes}: {resolved}"
            )
        missing = sorted(set(V2_ARRAY_KEYS).difference(handle.keys()))
        if missing:
            raise RuntimeError(f"V2 feature cache is missing arrays {missing}: {resolved}")
        counts = {key: int(handle[key].shape[0]) for key in V2_ARRAY_KEYS}
        if len(set(counts.values())) != 1 or min(counts.values()) < windows:
            raise RuntimeError(
                f"V2 feature-cache window counts are invalid for required={windows}: "
                f"{counts}, path={resolved}"
            )
        expected_scalars = {
            "benchmark": benchmark,
            "task_key": task,
            "task_manifest_sha256": expected_task_manifest_sha256,
            "dataset_sha256": expected_dataset_sha256,
            "camera_key": expected_camera_key,
            "task_upstream_commits": expected_task_upstream_commits,
            "dataset_format": expected_dataset_format,
            "dataset_source_identifier": expected_dataset_source_identifier,
            "dataset_revision": expected_dataset_revision,
            "robot": expected_robot,
            "gripper": expected_gripper,
            "controller_contract": expected_controller_contract,
            "base_checkpoint_sha256": expected_base_checkpoint_sha256,
            "dinov3_checkpoint_sha256": expected_dinov3_checkpoint_sha256,
            "num_frames": 6,
            "context_frames": 3,
            "future_frames": 3,
        }
        expected_scalars = {
            key: _decoded_hdf5_attr(value)
            for key, value in expected_scalars.items()
        }
        mismatch = {
            key: {"expected": value, "actual": _decoded_hdf5_attr(handle.attrs.get(key))}
            for key, value in expected_scalars.items()
            if _decoded_hdf5_attr(handle.attrs.get(key)) != value
        }
        num_blocks = int(_decoded_hdf5_attr(handle.attrs.get("num_encoder_blocks", -1)))
        middle = int(_decoded_hdf5_attr(handle.attrs.get("middle_site_index", -1)))
        late = int(_decoded_hdf5_attr(handle.attrs.get("late_site_index", -1)))
        if middle != num_blocks // 2 or late != num_blocks - 1:
            mismatch["adapter_sites"] = {
                "expected": [num_blocks // 2, num_blocks - 1],
                "actual": [middle, late],
            }
        actual_transform = _decoded_hdf5_attr(handle.attrs.get("action_transform"))
        if actual_transform != expected_action_transform:
            mismatch["action_transform"] = {
                "expected": expected_action_transform,
                "actual": actual_transform,
            }
        actual_camera = {
            key: _decoded_hdf5_attr(handle.attrs.get(key))
            for key in expected_camera_contract
        }
        if actual_camera != expected_camera_contract:
            mismatch["camera_contract"] = {
                "expected": expected_camera_contract,
                "actual": actual_camera,
            }
        array_hashes = _decoded_hdf5_attr(
            handle.attrs.get("array_content_sha256")
        )
        if not isinstance(array_hashes, dict) or set(array_hashes) != set(V2_ARRAY_KEYS):
            mismatch["array_content_sha256"] = {
                "expected": sorted(V2_ARRAY_KEYS),
                "actual": array_hashes,
            }
        elif any(
            not isinstance(value, str) or len(value) != 64
            for value in array_hashes.values()
        ):
            mismatch["array_content_sha256"] = {
                "expected": "one SHA256 per required array",
                "actual": array_hashes,
            }
        expected_content_hash = (
            hashlib.sha256(
                json.dumps(
                    array_hashes, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if isinstance(array_hashes, dict)
            else None
        )
        if str(handle.attrs.get("content_sha256", "")) != expected_content_hash:
            mismatch["content_sha256"] = {
                "expected": expected_content_hash,
                "actual": str(handle.attrs.get("content_sha256", "")),
            }
        expected_fingerprint = cache_fingerprint_v2(handle)
        if str(handle.attrs.get("cache_fingerprint", "")) != expected_fingerprint:
            mismatch["cache_fingerprint"] = {
                "expected": expected_fingerprint,
                "actual": str(handle.attrs.get("cache_fingerprint", "")),
            }
        if mismatch:
            raise RuntimeError(f"V2 feature-cache contract mismatch at {resolved}: {mismatch}")
        validation = {
            "path": str(resolved),
            "cache_fingerprint": str(handle.attrs["cache_fingerprint"]),
            "available_windows": min(counts.values()),
            "used_windows": windows,
            "schema_version": CACHE_SCHEMA_VERSION_V2,
            "middle_site_index": middle,
            "late_site_index": late,
            "content_sha256": str(handle.attrs["content_sha256"]),
        }
    stat = resolved.stat()
    validation.update(
        cache_file_size=int(stat.st_size),
        cache_file_mtime_ns=int(stat.st_mtime_ns),
    )
    if include_file_sha256:
        validation["cache_file_sha256"] = sha256_file(resolved)
        validation["sha256"] = validation["cache_file_sha256"]
    if deep_verify:
        validation.update(
            verify_cache_content_v2(
                resolved, chunk_windows=content_verification_chunk_windows
            )
        )
        validation["sha256"] = validation["cache_file_sha256"]
    return validation


def training_contract_v2(training: dict[str, Any]) -> dict[str, Any]:
    """Return the single normalized contract used by v2 writer and validator."""
    required = {
        "max_optimizer_steps",
        "microbatch_windows",
        "views_per_window",
        "gradient_accumulation",
        "lr",
        "betas",
        "epsilon",
        "weight_decay",
        "gradient_clip_norm",
        "precision",
        "seed",
        "warmup_steps",
        "minimum_lr",
        "scheduler",
        "loss_name",
    }
    missing = sorted(required.difference(training))
    if missing:
        raise ValueError(f"V2 training contract is missing fields {missing}")
    microbatch = int(training["microbatch_windows"])
    views = int(training["views_per_window"])
    accumulation = int(training["gradient_accumulation"])
    return {
        "loss_name": str(training["loss_name"]),
        "goal_encoder": "frozen_base",
        "max_optimizer_steps": int(training["max_optimizer_steps"]),
        "completed_optimizer_steps": int(training["max_optimizer_steps"]),
        "training_seed": int(training["seed"]),
        "optimizer_config": {
            "name": "AdamW",
            "lr": float(training["lr"]),
            "betas": [float(value) for value in training["betas"]],
            "epsilon": float(training["epsilon"]),
            "weight_decay": float(training["weight_decay"]),
        },
        "scheduler_config": {
            "name": str(training["scheduler"]),
            "warmup_steps": int(training["warmup_steps"]),
            "minimum_lr": float(training["minimum_lr"]),
        },
        "training_config": {
            "max_optimizer_steps": int(training["max_optimizer_steps"]),
            "microbatch_windows": microbatch,
            "views_per_window": views,
            "gradient_accumulation": accumulation,
            "lr": float(training["lr"]),
            "betas": [float(value) for value in training["betas"]],
            "epsilon": float(training["epsilon"]),
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
            "precision": str(training["precision"]),
            "seed": int(training["seed"]),
            "warmup_steps": int(training["warmup_steps"]),
            "minimum_lr": float(training["minimum_lr"]),
            "scheduler": str(training["scheduler"]),
            "loss_name": str(training["loss_name"]),
        },
        "effective_view_batch": microbatch * views * accumulation,
    }


def training_contract_mismatches_v2(
    payload: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    mismatch: dict[str, Any] = {}
    for key in (
        "loss_name",
        "max_optimizer_steps",
        "completed_optimizer_steps",
        "training_seed",
        "goal_encoder",
    ):
        if payload.get(key) != expected[key]:
            mismatch[key] = {
                "expected": expected[key],
                "actual": payload.get(key),
            }
    for section in ("optimizer_config", "scheduler_config"):
        actual_section = dict(payload.get(section, {}))
        if actual_section != expected[section]:
            mismatch[section] = {
                "expected": expected[section],
                "actual": actual_section,
            }
    actual_training = dict(payload.get("training_config", {}))
    training_mismatch = {
        key: {"expected": value, "actual": actual_training.get(key)}
        for key, value in expected["training_config"].items()
        if actual_training.get(key) != value
    }
    if training_mismatch:
        mismatch["training_config"] = training_mismatch
    actual_effective_batch = (
        int(actual_training.get("microbatch_windows", -1))
        * int(actual_training.get("views_per_window", -1))
        * int(actual_training.get("gradient_accumulation", -1))
    )
    if actual_effective_batch != int(expected["effective_view_batch"]):
        mismatch["effective_view_batch"] = {
            "expected": expected["effective_view_batch"],
            "actual": actual_effective_batch,
        }
    return mismatch


def validate_checkpoint(
    path: str | Path,
    method: str,
    cache_fingerprint: str,
    *,
    benchmark: str,
    task: str,
    allow_legacy_place: bool = False,
    expected_training_seed: int | None = None,
    expected_method_config: dict[str, Any] | None = None,
    expected_loss_weights: tuple[float, float] | None = None,
    expected_action_transform: dict[str, Any] | None = None,
    expected_camera_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = load_method_checkpoint(resolved)
    if str(payload.get("method_name")) != method:
        raise RuntimeError(f"Checkpoint method mismatch at {resolved}")
    if str(payload.get("cache_fingerprint")) != cache_fingerprint:
        raise RuntimeError(f"Checkpoint cache fingerprint mismatch at {resolved}")
    training = dict(payload.get("training_config", {}))
    if expected_training_seed is not None and int(training.get("seed", -1)) != expected_training_seed:
        raise RuntimeError(
            f"Checkpoint training seed mismatch at {resolved}: "
            f"expected={expected_training_seed}, actual={training.get('seed')}"
        )
    if expected_method_config is not None:
        actual_options = dict(payload.get("method_config", {}))
        defaults = {
            "temporal_pool": "mean",
            "mask_type": "adaptive",
            "use_rms_norm": True,
        }
        option_mismatch = {
            key: {
                "expected": value,
                "actual": actual_options.get(key, defaults.get(key)),
            }
            for key, value in expected_method_config.items()
            if key != "name"
            and actual_options.get(key, defaults.get(key)) != value
        }
        if option_mismatch:
            raise RuntimeError(
                f"Checkpoint method configuration mismatch at {resolved}: "
                f"{option_mismatch}"
            )
    if expected_loss_weights is not None:
        actual_weights = (
            float(training.get("canonical_weight", 1.0)),
            float(training.get("dynamics_weight", 1.0)),
        )
        if actual_weights != expected_loss_weights:
            raise RuntimeError(
                f"Checkpoint loss weights mismatch at {resolved}: "
                f"expected={expected_loss_weights}, actual={actual_weights}"
            )
    metadata = payload.get("data_metadata", {})
    actual = (str(metadata.get("benchmark", "")), str(metadata.get("task_key", "")))
    if actual != (benchmark, task):
        legacy_ok = allow_legacy_place and task == "robocasa_place" and not actual[0]
        if not legacy_ok:
            raise RuntimeError(
                f"Checkpoint benchmark identity mismatch: expected={(benchmark, task)}, actual={actual}"
            )
    if expected_action_transform is not None and metadata.get("action_transform") != expected_action_transform:
        if not (allow_legacy_place and task == "robocasa_place" and metadata.get("action_transform") is None):
            raise RuntimeError(f"Checkpoint action-transform mismatch at {resolved}")
    if expected_camera_contract is not None:
        actual_camera = {key: metadata.get(key) for key in expected_camera_contract}
        if actual_camera != expected_camera_contract:
            if not (
                allow_legacy_place
                and task == "robocasa_place"
                and all(value is None for value in actual_camera.values())
            ):
                raise RuntimeError(f"Checkpoint camera-contract mismatch at {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "parameter_count": int(payload["trainable_parameter_count"]),
        "cache_fingerprint": str(payload["cache_fingerprint"]),
        "legacy": not bool(actual[0]),
    }


def validate_checkpoint_v2(
    path: str | Path,
    method: str,
    cache_fingerprint: str,
    cache_file_sha256: str,
    *,
    benchmark: str,
    task: str,
    expected_method_config: dict[str, Any],
    expected_training_contract: dict[str, Any],
    expected_action_transform: dict[str, Any] | None,
    expected_camera_contract: dict[str, Any],
    expected_data_contract: dict[str, Any],
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = load_method_checkpoint(resolved)
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_V2,
        "method_name": method,
        "cache_fingerprint": cache_fingerprint,
        "cache_file_sha256": cache_file_sha256,
        **{
            key: expected_training_contract[key]
            for key in (
                "loss_name",
                "max_optimizer_steps",
                "completed_optimizer_steps",
                "training_seed",
                "goal_encoder",
            )
        },
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    mismatch.update(
        training_contract_mismatches_v2(payload, expected_training_contract)
    )
    configured = {key: value for key, value in expected_method_config.items() if key != "name"}
    actual_method = dict(payload.get("method_config", {}))
    for key, value in configured.items():
        if actual_method.get(key) != value:
            mismatch[f"method_config.{key}"] = {
                "expected": value,
                "actual": actual_method.get(key),
            }
    metadata = dict(payload.get("data_metadata", {}))
    expected_metadata = {
        "benchmark": benchmark,
        "task_key": task,
        "action_transform": expected_action_transform,
        **expected_camera_contract,
        **expected_data_contract,
    }
    for key, value in expected_metadata.items():
        normalized_expected = _decoded_hdf5_attr(value)
        normalized_actual = _decoded_hdf5_attr(metadata.get(key))
        if normalized_actual != normalized_expected:
            mismatch[f"data_metadata.{key}"] = {
                "expected": normalized_expected,
                "actual": normalized_actual,
            }
    parameter_count = int(payload.get("trainable_parameter_count", -1))
    if parameter_count <= 0:
        mismatch["trainable_parameter_count"] = {"expected": ">0", "actual": parameter_count}
    if mismatch:
        raise RuntimeError(f"V2 checkpoint contract mismatch at {resolved}: {mismatch}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "parameter_count": parameter_count,
        "cache_fingerprint": cache_fingerprint,
        "cache_file_sha256": cache_file_sha256,
        "completed_optimizer_steps": expected_training_contract[
            "completed_optimizer_steps"
        ],
        "schema_version": CHECKPOINT_SCHEMA_V2,
    }


def validate_offline(
    path: str | Path,
    windows: int,
    *,
    benchmark: str,
    task: str,
    method: str,
    expected_action_transform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "benchmark": benchmark,
        "task": task,
        "method": method,
        "window_count": windows,
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Offline artifact mismatch at {resolved}: {mismatch}")
    if not {"clean", "ood"}.issubset(payload.get("domains", {})):
        raise RuntimeError(f"Offline artifact lacks clean/OOD metrics: {resolved}")
    if expected_action_transform is not None and payload.get("action_transform") != expected_action_transform:
        raise RuntimeError(f"Offline action-transform mismatch: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_offline_v2(
    path: str | Path,
    windows: int,
    *,
    benchmark: str,
    task: str,
    method: str,
    expected_cache_fingerprint: str,
    expected_cache_file_sha256: str,
    expected_checkpoint_sha256: str | None,
    expected_action_transform: dict[str, Any] | None,
    expected_task_manifest_sha256: str,
    expected_dataset_sha256: str,
    expected_camera_key: str,
    expected_task_upstream_commits: dict[str, str],
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "jepa_wm_offline_metrics_v2",
        "benchmark": benchmark,
        "task": task,
        "method": method,
        "window_count": windows,
        "goal_encoder": "frozen_base",
        "loss_name": "unified_trajectory_mse",
        "cache_fingerprint": expected_cache_fingerprint,
        "cache_file_sha256": expected_cache_file_sha256,
        "method_checkpoint_sha256": expected_checkpoint_sha256,
        "action_transform": expected_action_transform,
        "task_manifest_sha256": expected_task_manifest_sha256,
        "dataset_sha256": expected_dataset_sha256,
        "camera_key": expected_camera_key,
        "task_upstream_commits": expected_task_upstream_commits,
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    required_metrics = {
        "terminal_h1_mse", "terminal_h2_mse", "terminal_h3_mse",
        "mean_through_h1_mse", "mean_through_h2_mse", "mean_through_h3_mse",
        "true_action_rank", "action_top1_accuracy",
        "action_mean_reciprocal_rank", "true_vs_negative_cost_gap",
        "zero_action_gap",
    }
    for domain in ("clean", "ood"):
        actual = set(dict(payload.get("domains", {})).get(domain, {}))
        missing = sorted(required_metrics.difference(actual))
        if missing:
            mismatch[f"domains.{domain}"] = {"missing": missing}
    if mismatch:
        raise RuntimeError(f"V2 offline artifact mismatch at {resolved}: {mismatch}")
    return {
        "path": str(resolved), "sha256": sha256_file(resolved),
        "window_count": windows, "schema_version": expected["schema_version"],
    }


def validate_planning(
    path: str | Path,
    episodes: int,
    *,
    benchmark: str,
    task: str,
    method: str,
    domain: str,
    seed: int,
    severity: float,
    evaluation_manifest: str | Path,
    allow_legacy_place: bool = False,
    expected_task_manifest_sha256: str | None = None,
    expected_cache_fingerprint: str | None = None,
    expected_cache_file_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_action_convention: dict[str, Any] | None = None,
    expected_action_transform: dict[str, Any] | None = None,
    expected_camera_contract: dict[str, Any] | None = None,
    expected_camera_key: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_task_upstream_commits: dict[str, str] | None = None,
    formal_cem: bool = True,
    expected_base_checkpoint_sha256: str | None = None,
    expected_dinov3_checkpoint_sha256: str | None = None,
    expected_appearance_seed: int = 2026,
    expected_appearance_pipeline: str = "composed_photometric_v1",
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    success = payload.get("per_episode_success")
    if not isinstance(success, list) or len(success) < episodes:
        raise RuntimeError(f"Planning artifact has fewer than {episodes} episodes: {resolved}")
    expected = {
        "method": method,
        "domain": domain,
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    actual_task = payload.get("task")
    legacy_task_ok = allow_legacy_place and task == "robocasa_place" and actual_task == "place"
    if actual_task != task and not legacy_task_ok:
        mismatch["task"] = {"expected": task, "actual": actual_task}
    actual_benchmark = payload.get("benchmark")
    if actual_benchmark != benchmark and not (
        allow_legacy_place and task == "robocasa_place" and actual_benchmark in (None, "robocasa")
    ):
        mismatch["benchmark"] = {"expected": benchmark, "actual": actual_benchmark}
    actual_seed = payload.get("cem_seed", payload.get("evaluation_seed"))
    if int(actual_seed) != seed:
        mismatch["seed"] = {"expected": seed, "actual": actual_seed}
    appearance = payload.get("config", {}).get("appearance", payload.get("evaluation_appearance", {}))
    actual_severity = float(appearance.get("severity", payload.get("severity", 1.0)))
    if domain == "ood" and actual_severity != severity:
        mismatch["severity"] = {"expected": severity, "actual": actual_severity}
    if mismatch:
        raise RuntimeError(f"Planning artifact mismatch at {resolved}: {mismatch}")
    if expected_base_checkpoint_sha256 is not None and payload.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256:
        raise RuntimeError(f"Planning JEPA-WM checkpoint fingerprint mismatch: {resolved}")
    if expected_dinov3_checkpoint_sha256 is not None and payload.get("dinov3_checkpoint_sha256") != expected_dinov3_checkpoint_sha256:
        raise RuntimeError(f"Planning DINOv3 checkpoint fingerprint mismatch: {resolved}")
    if int(appearance.get("seed", expected_appearance_seed)) != expected_appearance_seed:
        raise RuntimeError(f"Planning appearance seed mismatch: {resolved}")
    if str(appearance.get("pipeline_version", expected_appearance_pipeline)) != expected_appearance_pipeline:
        raise RuntimeError(f"Planning appearance pipeline mismatch: {resolved}")
    manifest = json.loads(resolve_path(evaluation_manifest).read_text(encoding="utf-8"))
    manifest_hash = str(manifest["task_manifest_sha256"])
    if expected_task_manifest_sha256 is not None and manifest_hash != expected_task_manifest_sha256:
        raise RuntimeError(f"Evaluation manifest task fingerprint mismatch: {resolved}")
    artifact_manifest_hash = payload.get("task_manifest_sha256")
    if artifact_manifest_hash != manifest_hash and not (
        allow_legacy_place and task == "robocasa_place" and artifact_manifest_hash is None
    ):
        raise RuntimeError(f"Planning task-manifest fingerprint mismatch: {resolved}")
    expected_eval_hash = str(manifest["evaluation_manifest_sha256"])
    artifact_eval_hash = payload.get("evaluation_manifest_sha256")
    if artifact_eval_hash != expected_eval_hash and not (
        allow_legacy_place and task == "robocasa_place" and artifact_eval_hash is None
    ):
        raise RuntimeError(f"Planning evaluation-manifest fingerprint mismatch: {resolved}")
    if expected_cache_fingerprint is not None:
        actual_cache = payload.get("cache_fingerprint")
        if actual_cache != expected_cache_fingerprint and not (
            allow_legacy_place and task == "robocasa_place" and actual_cache is None
        ):
            raise RuntimeError(f"Planning cache fingerprint mismatch: {resolved}")
    if expected_cache_file_sha256 is not None:
        actual_cache_file = payload.get("cache_file_sha256")
        if actual_cache_file != expected_cache_file_sha256 and not (
            allow_legacy_place
            and task == "robocasa_place"
            and actual_cache_file is None
        ):
            raise RuntimeError(f"Planning cache-file fingerprint mismatch: {resolved}")
    if expected_checkpoint_sha256 is not None and payload.get("method_checkpoint_sha256") != expected_checkpoint_sha256:
        raise RuntimeError(f"Planning checkpoint fingerprint mismatch: {resolved}")
    if expected_action_convention is not None and payload.get("action_convention") != expected_action_convention:
        if not (allow_legacy_place and task == "robocasa_place" and payload.get("action_convention") is None):
            raise RuntimeError(f"Planning action convention mismatch: {resolved}")
    if expected_action_transform is not None and payload.get("action_transform") != expected_action_transform:
        if not (allow_legacy_place and task == "robocasa_place" and payload.get("action_transform") is None):
            raise RuntimeError(f"Planning action-transform mismatch: {resolved}")
    if expected_camera_contract is not None:
        actual_camera = {key: payload.get(key) for key in expected_camera_contract}
        if actual_camera != expected_camera_contract:
            if not (
                allow_legacy_place
                and task == "robocasa_place"
                and all(value is None for value in actual_camera.values())
            ):
                raise RuntimeError(f"Planning camera-contract mismatch: {resolved}")
    for field, expected_value in (
        ("camera_key", expected_camera_key),
        ("dataset_fingerprint", expected_dataset_fingerprint),
        ("benchmark_upstream_commits", expected_task_upstream_commits),
    ):
        if expected_value is None:
            continue
        actual_value = payload.get(field)
        if actual_value != expected_value and not (
            allow_legacy_place
            and task == "robocasa_place"
            and actual_value is None
        ):
            raise RuntimeError(
                f"Planning {field} contract mismatch at {resolved}: "
                f"expected={expected_value}, actual={actual_value}"
            )
    if formal_cem:
        cem = payload.get("cem", {})
        expected_cem = {
            "iterations": 15,
            "num_samples": 300,
            "num_elites": 10,
            "horizon": 3,
            "num_act_stepped": 1,
            "candidate_chunk_size": 300,
        }
        wrong_cem = {
            key: {"expected": value, "actual": cem.get(key)}
            for key, value in expected_cem.items()
            if int(cem.get(key, -1)) != value
        }
        if wrong_cem:
            raise RuntimeError(f"Planning CEM configuration mismatch at {resolved}: {wrong_cem}")
    expected_ids = [str(item["instance_id"]) for item in manifest["instances"][:episodes]]
    expected_source_ids = [
        str(item["source_trajectory_id"])
        for item in manifest["instances"][:episodes]
    ]
    expected_initialization_fingerprints = [
        str(item["initialization_fingerprint"])
        for item in manifest["instances"][:episodes]
    ]
    expected_goal_fingerprints = [
        str(item["goal_fingerprint"])
        for item in manifest["instances"][:episodes]
    ]
    actual_ids = payload.get("evaluation_instance_ids")
    legacy = actual_ids in (None, []) and allow_legacy_place and task == "robocasa_place"
    if not legacy and list(actual_ids or [])[:episodes] != expected_ids:
        raise RuntimeError(f"Planning evaluation-instance IDs mismatch at {resolved}")
    if not legacy and list(payload.get("initialization_fingerprints", []))[:episodes] != expected_initialization_fingerprints:
        raise RuntimeError(f"Planning initialization fingerprints mismatch at {resolved}")
    if not legacy and list(payload.get("goal_fingerprints", []))[:episodes] != expected_goal_fingerprints:
        raise RuntimeError(f"Planning goal fingerprints mismatch at {resolved}")
    selected = [bool(value) for value in success[:episodes]]
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "source_available_episodes": len(success),
        "used_episodes": episodes,
        "used_episode_indices": list(range(episodes)),
        "success_count": sum(selected),
        "success_rate": sum(selected) / episodes,
        "evaluation_instance_ids": expected_ids,
        "source_trajectory_ids": expected_source_ids,
        "legacy": legacy,
        "source_task": actual_task,
        "source_seed": int(actual_seed),
        "source_checkpoint_fingerprint": payload.get("method_checkpoint_sha256"),
    }


def validate_planning_v2(
    path: str | Path,
    episodes: int,
    **kwargs: Any,
) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    result = validate_planning(resolved, episodes, **kwargs)
    expected = {
        "goal_encoder": "frozen_base",
        "current_encoder": "configured_method",
    }
    mismatch = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    fingerprints = payload.get("goal_base_latent_fingerprint")
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) < episodes
        or any(not isinstance(value, str) or len(value) != 64 for value in fingerprints[:episodes])
    ):
        mismatch["goal_base_latent_fingerprint"] = {
            "expected": f"at least {episodes} SHA256 values",
            "actual": fingerprints,
        }
    manifest = json.loads(
        resolve_path(kwargs["evaluation_manifest"]).read_text(encoding="utf-8")
    )
    manifest_fingerprints = [
        item.get("goal_base_latent_fingerprint")
        for item in manifest.get("instances", [])[:episodes]
    ]
    if all(value is not None for value in manifest_fingerprints) and list(
        fingerprints or []
    )[:episodes] != manifest_fingerprints:
        mismatch["goal_base_latent_fingerprint_manifest"] = {
            "expected": manifest_fingerprints,
            "actual": list(fingerprints or [])[:episodes],
        }
    benchmark = str(kwargs["benchmark"])
    if benchmark == "libero":
        max_steps = int(payload.get("config", {}).get("evaluation", {}).get("max_episode_steps", -1))
        if max_steps != 600:
            mismatch["evaluation.max_episode_steps"] = {
                "expected": 600,
                "actual": max_steps,
            }
    if mismatch:
        raise RuntimeError(f"V2 planning artifact mismatch at {resolved}: {mismatch}")
    result.update(
        goal_encoder="frozen_base",
        goal_base_latent_fingerprints=fingerprints[:episodes],
    )
    return result


def _state_job(job: JobSpec, status: str) -> dict[str, Any]:
    value = job.state_fields()
    value["status"] = status
    return value


def block_job_for_failed_dependencies(
    job: JobSpec, state: dict[str, Any]
) -> bool:
    failed = [
        dependency
        for dependency in job.dependencies
        if state.get("jobs", {}).get(dependency, {}).get("status")
        in {"failed", "blocked", "stopped"}
    ]
    if not failed:
        return False
    entry = job.state_fields()
    entry.update(
        status="blocked",
        error="blocked because dependencies failed: " + ", ".join(failed),
    )
    state.setdefault("jobs", {})[job.job_id] = entry
    return True


def _write_state(path: str | Path, state: dict[str, Any]) -> None:
    state["updated_at_unix"] = time.time()
    atomic_json(path, state)


def _gpu_free_memory_mib(gpu_ids: Sequence[int]) -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    available: dict[int, int] = {}
    requested = set(int(value) for value in gpu_ids)
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"Cannot parse nvidia-smi GPU memory row: {line!r}")
        index, free_memory = int(fields[0]), int(fields[1])
        if index in requested:
            available[index] = free_memory
    missing = sorted(requested.difference(available))
    if missing:
        raise RuntimeError(
            f"nvidia-smi did not report configured GPUs {missing}; reported={available}"
        )
    return available


def _log_segment_has_cuda_oom(path: Path, start_offset: int) -> bool:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end_offset = handle.tell()
        handle.seek(max(start_offset, end_offset - 2 * 1024 * 1024))
        output = handle.read().lower()
    return any(signature in output for signature in _CUDA_OOM_SIGNATURES)


def run_gpu_phase(
    jobs: Sequence[JobSpec],
    gpu_ids: Sequence[int],
    state_path: str | Path,
    state: dict[str, Any],
    validator: Any,
    *,
    raise_on_failure: bool = True,
) -> set[str]:
    pending = list(jobs)
    running: dict[
        int,
        tuple[JobSpec, subprocess.Popen[str], Any, float, int],
    ] = {}
    attempted_gpus: dict[str, set[int]] = {job.job_id: set() for job in jobs}
    attempt_history: dict[str, list[dict[str, Any]]] = {
        job.job_id: [] for job in jobs
    }
    minimum_free_mib = int(
        os.environ.get("WM_ADAPTER_MIN_GPU_FREE_MIB", _DEFAULT_MIN_GPU_FREE_MIB)
    )
    if minimum_free_mib < 0:
        raise ValueError(
            "WM_ADAPTER_MIN_GPU_FREE_MIB must be non-negative, "
            f"received {minimum_free_mib}"
    )
    failed_jobs: set[str] = set()
    while pending or running:
        stop_launching = bool(failed_jobs) and raise_on_failure
        if not stop_launching:
            free_memory = _gpu_free_memory_mib(gpu_ids)
            idle_gpus = sorted(
                (int(gpu) for gpu in gpu_ids if int(gpu) not in running),
                key=lambda gpu: (-free_memory[gpu], gpu),
            )
            for gpu in idle_gpus:
                candidate_index = next(
                    (
                        index
                        for index, candidate in enumerate(pending)
                        if gpu not in attempted_gpus[candidate.job_id]
                        and (
                            candidate.kind == "analysis"
                            or free_memory[gpu] >= minimum_free_mib
                        )
                    ),
                    None,
                )
                if candidate_index is None:
                    continue
                job = pending.pop(candidate_index)
                attempted_gpus[job.job_id].add(gpu)
                log_path = resolve_path(job.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_start_offset = log_path.stat().st_size if log_path.exists() else 0
                log = log_path.open("a", encoding="utf-8")
                environment = benchmark_subprocess_environment(job.benchmark, gpu=gpu)
                process = subprocess.Popen(
                    list(job.command),
                    cwd=project_root(),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                started = time.time()
                entry = _state_job(job, "running")
                entry.update(
                    gpu=gpu,
                    pid=process.pid,
                    start_time=started,
                    gpu_attempts=sorted(attempted_gpus[job.job_id]),
                    oom_retries=len(attempt_history[job.job_id]),
                    gpu_free_memory_mib_at_launch=free_memory[gpu],
                    minimum_gpu_free_memory_mib=minimum_free_mib,
                    scheduler_status="running",
                    attempt_history=list(attempt_history[job.job_id]),
                )
                state["jobs"][job.job_id] = entry
                running[gpu] = (
                    job,
                    process,
                    log,
                    started,
                    log_start_offset,
                )
                _write_state(state_path, state)
        if not running:
            if pending and not stop_launching:
                free_memory = _gpu_free_memory_mib(gpu_ids)
                waiting_payload = {
                    "scheduler_status": "waiting_for_eligible_gpu",
                    "minimum_gpu_free_memory_mib": minimum_free_mib,
                    "gpu_free_memory_mib": {
                        str(gpu): free_memory[int(gpu)] for gpu in gpu_ids
                    },
                }
                changed = False
                for job in pending:
                    entry = state["jobs"].setdefault(
                        job.job_id, _state_job(job, "pending")
                    )
                    for key, value in waiting_payload.items():
                        if entry.get(key) != value:
                            entry[key] = value
                            changed = True
                if changed:
                    _write_state(state_path, state)
                time.sleep(5.0)
                continue
            break
        time.sleep(1.0)
        for gpu, (job, process, log, started, log_start_offset) in list(
            running.items()
        ):
            return_code = process.poll()
            if return_code is None:
                continue
            log.flush()
            log.close()
            cuda_oom = return_code != 0 and _log_segment_has_cuda_oom(
                resolve_path(job.log_path), log_start_offset
            )
            ended = time.time()
            attempt_history[job.job_id].append(
                {
                    "gpu": gpu,
                    "start_time": started,
                    "end_time": ended,
                    "elapsed_seconds": ended - started,
                    "return_code": return_code,
                    "cuda_oom": cuda_oom,
                }
            )
            entry = state["jobs"][job.job_id]
            entry["end_time"] = ended
            entry["elapsed_seconds"] = entry["end_time"] - started
            entry["return_code"] = return_code
            entry["attempt_history"] = list(attempt_history[job.job_id])
            if return_code == 0:
                try:
                    entry["artifact_validation"] = validator(job, job.artifact_path)
                    entry["status"] = "completed"
                    entry["scheduler_status"] = "completed"
                except Exception as error:
                    entry["status"] = "failed"
                    entry["error"] = f"{type(error).__name__}: {error}"
                    failed_jobs.add(job.job_id)
            elif cuda_oom and len(attempted_gpus[job.job_id]) < len(gpu_ids):
                remaining = sorted(
                    set(int(value) for value in gpu_ids).difference(
                        attempted_gpus[job.job_id]
                    )
                )
                with resolve_path(job.log_path).open("a", encoding="utf-8") as marker:
                    marker.write(
                        "\nGPU_SCHEDULER_RETRY reason=cuda_oom "
                        f"failed_gpu={gpu} remaining_gpus={remaining}\n"
                    )
                entry.update(
                    status="pending",
                    gpu=None,
                    pid=None,
                    error=None,
                    scheduler_status="retrying_after_cuda_oom",
                    last_oom_gpu=gpu,
                    oom_retries=sum(
                        bool(value["cuda_oom"])
                        for value in attempt_history[job.job_id]
                    ),
                    gpu_attempts=sorted(attempted_gpus[job.job_id]),
                )
                pending.append(job)
            else:
                entry["status"] = "failed"
                entry["scheduler_status"] = "failed"
                entry["error"] = (
                    "CUDA OOM on every configured GPU attempted by this job: "
                    f"{sorted(attempted_gpus[job.job_id])}"
                    if cuda_oom
                    else f"process exited with code {return_code}"
                )
                failed_jobs.add(job.job_id)
            del running[gpu]
            _write_state(state_path, state)
    if failed_jobs and raise_on_failure:
        for job in pending:
            state["jobs"][job.job_id] = _state_job(job, "blocked")
        _write_state(state_path, state)
        failed = [key for key, value in state["jobs"].items() if value["status"] == "failed"]
        raise RuntimeError(f"Cross-benchmark jobs failed: {failed}")
    return failed_jobs


def phase_summary(state: dict[str, Any], jobs: Sequence[JobSpec]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    phases = tuple(dict.fromkeys(job.phase for job in jobs))
    for phase in phases:
        phase_jobs = [job for job in jobs if job.phase == phase]
        statuses = [state.get("jobs", {}).get(job.job_id, {}).get("status", "pending") for job in phase_jobs]
        complete = sum(value in {"completed", "reused"} for value in statuses)
        failed = sum(value == "failed" for value in statuses)
        running = sum(value == "running" for value in statuses)
        summary[phase] = {
            "total": len(phase_jobs),
            "complete": complete,
            "failed": failed,
            "running": running,
        }
    return summary
