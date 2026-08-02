from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from wm_adapter.utils.reproducibility import resolve_path


EPISODE_SPLIT_STRATEGY = "deterministic_trajectory_partition_v1"
WINDOW_SELECTION_STRATEGY = "episode_balanced_round_robin_v1"
TASK_MANIFEST_SCHEMA = "cross_benchmark_task_manifest_v1"
EVALUATION_MANIFEST_SCHEMA = "cross_benchmark_evaluation_manifest_v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)


def split_trajectory_indices(
    num_trajectories: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_trajectories <= 0:
        raise ValueError(
            f"Dataset has no trajectories: num_trajectories={num_trajectories}"
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be in (0,1), received {train_fraction}"
        )
    shuffled = np.random.default_rng(seed).permutation(
        np.arange(num_trajectories, dtype=np.int64)
    )
    train_count = int(np.floor(num_trajectories * train_fraction))
    if train_count <= 0 or train_count >= num_trajectories:
        raise ValueError(
            "Trajectory split must leave non-empty train and evaluation partitions: "
            f"num_trajectories={num_trajectories}, train_fraction={train_fraction}"
        )
    return np.sort(shuffled[:train_count]), np.sort(shuffled[train_count:])


def episode_balanced_windows(
    candidates: list[tuple[int, int]],
    allowed_trajectory_indices: np.ndarray,
    max_windows: int,
    seed: int,
) -> list[tuple[int, int]]:
    if max_windows <= 0:
        raise ValueError(f"max_windows must be positive, received {max_windows}")
    allowed = {int(value) for value in allowed_trajectory_indices.tolist()}
    grouped: dict[int, list[tuple[int, int]]] = {}
    for trajectory, start in candidates:
        if trajectory in allowed:
            grouped.setdefault(trajectory, []).append((trajectory, start))
    generator = np.random.default_rng(seed)
    trajectories = list(grouped)
    generator.shuffle(trajectories)
    for trajectory in trajectories:
        order = generator.permutation(len(grouped[trajectory]))
        grouped[trajectory] = [
            grouped[trajectory][int(index)] for index in order
        ]
    target = min(max_windows, sum(len(values) for values in grouped.values()))
    selected: list[tuple[int, int]] = []
    round_index = 0
    while len(selected) < target:
        added = False
        for trajectory in trajectories:
            values = grouped[trajectory]
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        round_index += 1
    return selected


@dataclass(frozen=True)
class ActionConvention:
    dimension: int
    translation: str
    rotation: str
    gripper: str
    source_range: tuple[float, float]
    target_range: tuple[float, float]
    controller_type: str
    control_frequency_hz: float
    action_repeat: int
    transform: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedTask:
    task_key: str
    benchmark: str
    suite: str
    task_id: int | str
    task_name: str
    language_instruction: str | None
    bddl_path: str | None
    bddl_sha256: str | None
    problem_folder: str | None
    initial_states_sha256: str | None
    initial_states_count: int
    dataset_path: str
    dataset_sha256: str | None
    available_demonstrations: int
    selected_train_demonstrations: tuple[str, ...]
    selected_test_demonstrations: tuple[str, ...]
    camera_key: str
    action_convention: dict[str, Any]
    environment_implementation: str
    upstream_commits: dict[str, str]
    frameskip: int
    max_episode_steps: int
    episode_cap_basis: str
    status: str = "resolved"
    candidate_failures: dict[str, list[str]] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TASK_MANIFEST_SCHEMA
        payload["task_manifest_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class EvaluationInstance:
    instance_id: str
    source_trajectory_id: str
    source_trajectory_index: int
    segment_start: int
    segment_end: int
    initialization_fingerprint: str
    goal_fingerprint: str
    environment_seed: int
    cem_seed: int
    appearance_seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkAdapter(ABC):
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def name(self) -> str:
        raise RuntimeError("Benchmark adapters must define name")

    @abstractmethod
    def resolve_task(self, *, strict: bool) -> ResolvedTask:
        raise RuntimeError("Benchmark adapters must implement resolve_task")

    @abstractmethod
    def preflight(self, *, deep: bool) -> dict[str, Any]:
        raise RuntimeError("Benchmark adapters must implement preflight")

    @abstractmethod
    def build_source_dataset(self, *, output_environment_info: bool) -> Any:
        raise RuntimeError("Benchmark adapters must implement build_source_dataset")

    @abstractmethod
    def enumerate_window_candidates(
        self,
        source_dataset: Any,
        num_frames: int,
        frameskip: int,
    ) -> list[tuple[int, int]]:
        raise RuntimeError("Benchmark adapters must implement enumerate_window_candidates")

    @abstractmethod
    def make_window_dataset(
        self,
        source_dataset: Any,
        selections: list[tuple[int, int]],
        *,
        num_frames: int,
        frameskip: int,
        appearance_seed: int,
        appearance_severity: float,
    ) -> Dataset[Any]:
        raise RuntimeError("Benchmark adapters must implement make_window_dataset")

    @abstractmethod
    def build_evaluation_manifest(
        self,
        task: ResolvedTask,
        *,
        count: int,
        seed: int,
        appearance_seed: int,
    ) -> dict[str, Any]:
        raise RuntimeError("Benchmark adapters must implement build_evaluation_manifest")

    @abstractmethod
    def run_planning(
        self,
        *,
        backend: Any,
        method: Any,
        output_directory: str | Path,
    ) -> Any:
        raise RuntimeError("Benchmark adapters must implement run_planning")

    def split_trajectory_ids(
        self,
        source_dataset: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        return split_trajectory_indices(
            len(source_dataset),
            float(self.cfg.data.train_fraction),
            int(self.cfg.data.split_seed),
        )

    def select_windows(
        self,
        candidates: list[tuple[int, int]],
        allowed_trajectories: np.ndarray,
        max_windows: int,
        seed: int,
    ) -> list[tuple[int, int]]:
        return episode_balanced_windows(
            candidates,
            allowed_trajectories,
            max_windows,
            seed,
        )

    def task_manifest_path(self) -> Path:
        configured = self.cfg.paths.get("task_manifest")
        if configured:
            return resolve_path(str(configured))
        return resolve_path(
            f"outputs/cross_benchmark_v1/manifests/tasks/{self.cfg.benchmark.task_key}.json"
        )

    def existing_task_manifest(self) -> ResolvedTask | None:
        path = self.task_manifest_path()
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        supplied_hash = str(payload.pop("task_manifest_sha256", ""))
        payload.pop("schema_version", None)
        task = ResolvedTask(**payload)
        computed_hash = task.as_dict()["task_manifest_sha256"]
        if supplied_hash != computed_hash:
            raise RuntimeError(f"Resolved task manifest fingerprint is invalid: {path}")
        if task.status != "resolved":
            raise RuntimeError(f"Resolved task manifest is not usable: {path}")
        if task.task_key != str(self.cfg.benchmark.task_key) or task.benchmark != self.name:
            raise RuntimeError(
                f"Resolved task manifest identity does not match configuration: {path}"
            )
        return task

    def evaluation_manifest_path(self) -> Path:
        configured = self.cfg.paths.get("evaluation_manifest")
        if configured:
            return resolve_path(str(configured))
        return resolve_path(
            "outputs/cross_benchmark_v1/manifests/evaluation/"
            f"{self.cfg.benchmark.task_key}.json"
        )

    def write_task_manifest(self, task: ResolvedTask) -> dict[str, Any]:
        payload = task.as_dict()
        destination = self.task_manifest_path()
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError(
                    f"Resolved task manifest is immutable and differs from {destination}"
                )
            return existing
        atomic_json(destination, payload)
        return payload

    def write_evaluation_manifest(self, payload: dict[str, Any]) -> dict[str, Any]:
        destination = self.evaluation_manifest_path()
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError(
                    f"Evaluation manifest is immutable and differs from {destination}"
                )
            return existing
        atomic_json(destination, payload)
        return payload

    @staticmethod
    def finalize_evaluation_manifest(
        task: ResolvedTask,
        instances: list[EvaluationInstance],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": EVALUATION_MANIFEST_SCHEMA,
            "task_key": task.task_key,
            "benchmark": task.benchmark,
            "suite": task.suite,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_manifest_sha256": task.as_dict()["task_manifest_sha256"],
            "instances": [instance.as_dict() for instance in instances],
        }
        payload["evaluation_manifest_sha256"] = canonical_sha256(payload)
        return payload
