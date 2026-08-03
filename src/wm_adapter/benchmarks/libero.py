from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info

from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.benchmarks.base import (
    ActionConvention,
    ActionTransform,
    BenchmarkAdapter,
    EvaluationInstance,
    ResolvedTask,
    array_sha256,
    canonical_sha256,
    split_trajectory_indices,
)
from wm_adapter.utils.checkpoints import git_commit, sha256_file
from wm_adapter.utils.reproducibility import resolve_path


LIBERO_CAMERA_KEYS = (
    "agentview_rgb",
    "agentview_image",
    "obs/agentview_rgb",
    "obs/agentview_image",
)
JEPA_WM_CANONICAL_LOWER = (-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, -1.0)
JEPA_WM_CANONICAL_UPPER = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0)


def _dataset_node(group: h5py.Group, path: str) -> h5py.Dataset | None:
    node: Any = group
    for component in path.split("/"):
        if component not in node:
            return None
        node = node[component]
    return node if isinstance(node, h5py.Dataset) else None


def _camera_key(group: h5py.Group) -> str:
    matches = [key for key in LIBERO_CAMERA_KEYS if _dataset_node(group, key) is not None]
    if len(matches) != 1:
        raise RuntimeError(
            "LIBERO demonstration must expose exactly one supported agent-view RGB key; "
            f"found={matches}, supported={list(LIBERO_CAMERA_KEYS)}"
        )
    return matches[0]


def _demo_keys(handle: h5py.File) -> list[str]:
    if "data" not in handle or not isinstance(handle["data"], h5py.Group):
        raise RuntimeError("LIBERO HDF5 lacks root group 'data'")
    return sorted(
        (key for key in handle["data"] if key.startswith("demo_")),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )


def _camera_contract_from_shape(
    shape: tuple[int, ...] | list[int],
    *,
    vertical_flip: bool,
) -> dict[str, Any]:
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 4 or dimensions[-1] != 3:
        raise RuntimeError(
            f"LIBERO agent-view image must be [T,H,W,3], received {dimensions}"
        )
    return {
        "camera_height": dimensions[1],
        "camera_width": dimensions[2],
        "camera_channel_order": "RGB",
        "camera_vertical_flip": bool(vertical_flip),
    }


def _validated_split_indices(
    total_successful: int,
    train_fraction: float,
    split_seed: int,
    required_eval_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected_train = int(np.floor(total_successful * train_fraction))
    expected_heldout = total_successful - expected_train
    try:
        train, heldout = split_trajectory_indices(
            total_successful,
            train_fraction,
            split_seed,
        )
    except ValueError as error:
        raise RuntimeError(
            "LIBERO trajectory split cannot satisfy the requested evaluation: "
            f"total_successful_demos={total_successful}, "
            f"train_fraction={train_fraction}, train_count={expected_train}, "
            f"held_out_count={expected_heldout}, "
            f"required_evaluation_count={required_eval_count}"
        ) from error
    if len(train) < 1 or len(heldout) < required_eval_count:
        raise RuntimeError(
            "LIBERO trajectory split cannot satisfy the requested evaluation: "
            f"total_successful_demos={total_successful}, "
            f"train_fraction={train_fraction}, train_count={len(train)}, "
            f"held_out_count={len(heldout)}, "
            f"required_evaluation_count={required_eval_count}"
        )
    return train, heldout


@dataclass(frozen=True)
class LiberoTrajectoryIdentity:
    demo_key: str
    length: int


class LiberoTrajectoryDataset:
    def __init__(
        self,
        path: str | Path,
        *,
        camera_key: str,
        vertical_flip: bool,
        action_transform: ActionTransform,
        demonstration_keys: tuple[str, ...] | None = None,
    ) -> None:
        self.path = resolve_path(path)
        self.camera_key = camera_key
        self.vertical_flip = vertical_flip
        self.action_transform = action_transform
        self._handles: dict[int, h5py.File] = {}
        with h5py.File(self.path, "r") as handle:
            keys = list(demonstration_keys) if demonstration_keys is not None else _demo_keys(handle)
            unknown = [key for key in keys if key not in handle["data"]]
            if unknown:
                raise RuntimeError(
                    f"LIBERO selected demonstrations are absent from {self.path}: {unknown}"
                )
            self.trajectories = [
                LiberoTrajectoryIdentity(
                    demo_key=key,
                    length=int(handle["data"][key]["actions"].shape[0]),
                )
                for key in keys
            ]

    def __len__(self) -> int:
        return len(self.trajectories)

    def _file(self) -> h5py.File:
        worker = get_worker_info()
        worker_id = -1 if worker is None else worker.id
        handle = self._handles.get(worker_id)
        if handle is None:
            handle = h5py.File(self.path, "r", libver="latest", swmr=True)
            self._handles[worker_id] = handle
        return handle

    def get_seq_length(self, index: int) -> int:
        return self.trajectories[index].length

    def get_frames(
        self,
        index: int,
        frames: list[int] | range,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        identity = self.trajectories[index]
        group = self._file()["data"][identity.demo_key]
        frame_indices = np.asarray(list(frames), dtype=np.int64)
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError("LIBERO frame selection must be a non-empty 1-D sequence")
        if np.any(frame_indices < 0) or np.any(frame_indices >= identity.length):
            raise IndexError(
                f"LIBERO frames {frame_indices.tolist()} exceed {identity.demo_key} length {identity.length}"
            )
        image_node = _dataset_node(group, self.camera_key)
        if image_node is None:
            raise RuntimeError(
                f"Configured LIBERO camera key {self.camera_key!r} is missing in {identity.demo_key}"
            )
        images = np.asarray(image_node[frame_indices])
        if images.ndim != 4 or images.shape[-1] != 3:
            raise RuntimeError(
                f"LIBERO camera data must be [T,H,W,3], received {images.shape}"
            )
        if self.vertical_flip:
            images = images[:, ::-1].copy()
        if images.dtype == np.uint8:
            visual = torch.from_numpy(images).permute(0, 3, 1, 2)
        else:
            minimum = float(images.min())
            maximum = float(images.max())
            if minimum < 0.0 or maximum > 1.0:
                raise RuntimeError(
                    f"LIBERO floating RGB values must be in [0,1], found [{minimum},{maximum}]"
                )
            visual = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        raw_actions = np.asarray(group["actions"][frame_indices])
        actions = torch.from_numpy(
            self.action_transform.environment_to_canonical_action(raw_actions)
        ).float()
        states = torch.from_numpy(np.asarray(group["states"][frame_indices])).float()
        rewards = (
            torch.from_numpy(np.asarray(group["rewards"][frame_indices])).float()
            if "rewards" in group
            else torch.zeros(frame_indices.size, dtype=torch.float32)
        )
        return (
            {"visual": visual},
            actions,
            states,
            rewards,
            {
                "demo_key": identity.demo_key,
                "file_path": str(self.path),
                "camera_key": self.camera_key,
            },
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in self._handles.values():
            if handle.id.valid:
                handle.close()


class LiberoWindowDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        source: LiberoTrajectoryDataset,
        selections: list[tuple[int, int]],
        *,
        num_frames: int,
        frameskip: int,
        appearance_seed: int,
        appearance_severity: float,
    ) -> None:
        self.source = source
        self.selections = selections
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.appearance_seed = appearance_seed
        self.appearance_severity = appearance_severity
        self.appearance = ComposedPhotometricShift()

    def __len__(self) -> int:
        return len(self.selections)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        trajectory, start = self.selections[index]
        frames = [start + offset * self.frameskip for offset in range(self.num_frames)]
        observation, actions, _, _, _ = self.source.get_frames(trajectory, frames)
        clean = observation["visual"]
        if tuple(actions.shape) != (self.num_frames, 7):
            raise RuntimeError(
                f"LIBERO action window must be {(self.num_frames, 7)}, received {tuple(actions.shape)}"
            )
        spec_seed = int(self.appearance_seed + index)
        spec = self.appearance.sample_spec(spec_seed, self.appearance_severity)
        return {
            "clean_images": clean,
            "ood_images": self.appearance.apply(clean, spec),
            "actions": actions,
            "episode_id": torch.tensor(trajectory, dtype=torch.int64),
            "window_id": torch.tensor(start, dtype=torch.int64),
            "appearance_seed": torch.tensor(spec_seed, dtype=torch.int64),
        }


class LiberoBenchmark(BenchmarkAdapter):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._strict_resolution: ResolvedTask | None = None

    @property
    def name(self) -> str:
        return "libero"

    def _repo(self) -> Path:
        value = str(self.cfg.paths.get("libero_root", os.environ.get("LIBERO_ROOT", ""))).strip()
        if not value:
            raise FileNotFoundError(
                "LIBERO_ROOT is empty; point it to an independent official LIBERO checkout"
            )
        repo = resolve_path(value)
        if not repo.is_dir():
            raise FileNotFoundError(f"Official LIBERO checkout does not exist: {repo}")
        if not (repo / ".git").exists():
            raise RuntimeError(f"LIBERO checkout is not an independent Git repository: {repo}")
        return repo

    def _official_suite(self) -> tuple[Any, Any, Path, str]:
        repo = self._repo()
        value = str(repo)
        if value not in sys.path:
            sys.path.insert(0, value)
        try:
            from libero.libero import benchmark
        except ImportError as error:
            raise RuntimeError(
                f"Cannot import official LIBERO package from {repo}: {error}"
            ) from error
        suites = benchmark.get_benchmark_dict()
        suite_name = str(self.cfg.benchmark.suite)
        if suite_name not in suites:
            raise RuntimeError(
                f"Official LIBERO registry lacks suite {suite_name!r}; available={sorted(suites)}"
            )
        task_suite = suites[suite_name]()
        task_id = int(self.cfg.benchmark.task_id)
        if task_id < 0 or task_id >= int(task_suite.n_tasks):
            raise IndexError(
                f"LIBERO task_id={task_id} is outside suite {suite_name} with {task_suite.n_tasks} tasks"
            )
        task = task_suite.get_task(task_id)
        bddl = Path(task_suite.get_task_bddl_file_path(task_id)).resolve()
        return task_suite, task, bddl, git_commit(repo)

    @staticmethod
    def _official_init_states(task: Any) -> Any:
        from libero.libero import get_libero_path

        init_states_path = (
            Path(get_libero_path("init_states"))
            / str(task.problem_folder)
            / str(task.init_states_file)
        ).resolve()
        if not init_states_path.is_file():
            raise FileNotFoundError(
                f"Official LIBERO init-state file does not exist: {init_states_path}"
            )
        # Official LIBERO init states contain NumPy objects rather than model
        # weights. PyTorch 2.6 requires the trusted-data mode to be explicit.
        return torch.load(init_states_path, weights_only=False)

    def _dataset_path(self, task: Any) -> Path:
        suite_name = str(self.cfg.benchmark.suite)
        suite_override_name = (
            "LIBERO_SPATIAL_DATA_ROOT"
            if suite_name == "libero_spatial"
            else "LIBERO_GOAL_DATA_ROOT"
        )
        suite_override = str(
            self.cfg.paths.get(
                "suite_data_root", os.environ.get(suite_override_name, "")
            )
        ).strip()
        common = str(
            self.cfg.paths.get(
                "libero_data_root", os.environ.get("LIBERO_DATA_ROOT", "")
            )
        ).strip()
        root_value = suite_override or common
        if not root_value:
            raise FileNotFoundError(
                f"Neither {suite_override_name} nor LIBERO_DATA_ROOT is configured"
            )
        root = resolve_path(root_value)
        filename = f"{task.name}_demo.hdf5"
        candidates = (root / filename, root / suite_name / filename)
        matches = [path.resolve() for path in candidates if path.is_file()]
        if len(set(matches)) != 1:
            raise FileNotFoundError(
                f"Expected one LIBERO dataset {filename} under {root} or {root / suite_name}; found={matches}"
            )
        return matches[0]

    @staticmethod
    def _inspect_dataset(path: Path) -> dict[str, Any]:
        with h5py.File(path, "r") as handle:
            demos = _demo_keys(handle)
            if not demos:
                raise RuntimeError(f"LIBERO dataset contains no demonstrations: {path}")
            first = handle["data"][demos[0]]
            camera = _camera_key(first)
            successful: list[str] = []
            action_min = np.full(7, np.inf, dtype=np.float64)
            action_max = np.full(7, -np.inf, dtype=np.float64)
            lengths: list[int] = []
            image_shape: tuple[int, ...] | None = None
            image_dtype: str | None = None
            for key in demos:
                group = handle["data"][key]
                if "actions" not in group or "states" not in group:
                    raise RuntimeError(f"LIBERO demonstration {key} lacks actions or states")
                actions = np.asarray(group["actions"])
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise RuntimeError(
                        f"LIBERO demonstration {key} action shape must be [T,7], received {actions.shape}"
                    )
                if not np.isfinite(actions).all():
                    raise RuntimeError(f"LIBERO demonstration {key} has non-finite actions")
                action_min = np.minimum(action_min, actions.min(axis=0))
                action_max = np.maximum(action_max, actions.max(axis=0))
                lengths.append(int(actions.shape[0]))
                current_camera = _camera_key(group)
                if current_camera != camera:
                    raise RuntimeError(
                        f"LIBERO camera key changed in {key}: expected={camera}, "
                        f"actual={current_camera}"
                    )
                image = _dataset_node(group, camera)
                if image is None:
                    raise RuntimeError(f"LIBERO camera {camera!r} is missing in {key}")
                _camera_contract_from_shape(
                    image.shape,
                    vertical_flip=False,
                )
                current_shape = tuple(int(value) for value in image.shape[1:])
                current_dtype = str(image.dtype)
                if image_shape is not None and current_shape != image_shape:
                    raise RuntimeError(
                        f"LIBERO camera shape changed in {key}: "
                        f"expected={image_shape}, actual={current_shape}"
                    )
                if image_dtype is not None and current_dtype != image_dtype:
                    raise RuntimeError(
                        f"LIBERO camera dtype changed in {key}: "
                        f"expected={image_dtype}, actual={current_dtype}"
                    )
                image_shape = current_shape
                image_dtype = current_dtype
                rewards = np.asarray(group["rewards"]) if "rewards" in group else None
                dones = np.asarray(group["dones"]) if "dones" in group else None
                if (rewards is not None and bool(np.any(rewards > 0))) or (
                    dones is not None and bool(dones[-1])
                ):
                    successful.append(key)
            if np.any(action_min < -1.0001) or np.any(action_max > 1.0001):
                raise RuntimeError(
                    "LIBERO demonstration actions exceed the environment [-1,1] range: "
                    f"min={action_min.tolist()}, max={action_max.tolist()}"
                )
            image = _dataset_node(first, camera)
            if image is None:
                raise RuntimeError(f"LIBERO camera {camera!r} disappeared from first demo")
            camera_contract = _camera_contract_from_shape(
                image.shape,
                vertical_flip=False,
            )
            return {
                "demonstrations": demos,
                "successful_demonstrations": successful,
                "camera_key": camera,
                "lengths": lengths,
                "action_min": action_min.tolist(),
                "action_max": action_max.tolist(),
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                **camera_contract,
            }

    def action_convention(self) -> ActionConvention:
        return ActionConvention(
            dimension=7,
            translation="JEPA-WM physical delta Cartesian position",
            rotation="JEPA-WM physical delta axis-angle vector",
            gripper="scalar command; -1=open and +1=close",
            source_range=(-1.0, 1.0),
            target_range=(-1.0, 1.0),
            controller_type="resolved from the live LIBERO environment",
            control_frequency_hz=float(self.cfg.benchmark.get("control_frequency_hz", 20.0)),
            action_repeat=int(self.cfg.data.frameskip),
            transform=(
                "verified affine JEPA-WM physical-delta <-> LIBERO controller-input "
                "mapping stored in action_transform"
            ),
        )

    @staticmethod
    def _vector(value: Any, length: int, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size == 1:
            array = np.full(length, float(array[0]), dtype=np.float64)
        if array.size != length or not np.isfinite(array).all():
            raise RuntimeError(
                f"LIBERO controller {label} must contain {length} finite values, "
                f"received shape={array.shape}, values={array.tolist()}"
            )
        return array

    @staticmethod
    def _unwrapped_environments(environment: Any) -> list[Any]:
        values: list[Any] = []
        current = environment
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            values.append(current)
            current = getattr(current, "env", None)
        return values

    def _environment_action_spec(
        self,
        environment: Any,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        for candidate in self._unwrapped_environments(environment):
            specification = getattr(candidate, "action_spec", None)
            if callable(specification):
                specification = specification()
            if isinstance(specification, (tuple, list)) and len(specification) == 2:
                lower = np.asarray(specification[0], dtype=np.float64).reshape(-1)
                upper = np.asarray(specification[1], dtype=np.float64).reshape(-1)
                if lower.size == upper.size == 7:
                    return lower, upper, f"{type(candidate).__name__}.action_spec"
            action_space = getattr(candidate, "action_space", None)
            if action_space is not None and hasattr(action_space, "low") and hasattr(
                action_space, "high"
            ):
                lower = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
                upper = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
                if lower.size == upper.size == 7:
                    return lower, upper, f"{type(candidate).__name__}.action_space"
        raise RuntimeError(
            "Cannot resolve the 7-D LIBERO environment action_spec from the official "
            "OffScreenRenderEnv or its wrapped robosuite environment"
        )

    def _arm_controller(self, environment: Any) -> tuple[Any, str]:
        candidates: list[tuple[str, Any]] = []
        seen_controllers: set[int] = set()

        def append_candidate(path: str, controller: Any) -> None:
            identity = id(controller)
            if identity in seen_controllers:
                return
            seen_controllers.add(identity)
            candidates.append((path, controller))

        for wrapped_index, candidate in enumerate(
            self._unwrapped_environments(environment)
        ):
            robots = getattr(candidate, "robots", None)
            if not robots:
                continue
            for robot_index, robot in enumerate(robots):
                for attribute in ("controller", "composite_controller"):
                    controller = getattr(robot, attribute, None)
                    if controller is None:
                        continue
                    path = f"env[{wrapped_index}].robots[{robot_index}].{attribute}"
                    append_candidate(path, controller)
                    parts = getattr(controller, "part_controllers", None)
                    if isinstance(parts, dict):
                        for name in sorted(parts):
                            append_candidate(
                                f"{path}.part_controllers[{name!r}]",
                                parts[name],
                            )
        viable: list[tuple[str, Any]] = []
        for path, controller in candidates:
            required = ("input_min", "input_max", "output_min", "output_max")
            if not all(hasattr(controller, name) for name in required):
                continue
            try:
                self._vector(controller.input_min, 6, "input_min")
                self._vector(controller.input_max, 6, "input_max")
                self._vector(controller.output_min, 6, "output_min")
                self._vector(controller.output_max, 6, "output_max")
            except RuntimeError:
                continue
            identity = " ".join(
                str(value)
                for value in (
                    type(controller).__name__,
                    getattr(controller, "name", ""),
                    getattr(controller, "type", ""),
                )
            ).lower()
            if "osc" in identity or "operationalspace" in identity.replace("_", ""):
                viable.append((path, controller))
        if len(viable) != 1:
            available = [
                f"{path}:{type(controller).__name__}" for path, controller in candidates
            ]
            raise RuntimeError(
                "Cannot uniquely resolve the LIBERO 6-D OSC_POSE arm controller; "
                f"viable={[path for path, _ in viable]}, available={available}"
            )
        return viable[0]

    @staticmethod
    def _gripper_width(observation: dict[str, Any]) -> float:
        values = np.asarray(
            observation.get("robot0_gripper_qpos", []), dtype=np.float64
        ).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(
                "LIBERO observation does not expose finite robot0_gripper_qpos for "
                "gripper-convention verification"
            )
        return float(np.abs(values).sum())

    def _infer_gripper_mapping(
        self,
        environment: Any,
        initial_state: np.ndarray,
    ) -> tuple[str, dict[str, float]]:
        widths: dict[str, float] = {}
        for command, label in ((1.0, "positive"), (-1.0, "negative")):
            environment.reset()
            environment.set_init_state(np.asarray(initial_state))
            action = np.zeros(7, dtype=np.float32)
            action[-1] = command
            observation: dict[str, Any] | None = None
            for _ in range(max(1, int(self.cfg.data.frameskip))):
                observation, _, _, _ = environment.step(action)
            if observation is None:
                raise RuntimeError("LIBERO gripper probe produced no observation")
            widths[label] = self._gripper_width(observation)
        scale = max(abs(widths["positive"]), abs(widths["negative"]), 1.0)
        if math.isclose(
            widths["positive"],
            widths["negative"],
            rel_tol=1.0e-6,
            abs_tol=np.finfo(np.float64).eps * scale * 32.0,
        ):
            raise RuntimeError(
                "LIBERO positive/negative gripper probes are indistinguishable; "
                f"responses={widths}"
            )
        # The canonical convention is +1=close. Smaller absolute joint spread is closed.
        mapping = (
            "identity"
            if widths["positive"] < widths["negative"]
            else "inverted"
        )
        return mapping, widths

    def _action_transform_from_environment(
        self,
        environment: Any,
        initial_state: np.ndarray,
    ) -> tuple[ActionTransform, dict[str, Any]]:
        lower, upper, action_spec_source = self._environment_action_spec(environment)
        if np.any(upper <= lower):
            raise RuntimeError(
                f"LIBERO action_spec has non-positive span: lower={lower}, upper={upper}"
            )
        controller_path, controller = self._arm_controller(environment)
        input_type = getattr(controller, "input_type", None)
        input_reference_frame = getattr(controller, "input_ref_frame", None)
        controls_orientation = getattr(controller, "use_ori", None)
        if input_type != "delta" or controls_orientation is not True:
            raise RuntimeError(
                "LIBERO controller is not a verified 6-D delta OSC_POSE contract: "
                f"controller={type(controller).__name__}, input_type={input_type!r}, "
                f"use_ori={controls_orientation!r}"
            )
        if input_reference_frame not in {"base", "world"}:
            raise RuntimeError(
                "LIBERO controller action reference frame is unavailable or unsupported: "
                f"input_ref_frame={input_reference_frame!r}"
            )
        input_lower = self._vector(controller.input_min, 6, "input_min")
        input_upper = self._vector(controller.input_max, 6, "input_max")
        output_lower = self._vector(controller.output_min, 6, "output_min")
        output_upper = self._vector(controller.output_max, 6, "output_max")
        if not np.allclose(lower[:6], input_lower) or not np.allclose(
            upper[:6], input_upper
        ):
            raise RuntimeError(
                "LIBERO environment action_spec arm bounds do not match the resolved "
                f"controller input bounds: action_spec=({lower[:6]}, {upper[:6]}), "
                f"controller=({input_lower}, {input_upper})"
            )
        gripper_mapping, gripper_response = self._infer_gripper_mapping(
            environment,
            initial_state,
        )
        translation_scale = tuple(
            float(value) for value in np.maximum(np.abs(output_lower[:3]), np.abs(output_upper[:3]))
        )
        rotation_scale = tuple(
            float(value) for value in np.maximum(np.abs(output_lower[3:]), np.abs(output_upper[3:]))
        )
        expected_frequency = float(
            self.cfg.benchmark.get("control_frequency_hz", 20.0)
        )
        actual_frequency: float | None = None
        frequency_source = ""
        for candidate in self._unwrapped_environments(environment):
            value = getattr(candidate, "control_freq", None)
            if value is not None:
                actual_frequency = float(value)
                frequency_source = f"{type(candidate).__name__}.control_freq"
                break
        if actual_frequency is None:
            raise RuntimeError(
                "Cannot resolve LIBERO control frequency from the live environment"
            )
        if not math.isclose(actual_frequency, expected_frequency):
            raise RuntimeError(
                "LIBERO control frequency differs from configuration: "
                f"configured={expected_frequency}, actual={actual_frequency}"
            )
        verified_identity = bool(
            np.allclose(output_lower, np.asarray(JEPA_WM_CANONICAL_LOWER[:6]))
            and np.allclose(output_upper, np.asarray(JEPA_WM_CANONICAL_UPPER[:6]))
            and gripper_mapping == "identity"
            and np.allclose(lower, np.asarray(JEPA_WM_CANONICAL_LOWER))
            and np.allclose(upper, np.asarray(JEPA_WM_CANONICAL_UPPER))
        )
        source = (
            f"{action_spec_source}; {controller_path}; {frequency_source}; "
            "positive/negative gripper response verified from a common simulator state"
        )
        transform = ActionTransform(
            canonical_lower=JEPA_WM_CANONICAL_LOWER,
            canonical_upper=JEPA_WM_CANONICAL_UPPER,
            environment_lower=tuple(float(value) for value in lower),
            environment_upper=tuple(float(value) for value in upper),
            controller_input_lower=tuple(float(value) for value in input_lower),
            controller_input_upper=tuple(float(value) for value in input_upper),
            controller_output_lower=tuple(float(value) for value in output_lower),
            controller_output_upper=tuple(float(value) for value in output_upper),
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            gripper_mapping=gripper_mapping,
            transform_name="jepa_wm_physical_delta_to_libero_controller_v1",
            verified_identity=verified_identity,
            verification_source=source,
            controller_type=type(controller).__name__,
            control_frequency_hz=actual_frequency,
            action_repeat=int(self.cfg.data.frameskip),
        )
        return transform, {
            "action_spec_lower": lower.tolist(),
            "action_spec_upper": upper.tolist(),
            "action_spec_source": action_spec_source,
            "controller_path": controller_path,
            "controller_type": type(controller).__name__,
            "controller_input_type": input_type,
            "controller_input_reference_frame": input_reference_frame,
            "rotation_representation": "delta axis-angle",
            "controller_input_lower": input_lower.tolist(),
            "controller_input_upper": input_upper.tolist(),
            "controller_output_lower": output_lower.tolist(),
            "controller_output_upper": output_upper.tolist(),
            "translation_scale": list(translation_scale),
            "rotation_scale": list(rotation_scale),
            "gripper_response": gripper_response,
            "gripper_mapping": gripper_mapping,
            "control_frequency_hz": actual_frequency,
        }

    def resolve_task(self, *, strict: bool) -> ResolvedTask:
        if strict and self._strict_resolution is not None:
            return self._strict_resolution
        required_eval_count = int(self.cfg.evaluation.num_episodes)
        train_fraction = float(self.cfg.data.train_fraction)
        split_seed = int(self.cfg.data.split_seed)
        if strict:
            existing = self.existing_task_manifest()
            if existing is not None:
                expected_train = int(
                    np.floor(existing.available_demonstrations * train_fraction)
                )
                split_mismatch = (
                    len(existing.selected_train_demonstrations) != expected_train
                    or len(existing.selected_test_demonstrations)
                    != existing.available_demonstrations - expected_train
                    or len(existing.selected_test_demonstrations) < required_eval_count
                    or existing.initial_states_count < required_eval_count
                    or existing.camera_height is None
                    or existing.camera_width is None
                    or existing.camera_channel_order is None
                    or existing.camera_vertical_flip is None
                    or existing.action_transform is None
                )
                if split_mismatch:
                    raise RuntimeError(
                        "Immutable LIBERO task manifest mismatch after the 60/40 "
                        "split/camera/action-contract upgrade: "
                        f"path={self.task_manifest_path()}, "
                        f"total_successful_demos={existing.available_demonstrations}, "
                        f"train_fraction={train_fraction}, "
                        f"train_count={len(existing.selected_train_demonstrations)}, "
                        f"held_out_count={len(existing.selected_test_demonstrations)}, "
                        f"required_evaluation_count={required_eval_count}. Delete or "
                        "archive this not-yet-used cross_benchmark_v1 LIBERO manifest "
                        "before rerunning preflight; completed artifacts must be retained."
                    )
                self._strict_resolution = existing
                return existing
        failures: dict[str, list[str]] = {}
        try:
            task_suite, task, bddl, commit = self._official_suite()
            dataset = self._dataset_path(task)
            inspected = self._inspect_dataset(dataset)
            if not bddl.is_file():
                raise FileNotFoundError(f"LIBERO BDDL file does not exist: {bddl}")
            init_states = self._official_init_states(task)
            successful = list(inspected["successful_demonstrations"])
            train_indices, evaluation_indices = _validated_split_indices(
                len(successful),
                train_fraction,
                split_seed,
                required_eval_count,
            )
            if len(init_states) < required_eval_count:
                raise RuntimeError(
                    "LIBERO fixed init-state file cannot satisfy evaluation: "
                    f"total_successful_demos={len(successful)}, "
                    f"train_fraction={train_fraction}, "
                    f"train_count={len(train_indices)}, "
                    f"held_out_count={len(evaluation_indices)}, "
                    f"required_evaluation_count={required_eval_count}, "
                    f"init_state_count={len(init_states)}"
                )
            selected_train = tuple(successful[int(index)] for index in train_indices.tolist())
            selected_test = tuple(successful[int(index)] for index in evaluation_indices.tolist())
            lengths = np.asarray(inspected["lengths"], dtype=np.int64)
            episode_cap_basis = (
                f"{self.cfg.evaluation.episode_cap_basis}; demonstration lengths "
                f"min/median/max={int(lengths.min())}/{float(np.median(lengths)):.1f}/"
                f"{int(lengths.max())}; control_frequency_hz="
                f"{self.action_convention().control_frequency_hz}"
            )
            camera_contract = _camera_contract_from_shape(
                inspected["image_shape"],
                vertical_flip=bool(self.cfg.data.get("vertical_flip", False)),
            )
            resolved = ResolvedTask(
                task_key=str(self.cfg.benchmark.task_key),
                benchmark="libero",
                suite=str(self.cfg.benchmark.suite),
                task_id=int(self.cfg.benchmark.task_id),
                task_name=str(task.name),
                language_instruction=str(task.language),
                bddl_path=str(bddl),
                bddl_sha256=sha256_file(bddl),
                problem_folder=str(task.problem_folder),
                initial_states_sha256=array_sha256(
                    np.asarray(init_states)
                ),
                initial_states_count=len(init_states),
                dataset_path=str(dataset),
                dataset_sha256=sha256_file(dataset) if strict else None,
                available_demonstrations=len(successful),
                selected_train_demonstrations=selected_train,
                selected_test_demonstrations=selected_test,
                camera_key=str(inspected["camera_key"]),
                action_convention=self.action_convention().as_dict(),
                environment_implementation="libero.libero.envs.OffScreenRenderEnv",
                upstream_commits={"libero": commit},
                frameskip=int(self.cfg.data.frameskip),
                max_episode_steps=int(self.cfg.evaluation.max_episode_steps),
                episode_cap_basis=episode_cap_basis,
                **camera_contract,
            )
            if strict:
                with h5py.File(dataset, "r") as handle:
                    initial_state = np.asarray(
                        handle["data"][successful[0]]["states"][0]
                    )
                environment = self._create_environment(resolved)
                try:
                    environment.seed(int(self.cfg.evaluation.eval_seed))
                    environment.reset()
                    action_transform, _ = self._action_transform_from_environment(
                        environment,
                        initial_state,
                    )
                finally:
                    environment.close()
                resolved = replace(
                    resolved,
                    action_transform=action_transform.as_dict(),
                )
                self._strict_resolution = resolved
            return resolved
        except Exception as error:
            if strict:
                raise
            failures["task_id_0"] = [f"{type(error).__name__}: {error}"]
            return ResolvedTask(
                task_key=str(self.cfg.benchmark.task_key),
                benchmark="libero",
                suite=str(self.cfg.benchmark.suite),
                task_id=int(self.cfg.benchmark.task_id),
                task_name=f"unresolved_{self.cfg.benchmark.suite}_task_{self.cfg.benchmark.task_id}",
                language_instruction=None,
                bddl_path=None,
                bddl_sha256=None,
                problem_folder=None,
                initial_states_sha256=None,
                initial_states_count=0,
                dataset_path="",
                dataset_sha256=None,
                available_demonstrations=0,
                selected_train_demonstrations=(),
                selected_test_demonstrations=(),
                camera_key="unresolved_agentview",
                action_convention=self.action_convention().as_dict(),
                environment_implementation="libero.libero.envs.OffScreenRenderEnv",
                upstream_commits={},
                frameskip=int(self.cfg.data.frameskip),
                max_episode_steps=int(self.cfg.evaluation.max_episode_steps),
                episode_cap_basis=str(self.cfg.evaluation.episode_cap_basis),
                status="unresolved",
                candidate_failures=failures,
                camera_height=None,
                camera_width=None,
                camera_channel_order=None,
                camera_vertical_flip=None,
                action_transform=None,
            )

    def _create_environment(self, task: ResolvedTask) -> Any:
        self._repo()
        from libero.libero.envs import OffScreenRenderEnv

        if task.camera_height is None or task.camera_width is None:
            raise RuntimeError(
                "Resolved LIBERO task lacks dataset-derived camera dimensions: "
                f"height={task.camera_height}, width={task.camera_width}"
            )
        return OffScreenRenderEnv(
            bddl_file_name=task.bddl_path,
            camera_names=["agentview"],
            camera_heights=int(task.camera_height),
            camera_widths=int(task.camera_width),
            control_freq=int(self.action_convention().control_frequency_hz),
            horizon=int(task.max_episode_steps),
        )

    def _observation_image(self, observation: dict[str, Any]) -> np.ndarray:
        task = self.resolve_task(strict=True)
        for key in ("agentview_image", "agentview_rgb"):
            if key in observation:
                image = np.asarray(observation[key])
                if image.ndim == 3 and image.shape[-1] == 3:
                    if bool(task.camera_vertical_flip):
                        return image[::-1].copy()
                    return image
        raise RuntimeError(
            f"LIBERO simulator observation lacks agent-view RGB; keys={sorted(observation)}"
        )

    def _deep_environment_preflight(
        self,
        task: ResolvedTask,
        source: LiberoTrajectoryDataset,
    ) -> dict[str, Any]:
        observation, _, states, _, _ = source.get_frames(0, [0])
        with h5py.File(source.path, "r") as handle:
            demo_key = source.trajectories[0].demo_key
            raw_demo_actions = np.asarray(handle["data"][demo_key]["actions"])
        action_transform = ActionTransform.from_dict(task.action_transform or {})
        canonical_demo_actions = action_transform.environment_to_canonical_action(
            raw_demo_actions
        )
        round_trip_actions = action_transform.canonical_to_environment_action(
            canonical_demo_actions
        )
        round_trip_error = float(
            np.max(np.abs(round_trip_actions.astype(np.float64) - raw_demo_actions))
        )
        if round_trip_error > 2.0e-6:
            raise RuntimeError(
                "LIBERO environment/canonical action round trip is not stable: "
                f"max_abs_error={round_trip_error}"
            )
        motion_candidates = np.flatnonzero(
            np.linalg.norm(canonical_demo_actions[:, :6], axis=1) > 1.0e-6
        )
        if motion_candidates.size == 0:
            raise RuntimeError(
                "LIBERO action-convention preflight needs a nonzero translation or "
                f"axis-angle action in {demo_key}"
            )
        action_index = int(motion_candidates[0])
        _, canonical_action_probe, action_state, _, _ = source.get_frames(
            0, [action_index]
        )
        environment = self._create_environment(task)
        try:
            environment.seed(int(self.cfg.evaluation.eval_seed))
            environment.reset()
            actual_transform, controller_report = (
                self._action_transform_from_environment(
                    environment,
                    states[0].numpy(),
                )
            )
            if actual_transform.as_dict() != action_transform.as_dict():
                raise RuntimeError(
                    "LIBERO live action contract differs from the immutable task "
                    f"manifest: expected={action_transform.as_dict()}, "
                    f"actual={actual_transform.as_dict()}"
                )
            # Camera parity must compare the dataset image with the same recorded
            # simulator state, not an unrelated task-level fixed initial state.
            environment.reset()
            simulator_observation = environment.set_init_state(states[0].numpy())
            simulator_image = self._observation_image(simulator_observation)
            dataset_image = observation["visual"][0].permute(1, 2, 0).numpy()
            if simulator_image.shape != dataset_image.shape:
                raise RuntimeError(
                    "LIBERO camera parity shape mismatch: "
                    f"simulator={simulator_image.shape}, dataset={dataset_image.shape}"
                )
            for label, image in (
                ("simulator", simulator_image),
                ("dataset", dataset_image),
            ):
                if image.dtype == np.uint8:
                    minimum, maximum = float(image.min()), float(image.max())
                    valid_range = minimum >= 0.0 and maximum <= 255.0
                elif np.issubdtype(image.dtype, np.floating):
                    minimum, maximum = float(image.min()), float(image.max())
                    valid_range = minimum >= 0.0 and maximum <= 1.0
                else:
                    raise RuntimeError(
                        f"LIBERO {label} camera dtype is unsupported: {image.dtype}"
                    )
                if not valid_range:
                    raise RuntimeError(
                        f"LIBERO {label} camera range is invalid: [{minimum}, {maximum}]"
                    )
            sim_float = simulator_image.astype(np.float32)
            data_float = dataset_image.astype(np.float32)
            parity_errors = {
                "direct": float(np.abs(sim_float - data_float).mean()),
                "vertical_flip": float(np.abs(sim_float - data_float[::-1]).mean()),
                "bgr": float(np.abs(sim_float - data_float[..., ::-1]).mean()),
                "vertical_flip_bgr": float(
                    np.abs(sim_float - data_float[::-1, :, ::-1]).mean()
                ),
            }
            ordered_parity = sorted(parity_errors.items(), key=lambda item: item[1])
            if ordered_parity[0][0] != "direct":
                raise RuntimeError(
                    "LIBERO camera parity indicates an orientation/channel mismatch: "
                    f"errors={parity_errors}, best={ordered_parity[0][0]}"
                )
            if math.isclose(
                ordered_parity[0][1],
                ordered_parity[1][1],
                rel_tol=1.0e-3,
                abs_tol=np.finfo(np.float32).eps
                * max(ordered_parity[0][1], ordered_parity[1][1], 1.0),
            ):
                raise RuntimeError(
                    "LIBERO camera parity is ambiguous between direct and an "
                    f"incorrect orientation/channel candidate: errors={parity_errors}"
                )
            probe_state_changes: dict[str, float] = {}
            for label, dimension, magnitude in (
                ("translation", 0, 0.01),
                ("rotation_axis_angle", 3, 0.01),
            ):
                environment.reset()
                environment.set_init_state(states[0].numpy())
                before_probe = np.asarray(environment.get_sim_state()).copy()
                canonical_probe = np.zeros(7, dtype=np.float32)
                canonical_probe[dimension] = magnitude
                environment_probe = (
                    action_transform.canonical_to_environment_action(
                        canonical_probe
                    )
                )
                for _ in range(max(1, int(self.cfg.data.frameskip))):
                    environment.step(environment_probe)
                after_probe = np.asarray(environment.get_sim_state())
                delta = float(np.linalg.norm(after_probe - before_probe))
                if not np.isfinite(after_probe).all() or delta <= np.finfo(
                    np.float64
                ).eps * max(float(np.linalg.norm(before_probe)), 1.0) * 32.0:
                    raise RuntimeError(
                        f"LIBERO canonical {label} probe did not produce a finite, "
                        f"observable simulator-state change: delta_l2={delta}, "
                        f"canonical_action={canonical_probe.tolist()}, "
                        f"environment_action={environment_probe.tolist()}"
                    )
                probe_state_changes[label] = delta
            environment.reset()
            simulator_observation = environment.set_init_state(
                action_state[0].numpy()
            )
            before_state = np.asarray(environment.get_sim_state()).copy()
            canonical_action = canonical_action_probe[0].numpy()
            environment_action = action_transform.canonical_to_environment_action(
                canonical_action
            )
            after_observation = simulator_observation
            reward = 0.0
            done = False
            info: dict[str, Any] = {}
            for _ in range(max(1, int(self.cfg.data.frameskip))):
                after_observation, reward, done, info = environment.step(
                    environment_action
                )
            after_state = np.asarray(environment.get_sim_state())
            if not np.isfinite(after_state).all() or np.allclose(before_state, after_state):
                raise RuntimeError("LIBERO demonstration action did not produce finite simulator state change")
            success = bool(
                reward > 0
                or info.get("success", False)
                or environment.check_success()
            )
            return {
                "camera_parity_mae": parity_errors,
                "camera_parity_best_candidate": ordered_parity[0][0],
                "camera_parity_ambiguous": False,
                "camera_shape": list(simulator_image.shape),
                "camera_dtype": str(simulator_image.dtype),
                "action_state_delta_l2": float(np.linalg.norm(after_state - before_state)),
                "canonical_probe_state_delta_l2": probe_state_changes,
                "action_round_trip_max_abs_error": round_trip_error,
                "action_contract": action_transform.as_dict(),
                "controller_contract": controller_report,
                "success_signal_type": "reward/info/check_success; done terminates only",
                "success_on_probe": success,
                "dataset_state_shape": list(states.shape),
                "action_probe_index": action_index,
                "canonical_action_probe": canonical_action.tolist(),
                "environment_action_probe": environment_action.tolist(),
            }
        finally:
            environment.close()

    def preflight(self, *, deep: bool) -> dict[str, Any]:
        task = self.resolve_task(strict=True)
        source = self.build_source_dataset(output_environment_info=deep)
        train, evaluation = self.split_trajectory_ids(source)
        train_ids = tuple(
            source.trajectories[int(index)].demo_key for index in train.tolist()
        )
        evaluation_ids = tuple(
            source.trajectories[int(index)].demo_key
            for index in evaluation.tolist()
        )
        if train_ids != tuple(task.selected_train_demonstrations) or evaluation_ids != tuple(task.selected_test_demonstrations):
            raise RuntimeError(
                "LIBERO deterministic split no longer matches the immutable task manifest"
            )
        required_eval_count = int(self.cfg.evaluation.num_episodes)
        if len(train) < 1 or len(evaluation) < required_eval_count:
            raise RuntimeError(
                "LIBERO deterministic split cannot satisfy this run: "
                f"total_successful_demos={len(source)}, "
                f"train_fraction={float(self.cfg.data.train_fraction)}, "
                f"train_count={len(train)}, held_out_count={len(evaluation)}, "
                f"required_evaluation_count={required_eval_count}"
            )
        candidates = self.enumerate_window_candidates(
            source, int(self.cfg.data.num_frames), int(self.cfg.data.frameskip)
        )
        selected = self.select_windows(
            candidates,
            train,
            int(self.cfg.data.num_train_windows),
            int(self.cfg.data.window_seed),
        )
        if len(selected) != int(self.cfg.data.num_train_windows):
            raise RuntimeError(
                f"LIBERO selected {len(selected)} training windows; expected {self.cfg.data.num_train_windows}"
            )
        report = {
            "benchmark": self.name,
            "task": task.as_dict(),
            "train_demonstrations": train.tolist(),
            "evaluation_demonstrations": evaluation.tolist(),
            "selected_train_windows": len(selected),
            "deep_environment_check": deep,
        }
        if deep:
            report["environment_parity"] = self._deep_environment_preflight(task, source)
        return report

    def build_source_dataset(self, *, output_environment_info: bool) -> LiberoTrajectoryDataset:
        del output_environment_info
        task = self.resolve_task(strict=True)
        inspected = self._inspect_dataset(resolve_path(task.dataset_path))
        successful = tuple(str(key) for key in inspected["successful_demonstrations"])
        return LiberoTrajectoryDataset(
            task.dataset_path,
            camera_key=task.camera_key,
            vertical_flip=bool(task.camera_vertical_flip),
            action_transform=ActionTransform.from_dict(task.action_transform or {}),
            demonstration_keys=successful,
        )

    def enumerate_window_candidates(
        self,
        source_dataset: LiberoTrajectoryDataset,
        num_frames: int,
        frameskip: int,
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        span = (num_frames - 1) * frameskip + 1
        for trajectory in range(len(source_dataset)):
            latest = source_dataset.get_seq_length(trajectory) - span
            if latest >= 0:
                candidates.extend((trajectory, start) for start in range(latest + 1))
        return candidates

    def make_window_dataset(
        self,
        source_dataset: LiberoTrajectoryDataset,
        selections: list[tuple[int, int]],
        *,
        num_frames: int,
        frameskip: int,
        appearance_seed: int,
        appearance_severity: float,
    ) -> LiberoWindowDataset:
        return LiberoWindowDataset(
            source_dataset,
            selections,
            num_frames=num_frames,
            frameskip=frameskip,
            appearance_seed=appearance_seed,
            appearance_severity=appearance_severity,
        )

    def build_evaluation_manifest(
        self,
        task: ResolvedTask,
        *,
        count: int,
        seed: int,
        appearance_seed: int,
    ) -> dict[str, Any]:
        source = self.build_source_dataset(output_environment_info=True)
        _, evaluation = self.split_trajectory_ids(source)
        if len(evaluation) < count:
            raise RuntimeError(
                f"LIBERO evaluation needs {count} distinct held-out demonstrations, found {len(evaluation)}"
            )
        generator = np.random.default_rng(seed)
        selected = generator.permutation(evaluation)[:count]
        instances: list[EvaluationInstance] = []
        for position, trajectory_value in enumerate(selected.tolist()):
            trajectory = int(trajectory_value)
            end = source.get_seq_length(trajectory) - 1
            observation, _, states, rewards, info = source.get_frames(
                trajectory, [0, end]
            )
            if not bool(rewards[-1] > 0):
                with h5py.File(source.path, "r") as handle:
                    group = handle["data"][info["demo_key"]]
                    dones = np.asarray(group["dones"]) if "dones" in group else np.zeros(1)
                if not bool(dones[-1]):
                    raise RuntimeError(
                        f"LIBERO held-out demonstration is not marked successful: {info['demo_key']}"
                    )
            identity = {
                "suite": task.suite,
                "task_id": task.task_id,
                "demo": info["demo_key"],
            }
            instances.append(
                EvaluationInstance(
                    instance_id=canonical_sha256(identity)[:24],
                    source_trajectory_id=str(info["demo_key"]),
                    source_trajectory_index=trajectory,
                    segment_start=0,
                    segment_end=end,
                    initialization_fingerprint=array_sha256(states[0].numpy()),
                    goal_fingerprint=array_sha256(observation["visual"][-1].numpy()),
                    environment_seed=int(
                        (seed * seed + position * seed) % (2**32 - 2)
                    ),
                    cem_seed=int(seed + position),
                    appearance_seed=int(appearance_seed + position),
                )
            )
        return self.finalize_evaluation_manifest(
            task,
            instances,
            {"distinct_demonstrations": True},
        )

    def canonical_to_environment_action(self, action: np.ndarray) -> np.ndarray:
        task = self.resolve_task(strict=True)
        return ActionTransform.from_dict(
            task.action_transform or {}
        ).canonical_to_environment_action(action)

    def environment_to_canonical_action(self, action: np.ndarray) -> np.ndarray:
        task = self.resolve_task(strict=True)
        return ActionTransform.from_dict(
            task.action_transform or {}
        ).environment_to_canonical_action(action)

    def run_planning(
        self,
        *,
        backend: Any,
        method: Any,
        output_directory: str | Path,
    ) -> Any:
        from wm_adapter.planning.libero_planner import run_libero_planning

        return run_libero_planning(
            experiment_config=self.cfg,
            backend=backend,
            method=method,
            output_directory=output_directory,
        )
