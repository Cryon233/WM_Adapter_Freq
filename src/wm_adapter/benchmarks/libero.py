from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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
    BenchmarkAdapter,
    EvaluationInstance,
    ResolvedTask,
    array_sha256,
    canonical_sha256,
)
from wm_adapter.utils.checkpoints import git_commit, sha256_file
from wm_adapter.utils.reproducibility import resolve_path


LIBERO_CAMERA_KEYS = (
    "agentview_rgb",
    "agentview_image",
    "obs/agentview_rgb",
    "obs/agentview_image",
)


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
        demonstration_keys: tuple[str, ...] | None = None,
    ) -> None:
        self.path = resolve_path(path)
        self.camera_key = camera_key
        self.vertical_flip = vertical_flip
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
        actions = torch.from_numpy(np.asarray(group["actions"][frame_indices])).float()
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
                rewards = np.asarray(group["rewards"]) if "rewards" in group else None
                dones = np.asarray(group["dones"]) if "dones" in group else None
                if (rewards is not None and bool(np.any(rewards > 0))) or (
                    dones is not None and bool(dones[-1])
                ):
                    successful.append(key)
            if np.any(action_min < -1.0001) or np.any(action_max > 1.0001):
                raise RuntimeError(
                    "LIBERO demonstration actions exceed the canonical [-1,1] range: "
                    f"min={action_min.tolist()}, max={action_max.tolist()}"
                )
            image = _dataset_node(first, camera)
            if image is None or image.ndim != 4 or image.shape[-1] != 3:
                raise RuntimeError(
                    f"LIBERO agent-view image must be [T,H,W,3], found {None if image is None else image.shape}"
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
            }

    def action_convention(self) -> ActionConvention:
        return ActionConvention(
            dimension=7,
            translation="normalized OSC_POSE delta Cartesian position",
            rotation="normalized OSC_POSE delta axis-angle vector",
            gripper="scalar command; -1=open and +1=close",
            source_range=(-1.0, 1.0),
            target_range=(-1.0, 1.0),
            controller_type="LIBERO official OSC_POSE",
            control_frequency_hz=float(self.cfg.benchmark.get("control_frequency_hz", 20.0)),
            action_repeat=int(self.cfg.data.frameskip),
            transform="identity LIBERO normalized OSC_POSE action <-> JEPA-WM canonical 7-D action",
        )

    def resolve_task(self, *, strict: bool) -> ResolvedTask:
        if strict:
            existing = self.existing_task_manifest()
            if existing is not None:
                return existing
        failures: dict[str, list[str]] = {}
        try:
            task_suite, task, bddl, commit = self._official_suite()
            dataset = self._dataset_path(task)
            inspected = self._inspect_dataset(dataset)
            if not bddl.is_file():
                raise FileNotFoundError(f"LIBERO BDDL file does not exist: {bddl}")
            init_states = task_suite.get_task_init_states(int(self.cfg.benchmark.task_id))
            if len(init_states) < 20:
                raise RuntimeError(
                    f"LIBERO fixed init-state file has {len(init_states)} states; 20 are required"
                )
            successful = list(inspected["successful_demonstrations"])
            if len(successful) < 20:
                raise RuntimeError(
                    f"LIBERO dataset has {len(successful)} successful demonstrations; 20 are required"
                )
            train_indices, evaluation_indices = self.split_trajectory_ids(successful)
            selected_train = tuple(successful[int(index)] for index in train_indices.tolist())
            selected_test = tuple(successful[int(index)] for index in evaluation_indices.tolist())
            if len(selected_test) < 20:
                raise RuntimeError(
                    "The deterministic 80/20 split leaves fewer than 20 distinct held-out "
                    f"successful demonstrations: {len(selected_test)}"
                )
            lengths = np.asarray(inspected["lengths"], dtype=np.int64)
            episode_cap_basis = (
                f"{self.cfg.evaluation.episode_cap_basis}; demonstration lengths "
                f"min/median/max={int(lengths.min())}/{float(np.median(lengths)):.1f}/"
                f"{int(lengths.max())}; control_frequency_hz="
                f"{self.action_convention().control_frequency_hz}"
            )
            return ResolvedTask(
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
            )
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
            )

    def _create_environment(self, task: ResolvedTask) -> Any:
        self._repo()
        from libero.libero.envs import OffScreenRenderEnv

        return OffScreenRenderEnv(
            bddl_file_name=task.bddl_path,
            camera_names=["agentview"],
            camera_heights=256,
            camera_widths=256,
            control_freq=int(self.action_convention().control_frequency_hz),
            horizon=int(task.max_episode_steps),
        )

    def _observation_image(self, observation: dict[str, Any]) -> np.ndarray:
        for key in ("agentview_image", "agentview_rgb"):
            if key in observation:
                image = np.asarray(observation[key])
                if image.ndim == 3 and image.shape[-1] == 3:
                    if bool(self.cfg.data.get("vertical_flip", False)):
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
            demo_actions = np.asarray(handle["data"][demo_key]["actions"])
        gripper_candidates = np.flatnonzero(np.abs(demo_actions[:, -1]) > 0.5)
        motion_candidates = np.flatnonzero(
            np.linalg.norm(demo_actions[:, :6], axis=1) > 1.0e-4
        )
        if gripper_candidates.size == 0 or motion_candidates.size == 0:
            raise RuntimeError(
                "LIBERO action-convention preflight needs at least one meaningful "
                f"gripper and Cartesian/rotation action in {demo_key}"
            )
        action_index = int(gripper_candidates[0])
        _, action_probe, action_state, _, _ = source.get_frames(0, [action_index])
        environment = self._create_environment(task)
        try:
            environment.seed(int(self.cfg.evaluation.eval_seed))
            environment.reset()
            # Camera parity must compare the dataset image with the same recorded
            # simulator state, not an unrelated task-level fixed initial state.
            simulator_observation = environment.set_init_state(states[0].numpy())
            simulator_image = self._observation_image(simulator_observation)
            dataset_image = observation["visual"][0].permute(1, 2, 0).numpy()
            if simulator_image.shape != dataset_image.shape:
                raise RuntimeError(
                    "LIBERO camera parity shape mismatch: "
                    f"simulator={simulator_image.shape}, dataset={dataset_image.shape}"
                )
            sim_float = simulator_image.astype(np.float32)
            data_float = dataset_image.astype(np.float32)
            direct = float(np.abs(sim_float - data_float).mean())
            flipped = float(np.abs(sim_float - data_float[::-1]).mean())
            bgr = float(np.abs(sim_float - data_float[..., ::-1]).mean())
            if direct > flipped or direct > bgr:
                raise RuntimeError(
                    "LIBERO camera parity indicates an orientation/channel mismatch: "
                    f"direct_mae={direct}, vertical_flip_mae={flipped}, bgr_mae={bgr}"
                )
            simulator_observation = environment.set_init_state(
                action_state[0].numpy()
            )
            before_state = np.asarray(environment.get_sim_state()).copy()
            action = action_probe[0].numpy()
            before_gripper = np.asarray(
                simulator_observation.get("robot0_gripper_qpos", []), dtype=np.float64
            )
            after_observation = simulator_observation
            reward = 0.0
            done = False
            info: dict[str, Any] = {}
            for _ in range(max(1, int(self.cfg.data.frameskip))):
                after_observation, reward, done, info = environment.step(action)
            after_state = np.asarray(environment.get_sim_state())
            if not np.isfinite(after_state).all() or np.allclose(before_state, after_state):
                raise RuntimeError("LIBERO demonstration action did not produce finite simulator state change")
            after_gripper = np.asarray(
                after_observation.get("robot0_gripper_qpos", []), dtype=np.float64
            )
            if before_gripper.size and after_gripper.shape == before_gripper.shape and abs(action[-1]) > 0.5:
                before_width = float(np.abs(before_gripper).sum())
                after_width = float(np.abs(after_gripper).sum())
                if (action[-1] > 0 and after_width > before_width + 1.0e-4) or (
                    action[-1] < 0 and after_width < before_width - 1.0e-4
                ):
                    raise RuntimeError(
                        "LIBERO gripper response is reversed relative to the declared -1=open,+1=close convention"
                    )
            success = bool(
                reward > 0
                or info.get("success", False)
                or environment.check_success()
            )
            return {
                "camera_direct_mae": direct,
                "camera_vertical_flip_mae": flipped,
                "camera_bgr_mae": bgr,
                "action_state_delta_l2": float(np.linalg.norm(after_state - before_state)),
                "success_signal_type": "reward/info/check_success; done terminates only",
                "success_on_probe": success,
                "dataset_state_shape": list(states.shape),
                "action_probe_index": action_index,
                "action_probe": action.tolist(),
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
        if len(evaluation) < 20:
            raise RuntimeError(
                f"LIBERO held-out partition has {len(evaluation)} distinct demonstrations; 20 are required"
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
            vertical_flip=bool(self.cfg.data.get("vertical_flip", False)),
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
        payload = self.finalize_evaluation_manifest(task, instances)
        payload["distinct_demonstrations"] = True
        payload["evaluation_manifest_sha256"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "evaluation_manifest_sha256"}
        )
        return payload

    @staticmethod
    def canonical_to_environment_action(action: np.ndarray) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (7,):
            raise ValueError(
                f"LIBERO canonical action must have shape (7,), received {values.shape}"
            )
        if not np.isfinite(values).all() or np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError(
                f"LIBERO canonical action must be finite in [-1,1], received {values}"
            )
        return values.copy()

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
