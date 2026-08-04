from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from wm_adapter.benchmarks.base import (
    ActionConvention,
    ActionTransform,
    BenchmarkAdapter,
    EvaluationInstance,
    ResolvedTask,
    array_sha256,
    canonical_sha256,
)
from wm_adapter.data.robocasa_windows import (
    RoboCasaWindowDataset,
    build_robocasa_dataset,
)
from wm_adapter.data.robocasa_lerobot import (
    LEROBOT_DATASET_FORMAT,
    RoboCasaLeRobotDataset,
    inspect_robocasa_lerobot,
)
from wm_adapter.utils.checkpoints import (
    UPSTREAM_COMMITS,
    sha256_dataset_path,
)
from wm_adapter.utils.reproducibility import resolve_path


ARTICULATED_TASK_PRIORITY = (
    "OpenDrawer",
    "CloseDrawer",
    "OpenCabinet",
    "CloseCabinet",
)


class RoboCasaBenchmark(BenchmarkAdapter):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._strict_resolution: ResolvedTask | None = None

    @property
    def name(self) -> str:
        return "robocasa"

    def action_convention(self) -> ActionConvention:
        return ActionConvention(
            dimension=7,
            translation="JEPA-WM physical delta Cartesian position in the robot base frame",
            rotation="JEPA-WM physical delta axis-angle rotation",
            gripper="scalar command; -1=open and +1=close",
            source_range=(-1.0, 1.0),
            target_range=(-1.0, 1.0),
            controller_type="DROID canonical action -> RoboCasa OSC_POSE",
            control_frequency_hz=float(self.cfg.benchmark.get("control_frequency_hz", 20.0)),
            action_repeat=int(self.cfg.data.frameskip),
            transform=(
                "JEPA-WM physical deltas are mapped to normalized RoboCasa "
                "OSC_POSE controller input by the pinned wrapper"
            ),
        )

    def action_transform(self) -> ActionTransform:
        output_scale = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
        return ActionTransform(
            canonical_lower=tuple(-value for value in (*output_scale, 1.0)),
            canonical_upper=(*output_scale, 1.0),
            environment_lower=(-1.0,) * 7,
            environment_upper=(1.0,) * 7,
            controller_input_lower=(-1.0,) * 6,
            controller_input_upper=(1.0,) * 6,
            controller_output_lower=tuple(-value for value in output_scale),
            controller_output_upper=output_scale,
            translation_scale=output_scale[:3],
            rotation_scale=output_scale[3:],
            gripper_mapping="identity",
            transform_name="jepa_wm_physical_delta_to_robocasa_osc_pose_v1",
            verified_identity=False,
            verification_source=(
                "pinned JEPA-WM RoboCasaWrapper RCASA_CONTROLLER_OUTPUT_LIMS "
                "and RoboCasa OSC_POSE action contract"
            ),
            controller_type="RoboCasa OSC_POSE via JEPA-WM RoboCasaWrapper",
            control_frequency_hz=float(
                self.cfg.benchmark.get("control_frequency_hz", 20.0)
            ),
            action_repeat=int(self.cfg.data.frameskip),
        )

    def _candidate_dataset(self, candidate: str) -> Path | None:
        mappings = self.cfg.paths.get("candidate_datasets", {})
        value = mappings.get(candidate) if mappings else None
        if value is None or not str(value).strip():
            value = self.cfg.paths.get("robocasa_hdf5", "")
        if not str(value).strip():
            return None
        return resolve_path(str(value))

    def _candidate_lerobot_dataset(self, candidate: str) -> Path | None:
        mappings = self.cfg.paths.get("candidate_lerobot_datasets", {})
        value = mappings.get(candidate) if mappings else None
        if value is None or not str(value).strip():
            return None
        return resolve_path(str(value))

    def _candidate_sources(self, candidate: str) -> list[tuple[str, Path]]:
        sources: list[tuple[str, Path]] = []
        lerobot = self._candidate_lerobot_dataset(candidate)
        if lerobot is not None:
            sources.append((LEROBOT_DATASET_FORMAT, lerobot))
        hdf5 = self._candidate_dataset(candidate)
        if hdf5 is not None:
            sources.append(("hdf5", hdf5))
        return sources

    @staticmethod
    def _inspect_dataset(
        path: Path,
        expected_task: str,
        camera_view: str,
    ) -> tuple[list[str], dict[str, Any]]:
        errors: list[str] = []
        details: dict[str, Any] = {"dataset_path": str(path)}
        if not path.is_file():
            return [f"demonstration HDF5 does not exist: {path}"], details
        try:
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    return ["HDF5 lacks root group 'data'"], details
                data = handle["data"]
                try:
                    env_args = json.loads(str(data.attrs.get("env_args", "{}")))
                except json.JSONDecodeError as error:
                    return [f"data.env_args is invalid JSON: {error}"], details
                actual_task = str(env_args.get("env_name", ""))
                details["source_task"] = actual_task
                environment_kwargs = env_args.get("env_kwargs", {})
                details["robot"] = str(
                    environment_kwargs.get("robots", "unavailable in HDF5 metadata")
                )
                details["gripper"] = str(
                    environment_kwargs.get(
                        "gripper_types",
                        "unavailable in HDF5 metadata; strict simulator XML preflight required",
                    )
                )
                details["controller"] = json.dumps(
                    environment_kwargs.get(
                        "controller_configs",
                        "unavailable in HDF5 metadata",
                    ),
                    sort_keys=True,
                )
                if actual_task != expected_task:
                    errors.append(
                        f"dataset env_name is {actual_task!r}, expected {expected_task!r}"
                    )
                demos = sorted(
                    (key for key in data if key.startswith("demo_")),
                    key=lambda value: int(value.rsplit("_", 1)[-1]),
                )
                details["demonstration_ids"] = demos
                details["available_demonstrations"] = len(demos)
                if len(demos) < 2:
                    errors.append(
                        f"dataset needs at least two trajectories, found {len(demos)}"
                    )
                window_candidates = 0
                lengths: list[int] = []
                camera_key: str | None = None
                camera_shape: tuple[int, ...] | None = None
                for demo in demos:
                    group = data[demo]
                    if "actions" not in group or "states" not in group:
                        errors.append(f"{demo} lacks actions or states")
                        continue
                    if "rewards" not in group:
                        errors.append(
                            f"{demo} lacks the official task success/reward signal"
                        )
                    actions = group["actions"]
                    lengths.append(int(actions.shape[0]))
                    if actions.ndim != 2 or actions.shape[1] < 7:
                        errors.append(
                            f"{demo} action shape is {actions.shape}, expected [T,>=7]"
                        )
                        continue
                    if "obs" not in group:
                        errors.append(f"{demo} lacks obs group")
                        continue
                    image_keys = sorted(
                        key for key in group["obs"] if key.endswith("_image")
                    )
                    if not image_keys:
                        errors.append(f"{demo} has no camera image dataset")
                        continue
                    expected_image_key = (
                        camera_view
                        if camera_view.endswith("_image")
                        else f"{camera_view}_image"
                    )
                    if expected_image_key not in group["obs"]:
                        errors.append(
                            f"{demo} lacks configured camera {expected_image_key!r}; "
                            f"available={image_keys}"
                        )
                        continue
                    image = group["obs"][expected_image_key]
                    if image.ndim != 4 or image.shape[-1] != 3:
                        errors.append(
                            f"{demo}/{expected_image_key} must be [T,H,W,3], "
                            f"received {image.shape}"
                        )
                        continue
                    current_shape = tuple(int(value) for value in image.shape[1:])
                    if camera_shape is not None and current_shape != camera_shape:
                        errors.append(
                            f"{demo}/{expected_image_key} shape {current_shape} differs "
                            f"from {camera_shape}"
                        )
                        continue
                    camera_key = f"obs/{expected_image_key}"
                    camera_shape = current_shape
                    window_candidates += max(0, int(actions.shape[0]) - 16)
                details["window_candidates_lower_bound"] = window_candidates
                details["demonstration_lengths"] = lengths
                details["camera_key"] = camera_key
                details["camera_shape"] = camera_shape
                if window_candidates < 2000:
                    errors.append(
                        "dataset cannot provide 2000 four-frame training windows: "
                        f"lower_bound={window_candidates}"
                    )
        except OSError as error:
            errors.append(f"cannot open HDF5: {error}")
        return errors, details

    def resolve_task(self, *, strict: bool) -> ResolvedTask:
        if strict and self._strict_resolution is not None:
            return self._strict_resolution
        if strict:
            existing = self.existing_task_manifest()
            if existing is not None:
                self._strict_resolution = existing
                return existing
        task_key = str(self.cfg.benchmark.task_key)
        requested = str(self.cfg.benchmark.get("task_name", self.cfg.data.task_name))
        failures: dict[str, list[str]] = {}
        selected: str | None = None
        selected_path: Path | None = None
        details: dict[str, Any] = {}
        candidates = (
            ARTICULATED_TASK_PRIORITY
            if requested in {"auto_articulated", "articulated"}
            else (requested,)
        )
        for candidate in candidates:
            sources = self._candidate_sources(candidate)
            if not sources:
                failures[candidate] = ["no dataset path is configured"]
                continue
            source_failures: list[str] = []
            for dataset_format, path in sources:
                if dataset_format == LEROBOT_DATASET_FORMAT:
                    reasons, candidate_details = inspect_robocasa_lerobot(
                        path,
                        task_name=candidate,
                        camera_view=str(self.cfg.data.camera_view),
                    )
                else:
                    reasons, candidate_details = self._inspect_dataset(
                        path,
                        candidate,
                        str(self.cfg.data.camera_view),
                    )
                    candidate_details["dataset_format"] = "hdf5"
                    candidate_details["official_source_identifier"] = (
                        "facebook/jepa-wms"
                        if candidate == "PnPCounterTop"
                        else "robocasa/robocasa official dataset registry"
                    )
                if reasons:
                    source_failures.extend(
                        f"{dataset_format}:{path}: {reason}" for reason in reasons
                    )
                    continue
                if strict and requested in {"auto_articulated", "articulated"}:
                    try:
                        source = self._build_source(
                            path=path,
                            task_name=candidate,
                            dataset_format=dataset_format,
                            output_environment_info=True,
                        )
                        _, heldout = self.split_trajectory_ids(source)
                        if len(heldout) < int(self.cfg.evaluation.num_episodes):
                            raise RuntimeError(
                                "held-out RoboCasa trajectories are insufficient: "
                                f"available={len(heldout)}, "
                                f"required={self.cfg.evaluation.num_episodes}"
                            )
                        candidate_details["resolver_environment_check"] = (
                            self._deep_environment_preflight(
                                candidate, source, heldout
                            )
                        )
                    except Exception as error:
                        source_failures.append(
                            f"{dataset_format}:{path}: environment/state/action/"
                            "success preflight failed: "
                            f"{type(error).__name__}: {error}"
                        )
                        continue
                selected = candidate
                selected_path = path
                details = candidate_details
                break
            if selected is not None:
                if source_failures:
                    details["rejected_higher_priority_sources"] = source_failures
                break
            failures[candidate] = source_failures
        if selected is None or selected_path is None:
            message = (
                "No RoboCasa task candidate satisfies the fixed resolver priority: "
                + json.dumps(failures, sort_keys=True)
            )
            if strict:
                raise RuntimeError(message)
            selected = requested if requested not in {"auto_articulated", "articulated"} else ARTICULATED_TASK_PRIORITY[0]
            configured_sources = self._candidate_sources(selected)
            selected_path = (
                configured_sources[0][1]
                if configured_sources
                else resolve_path("missing-robocasa-dataset.hdf5")
            )
            details = {"available_demonstrations": 0}
        camera_key = str(details.get("camera_key") or self.cfg.data.camera_view)
        camera_shape = details.get("camera_shape")
        status = "resolved" if not failures or selected not in failures else "unresolved"
        if not strict and requested in {"auto_articulated", "articulated"} and status == "resolved":
            status = "candidate_requires_deep_preflight"
        available = int(details.get("available_demonstrations", 0))
        if available >= 2:
            if str(details.get("dataset_format")) == LEROBOT_DATASET_FORMAT:
                train_count = int(
                    np.floor(available * float(self.cfg.data.train_fraction))
                )
                train = np.arange(train_count, dtype=np.int64)
                evaluation = np.arange(train_count, available, dtype=np.int64)
            else:
                train, evaluation = self.split_trajectory_ids(range(available))
            demonstration_ids = list(details.get("demonstration_ids", []))
            train_ids = tuple(
                str(demonstration_ids[int(index)]) for index in train.tolist()
            )
            evaluation_ids = tuple(
                str(demonstration_ids[int(index)]) for index in evaluation.tolist()
            )
        else:
            train_ids = ()
            evaluation_ids = ()
        lengths = np.asarray(details.get("demonstration_lengths", []), dtype=np.int64)
        length_summary = (
            f"demonstration lengths min/median/max="
            f"{int(lengths.min())}/{float(np.median(lengths)):.1f}/{int(lengths.max())}"
            if lengths.size
            else "demonstration lengths unavailable"
        )
        result = ResolvedTask(
            task_key=task_key,
            benchmark="robocasa",
            suite=str(self.cfg.benchmark.get("suite", "single_stage")),
            task_id=str(selected),
            task_name=str(selected),
            language_instruction=None,
            bddl_path=None,
            bddl_sha256=None,
            problem_folder=None,
            initial_states_sha256=None,
            initial_states_count=0,
            dataset_path=str(selected_path),
            dataset_sha256=(
                sha256_dataset_path(selected_path) if strict else None
            ),
            available_demonstrations=available,
            selected_train_demonstrations=train_ids,
            selected_test_demonstrations=evaluation_ids,
            camera_key=camera_key,
            action_convention=(
                json.loads(
                    json.dumps(
                        {
                            **self.action_convention().as_dict(),
                            "dataset_raw_action": details.get(
                                "canonical_action_mapping"
                            ),
                            "dataset_raw_action_segments": details.get(
                                "raw_action_segments"
                            ),
                            "dataset_raw_action_indices": details.get(
                                "raw_action_indices"
                            ),
                        }
                    )
                )
                if str(details.get("dataset_format")) == LEROBOT_DATASET_FORMAT
                else self.action_convention().as_dict()
            ),
            environment_implementation="robocasa.utils.env_utils.create_env via JEPA-WM RoboCasaWrapper",
            upstream_commits=dict(UPSTREAM_COMMITS),
            frameskip=int(self.cfg.data.frameskip),
            max_episode_steps=int(self.cfg.evaluation.max_episode_steps),
            episode_cap_basis=(
                f"{self.cfg.evaluation.episode_cap_basis}; {length_summary}; "
                f"control_frequency_hz={self.action_convention().control_frequency_hz}"
            ),
            status=status,
            candidate_failures=failures or None,
            camera_height=(int(camera_shape[0]) if camera_shape is not None else None),
            camera_width=(int(camera_shape[1]) if camera_shape is not None else None),
            camera_channel_order=("RGB" if camera_shape is not None else None),
            camera_vertical_flip=False if camera_shape is not None else None,
            action_transform=self.action_transform().as_dict(),
            dataset_format=str(details.get("dataset_format", "hdf5")),
            dataset_source_identifier=details.get(
                "official_source_identifier"
            ),
            dataset_revision=details.get("official_source_revision"),
            dataset_file_count=(
                len(details.get("required_relative_files", ()))
                if details.get("required_relative_files")
                else 1
            ),
            robot=str(details.get("robot", "unavailable")),
            gripper=str(details.get("gripper", "unavailable")),
            controller_contract=str(
                details.get(
                    "controller",
                    "RoboCasa OSC_POSE through pinned JEPA-WM action wrapper",
                )
            ),
        )
        if strict:
            self._strict_resolution = result
        return result

    def preflight(self, *, deep: bool) -> dict[str, Any]:
        task = self.resolve_task(strict=True)
        source = self.build_source_dataset(output_environment_info=deep)
        train, evaluation = self.split_trajectory_ids(source)
        train_ids = tuple(
            str(source.trajectories[int(index)].get("demo_key", index))
            for index in train.tolist()
        )
        evaluation_ids = tuple(
            str(source.trajectories[int(index)].get("demo_key", index))
            for index in evaluation.tolist()
        )
        if train_ids != tuple(task.selected_train_demonstrations) or evaluation_ids != tuple(task.selected_test_demonstrations):
            raise RuntimeError(
                "RoboCasa deterministic split no longer matches the immutable task manifest"
            )
        candidates = self.enumerate_window_candidates(
            source,
            int(self.cfg.data.num_frames),
            int(self.cfg.data.frameskip),
        )
        selected = self.select_windows(
            candidates,
            train,
            int(self.cfg.data.num_train_windows),
            int(self.cfg.data.window_seed),
        )
        errors: list[str] = []
        if len(selected) != int(self.cfg.data.num_train_windows):
            errors.append(
                f"selected {len(selected)} training windows, expected {self.cfg.data.num_train_windows}"
            )
        required_evaluation = (
            int(self.cfg.evaluation.num_episodes)
            if task.task_key == "robocasa_articulated"
            else 1
        )
        if len(evaluation) < required_evaluation:
            errors.append(
                "held-out trajectory partition is too small: "
                f"available={len(evaluation)}, required={required_evaluation}"
            )
        if errors:
            raise RuntimeError("RoboCasa preflight failed:\n- " + "\n- ".join(errors))
        report = {
            "benchmark": self.name,
            "task": task.as_dict(),
            "train_trajectories": train.tolist(),
            "evaluation_trajectories": evaluation.tolist(),
            "selected_train_windows": len(selected),
            "deep_environment_check": deep,
        }
        if deep:
            report["environment_restore"] = self._deep_environment_preflight(
                task.task_name, source, evaluation
            )
        return report

    def _deep_environment_preflight(
        self,
        task_name: str,
        source: Any,
        evaluation: np.ndarray,
    ) -> dict[str, Any]:
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.common.parser import parse_cfg

        official = OmegaConf.load(resolve_path(str(self.cfg.model.official_planning_config)))
        official = OmegaConf.create(
            OmegaConf.to_container(official, resolve=False)
        )
        official.folder = str(resolve_path("logs/cross_benchmark_v1/preflight"))
        OmegaConf.resolve(official)
        official.work_dir = resolve_path("logs/cross_benchmark_v1/preflight")
        official.task_specification.task = f"robocasa-{task_name}"
        subtask = self.cfg.planning.get("subtask")
        official.task_specification.env.subtask = subtask
        official.task_specification.env.sample_subtask_slice = bool(subtask)
        configured_gripper = self.cfg.benchmark.get("gripper_types")
        if configured_gripper is not None:
            official.task_specification.env.gripper_types = str(configured_gripper)
        official.model_kwargs.data.custom.filter_tasks = [task_name]
        official = parse_cfg(official)
        official.rank = 0
        official.world_size = 1
        official.device = "cpu"
        official.num_active_gpus = 1
        official.active_ranks = [0]
        official.local_seed = int(self.cfg.evaluation.eval_seed)
        official.frameskip = int(self.cfg.data.frameskip)
        generator = torch.Generator(device="cpu").manual_seed(
            int(self.cfg.evaluation.eval_seed)
        )
        sample: Any | None = None
        trajectory = -1
        for _ in range(100):
            trajectory = int(
                evaluation[
                    int(
                        torch.randint(
                            0, len(evaluation), (1,), generator=generator
                        ).item()
                    )
                ]
            )
            try:
                sample = source.__getitem__(
                    trajectory, subtask=str(subtask) if subtask else None
                )
            except ValueError:
                continue
            break
        if sample is None:
            raise RuntimeError(
                f"RoboCasa {task_name} has no usable held-out trajectory after 100 attempts"
            )
        observation, actions, states, rewards, environment_info = sample
        if states is None or environment_info is None:
            raise RuntimeError(
                f"RoboCasa {task_name} cannot restore simulator state/environment XML"
            )
        environment = make_env(official)
        try:
            environment.update_env(environment_info)
            restored_observation, info = environment.prepare(
                int(self.cfg.evaluation.eval_seed),
                np.asarray(states[0]),
                env_info=environment_info,
            )
            if isinstance(restored_observation, (torch.Tensor, np.ndarray)):
                visual = restored_observation
            elif isinstance(restored_observation, Mapping):
                visual = restored_observation.get("pixels")
                if visual is None:
                    visual = restored_observation.get("visual")
            else:
                raise RuntimeError(
                    "RoboCasa restored observation has an unsupported type: "
                    f"type={type(restored_observation).__name__}"
                )
            if visual is None:
                raise RuntimeError(
                    f"RoboCasa restored observation has no RGB value: keys={sorted(restored_observation)}"
                )
            if int(environment.action_dim) != 7 or int(actions.shape[-1]) != 7:
                raise RuntimeError(
                    f"RoboCasa action dimension mismatch: env={environment.action_dim}, dataset={actions.shape}"
                )
            if "success" not in info:
                raise RuntimeError(
                    f"RoboCasa official success signal is absent: keys={sorted(info)}"
                )
            return {
                "trajectory_index": trajectory,
                "dataset_visual_shape": list(observation["visual"].shape),
                "restored_visual_shape": list(np.asarray(visual).shape),
                "action_dim": int(environment.action_dim),
                "success_signal": "info.success",
                "reward_shape": None if rewards is None else list(rewards.shape),
            }
        finally:
            environment.close()

    def build_source_dataset(self, *, output_environment_info: bool) -> Any:
        task = self.resolve_task(strict=True)
        dataset_path = resolve_path(task.dataset_path)
        dataset_format = LEROBOT_DATASET_FORMAT if dataset_path.is_dir() else "hdf5"
        return self._build_source(
            path=dataset_path,
            task_name=task.task_name,
            dataset_format=dataset_format,
            output_environment_info=output_environment_info,
        )

    def _build_source(
        self,
        *,
        path: Path,
        task_name: str,
        dataset_format: str,
        output_environment_info: bool,
    ) -> Any:
        if dataset_format == LEROBOT_DATASET_FORMAT:
            return RoboCasaLeRobotDataset(
                path,
                task_name=task_name,
                camera_view=str(self.cfg.data.camera_view),
                action_transform=self.action_transform(),
                output_environment_info=output_environment_info,
            )
        if dataset_format != "hdf5":
            raise ValueError(
                f"Unsupported RoboCasa dataset format {dataset_format!r}: {path}"
            )
        return build_robocasa_dataset(
            jepa_wms_root=resolve_path(self.cfg.model.third_party_root) / "jepa-wms",
            dataset_root=path.parent.parent,
            hdf5_path=path,
            task_name=task_name,
            camera_view=str(self.cfg.data.camera_view),
            output_environment_info=output_environment_info,
            transform=None,
        )

    def split_trajectory_ids(
        self,
        source_dataset: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(source_dataset, RoboCasaLeRobotDataset):
            episode_count = len(source_dataset)
            train_count = int(
                np.floor(episode_count * float(self.cfg.data.train_fraction))
            )
            if train_count <= 0 or train_count >= episode_count:
                raise ValueError(
                    "RoboCasa365 ordered split must leave non-empty partitions: "
                    f"episodes={episode_count}, "
                    f"train_fraction={self.cfg.data.train_fraction}"
                )
            return (
                np.arange(train_count, dtype=np.int64),
                np.arange(train_count, episode_count, dtype=np.int64),
            )
        return super().split_trajectory_ids(source_dataset)

    def enumerate_window_candidates(
        self,
        source_dataset: Any,
        num_frames: int,
        frameskip: int,
    ) -> list[tuple[int, int]]:
        segment_code: int | None = None
        suite = self.cfg.get("suite", {})
        if str(suite.get("name", "")) == "cross_backend_adapter_v1":
            subtask = str(self.cfg.planning.get("subtask", ""))
            segment_codes = {"reach": 0, "place": 2}
            if subtask not in segment_codes:
                raise ValueError(
                    "cross_backend_adapter_v1 RoboCasa windows require planning.subtask "
                    f"to be reach or place, received {subtask!r}"
                )
            segment_code = segment_codes[subtask]
        return RoboCasaWindowDataset.all_candidates(
            source_dataset,
            num_frames,
            frameskip,
            segment_code=segment_code,
        )

    def select_windows(
        self,
        candidates: list[tuple[int, int]],
        allowed_trajectories: np.ndarray,
        max_windows: int,
        seed: int,
    ) -> list[tuple[int, int]]:
        selected = super().select_windows(
            candidates,
            allowed_trajectories,
            max_windows,
            seed,
        )
        suite = self.cfg.get("suite", {})
        if (
            str(suite.get("name", "")) != "cross_backend_adapter_v1"
            or len(selected) == max_windows
        ):
            return selected
        if not selected:
            raise RuntimeError(
                "RoboCasa subtask filtering produced no windows in the requested "
                f"trajectory partition: subtask={self.cfg.planning.get('subtask')!r}"
            )
        # The official PnPCounterTop release contains fewer unique per-subtask
        # windows than the fixed paper budget. Cycle the deterministic balanced
        # order so the configured 2000/200 window counts remain exact.
        return [selected[index % len(selected)] for index in range(max_windows)]

    def make_window_dataset(
        self,
        source_dataset: Any,
        selections: list[tuple[int, int]],
        *,
        num_frames: int,
        frameskip: int,
        appearance_seed: int,
        appearance_severity: float,
        appearance_severity_range: tuple[float, float] | None = None,
    ) -> RoboCasaWindowDataset:
        return RoboCasaWindowDataset(
            source_dataset,
            selections,
            num_frames=num_frames,
            frameskip=frameskip,
            appearance_seed=appearance_seed,
            appearance_severity=appearance_severity,
            appearance_severity_range=appearance_severity_range,
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
        configured_goal_span = self.cfg.evaluation.get("goal_span_steps")
        goal_span = (
            int(configured_goal_span)
            if configured_goal_span is not None
            else 0
        )
        subtask = self.cfg.planning.get("subtask")
        official_segment_codes = {"reach": 0, "place": 2}
        if str(subtask) in official_segment_codes and task.task_key in {
            "robocasa_reach",
            "robocasa_place",
        }:
            subtask_name = str(subtask)
            segment_code = official_segment_codes[subtask_name]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            instances: list[EvaluationInstance] = []
            for position in range(count):
                selected: tuple[int, Any] | None = None
                errors: list[str] = []
                for _ in range(100):
                    subset_index = int(
                        torch.randint(
                            0, len(evaluation), (1,), generator=generator
                        ).item()
                    )
                    trajectory = int(evaluation[subset_index])
                    try:
                        result = source.__getitem__(
                            trajectory, subtask=subtask_name
                        )
                    except ValueError as error:
                        errors.append(str(error))
                        continue
                    selected = (trajectory, result)
                    break
                if selected is None:
                    raise RuntimeError(
                        f"RoboCasa {subtask_name} compatibility manifest could not reproduce "
                        f"the pinned evaluator sampling after 100 attempts: {errors[-3:]}"
                    )
                trajectory, result = selected
                observation, _, states, _, trajectory_info = result
                if states is None:
                    raise RuntimeError(
                        f"RoboCasa {subtask_name} trajectory {trajectory} returned no simulator state"
                    )
                trajectory_path = Path(str(trajectory_info["file_path"]))
                with h5py.File(trajectory_path, "r") as handle:
                    group = handle["data"][str(trajectory_info["demo_key"])]
                    segments = np.asarray(
                        group["meta_data_info/current_task_segment"]
                    )
                frame_indices = np.flatnonzero(segments == segment_code)
                if frame_indices.size == 0 or not np.array_equal(
                    frame_indices,
                    np.arange(frame_indices[0], frame_indices[-1] + 1),
                ):
                    raise RuntimeError(
                        f"RoboCasa {subtask_name} compatibility requires one contiguous "
                        f"segment code {segment_code}: trajectory={trajectory}, "
                        f"indices={frame_indices.tolist()}"
                    )
                if int(observation["visual"].shape[0]) != int(frame_indices.size):
                    raise RuntimeError(
                        f"RoboCasa {subtask_name} filtered length mismatch for trajectory {trajectory}"
                    )
                source_id = str(trajectory_info.get("demo_key", trajectory))
                identity = {
                    "task": task.task_key,
                    "source": source_id,
                    "start": int(frame_indices[0]),
                    "end": int(frame_indices[-1]),
                    "evaluation_position": position,
                }
                instances.append(
                    EvaluationInstance(
                        instance_id=canonical_sha256(identity)[:24],
                        source_trajectory_id=source_id,
                        source_trajectory_index=trajectory,
                        segment_start=int(frame_indices[0]),
                        segment_end=int(frame_indices[-1]),
                        initialization_fingerprint=array_sha256(states[0].numpy()),
                        goal_fingerprint=array_sha256(
                            observation["visual"][-1].numpy()
                        ),
                        environment_seed=int(
                            (seed * seed + position * seed) % (2**32 - 2)
                        ),
                        cem_seed=seed,
                        appearance_seed=appearance_seed,
                    )
                )
            return self.finalize_evaluation_manifest(
                task,
                instances,
                {
                    "source_trajectory_count": len(
                    {value.source_trajectory_id for value in instances}
                    ),
                    "segment_identity_note": (
                    "Sampling reproduces the pinned evaluator's seeded held-out "
                    "trajectory stream; repeated trajectories are clustered."
                    ),
                    "cem_seed_mode": "continuous_generator_stream",
                    "legacy_place_reuse_compatible": subtask_name == "place",
                    "official_subtask": subtask_name,
                    "official_subtask_segment_code": segment_code,
                },
            )
        primary_candidates: list[tuple[int, int, int]] = []
        for trajectory in evaluation.tolist():
            length = int(source.get_seq_length(int(trajectory)))
            end = length - 1
            start = max(0, end - goal_span)
            try:
                observation, _, _, _, _ = source.get_frames(
                    int(trajectory),
                    range(start, end + 1),
                    subtask=str(subtask) if subtask else None,
                )
            except ValueError:
                continue
            if int(observation["visual"].shape[0]) == end - start + 1:
                primary_candidates.append((int(trajectory), start, end))
        candidates = primary_candidates
        uses_distinct_source_trajectories = len(primary_candidates) >= count
        if not uses_distinct_source_trajectories:
            candidates = []
            # A small held-out partition may contribute distinct fixed-span
            # segments. Their shared source IDs remain explicit for clustering.
            for trajectory in evaluation.tolist():
                length = int(source.get_seq_length(int(trajectory)))
                for start in range(max(0, length - goal_span)):
                    end = start + goal_span
                    try:
                        observation, _, _, _, _ = source.get_frames(
                            int(trajectory),
                            range(start, end + 1),
                            subtask=str(subtask) if subtask else None,
                        )
                    except ValueError:
                        continue
                    if int(observation["visual"].shape[0]) != goal_span + 1:
                        continue
                    candidates.append((int(trajectory), start, end))
        generator = np.random.default_rng(seed)
        if len(candidates) < count:
            raise RuntimeError(
                f"RoboCasa held-out partition provides {len(candidates)} segments, expected {count}"
            )
        order = generator.permutation(len(candidates))[:count]
        instances: list[EvaluationInstance] = []
        for position, candidate_index in enumerate(order.tolist()):
            trajectory, start, end = candidates[int(candidate_index)]
            observation, _, states, _, trajectory_info = source.get_frames(
                trajectory,
                range(start, end + 1),
                subtask=str(subtask) if subtask else None,
            )
            if states is None:
                raise RuntimeError(
                    f"RoboCasa evaluation trajectory {trajectory} returned no simulator state"
                )
            visual = observation["visual"]
            source_id = str(trajectory_info.get("demo_key", trajectory))
            identity_payload = {
                "task": task.task_key,
                "source": source_id,
                "start": start,
                "end": end,
            }
            instances.append(
                EvaluationInstance(
                    instance_id=canonical_sha256(identity_payload)[:24],
                    source_trajectory_id=source_id,
                    source_trajectory_index=trajectory,
                    segment_start=start,
                    segment_end=end,
                    initialization_fingerprint=array_sha256(states[0].numpy()),
                    goal_fingerprint=array_sha256(visual[-1].numpy()),
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
            {
                "source_trajectory_count": len(
                    set(value.source_trajectory_id for value in instances)
                ),
                "distinct_source_trajectories": uses_distinct_source_trajectories,
                "segment_identity_note": (
                    "One final-goal segment per held-out trajectory is preferred; when "
                    "fewer than the requested count exist, segments sharing "
                    "source_trajectory_id are retained explicitly and clustered for "
                    "trajectory bootstrap"
                ),
                "cem_seed_mode": "per_instance",
            },
        )

    def run_planning(
        self,
        *,
        backend: Any,
        method: Any,
        output_directory: str | Path,
    ) -> Any:
        from wm_adapter.planning.jepa_wm_planner import run_robocasa_planning

        return run_robocasa_planning(
            experiment_config=self.cfg,
            backend=backend,
            method=method,
            output_directory=output_directory,
        )
