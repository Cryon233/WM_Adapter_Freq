from __future__ import annotations

import gzip
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wm_adapter.benchmarks.base import ActionTransform
from wm_adapter.utils.reproducibility import resolve_path


LEROBOT_SOURCE_ID = "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos"
LEROBOT_SOURCE_REVISION = "bf736c0cc8f9ea8740c812901eec02bce09517f1"
LEROBOT_SOURCE_FILE = "pretrain/atomic/OpenDrawer/20250819/lerobot.tar"
LEROBOT_DATASET_FORMAT = "robocasa365_lerobot_v2.1_task"

_REQUIRED_METADATA = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/stats.json",
    "meta/modality.json",
    "meta/embodiment.json",
    "extras/dataset_meta.json",
)
_REQUIRED_VIDEO_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)
_ACTION_SEGMENTS = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, received {type(value).__name__}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected a JSON object at {path}:{line_number}, "
                f"received {type(value).__name__}"
            )
        records.append(value)
    return records


def _load_metadata(root: Path) -> dict[str, Any]:
    missing = [str(root / relative) for relative in _REQUIRED_METADATA if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Official task-level RoboCasa365 metadata is incomplete; "
            f"missing={missing}, root={root}"
        )
    episodes = _load_jsonl(root / "meta" / "episodes.jsonl")
    episodes.sort(key=lambda value: int(value["episode_index"]))
    return {
        "info": _load_json(root / "meta" / "info.json"),
        "tasks": _load_jsonl(root / "meta" / "tasks.jsonl"),
        "episodes": episodes,
        "modality": _load_json(root / "meta" / "modality.json"),
        "embodiment": _load_json(root / "meta" / "embodiment.json"),
        "dataset_meta": _load_json(root / "extras" / "dataset_meta.json"),
    }


def _episode_path(root: Path, pattern: str, episode_id: int, chunks_size: int, **values: Any) -> Path:
    return root / pattern.format(
        episode_chunk=episode_id // chunks_size,
        episode_index=episode_id,
        **values,
    )


def _manipulator_action_indices(modality: dict[str, Any]) -> tuple[int, ...]:
    action = modality.get("action")
    if not isinstance(action, dict):
        raise ValueError("meta/modality.json has no action mapping")
    for name, (expected_start, expected_end) in _ACTION_SEGMENTS.items():
        segment = action.get(name)
        if not isinstance(segment, dict):
            raise ValueError(f"meta/modality.json action mapping lacks {name!r}")
        actual = (
            str(segment.get("original_key")),
            int(segment.get("start", -1)),
            int(segment.get("end", -1)),
        )
        expected = ("action", expected_start, expected_end)
        if actual != expected:
            raise ValueError(
                f"Unexpected RoboCasa365 action segment {name!r}: "
                f"expected={expected}, actual={actual}"
            )
    return tuple(range(5, 12))


def _controller_contract(metadata: dict[str, Any]) -> tuple[dict[str, Any], tuple[float, ...]]:
    dataset_meta = metadata["dataset_meta"]
    env_info = dataset_meta.get("env_info")
    if not isinstance(env_info, dict):
        raise ValueError("extras/dataset_meta.json has no env_info mapping")
    controller = env_info.get("controller_configs")
    if not isinstance(controller, dict):
        raise ValueError("extras/dataset_meta.json has no controller_configs mapping")
    body_parts = controller.get("body_parts")
    if not isinstance(body_parts, dict) or not isinstance(body_parts.get("right"), dict):
        raise ValueError("RoboCasa controller metadata has no right-arm controller")
    right = body_parts["right"]
    required = {
        "type": "OSC_POSE",
        "input_type": "delta",
        "input_ref_frame": "base",
    }
    mismatches = {
        key: {"expected": expected, "actual": right.get(key)}
        for key, expected in required.items()
        if right.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Unsupported RoboCasa right-arm controller contract: {mismatches}")
    input_min = np.broadcast_to(np.asarray(right.get("input_min"), dtype=np.float64), (6,))
    input_max = np.broadcast_to(np.asarray(right.get("input_max"), dtype=np.float64), (6,))
    output_min = np.asarray(right.get("output_min"), dtype=np.float64)
    output_max = np.asarray(right.get("output_max"), dtype=np.float64)
    if (
        not np.array_equal(input_min, np.full(6, -1.0))
        or not np.array_equal(input_max, np.full(6, 1.0))
        or output_min.shape != (6,)
        or output_max.shape != (6,)
        or not np.allclose(output_min, -output_max, atol=0.0, rtol=0.0)
    ):
        raise ValueError(
            "Unsupported RoboCasa OSC_POSE scaling: "
            f"input_min={input_min.tolist()}, input_max={input_max.tolist()}, "
            f"output_min={output_min.tolist()}, output_max={output_max.tolist()}"
        )
    expected_scale = np.asarray((0.05, 0.05, 0.05, 0.5, 0.5, 0.5))
    if not np.allclose(output_max, expected_scale, atol=0.0, rtol=0.0):
        raise ValueError(
            "RoboCasa365 controller output scale differs from JEPA-WM canonical action: "
            f"expected={expected_scale.tolist()}, actual={output_max.tolist()}"
        )
    return controller, tuple(float(value) for value in output_max.tolist())


def _gripper_from_xml(xml_text: str, path: Path) -> str:
    root = ET.fromstring(xml_text)
    names = {
        str(element.get("name"))
        for element in root.iter()
        if element.get("name") is not None
    }
    if {
        "gripper0_right_finger1_visual",
        "gripper0_right_finger2_visual",
    }.issubset(names):
        return "PandaGripper"
    if any("left_outer_finger_visual" in name for name in names) and any(
        "right_outer_finger_visual" in name for name in names
    ):
        return "Robotiq85Gripper"
    raise ValueError(
        "Cannot identify the RoboCasa gripper from model XML names: "
        f"{path}"
    )


def inspect_robocasa_lerobot(
    root: str | Path,
    *,
    task_name: str,
    camera_view: str,
) -> tuple[list[str], dict[str, Any]]:
    resolved = resolve_path(root)
    details: dict[str, Any] = {
        "dataset_path": str(resolved),
        "dataset_format": LEROBOT_DATASET_FORMAT,
        "official_source_identifier": LEROBOT_SOURCE_ID,
        "official_source_revision": LEROBOT_SOURCE_REVISION,
        "official_source_file": LEROBOT_SOURCE_FILE,
        "source_split": "pretrain",
        "source_type": "human",
    }
    if not resolved.is_dir():
        return [f"LeRobot dataset directory does not exist: {resolved}"], details
    try:
        metadata = _load_metadata(resolved)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        return [str(error)], details

    errors: list[str] = []
    info = metadata["info"]
    dataset_meta = metadata["dataset_meta"]
    episodes = metadata["episodes"]
    if str(info.get("codebase_version")) != "v2.1":
        errors.append(
            "unsupported task-level LeRobot schema: "
            f"codebase_version={info.get('codebase_version')!r}, expected='v2.1'"
        )
    actual_task = str(dataset_meta.get("env", ""))
    env_args = dataset_meta.get("env_args", {})
    if actual_task != task_name or str(env_args.get("env_name", "")) != task_name:
        errors.append(
            "RoboCasa365 task identity mismatch: "
            f"requested={task_name!r}, dataset_meta.env={actual_task!r}, "
            f"env_args.env_name={env_args.get('env_name')!r}"
        )
    episode_ids = [int(value["episode_index"]) for value in episodes]
    if len(episode_ids) != len(set(episode_ids)) or episode_ids != sorted(episode_ids):
        errors.append(f"episode_index values must be unique and sorted: {episode_ids}")
    if int(info.get("total_episodes", -1)) != len(episodes):
        errors.append(
            "meta/info.json total_episodes disagrees with meta/episodes.jsonl: "
            f"info={info.get('total_episodes')}, records={len(episodes)}"
        )
    if not episodes:
        errors.append("RoboCasa365 task dataset contains no episodes")

    feature_key = f"observation.images.{camera_view}"
    feature = info.get("features", {}).get(feature_key)
    shape: tuple[int, ...] = ()
    if not isinstance(feature, dict):
        errors.append(
            f"LeRobot camera feature {feature_key!r} is absent; "
            f"available={sorted(info.get('features', {}))}"
        )
    else:
        shape = tuple(int(value) for value in feature.get("shape", ()))
        if len(shape) != 3 or shape[-1] != 3:
            errors.append(
                f"LeRobot camera {feature_key!r} must be [H,W,3], received {shape}"
            )
    action_shape = tuple(
        int(value)
        for value in info.get("features", {}).get("action", {}).get("shape", ())
    )
    state_shape = tuple(
        int(value)
        for value in info.get("features", {}).get("observation.state", {}).get("shape", ())
    )
    if action_shape != (12,):
        errors.append(f"RoboCasa365 raw action must be 12-D, received {action_shape}")
    if state_shape != (16,):
        errors.append(f"RoboCasa365 policy observation.state must be 16-D, received {state_shape}")

    try:
        manipulator_indices = _manipulator_action_indices(metadata["modality"])
        controller, output_scale = _controller_contract(metadata)
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        manipulator_indices = ()
        controller = {}
        output_scale = ()

    embodiment = metadata["embodiment"]
    robot = str(embodiment.get("robot_name", info.get("robot_type", "")))
    if robot != "PandaOmron" or str(info.get("robot_type")) != robot:
        errors.append(
            "RoboCasa365 robot identity is not the expected PandaOmron: "
            f"embodiment={robot!r}, info={info.get('robot_type')!r}"
        )
    fps = float(info.get("fps", 0.0))
    frequencies = (
        float(embodiment.get("record_frequency", 0.0)),
        float(embodiment.get("body_controller_frequency", 0.0)),
        float(embodiment.get("hand_controller_frequency", 0.0)),
    )
    if fps <= 0.0 or any(not np.isclose(value, fps) for value in frequencies):
        errors.append(
            f"RoboCasa365 frequency contract is inconsistent: fps={fps}, "
            f"embodiment_frequencies={frequencies}"
        )

    chunks_size = int(info.get("chunks_size", 0))
    data_pattern = str(info.get("data_path", ""))
    video_pattern = str(info.get("video_path", ""))
    if chunks_size <= 0 or not data_pattern or not video_pattern:
        errors.append(
            "meta/info.json lacks a valid chunks_size/data_path/video_path contract"
        )
    required_files: set[str] = set(_REQUIRED_METADATA)
    grippers: set[str] = set()
    total_windows = 0
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        errors.append(f"Reading RoboCasa365 parquet metadata requires pyarrow: {error}")
        pq = None
    for record in episodes:
        episode_id = int(record["episode_index"])
        expected_length = int(record["length"])
        total_windows += max(0, expected_length - 16)
        data_path = _episode_path(
            resolved, data_pattern, episode_id, chunks_size
        )
        ep_root = resolved / "extras" / f"episode_{episode_id:06d}"
        ep_meta_path = ep_root / "ep_meta.json"
        xml_path = ep_root / "model.xml.gz"
        states_path = ep_root / "states.npz"
        episode_paths = [data_path, ep_meta_path, xml_path, states_path]
        for video_key in _REQUIRED_VIDEO_KEYS:
            episode_paths.append(
                _episode_path(
                    resolved,
                    video_pattern,
                    episode_id,
                    chunks_size,
                    video_key=video_key,
                )
            )
        required_files.update(str(path.relative_to(resolved)) for path in episode_paths)
        missing = [str(path) for path in episode_paths if not path.is_file()]
        if missing:
            errors.append(f"episode {episode_id} is incomplete; missing={missing}")
            continue
        try:
            ep_meta = _load_json(ep_meta_path)
            if not isinstance(ep_meta.get("cam_configs"), dict):
                raise ValueError("ep_meta.json has no cam_configs mapping")
            with gzip.open(xml_path, "rt", encoding="utf-8") as handle:
                xml_text = handle.read()
            grippers.add(_gripper_from_xml(xml_text, xml_path))
            with np.load(states_path, allow_pickle=False) as state_archive:
                if "states" not in state_archive.files:
                    raise ValueError(f"states.npz has keys={state_archive.files}, expected 'states'")
                simulator_states = np.asarray(state_archive["states"])
            if (
                simulator_states.ndim != 2
                or simulator_states.shape[0] != expected_length
                or not np.issubdtype(simulator_states.dtype, np.floating)
                or not np.isfinite(simulator_states).all()
            ):
                raise ValueError(
                    "invalid raw simulator state array: "
                    f"shape={simulator_states.shape}, dtype={simulator_states.dtype}, "
                    f"expected_frames={expected_length}"
                )
            if pq is not None:
                parquet = pq.ParquetFile(data_path)
                columns = set(parquet.schema_arrow.names)
                required_columns = {
                    "episode_index",
                    "frame_index",
                    "action",
                    "observation.state",
                    "next.reward",
                }
                if parquet.metadata.num_rows != expected_length or not required_columns.issubset(columns):
                    raise ValueError(
                        "invalid episode parquet contract: "
                        f"rows={parquet.metadata.num_rows}, expected={expected_length}, "
                        f"missing_columns={sorted(required_columns.difference(columns))}"
                    )
        except (OSError, ValueError, ET.ParseError, json.JSONDecodeError) as error:
            errors.append(f"episode {episode_id} metadata is invalid: {error}")

    if total_windows < 2000:
        errors.append(
            "LeRobot dataset cannot provide 2000 four-frame windows: "
            f"lower_bound={total_windows}"
        )
    if grippers != {"PandaGripper"}:
        errors.append(
            "RoboCasa365 task episodes do not share the expected PandaGripper XML "
            f"contract: detected={sorted(grippers)}"
        )
    details.update(
        {
            "source_task": actual_task,
            "demonstration_ids": [str(value) for value in episode_ids],
            "available_demonstrations": len(episode_ids),
            "demonstration_lengths": [int(value["length"]) for value in episodes],
            "camera_key": feature_key,
            "camera_shape": shape,
            "camera_codec": feature.get("video_info", {}).get("video.codec")
            if isinstance(feature, dict)
            else None,
            "camera_keys": list(_REQUIRED_VIDEO_KEYS),
            "action_shape": action_shape,
            "policy_state_shape": state_shape,
            "raw_action_indices": list(manipulator_indices),
            "raw_action_segments": dict(_ACTION_SEGMENTS),
            "canonical_action_mapping": (
                "raw normalized action[5:8] translation + action[8:11] axis-angle "
                "+ action[11] gripper_close -> verified RoboCasa OSC_POSE input -> "
                "JEPA-WM physical delta canonical action"
            ),
            "controller": json.dumps(controller, sort_keys=True, separators=(",", ":")),
            "controller_output_scale": list(output_scale),
            "window_candidates_lower_bound": total_windows,
            "required_relative_files": sorted(required_files),
            "environment_restore_capable": not any(
                "is incomplete" in value or "metadata is invalid" in value
                for value in errors
            ),
            "robot": robot,
            "mobile_base": "PandaOmron mobile base; base_motion action[0:4] is excluded",
            "gripper": next(iter(grippers)) if len(grippers) == 1 else "unresolved",
            "fps": fps,
            "dataset_version": str(dataset_meta.get("date", "20250819")),
        }
    )
    return errors, details


class RoboCasaLeRobotDataset:
    """Official task-level RoboCasa365 episodes with exact replay metadata."""

    def __init__(
        self,
        root: str | Path,
        *,
        task_name: str,
        camera_view: str,
        action_transform: ActionTransform,
        output_environment_info: bool,
    ) -> None:
        self.root = resolve_path(root)
        self.task_name = task_name
        self.camera_view = camera_view
        self.action_transform = action_transform
        self.output_environment_info = output_environment_info
        errors, details = inspect_robocasa_lerobot(
            self.root,
            task_name=task_name,
            camera_view=camera_view,
        )
        if errors:
            raise RuntimeError("Invalid RoboCasa task-level LeRobot dataset:\n- " + "\n- ".join(errors))
        if output_environment_info and not details["environment_restore_capable"]:
            raise RuntimeError(
                "RoboCasa task-level data lacks complete per-episode model XML or raw "
                f"simulator state: dataset={self.root}"
            )
        metadata = _load_metadata(self.root)
        self.info = metadata["info"]
        self.details = details
        self.modality = metadata["modality"]
        self.action_indices = np.asarray(
            _manipulator_action_indices(self.modality), dtype=np.int64
        )
        self.trajectories: list[dict[str, Any]] = []
        self.seq_lengths: list[int] = []
        chunks_size = int(self.info["chunks_size"])
        data_pattern = str(self.info["data_path"])
        video_pattern = str(self.info["video_path"])
        feature_key = f"observation.images.{camera_view}"
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError(
                "Reading RoboCasa task-level parquet data requires pandas and pyarrow"
            ) from error
        for record in metadata["episodes"]:
            episode_id = int(record["episode_index"])
            expected_length = int(record["length"])
            parquet_path = _episode_path(
                self.root, data_pattern, episode_id, chunks_size
            )
            frames = pd.read_parquet(
                parquet_path,
                columns=[
                    "episode_index",
                    "frame_index",
                    "action",
                    "observation.state",
                    "next.reward",
                ],
            ).sort_values("frame_index")
            if len(frames) != expected_length:
                raise RuntimeError(
                    f"LeRobot episode {episode_id} has {len(frames)} parquet frames, "
                    f"expected {expected_length}"
                )
            if set(int(value) for value in frames["episode_index"].unique()) != {episode_id}:
                raise RuntimeError(
                    f"LeRobot parquet episode identity mismatch: path={parquet_path}"
                )
            raw_actions = np.stack(frames["action"].to_numpy()).astype(np.float64)
            normalized_arm_actions = raw_actions[:, self.action_indices]
            canonical_actions = self.action_transform.environment_to_canonical_action(
                normalized_arm_actions
            )
            policy_states = np.stack(frames["observation.state"].to_numpy()).astype(
                np.float32
            )
            rewards = np.asarray(frames["next.reward"], dtype=np.float32)
            video_path = _episode_path(
                self.root,
                video_pattern,
                episode_id,
                chunks_size,
                video_key=feature_key,
            )
            extras = self.root / "extras" / f"episode_{episode_id:06d}"
            self.trajectories.append(
                {
                    "demo_key": str(episode_id),
                    "episode_index": episode_id,
                    "task_name": task_name,
                    "language_instruction": list(record.get("tasks", [])),
                    "actions": canonical_actions,
                    "raw_actions": raw_actions,
                    "policy_states": policy_states,
                    "rewards": rewards,
                    "video_path": str(video_path),
                    "state_path": str(extras / "states.npz"),
                    "model_xml_path": str(extras / "model.xml.gz"),
                    "ep_meta_path": str(extras / "ep_meta.json"),
                    "file_path": str(self.root),
                    "dataset_format": LEROBOT_DATASET_FORMAT,
                }
            )
            self.seq_lengths.append(expected_length)

    def __len__(self) -> int:
        return len(self.trajectories)

    def get_seq_length(self, index: int) -> int:
        return self.seq_lengths[index]

    def _decode_frames(self, trajectory: dict[str, Any], frames: list[int]) -> Tensor:
        if not frames:
            raise ValueError("At least one LeRobot video frame must be requested")
        if frames != sorted(frames) or len(frames) != len(set(frames)):
            raise ValueError(
                f"LeRobot frame indices must be unique and sorted, received {frames}"
            )
        height, width, channels = self.details["camera_shape"]
        if channels != 3:
            raise RuntimeError(
                f"LeRobot RGB camera has invalid shape {self.details['camera_shape']}"
            )
        fps = float(self.info["fps"])
        output_count = frames[-1] - frames[0] + 1
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(trajectory["video_path"]),
            "-ss",
            f"{frames[0] / fps:.9f}",
            "-frames:v",
            str(output_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg could not decode official RoboCasa365 H.264 video "
                f"{trajectory['video_path']}: "
                f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
            )
        expected_bytes = output_count * height * width * channels
        if len(completed.stdout) != expected_bytes:
            raise RuntimeError(
                "Decoded RoboCasa365 window has an unexpected byte count: "
                f"path={trajectory['video_path']}, expected={expected_bytes}, "
                f"actual={len(completed.stdout)}, frames={frames}"
            )
        decoded = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
            output_count, height, width, channels
        )
        offsets = [frame - frames[0] for frame in frames]
        selected = np.ascontiguousarray(decoded[offsets])
        return torch.from_numpy(selected).permute(0, 3, 1, 2).float() / 255.0

    @staticmethod
    def _environment_info(trajectory: dict[str, Any]) -> dict[str, Any]:
        with gzip.open(trajectory["model_xml_path"], "rt", encoding="utf-8") as handle:
            model_xml = handle.read()
        ep_meta = _load_json(Path(trajectory["ep_meta_path"]))
        return {
            "file_path": trajectory["file_path"],
            "demo_key": trajectory["demo_key"],
            "episode_index": trajectory["episode_index"],
            "task_name": trajectory["task_name"],
            "dataset_format": trajectory["dataset_format"],
            "model_xml": model_xml,
            "ep_meta": ep_meta,
        }

    def get_frames(
        self,
        index: int,
        frames: list[int] | range,
        subtask: str | None = None,
    ) -> tuple[
        dict[str, Tensor],
        Tensor,
        Tensor | None,
        Tensor,
        dict[str, Any],
    ]:
        if subtask is not None:
            raise ValueError(
                "RoboCasa365 atomic task episodes do not expose PnPCounterTop "
                f"segment labels; subtask={subtask!r}"
            )
        indices = [int(value) for value in frames]
        trajectory = self.trajectories[index]
        if any(value < 0 or value >= self.seq_lengths[index] for value in indices):
            raise IndexError(
                f"LeRobot episode {trajectory['episode_index']} frame indices out of range: "
                f"frames={indices}, length={self.seq_lengths[index]}"
            )
        images = self._decode_frames(trajectory, indices)
        actions = torch.from_numpy(trajectory["actions"][indices]).float()
        proprio = torch.from_numpy(trajectory["policy_states"][indices]).float()
        rewards = torch.from_numpy(trajectory["rewards"][indices]).float()
        simulator_states: Tensor | None = None
        environment_info: dict[str, Any] = trajectory
        if self.output_environment_info:
            with np.load(trajectory["state_path"], allow_pickle=False) as archive:
                raw_states = np.asarray(archive["states"])
            simulator_states = torch.from_numpy(
                np.ascontiguousarray(raw_states[indices])
            )
            environment_info = self._environment_info(trajectory)
        return (
            {"visual": images, "proprio": proprio},
            actions,
            simulator_states,
            rewards,
            environment_info,
        )

    def __getitem__(
        self,
        index: int,
        subtask: str | None = None,
    ) -> tuple[
        dict[str, Tensor],
        Tensor,
        Tensor | None,
        Tensor,
        dict[str, Any],
    ]:
        return self.get_frames(
            index,
            range(self.get_seq_length(index)),
            subtask=subtask,
        )
