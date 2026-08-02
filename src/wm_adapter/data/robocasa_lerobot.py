from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wm_adapter.benchmarks.base import ActionTransform
from wm_adapter.utils.reproducibility import resolve_path


LEROBOT_SOURCE_ID = "lerobot/robocasa_target_human_unified"
LEROBOT_SOURCE_REVISION = "a1c7cba1a128f5dd8d3084012ba366d10ce174d9"


def _load_metadata(root: Path) -> tuple[dict[str, Any], Any, Any]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Reading official RoboCasa LeRobot data requires pandas and pyarrow"
        ) from error
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    missing = [
        str(path)
        for path in (info_path, tasks_path, episodes_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Official RoboCasa LeRobot metadata is incomplete; missing="
            f"{missing}, root={root}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks = pd.read_parquet(tasks_path)
    episodes = pd.read_parquet(episodes_path)
    return info, tasks, episodes


def inspect_robocasa_lerobot(
    root: str | Path,
    *,
    task_name: str,
    camera_view: str,
) -> tuple[list[str], dict[str, Any]]:
    resolved = resolve_path(root)
    details: dict[str, Any] = {
        "dataset_path": str(resolved),
        "dataset_format": "lerobot_v3",
        "official_source_identifier": LEROBOT_SOURCE_ID,
        "official_source_revision": LEROBOT_SOURCE_REVISION,
    }
    if not resolved.is_dir():
        return [f"LeRobot dataset directory does not exist: {resolved}"], details
    try:
        info, tasks, episodes = _load_metadata(resolved)
    except (FileNotFoundError, OSError, ValueError) as error:
        return [str(error)], details
    errors: list[str] = []
    if str(info.get("codebase_version")) != "v3.0":
        errors.append(
            "unsupported LeRobot schema: "
            f"codebase_version={info.get('codebase_version')!r}"
        )
    if task_name not in tasks.index:
        errors.append(
            f"LeRobot task {task_name!r} is absent; available={list(tasks.index)}"
        )
        return errors, details
    task_index = int(tasks.loc[task_name, "task_index"])
    task_episodes = episodes[
        episodes["tasks"].apply(
            lambda values: isinstance(values, (list, tuple, np.ndarray))
            and task_name in values
        )
    ].sort_values("episode_index")
    episode_ids = [str(int(value)) for value in task_episodes["episode_index"]]
    details.update(
        {
            "source_task": task_name,
            "task_index": task_index,
            "demonstration_ids": episode_ids,
            "available_demonstrations": len(episode_ids),
            "demonstration_lengths": [
                int(value) for value in task_episodes["length"]
            ],
        }
    )
    if not episode_ids:
        errors.append(f"LeRobot dataset contains no {task_name} episodes")
        return errors, details
    feature_key = f"observation.images.{camera_view}"
    feature = info.get("features", {}).get(feature_key)
    if not isinstance(feature, dict):
        errors.append(
            f"LeRobot camera feature {feature_key!r} is absent; "
            f"available={sorted(info.get('features', {}))}"
        )
        return errors, details
    shape = tuple(int(value) for value in feature.get("shape", ()))
    if len(shape) != 3 or shape[-1] != 3:
        errors.append(
            f"LeRobot camera {feature_key!r} must be [H,W,3], received {shape}"
        )
    action_feature = info.get("features", {}).get("action", {})
    action_shape = tuple(int(value) for value in action_feature.get("shape", ()))
    if action_shape != (12,):
        errors.append(
            "Official RoboCasa365 LeRobot action must contain 12 values "
            f"(base/control + 7-D manipulator), received {action_shape}"
        )
    required_files: set[str] = {
        "README.md",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
    }
    data_pattern = str(info.get("data_path", ""))
    video_pattern = str(info.get("video_path", ""))
    for _, row in task_episodes.iterrows():
        required_files.add(
            data_pattern.format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
        )
        prefix = f"videos/{feature_key}"
        required_files.add(
            video_pattern.format(
                video_key=feature_key,
                chunk_index=int(row[f"{prefix}/chunk_index"]),
                file_index=int(row[f"{prefix}/file_index"]),
            )
        )
    missing_files = [
        str(resolved / relative)
        for relative in sorted(required_files)
        if not (resolved / relative).is_file()
    ]
    if missing_files:
        errors.append(
            "LeRobot OpenDrawer subset is incomplete; missing files="
            f"{missing_files}"
        )
    total_windows = sum(
        max(0, int(length) - 16)
        for length in details["demonstration_lengths"]
    )
    if total_windows < 2000:
        errors.append(
            "LeRobot dataset cannot provide 2000 four-frame windows: "
            f"lower_bound={total_windows}"
        )
    environment_features = set(info.get("features", {}))
    restore_keys = {
        "model_xml",
        "environment.model_xml",
        "simulator_state",
        "environment.simulator_state",
    }
    restore_capable = bool(environment_features.intersection(restore_keys))
    details.update(
        {
            "camera_key": feature_key,
            "camera_shape": shape,
            "camera_codec": feature.get("video_info", {}).get("video.codec"),
            "action_shape": action_shape,
            "window_candidates_lower_bound": total_windows,
            "required_relative_files": sorted(required_files),
            "environment_restore_capable": restore_capable,
            "environment_restore_missing": sorted(
                restore_keys.difference(environment_features)
            ),
            "robot": str(info.get("robot_type", "robocasa")),
            "gripper": "unavailable in LeRobot metadata; strict simulator preflight required",
            "controller": (
                "12-D RoboCasa policy action with a 7-D manipulator suffix; "
                "controller identity is unavailable in LeRobot metadata"
            ),
        }
    )
    return errors, details


class RoboCasaLeRobotDataset:
    """Read an official task subset without pretending policy state is MuJoCo state."""

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
            raise RuntimeError("Invalid RoboCasa LeRobot dataset:\n- " + "\n- ".join(errors))
        if output_environment_info and not details["environment_restore_capable"]:
            raise RuntimeError(
                "Official RoboCasa LeRobot data cannot initialize closed-loop planning: "
                "the release contains policy observation.state but no external model_xml "
                "or full MuJoCo simulator state. Obtain the official raw RoboCasa task "
                "dataset rather than treating the 16-D policy state as simulator state. "
                f"dataset={self.root}, source={LEROBOT_SOURCE_ID}@{LEROBOT_SOURCE_REVISION}"
            )
        info, _, episodes = _load_metadata(self.root)
        selected = episodes[
            episodes["tasks"].apply(
                lambda values: isinstance(values, (list, tuple, np.ndarray))
                and task_name in values
            )
        ].sort_values("episode_index")
        data_files = sorted(
            {
                self.root
                / str(info["data_path"]).format(
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
                for _, row in selected.iterrows()
            }
        )
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError(
                "Reading RoboCasa LeRobot parquet data requires pandas and pyarrow"
            ) from error
        frames = pd.concat(
            [
                pd.read_parquet(
                    path,
                    columns=[
                        "episode_index",
                        "frame_index",
                        "action",
                        "observation.state",
                    ],
                )
                for path in data_files
            ],
            ignore_index=True,
        )
        selected_ids = {int(value) for value in selected["episode_index"]}
        frames = frames[frames["episode_index"].isin(selected_ids)]
        self.info = info
        self.details = details
        self.trajectories: list[dict[str, Any]] = []
        self.seq_lengths: list[int] = []
        feature_key = f"observation.images.{camera_view}"
        video_prefix = f"videos/{feature_key}"
        for _, row in selected.iterrows():
            episode_id = int(row["episode_index"])
            episode_frames = frames[frames["episode_index"] == episode_id].sort_values(
                "frame_index"
            )
            expected_length = int(row["length"])
            if len(episode_frames) != expected_length:
                raise RuntimeError(
                    f"LeRobot episode {episode_id} has {len(episode_frames)} parquet "
                    f"frames, expected {expected_length}"
                )
            raw_actions = np.stack(episode_frames["action"].to_numpy()).astype(
                np.float32
            )
            raw_states = np.stack(
                episode_frames["observation.state"].to_numpy()
            ).astype(np.float32)
            video_path = self.root / str(info["video_path"]).format(
                video_key=feature_key,
                chunk_index=int(row[f"{video_prefix}/chunk_index"]),
                file_index=int(row[f"{video_prefix}/file_index"]),
            )
            self.trajectories.append(
                {
                    "demo_key": str(episode_id),
                    "episode_index": episode_id,
                    "actions": self.action_transform.environment_to_canonical_action(
                        raw_actions[:, -7:]
                    ),
                    "policy_states": raw_states,
                    "video_path": str(video_path),
                    "video_start_frame": int(
                        round(float(row[f"{video_prefix}/from_timestamp"]) * float(info["fps"]))
                    ),
                    "file_path": str(self.root),
                    "dataset_format": "lerobot_v3",
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
        absolute_start = int(trajectory["video_start_frame"]) + frames[0]
        output_count = frames[-1] - frames[0] + 1
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{absolute_start / fps:.9f}",
            "-i",
            str(trajectory["video_path"]),
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
                "ffmpeg could not decode official RoboCasa LeRobot AV1 video "
                f"{trajectory['video_path']}: {completed.stderr.decode('utf-8', errors='replace').strip()}"
            )
        expected_bytes = output_count * height * width * channels
        if len(completed.stdout) != expected_bytes:
            raise RuntimeError(
                "Decoded RoboCasa LeRobot window has an unexpected byte count: "
                f"path={trajectory['video_path']}, expected={expected_bytes}, "
                f"actual={len(completed.stdout)}, frames={frames}"
            )
        decoded = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
            output_count, height, width, channels
        )
        offsets = [frame - frames[0] for frame in frames]
        selected = np.ascontiguousarray(decoded[offsets])
        return torch.from_numpy(selected).permute(0, 3, 1, 2).float() / 255.0

    def get_frames(
        self,
        index: int,
        frames: list[int] | range,
        subtask: str | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, None, None, dict[str, Any]]:
        if subtask is not None:
            raise ValueError(
                "RoboCasa365 atomic LeRobot episodes do not expose PnPCounterTop "
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
        return {"visual": images, "proprio": proprio}, actions, None, None, trajectory

    def __getitem__(
        self,
        index: int,
        subtask: str | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, None, None, dict[str, Any]]:
        return self.get_frames(
            index,
            range(self.get_seq_length(index)),
            subtask=subtask,
        )
