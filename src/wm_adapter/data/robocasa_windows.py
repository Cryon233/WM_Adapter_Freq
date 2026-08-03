from __future__ import annotations

import os
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.utils.reproducibility import resolve_path


EPISODE_SPLIT_STRATEGY = "deterministic_episode_partition_v1"
WINDOW_SELECTION_STRATEGY = "episode_balanced_round_robin_v1"


def split_episode_indices(
    num_episodes: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_episodes <= 0:
        raise ValueError(f"RoboCasa dataset has no episodes: num_episodes={num_episodes}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0,1), received {train_fraction}")
    indices = np.arange(num_episodes, dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(indices)
    train_count = int(np.floor(num_episodes * train_fraction))
    if train_count == 0 or train_count == num_episodes:
        raise ValueError(
            f"Episode split must leave non-empty train and evaluation partitions: "
            f"num_episodes={num_episodes}, train_fraction={train_fraction}"
        )
    return np.sort(shuffled[:train_count]), np.sort(shuffled[train_count:])


def select_episode_balanced_windows(
    candidates: list[tuple[int, int]],
    allowed_episode_indices: np.ndarray,
    max_windows: int,
    seed: int,
) -> list[tuple[int, int]]:
    allowed = set(int(value) for value in allowed_episode_indices.tolist())
    grouped: dict[int, list[tuple[int, int]]] = {episode: [] for episode in allowed}
    for episode, start in candidates:
        if episode in allowed:
            grouped[episode].append((episode, start))
    grouped = {episode: windows for episode, windows in grouped.items() if windows}
    generator = np.random.default_rng(seed)
    episodes = list(grouped)
    generator.shuffle(episodes)
    for episode in episodes:
        order = generator.permutation(len(grouped[episode]))
        grouped[episode] = [grouped[episode][int(index)] for index in order]
    target = min(max_windows, sum(len(windows) for windows in grouped.values()))
    selected: list[tuple[int, int]] = []
    round_index = 0
    while len(selected) < target:
        added = False
        for episode in episodes:
            windows = grouped[episode]
            if round_index < len(windows):
                selected.append(windows[round_index])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        round_index += 1
    return selected


def build_robocasa_dataset(
    *,
    jepa_wms_root: str | Path,
    dataset_root: str | Path,
    hdf5_path: str | Path,
    task_name: str,
    camera_view: str,
    output_environment_info: bool,
    transform: Any | None,
) -> Any:
    source_root = resolve_path(jepa_wms_root)
    source_value = str(source_root)
    if source_value not in sys.path:
        sys.path.insert(0, source_value)
    root = resolve_path(dataset_root)
    hdf5 = resolve_path(hdf5_path)
    if not hdf5.is_file():
        raise FileNotFoundError(f"Official RoboCasa offline HDF5 does not exist: {hdf5}")
    expected_parent = root / "robocasa"
    if hdf5.parent != expected_parent.resolve():
        raise ValueError(
            f"Official custom RoboCasa data must be inside {expected_parent}; received {hdf5}"
        )
    matching = sorted(path.resolve() for path in expected_parent.rglob("*im256.hdf5") if path.is_file())
    if hdf5 not in matching:
        raise RuntimeError(
            f"JEPA-WM custom RoboCasa loader cannot see the configured HDF5 {hdf5}; "
            f"found {[str(path) for path in matching]}"
        )
    os.environ["JEPAWM_DSET"] = str(root)
    from app.plan_common.datasets.robocasa_dset import RoboCasaDataset

    dataset = RoboCasaDataset(
        transform=transform,
        filter_tasks=[task_name],
        filter_first_episodes=None,
        camera_views=[camera_view],
        normalize_action=False,
        use_human=True,
        use_mg=True,
        manip_only=True,
        with_reward=True,
        output_rcasa_state=output_environment_info,
        output_rcasa_info=output_environment_info,
        rcasa_to_droid_action_format=False,
        custom_teleop_dset=True,
    )
    source_files = {
        resolve_path(str(trajectory["file_path"]))
        for trajectory in dataset.trajectories
    }
    if source_files != {hdf5}:
        raise RuntimeError(
            "The filtered RoboCasa task did not resolve exclusively to its configured "
            f"HDF5: expected={hdf5}, actual={sorted(str(path) for path in source_files)}"
        )
    return dataset


@dataclass(frozen=True)
class WindowIdentity:
    episode_id: int
    start_step: int


class RoboCasaWindowDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        source_dataset: Any,
        windows: list[tuple[int, int]],
        *,
        num_frames: int,
        frameskip: int,
        appearance_seed: int,
        appearance_severity: float,
        appearance_severity_range: tuple[float, float] | None = None,
    ) -> None:
        self.source_dataset = source_dataset
        self.windows = [WindowIdentity(int(episode), int(start)) for episode, start in windows]
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.appearance_seed = appearance_seed
        self.appearance_severity = appearance_severity
        self.appearance_severity_range = appearance_severity_range
        self.appearance = ComposedPhotometricShift()

    @staticmethod
    def all_candidates(source_dataset: Any, num_frames: int, frameskip: int) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for episode in range(len(source_dataset)):
            length = int(source_dataset.get_seq_length(episode))
            latest_start = length - num_frames * frameskip
            if latest_start >= 0:
                candidates.extend((episode, start) for start in range(latest_start + 1))
        return candidates

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        identity = self.windows[index]
        frame_indices = [identity.start_step + offset * self.frameskip for offset in range(self.num_frames)]
        observation, actions, _, _, _ = self.source_dataset.get_frames(identity.episode_id, frame_indices)
        clean = observation["visual"]
        if clean.shape[0] != self.num_frames or clean.shape[1] != 3:
            raise RuntimeError(
                f"RoboCasa window {identity} returned visual shape {tuple(clean.shape)}, "
                f"expected [{self.num_frames},3,H,W]"
            )
        if actions.shape != (self.num_frames, 7):
            raise RuntimeError(
                f"RoboCasa window {identity} returned action shape {tuple(actions.shape)}, "
                f"expected {(self.num_frames, 7)}"
            )
        if self.appearance_severity_range is None:
            spec_seed = int(self.appearance_seed + index)
            severity = self.appearance_severity
        else:
            low, high = self.appearance_severity_range
            if not 0.0 <= low <= high:
                raise ValueError(f"Invalid appearance severity range {(low, high)}")
            digest = hashlib.sha256(
                f"{self.appearance_seed}:{identity.episode_id}:{identity.start_step}".encode("utf-8")
            ).digest()
            spec_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            unit = int.from_bytes(digest[8:16], "little") / float(2**64 - 1)
            severity = low + (high - low) * unit
        spec = self.appearance.sample_spec(spec_seed, severity)
        shifted = self.appearance.apply(clean, spec)
        return {
            "clean_images": clean,
            "ood_images": shifted,
            "actions": actions.float(),
            "episode_id": torch.tensor(identity.episode_id, dtype=torch.int64),
            "window_id": torch.tensor(identity.start_step, dtype=torch.int64),
            "appearance_seed": torch.tensor(spec_seed, dtype=torch.int64),
            "appearance_severity": torch.tensor(severity, dtype=torch.float32),
        }
