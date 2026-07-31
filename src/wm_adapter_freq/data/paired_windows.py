from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_pretraining import data as spt_data
from torch import Tensor
from torch.utils.data import Dataset

from wm_adapter_freq.data.appearance_shift import (
    SHIFT_NAMES,
    TwoRoomAppearanceShift,
)


EPISODE_SPLIT_STRATEGY = "deterministic_episode_partition_v1"
WINDOW_SELECTION_STRATEGY = "episode_balanced_round_robin_v1"


def split_episode_indices(
    num_episodes: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition dataset episode indices into deterministic train/eval sets."""
    episode_indices = np.arange(num_episodes, dtype=np.int64)
    generator = np.random.default_rng(seed)
    permuted_indices = generator.permutation(episode_indices)
    train_count = int(np.floor(num_episodes * train_fraction))
    train_episode_indices = np.sort(permuted_indices[:train_count])
    eval_episode_indices = np.sort(permuted_indices[train_count:])
    return train_episode_indices, eval_episode_indices


def select_episode_balanced_window_indices(
    dataset: Any,
    allowed_episode_indices: np.ndarray,
    max_windows: int,
    seed: int,
) -> list[int]:
    """Select clip indices evenly across episodes with deterministic shuffling."""
    clip_indices = dataset.clip_indices
    allowed_episodes = set(
        np.asarray(allowed_episode_indices, dtype=np.int64).tolist()
    )
    windows_by_episode: dict[int, list[int]] = {}
    for window_index, (episode_index, _) in enumerate(clip_indices):
        episode = int(episode_index)
        if episode in allowed_episodes:
            windows_by_episode.setdefault(episode, []).append(window_index)

    candidate_count = sum(map(len, windows_by_episode.values()))
    target_count = min(int(max_windows), candidate_count)
    if target_count <= 0:
        return []

    generator = np.random.default_rng(int(seed))
    episode_order = generator.permutation(
        np.asarray(list(windows_by_episode), dtype=np.int64)
    ).tolist()
    shuffled_windows = {
        episode_index: generator.permutation(
            np.asarray(
                windows_by_episode[episode_index],
                dtype=np.int64,
            )
        ).tolist()
        for episode_index in episode_order
    }

    selected: list[int] = []
    round_index = 0
    while len(selected) < target_count:
        added_in_round = False
        for episode_index in episode_order:
            episode_windows = shuffled_windows[episode_index]
            if round_index < len(episode_windows):
                selected.append(int(episode_windows[round_index]))
                added_in_round = True
                if len(selected) == target_count:
                    break
        if not added_in_round:
            break
        round_index += 1
    return selected


class PairedAppearanceWindowDataset(Dataset[dict[str, Tensor]]):
    """Wrap one clean physical trajectory window with four shifted views."""

    def __init__(
        self,
        dataset: Any,
        appearance_shift: TwoRoomAppearanceShift | None = None,
        seed: int = 0,
        severity: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.appearance_shift = appearance_shift or TwoRoomAppearanceShift()
        self.seed = int(seed)
        self.severity = float(severity)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.dataset[index]
        clean_pixels = torch.as_tensor(sample["pixels"]).clone()

        shifted_windows: list[Tensor] = []
        shift_seeds: list[int] = []
        for shift_index, shift_type in enumerate(SHIFT_NAMES):
            shift_seed = (
                self.seed
                + 1_000_003 * int(index)
                + 97_409 * shift_index
            ) % (2**63 - 1)
            spec = self.appearance_shift.sample_spec(
                shift_type=shift_type,
                seed=shift_seed,
                severity=self.severity,
            )
            shifted_windows.append(
                self.appearance_shift.apply(clean_pixels, spec)
            )
            shift_seeds.append(shift_seed)

        return {
            "clean_pixels": clean_pixels,
            "shifted_pixels": torch.stack(shifted_windows),
            "action": torch.as_tensor(sample["action"]).float(),
            "proprio": torch.as_tensor(sample["proprio"]).float(),
            "shift_type": torch.arange(
                len(SHIFT_NAMES),
                dtype=torch.int64,
            ),
            "shift_seed": torch.tensor(shift_seeds, dtype=torch.int64),
        }


def load_paired_two_room_windows(
    dataset_name: str,
    cache_dir: str | Path | None = None,
    seed: int = 0,
    severity: float = 1.0,
) -> tuple[PairedAppearanceWindowDataset, Any]:
    """Load stable-worldmodel TwoRoom clips with the final four-step contract."""
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        num_steps=4,
        frameskip=5,
        transform=None,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"],
    )
    return (
        PairedAppearanceWindowDataset(
            dataset,
            seed=seed,
            severity=severity,
        ),
        dataset,
    )


def build_image_preprocessor(
    image_size: int = 224,
) -> Callable[[Tensor], Tensor]:
    """Build the exact ImageNet preprocessing used by upstream training."""
    transform = spt_data.transforms.Compose(
        spt_data.transforms.ToImage(
            **spt_data.dataset_stats.ImageNet,
            source="pixels",
            target="pixels",
        ),
        spt_data.transforms.Resize(
            image_size,
            source="pixels",
            target="pixels",
        ),
    )

    def preprocess(pixels: Tensor) -> Tensor:
        return transform({"pixels": pixels})["pixels"]

    return preprocess
