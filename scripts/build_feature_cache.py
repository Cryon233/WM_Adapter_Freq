from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_adapter.adapters.base import BaseMethod
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.data.feature_cache import FeatureCacheWriter
from wm_adapter.data.robocasa_windows import (
    EPISODE_SPLIT_STRATEGY,
    WINDOW_SELECTION_STRATEGY,
    RoboCasaWindowDataset,
    build_robocasa_dataset,
    select_episode_balanced_windows,
    split_episode_indices,
)
from wm_adapter.utils.checkpoints import sha256_file
from wm_adapter.utils.reproducibility import load_experiment_config, resolve_path, seed_everything


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.int64).tobytes()).hexdigest()


def _backend(cfg: Any) -> JEPAWMDroidBackend:
    return JEPAWMDroidBackend(
        third_party_root=cfg.model.third_party_root,
        jepa_checkpoint=cfg.model.jepa_checkpoint,
        dinov3_checkpoint=cfg.model.dinov3_checkpoint,
        official_planning_config=cfg.model.official_planning_config,
        device=cfg.device,
    )


@torch.no_grad()
def _encode_batch(
    backend: JEPAWMDroidBackend,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    clean_images = batch["clean_images"]
    ood_images = batch["ood_images"]
    batch_size, sequence_length = clean_images.shape[:2]
    clean_prefix = backend.encode_prefix(clean_images)
    ood_prefix = backend.encode_prefix(ood_images)
    clean_latent = backend.encode_from_prefix(
        clean_prefix, BaseMethod().to(backend.device), batch_size, sequence_length
    )
    return {
        "clean_prefix_tokens": clean_prefix,
        "ood_prefix_tokens": ood_prefix,
        "clean_context_final_latent": clean_latent[:, :3],
        "clean_future_latent": clean_latent[:, 3:4],
        "actions": batch["actions"],
        "episode_id": batch["episode_id"],
        "window_id": batch["window_id"],
        "appearance_seed": batch["appearance_seed"],
    }


def main() -> None:
    cfg = load_experiment_config()
    if str(cfg.appearance.pipeline_version) != "composed_photometric_v1":
        raise ValueError(
            f"Unsupported appearance pipeline version: {cfg.appearance.pipeline_version}"
        )
    seed_everything(int(cfg.data.window_seed))
    backend = _backend(cfg)
    source = build_robocasa_dataset(
        jepa_wms_root=backend.jepa_repo,
        dataset_root=cfg.paths.dataset_root,
        hdf5_path=cfg.paths.robocasa_hdf5,
        task_name=str(cfg.data.task_name),
        camera_view=str(cfg.data.camera_view),
        output_environment_info=False,
        transform=None,
    )
    train_episodes, evaluation_episodes = split_episode_indices(
        len(source), float(cfg.data.train_fraction), int(cfg.data.split_seed)
    )
    candidates = RoboCasaWindowDataset.all_candidates(
        source, int(cfg.data.num_frames), int(cfg.data.frameskip)
    )
    selected = select_episode_balanced_windows(
        candidates,
        train_episodes,
        int(cfg.data.num_train_windows),
        int(cfg.data.window_seed),
    )
    if len(selected) != int(cfg.data.num_train_windows):
        raise RuntimeError(
            f"Requested {cfg.data.num_train_windows} train windows, but only {len(selected)} "
            "eligible episode-disjoint RoboCasa windows are available"
        )
    windows = RoboCasaWindowDataset(
        source,
        selected,
        num_frames=int(cfg.data.num_frames),
        frameskip=int(cfg.data.frameskip),
        appearance_seed=int(cfg.appearance.training_seed),
        appearance_severity=float(cfg.appearance.severity),
    )
    loader = DataLoader(
        windows,
        batch_size=int(cfg.cache.encoder_batch_size),
        shuffle=False,
        num_workers=int(cfg.cache.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.cache.num_workers) > 0,
    )
    iterator = iter(loader)
    first_batch = next(iterator)
    first_encoded = _encode_batch(backend, first_batch)
    layout = backend.token_layout
    appearance_metadata = ComposedPhotometricShift.metadata(
        float(cfg.appearance.severity), int(cfg.appearance.training_seed)
    )
    preprocessing_metadata = {
        "implementation": "jepa-wms.app.plan_common.datasets.transforms.make_transforms",
        "image_size": backend.image_size,
        "patch_size": backend.patch_size,
        "normalize": OmegaConf.to_container(backend.official_planning_template.model_kwargs.data_aug.normalize),
        "random_horizontal_flip": False,
        "random_resize_scale": [1.0, 1.0],
        "random_resize_aspect_ratio": [1.0, 1.0],
    }
    metadata = {
        "backend": "jepa_wm_droid",
        "dataset": str(resolve_path(cfg.paths.robocasa_hdf5)),
        "dataset_sha256": sha256_file(resolve_path(cfg.paths.robocasa_hdf5)),
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "upstream_commits": backend.upstream_commits,
        "appearance_metadata": appearance_metadata,
        "preprocessing_metadata": preprocessing_metadata,
        "token_layout": {
            "total_tokens": layout.total_tokens,
            "prefix_tokens": layout.prefix_tokens,
            "patch_tokens": layout.patch_tokens,
            "grid_height": layout.grid_height,
            "grid_width": layout.grid_width,
            "token_dim": layout.token_dim,
        },
        "episode_split_strategy": EPISODE_SPLIT_STRATEGY,
        "episode_split_seed": int(cfg.data.split_seed),
        "episode_split_train_fraction": float(cfg.data.train_fraction),
        "source_episode_count": len(source),
        "train_episode_count": len(train_episodes),
        "evaluation_episode_count": len(evaluation_episodes),
        "train_episode_indices_sha256": _sha256_array(train_episodes),
        "evaluation_episode_indices_sha256": _sha256_array(evaluation_episodes),
        "window_selection_strategy": WINDOW_SELECTION_STRATEGY,
        "window_selection_seed": int(cfg.data.window_seed),
        "selected_window_pairs_sha256": _sha256_array(np.asarray(selected, dtype=np.int64)),
    }
    output_path = resolve_path(cfg.paths.feature_cache)
    writer = FeatureCacheWriter(output_path, metadata)
    try:
        writer.append(first_encoded)
        for batch in tqdm(iterator, total=len(loader) - 1, desc="building feature cache"):
            writer.append(_encode_batch(backend, batch))
        fingerprint = writer.finalize()
    except BaseException:
        writer.close_unfinalized()
        raise
    print(f"Feature cache written: {output_path}")
    print(f"Cache fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
