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
from wm_adapter.benchmarks.base import (
    EPISODE_SPLIT_STRATEGY,
    WINDOW_SELECTION_STRATEGY,
)
from wm_adapter.benchmarks.factory import build_benchmark
from wm_adapter.data.feature_cache import FeatureCacheWriter
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
        planning_tag=cfg.model.get("planning_tag"),
        planning_subtask=cfg.model.get("planning_subtask"),
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
    return _encoded_batch(batch, clean_prefix, ood_prefix, clean_latent)


def _encoded_batch(
    batch: dict[str, torch.Tensor],
    clean_prefix: torch.Tensor,
    ood_prefix: torch.Tensor,
    clean_latent: torch.Tensor,
) -> dict[str, torch.Tensor]:
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


@torch.no_grad()
def _assert_split_encoder_parity(
    backend: JEPAWMDroidBackend,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = backend._normalize_images(images)
    batch_size, sequence_length = normalized.shape[:2]
    flattened = normalized.reshape(
        batch_size * sequence_length,
        3,
        backend.image_size,
        backend.image_size,
    )
    official_features = backend.encoder.forward_features(flattened)
    if not isinstance(official_features, dict) or "x_norm_patchtokens" not in official_features:
        raise RuntimeError(
            "Official DINOv3 forward_features did not return x_norm_patchtokens: "
            f"type={type(official_features).__name__}"
        )
    official_flat = official_features["x_norm_patchtokens"]
    expected_flat_shape = (
        batch_size * sequence_length,
        backend.num_patch_tokens,
        backend.token_dim,
    )
    if tuple(official_flat.shape) != expected_flat_shape:
        raise RuntimeError(
            "Official DINOv3 patch latent shape mismatch: "
            f"expected={expected_flat_shape}, actual={tuple(official_flat.shape)}"
        )
    official = official_flat.reshape(
        batch_size,
        sequence_length,
        backend.num_patch_tokens,
        backend.token_dim,
    )
    split_prefix = backend._encode_normalized_prefix(normalized)
    split = backend.encode_from_prefix(
        split_prefix,
        BaseMethod().to(backend.device),
        batch_size,
        sequence_length,
    )
    if official.shape != split.shape:
        raise RuntimeError(
            "Split encoder parity shape mismatch: "
            f"official={tuple(official.shape)}, split={tuple(split.shape)}"
        )
    low_precision = official.dtype in {torch.float16, torch.bfloat16} or split.dtype in {
        torch.float16,
        torch.bfloat16,
    }
    atol, rtol = (2.0e-3, 2.0e-3) if low_precision else (1.0e-6, 1.0e-5)
    official_float = official.float()
    split_float = split.float()
    max_error = float((official_float - split_float).abs().max().cpu())
    if not torch.allclose(official_float, split_float, atol=atol, rtol=rtol):
        raise RuntimeError(
            "Split encoder parity failed: "
            f"max_abs_error={max_error}, official_shape={tuple(official.shape)}, "
            f"split_shape={tuple(split.shape)}, atol={atol}, rtol={rtol}"
        )
    print(f"Split encoder parity passed: max_abs_error={max_error}")
    return split_prefix, split


def main() -> None:
    cfg = load_experiment_config()
    if str(cfg.appearance.pipeline_version) != "composed_photometric_v1":
        raise ValueError(
            f"Unsupported appearance pipeline version: {cfg.appearance.pipeline_version}"
        )
    seed_everything(int(cfg.data.window_seed))
    backend = _backend(cfg)
    benchmark = build_benchmark(cfg)
    resolved_task = benchmark.resolve_task(strict=True)
    task_manifest = benchmark.write_task_manifest(resolved_task)
    source = benchmark.build_source_dataset(output_environment_info=False)
    train_episodes, evaluation_episodes = benchmark.split_trajectory_ids(source)
    candidates = benchmark.enumerate_window_candidates(
        source,
        int(cfg.data.num_frames),
        int(cfg.data.frameskip),
    )
    selected = benchmark.select_windows(
        candidates,
        train_episodes,
        int(cfg.data.num_train_windows),
        int(cfg.data.window_seed),
    )
    if len(selected) != int(cfg.data.num_train_windows):
        raise RuntimeError(
            f"Requested {cfg.data.num_train_windows} train windows, but only {len(selected)} "
            f"eligible episode-disjoint {benchmark.name} windows are available"
        )
    windows = benchmark.make_window_dataset(
        source,
        selected,
        num_frames=int(cfg.data.num_frames),
        frameskip=int(cfg.data.frameskip),
        appearance_seed=int(cfg.appearance.training_seed),
        appearance_severity=float(
            cfg.appearance.get("training_severity", cfg.appearance.severity)
        ),
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
    clean_prefix, clean_latent = _assert_split_encoder_parity(
        backend,
        first_batch["clean_images"],
    )
    ood_prefix = backend.encode_prefix(first_batch["ood_images"])
    first_encoded = _encoded_batch(first_batch, clean_prefix, ood_prefix, clean_latent)
    layout = backend.token_layout
    appearance_metadata = ComposedPhotometricShift.metadata(
        float(cfg.appearance.get("training_severity", cfg.appearance.severity)),
        int(cfg.appearance.training_seed),
    )
    preprocessing_metadata = {
        "implementation": "jepa-wms.app.plan_common.datasets.transforms.make_transforms",
        "image_size": backend.image_size,
        "patch_size": backend.patch_size,
        "normalize": OmegaConf.to_container(backend.official_planning_template.model_kwargs.data_aug.normalize),
        "random_horizontal_flip": False,
        "random_resize_scale": [1.0, 1.0],
        "random_resize_aspect_ratio": [1.0, 1.0],
        "source_camera_key": resolved_task.camera_key,
        "source_camera_height": resolved_task.camera_height,
        "source_camera_width": resolved_task.camera_width,
        "source_vertical_flip": resolved_task.camera_vertical_flip,
        "source_channel_order": resolved_task.camera_channel_order,
    }
    metadata = {
        "backend": "jepa_wm_droid",
        "benchmark": resolved_task.benchmark,
        "benchmark_suite": resolved_task.suite,
        "task_id": resolved_task.task_id,
        "task_name": resolved_task.task_name,
        "task_key": resolved_task.task_key,
        "task_manifest_sha256": task_manifest["task_manifest_sha256"],
        "task_manifest_path": str(benchmark.task_manifest_path()),
        "dataset": resolved_task.dataset_path,
        "dataset_sha256": resolved_task.dataset_sha256,
        "camera_key": resolved_task.camera_key,
        "camera_height": resolved_task.camera_height,
        "camera_width": resolved_task.camera_width,
        "camera_channel_order": resolved_task.camera_channel_order,
        "camera_vertical_flip": resolved_task.camera_vertical_flip,
        "action_convention": resolved_task.action_convention,
        "action_transform": resolved_task.action_transform,
        "source_trajectory_ids": list(
            resolved_task.selected_train_demonstrations
        ),
        "train_test_split": {
            "strategy": EPISODE_SPLIT_STRATEGY,
            "seed": int(cfg.data.split_seed),
            "train_fraction": float(cfg.data.train_fraction),
            "train_source_trajectory_ids": list(
                resolved_task.selected_train_demonstrations
            ),
            "evaluation_source_trajectory_ids": list(
                resolved_task.selected_test_demonstrations
            ),
        },
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "upstream_commits": backend.upstream_commits,
        "task_upstream_commits": resolved_task.upstream_commits,
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
        "window_identity": ["source_trajectory_index", "start_step"],
        "source_trajectory_identity": (
            "benchmark trajectory index plus immutable source trajectory ID"
        ),
    }
    for optional_contract_key in (
        "camera_height",
        "camera_width",
        "camera_channel_order",
        "camera_vertical_flip",
        "action_transform",
    ):
        if metadata[optional_contract_key] is None:
            metadata.pop(optional_contract_key)
    output_path = resolve_path(cfg.paths.feature_cache)
    if output_path.exists():
        raise FileExistsError(
            f"Feature cache already exists and will not be overwritten: {output_path}"
        )
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
