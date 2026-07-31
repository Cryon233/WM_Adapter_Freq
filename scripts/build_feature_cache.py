from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from wm_adapter_freq.backends.base import BaseWorldModelBackend, build_backend
from wm_adapter_freq.data.appearance_shift import (
    SHIFT_NAMES,
    SHIFT_PIPELINE_VERSION,
)
from wm_adapter_freq.data.feature_cache import FeatureCacheWriter
from wm_adapter_freq.data.paired_windows import (
    WINDOW_SELECTION_STRATEGY,
    build_image_preprocessor,
    load_paired_two_room_windows,
    select_episode_balanced_window_indices,
)
from wm_adapter_freq.io.fingerprint import (
    STABLE_WORLDMODEL_COMMIT,
    resolve_base_model_identity,
)


def _fit_scaler(dataset: Any, key: str) -> tuple[Any, int]:
    from stable_worldmodel.data.normalization import get_scaler

    values = np.asarray(dataset.get_col_data(key))
    feature_dim = int(values.shape[-1])
    values = values.reshape(-1, feature_dim)
    values = values[~np.isnan(values).any(axis=1)]
    scaler = get_scaler("zscore")
    scaler.fit(values)
    return scaler, feature_dim


def _normalize(values: Tensor, scaler: Any, feature_dim: int) -> Tensor:
    shape = values.shape
    normalized = scaler.transform(values.reshape(-1, feature_dim))
    return normalized.reshape(shape).float()


def _scaler_metadata(scaler: Any, feature_dim: int) -> dict[str, object]:
    return {
        "method": "zscore",
        "mean": np.asarray(scaler.mean).reshape(-1).tolist(),
        "std": np.asarray(scaler.std).reshape(-1).tolist(),
        "eps": float(scaler.eps),
        "feature_dim": int(feature_dim),
    }


def _preprocess(
    pixels: Tensor, image_size: int
) -> Tensor:
    preprocess = build_image_preprocessor(image_size)
    flat = pixels.reshape(-1, *pixels.shape[-3:])
    processed = preprocess(flat)
    return processed.reshape(*pixels.shape[:-3], *processed.shape[-3:])


def _encode_chunks(
    backend: BaseWorldModelBackend,
    pixels: Tensor,
    chunk_size: int,
    device: torch.device,
    clean_target: bool,
) -> Tensor:
    outputs = []
    for start in range(0, pixels.shape[0], chunk_size):
        chunk = pixels[start : start + chunk_size].to(
            device, non_blocking=True
        )
        if clean_target:
            output = backend.encode_clean_target(chunk)
        else:
            output = backend.encode_prefix(chunk)
        outputs.append(output.cpu())
    return torch.cat(outputs)


@hydra.main(
    version_base=None,
    config_path="../configs/cache",
    config_name="prejepa_tworoom",
)
def main(cfg: DictConfig) -> None:
    from stable_worldmodel.wm.utils import load_pretrained

    if str(cfg.window_selection.strategy) != WINDOW_SELECTION_STRATEGY:
        raise ValueError(
            "window_selection.strategy must be "
            f"'{WINDOW_SELECTION_STRATEGY}'"
        )

    torch.manual_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    identity = resolve_base_model_identity(str(cfg.base_model_ref))
    base_model = load_pretrained(identity.resolved_weights_path)
    base_model.eval()
    base_model.requires_grad_(False)
    base_model.to(device)
    backend = build_backend(str(cfg.backend), base_model)

    paired_dataset, source_dataset = load_paired_two_room_windows(
        str(cfg.dataset_name),
        cache_dir=cfg.get("dataset_cache_dir"),
        seed=int(cfg.seed),
        severity=float(cfg.appearance.severity),
    )
    max_windows = min(int(cfg.max_windows), len(paired_dataset))
    selected_window_indices = select_episode_balanced_window_indices(
        source_dataset,
        max_windows=max_windows,
        seed=int(cfg.window_selection.seed),
    )
    windows = Subset(paired_dataset, selected_window_indices)
    data_loader = DataLoader(
        windows,
        batch_size=int(cfg.encoder_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg.num_workers) > 0,
    )
    action_scaler, action_dim = _fit_scaler(source_dataset, "action")
    proprio_scaler, proprio_dim = _fit_scaler(source_dataset, "proprio")
    action_normalization = _scaler_metadata(action_scaler, action_dim)
    proprio_normalization = _scaler_metadata(
        proprio_scaler,
        proprio_dim,
    )

    output_path = Path(str(cfg.output_path)).expanduser()
    selected_indices_sha256 = hashlib.sha256(
        np.asarray(
            selected_window_indices,
            dtype=np.int64,
        ).tobytes()
    ).hexdigest()
    metadata: dict[str, str | int | float] = {
        "backend": str(cfg.backend),
        "base_model_ref": str(cfg.base_model_ref),
        "base_model_fingerprint": identity.combined_fingerprint,
        "stable_worldmodel_commit": STABLE_WORLDMODEL_COMMIT,
        "dataset_name": str(cfg.dataset_name),
        "image_size": int(cfg.image_size),
        "patch_size": 14,
        "token_dim": backend.token_dim,
        "latent_dim": backend.latent_dim,
        "history_size": 3,
        "num_preds": 1,
        "frameskip": 5,
        "variants_per_window": len(SHIFT_NAMES),
        "appearance_severity": float(cfg.appearance.severity),
        "appearance_shift_names": json.dumps(
            list(SHIFT_NAMES),
            separators=(",", ":"),
        ),
        "appearance_pipeline_version": SHIFT_PIPELINE_VERSION,
        "window_selection_strategy": WINDOW_SELECTION_STRATEGY,
        "window_selection_seed": int(cfg.window_selection.seed),
        "source_window_count": len(source_dataset.clip_indices),
        "selected_window_count": len(selected_window_indices),
        "selected_window_indices_sha256": selected_indices_sha256,
        "normalization_method": "zscore",
        "action_normalization": json.dumps(
            action_normalization,
            separators=(",", ":"),
        ),
        "proprio_normalization": json.dumps(
            proprio_normalization,
            separators=(",", ":"),
        ),
    }
    with (
        torch.inference_mode(),
        FeatureCacheWriter(
            output_path,
            metadata=metadata,
            chunk_size=int(cfg.writer_chunk_size),
        ) as writer,
    ):
        for batch in tqdm(data_loader, desc=f"Caching {cfg.backend} features"):
            clean_pixels = _preprocess(batch["clean_pixels"], int(cfg.image_size))
            shifted_pixels = _preprocess(
                batch["shifted_pixels"], int(cfg.image_size)
            )
            batch_size, num_views, sequence_length = shifted_pixels.shape[:3]

            clean_prefix = _encode_chunks(
                backend,
                clean_pixels,
                int(cfg.encoder_batch_size),
                device,
                clean_target=False,
            )
            clean_targets = _encode_chunks(
                backend,
                clean_pixels,
                int(cfg.encoder_batch_size),
                device,
                clean_target=True,
            )
            shifted_flat = shifted_pixels.reshape(
                batch_size * num_views,
                sequence_length,
                *shifted_pixels.shape[-3:],
            )
            shifted_prefix = _encode_chunks(
                backend,
                shifted_flat,
                int(cfg.encoder_batch_size),
                device,
                clean_target=False,
            ).reshape(
                batch_size,
                num_views,
                sequence_length,
                clean_prefix.shape[-2],
                clean_prefix.shape[-1],
            )

            writer.append(
                {
                    "clean_prefix_tokens": clean_prefix,
                    "shifted_prefix_tokens": shifted_prefix,
                    "clean_targets": clean_targets,
                    "action": _normalize(
                        batch["action"], action_scaler, action_dim
                    ),
                    "proprio": _normalize(
                        batch["proprio"], proprio_scaler, proprio_dim
                    ),
                    "shift_type": batch["shift_type"],
                    "shift_seed": batch["shift_seed"],
                }
            )


if __name__ == "__main__":
    main()
