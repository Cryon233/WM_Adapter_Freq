from __future__ import annotations

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
from wm_adapter_freq.data.feature_cache import FeatureCacheWriter
from wm_adapter_freq.data.paired_windows import (
    build_image_preprocessor,
    load_paired_two_room_windows,
)


def _model_reference(value: str) -> str:
    path = Path(value).expanduser()
    return str(path) if path.exists() or value.startswith(("~", ".")) else value


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

    torch.manual_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    base_model = load_pretrained(_model_reference(str(cfg.base_model_ref)))
    base_model.eval()
    base_model.requires_grad_(False)
    base_model.to(device)
    backend = build_backend(str(cfg.backend), base_model)

    paired_dataset, source_dataset = load_paired_two_room_windows(
        str(cfg.dataset_name),
        cache_dir=cfg.get("dataset_cache_dir"),
        seed=int(cfg.seed),
    )
    max_windows = min(int(cfg.max_windows), len(paired_dataset))
    windows = Subset(paired_dataset, range(max_windows))
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

    output_path = Path(str(cfg.output_path)).expanduser()
    with (
        torch.inference_mode(),
        FeatureCacheWriter(
            output_path,
            backend=str(cfg.backend),
            chunk_size=1,
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
