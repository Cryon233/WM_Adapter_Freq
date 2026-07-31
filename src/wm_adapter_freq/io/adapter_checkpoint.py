from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)


def save_adapter_checkpoint(
    path: str | Path,
    adapter: SequenceStableAdaptiveDCTAdapter,
    backend: str,
    base_model_ref: str,
    token_dim: int,
    latent_dim: int,
    history_size: int = 3,
    image_size: int = 224,
    patch_size: int = 14,
) -> None:
    """Save only adapter weights and the metadata needed to reattach them."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "adapter_state_dict": {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()
        },
        "adapter_config": {
            "embed_dim": adapter.embed_dim,
            "grid_height": adapter.grid_height,
            "grid_width": adapter.grid_width,
            "rank": adapter.rank,
            "modulation_range": adapter.modulation_range,
            "eps": adapter.eps,
        },
        "backend": backend,
        "base_model_ref": base_model_ref,
        "history_size": history_size,
        "image_size": image_size,
        "patch_size": patch_size,
        "token_dim": token_dim,
        "latent_dim": latent_dim,
    }
    torch.save(checkpoint, output_path)


def load_adapter_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[SequenceStableAdaptiveDCTAdapter, dict[str, Any]]:
    """Load an adapter and return it together with checkpoint metadata."""
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )
    adapter = SequenceStableAdaptiveDCTAdapter(
        **checkpoint["adapter_config"]
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.to(device)
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"adapter_state_dict"}
    }
    return adapter, metadata
