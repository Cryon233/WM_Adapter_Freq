from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from wm_adapter.adapters.base import BaseMethod, PEFTMethod
from wm_adapter.adapters.dct_adapter import SequenceStableAdaptiveDCTAdapter
from wm_adapter.adapters.hfra import HFRACoreOnlyAdapter, HybridFourierResidualAdapter
from wm_adapter.adapters.lora import LastBlockAttentionLoRA
from wm_adapter.adapters.token_mlp import TokenMLPAdapter


def _plain_config(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
        if not isinstance(value, dict):
            raise TypeError(f"Method config must resolve to a mapping, found {type(value).__name__}")
        return value
    return dict(config)


def build_method(
    method_name: str,
    backend: Any,
    config: DictConfig | dict[str, Any],
) -> PEFTMethod:
    values = _plain_config(config)
    declared_name = str(values.pop("name", method_name))
    if declared_name != method_name:
        raise ValueError(f"Requested method {method_name!r}, but config declares {declared_name!r}")
    if method_name == "base":
        method: PEFTMethod = BaseMethod()
    elif method_name == "dct_adapter":
        method = SequenceStableAdaptiveDCTAdapter(
            embed_dim=backend.token_dim,
            grid_height=backend.grid_height,
            grid_width=backend.grid_width,
            rank=int(values["rank"]),
            mask_scale=float(values["mask_scale"]),
            eps=float(values["eps"]),
            temporal_pool=str(values.get("temporal_pool", "mean")),
            mask_type=str(values.get("mask_type", "adaptive")),
            use_rms_norm=bool(values.get("use_rms_norm", True)),
        )
    elif method_name == "token_mlp":
        method = TokenMLPAdapter(embed_dim=backend.token_dim, rank=int(values["rank"]))
    elif method_name == "lora":
        method = LastBlockAttentionLoRA(
            embed_dim=backend.token_dim,
            rank=int(values["rank"]),
            alpha=float(values["alpha"]),
            dropout=float(values["dropout"]),
        )
    elif method_name in {"hfra", "hfra_core_only"}:
        adapter_class = (
            HybridFourierResidualAdapter
            if method_name == "hfra"
            else HFRACoreOnlyAdapter
        )
        method = adapter_class(
            embed_dim=backend.token_dim,
            grid_height=backend.grid_height,
            grid_width=backend.grid_width,
            num_encoder_blocks=backend.num_encoder_blocks,
            rank=int(values["rank"]),
        )
    else:
        raise ValueError(
            f"Unknown method {method_name!r}; expected base, dct_adapter, "
            "token_mlp, lora, hfra, or hfra_core_only"
        )
    method.attach_backend(backend)
    return method
