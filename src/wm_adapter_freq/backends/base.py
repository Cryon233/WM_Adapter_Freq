from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)


class BaseWorldModelBackend(ABC):
    """Small common surface for the two supported world-model families."""

    latent_type: str
    latent_dim: int
    token_dim: int
    grid_height: int = 16
    grid_width: int = 16

    def __init__(self, base_model: nn.Module) -> None:
        self.base_model = base_model
        self.base_model.eval()
        self.base_model.requires_grad_(False)

    @abstractmethod
    def encode_prefix(self, pixels: Tensor) -> Tensor:
        """Return visual tokens immediately before the final ViT block."""

    @abstractmethod
    def apply_adapter_and_tail(
        self,
        prefix_tokens: Tensor,
        adapter: SequenceStableAdaptiveDCTAdapter,
    ) -> Tensor:
        """Adapt patches, run the frozen tail, and perform latent readout."""

    @abstractmethod
    def encode_clean_target(self, pixels: Tensor) -> Tensor:
        """Encode clean pixels without an adapter."""

    @abstractmethod
    def predict_next(
        self,
        adapted_latent_history: Tensor,
        action: Tensor,
        proprio: Tensor | None = None,
    ) -> Tensor:
        """Run the original frozen dynamics path."""

    @abstractmethod
    def build_online_model(
        self, adapter: SequenceStableAdaptiveDCTAdapter
    ) -> nn.Module:
        """Build an upstream planning-compatible adapted model."""

    @abstractmethod
    def training_modules(self) -> Iterable[nn.Module]:
        """Modules needed on the training device for cached-prefix training."""

    def move_training_modules(self, device: torch.device | str) -> None:
        for module in self.training_modules():
            module.to(device)
            module.eval()
            module.requires_grad_(False)


def build_backend(
    backend: str, base_model: nn.Module
) -> BaseWorldModelBackend:
    """Construct a backend for a loaded stable-worldmodel checkpoint."""
    if backend == "prejepa":
        from wm_adapter_freq.backends.prejepa import PreJEPABackend

        return PreJEPABackend(base_model)
    if backend == "lewm":
        from wm_adapter_freq.backends.lewm import LeWMBackend

        return LeWMBackend(base_model)
    raise ValueError(f"Unsupported backend: {backend}")
