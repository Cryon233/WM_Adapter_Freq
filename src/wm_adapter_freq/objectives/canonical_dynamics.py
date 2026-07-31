from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.backends.base import BaseWorldModelBackend


@dataclass(frozen=True)
class CanonicalDynamicsLoss:
    total: Tensor
    canonical: Tensor
    dynamics: Tensor


class CanonicalDynamicsObjective(nn.Module):
    """Canonical alignment plus frozen world-model dynamics alignment."""

    def __init__(
        self,
        backend: BaseWorldModelBackend,
        adapter: SequenceStableAdaptiveDCTAdapter,
        canonical_weight: float = 1.0,
        dynamics_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.adapter = adapter
        self.canonical_weight = canonical_weight
        self.dynamics_weight = dynamics_weight

    def forward(self, batch: dict[str, Tensor]) -> CanonicalDynamicsLoss:
        prefix_tokens = batch["prefix_tokens"]
        clean_targets = batch["clean_targets"]

        context_latent = self.backend.apply_adapter_and_tail(
            prefix_tokens[:, :3], self.adapter
        )
        goal_latent = self.backend.apply_adapter_and_tail(
            prefix_tokens[:, 3:4], self.adapter
        )
        adapted_latent = torch.cat((context_latent, goal_latent), dim=1)
        target = clean_targets.to(adapted_latent.dtype)
        per_frame = F.mse_loss(
            adapted_latent,
            target,
            reduction="none",
        ).flatten(start_dim=2).mean(dim=2)
        canonical_loss = per_frame.mean()

        predicted_visual = self.backend.predict_next(
            context_latent,
            batch["action"][:, :3],
            batch["proprio"][:, :3],
        )
        dynamics_target = clean_targets[:, 1:4].to(predicted_visual.dtype)
        dynamics_loss = F.mse_loss(predicted_visual, dynamics_target)

        total_loss = (
            self.canonical_weight * canonical_loss
            + self.dynamics_weight * dynamics_loss
        )
        return CanonicalDynamicsLoss(
            total=total_loss,
            canonical=canonical_loss,
            dynamics=dynamics_loss,
        )
