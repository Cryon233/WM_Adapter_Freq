from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.backends.base import BaseWorldModelBackend
from wm_adapter_freq.encoders.lewm_vit_split import LeWMViTSplitEncoder


class LeWMBackend(BaseWorldModelBackend):
    """Adapter integration for stable-worldmodel's LeWM."""

    latent_type = "global"

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__(base_model)
        self.split_encoder = LeWMViTSplitEncoder(base_model.encoder)
        self.token_dim = int(base_model.encoder.config.hidden_size)
        self.latent_dim = int(
            getattr(base_model.projector, "output_dim", self.token_dim)
        )

    def encode_prefix(self, pixels: Tensor) -> Tensor:
        return self.split_encoder.encode_prefix(pixels)

    def apply_adapter_and_tail(
        self,
        prefix_tokens: Tensor,
        adapter: SequenceStableAdaptiveDCTAdapter,
    ) -> Tensor:
        cls_tokens = prefix_tokens[:, :, :1]
        patch_tokens = adapter(prefix_tokens[:, :, 1:])
        final_tokens = self.split_encoder.forward_tail(
            torch.cat((cls_tokens, patch_tokens), dim=2)
        )
        cls_latent = final_tokens[:, :, 0]
        flat_latent = cls_latent.reshape(-1, cls_latent.shape[-1])
        projected = self.base_model.projector(flat_latent)
        return projected.reshape(
            cls_latent.shape[0], cls_latent.shape[1], -1
        )

    def encode_clean_target(self, pixels: Tensor) -> Tensor:
        final_tokens = self.split_encoder.forward_tail(
            self.split_encoder.encode_prefix(pixels)
        )
        cls_latent = final_tokens[:, :, 0]
        projected = self.base_model.projector(
            cls_latent.reshape(-1, cls_latent.shape[-1])
        )
        return projected.reshape(
            cls_latent.shape[0], cls_latent.shape[1], -1
        )

    def predict_next(
        self,
        adapted_latent_history: Tensor,
        action: Tensor,
        proprio: Tensor | None = None,
    ) -> Tensor:
        action_embedding = self.base_model.action_encoder(action.float())
        return self.base_model.predict(
            adapted_latent_history, action_embedding
        )

    def build_online_model(
        self, adapter: SequenceStableAdaptiveDCTAdapter
    ) -> nn.Module:
        from wm_adapter_freq.models.adapted_lewm import AdaptedLeWM

        return AdaptedLeWM(self.base_model, adapter)

    def training_modules(self) -> Iterable[nn.Module]:
        return (
            self.split_encoder.last_block,
            self.split_encoder.final_norm,
            self.base_model.projector,
            self.base_model.predictor,
            self.base_model.action_encoder,
            self.base_model.pred_proj,
        )
