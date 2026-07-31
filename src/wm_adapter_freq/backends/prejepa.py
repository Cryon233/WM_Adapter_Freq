from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.backends.base import BaseWorldModelBackend
from wm_adapter_freq.encoders.dinov2_split import DINOv2SplitEncoder


class PreJEPABackend(BaseWorldModelBackend):
    """Adapter integration for stable-worldmodel's PreJEPA/DINO-WM."""

    latent_type = "patch"

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__(base_model)
        self.split_encoder = DINOv2SplitEncoder(base_model.backbone)
        self.token_dim = int(base_model.backbone.config.hidden_size)
        self.latent_dim = self.token_dim

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
        return final_tokens[:, :, 1:]

    def encode_clean_target(self, pixels: Tensor) -> Tensor:
        final_tokens = self.split_encoder.forward_tail(
            self.split_encoder.encode_prefix(pixels)
        )
        return final_tokens[:, :, 1:]

    def predict_next(
        self,
        adapted_latent_history: Tensor,
        action: Tensor,
        proprio: Tensor | None = None,
    ) -> Tensor:
        num_patches = adapted_latent_history.shape[2]
        embedding_parts = [adapted_latent_history]
        inputs = {"action": action, "proprio": proprio}
        for key, encoder in self.base_model.extra_encoders.items():
            extra = inputs.get(key)
            if extra is None:
                raise ValueError(f"Missing {key} input required by PreJEPA.")
            extra_embedding = encoder(extra.float())
            extra_embedding = extra_embedding.unsqueeze(2).expand(
                -1, -1, num_patches, -1
            )
            embedding_parts.append(extra_embedding)
        predictor_input = torch.cat(embedding_parts, dim=-1)
        return self.base_model.predict(predictor_input)[..., : self.latent_dim]

    def build_online_model(
        self, adapter: SequenceStableAdaptiveDCTAdapter
    ) -> nn.Module:
        from wm_adapter_freq.models.adapted_prejepa import AdaptedPreJEPA

        return AdaptedPreJEPA(self.base_model, adapter)

    def training_modules(self) -> Iterable[nn.Module]:
        return (
            self.split_encoder.last_block,
            self.split_encoder.final_norm,
            self.base_model.predictor,
            self.base_model.extra_encoders,
        )
