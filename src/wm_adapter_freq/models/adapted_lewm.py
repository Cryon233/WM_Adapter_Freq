from __future__ import annotations

import torch
from torch import Tensor
from stable_worldmodel.wm.lewm.lewm import LeWM

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.encoders.lewm_vit_split import LeWMViTSplitEncoder


class AdaptedLeWM(LeWM):
    """LeWM with the shared DCT adapter before Tiny ViT's final block."""

    def __init__(
        self,
        base_model: LeWM,
        adapter: SequenceStableAdaptiveDCTAdapter,
    ) -> None:
        super().__init__(
            encoder=base_model.encoder,
            predictor=base_model.predictor,
            action_encoder=base_model.action_encoder,
            projector=base_model.projector,
            pred_proj=base_model.pred_proj,
        )
        self.split_encoder = LeWMViTSplitEncoder(self.encoder)
        self.adapter = adapter
        self.requires_grad_(False)
        self.adapter.requires_grad_(True)
        self.eval()

    def encode(self, info: dict[str, Tensor]) -> dict[str, Tensor]:
        pixels = info["pixels"].to(next(self.encoder.parameters()).dtype)
        prefix_tokens = self.split_encoder.encode_prefix(pixels)
        cls_tokens = prefix_tokens[:, :, :1]
        patch_tokens = self.adapter(prefix_tokens[:, :, 1:])
        final_tokens = self.split_encoder.forward_tail(
            torch.cat((cls_tokens, patch_tokens), dim=2)
        )
        cls_latent = final_tokens[:, :, 0]
        projected = self.projector(
            cls_latent.reshape(-1, cls_latent.shape[-1])
        )
        info["emb"] = projected.reshape(
            cls_latent.shape[0], cls_latent.shape[1], -1
        )
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info
