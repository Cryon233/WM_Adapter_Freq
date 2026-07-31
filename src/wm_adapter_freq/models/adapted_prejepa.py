from __future__ import annotations

import torch
from torch import Tensor
from stable_worldmodel.wm.prejepa.prejepa import PreJEPA

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.encoders.dinov2_split import DINOv2SplitEncoder
from wm_adapter_freq.models.prejepa_visual_goal import (
    visual_terminal_goal_mse,
)


class AdaptedPreJEPA(PreJEPA):
    """PreJEPA with the shared DCT adapter before DINOv2's final block."""

    def __init__(
        self,
        base_model: PreJEPA,
        adapter: SequenceStableAdaptiveDCTAdapter,
    ) -> None:
        super().__init__(
            encoder=base_model.backbone,
            predictor=base_model.predictor,
            extra_encoders=base_model.extra_encoders,
            decoder=base_model.decoder,
            history_size=base_model.history_size,
            num_pred=base_model.num_pred,
            interpolate_pos_encoding=base_model.interpolate_pos_encoding,
        )
        self.split_encoder = DINOv2SplitEncoder(self.backbone)
        self.adapter = adapter
        self.requires_grad_(False)
        self.adapter.requires_grad_(True)
        self.eval()

    def _encode_image(self, pixels: Tensor) -> Tensor:
        prefix_tokens = self.split_encoder.encode_prefix(pixels)
        cls_tokens = prefix_tokens[:, :, :1]
        patch_tokens = self.adapter(prefix_tokens[:, :, 1:])
        final_tokens = self.split_encoder.forward_tail(
            torch.cat((cls_tokens, patch_tokens), dim=2)
        )
        return final_tokens[:, :, 1:]

    def criterion(
        self,
        info_dict: dict[str, Tensor],
        action_candidates: Tensor,
    ) -> Tensor:
        return visual_terminal_goal_mse(info_dict)
