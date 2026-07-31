from __future__ import annotations

import torch
import torch.nn.functional as F
from stable_worldmodel.wm.prejepa.prejepa import PreJEPA
from torch import Tensor


def visual_terminal_goal_mse(
    info_dict: dict[str, Tensor],
) -> Tensor:
    """Compute terminal MSE using only the visual latent."""
    predicted = info_dict["predicted_pixels_emb"]
    goal = info_dict["pixels_goal_emb"]
    cost = F.mse_loss(
        predicted[:, :, -1:],
        goal,
        reduction="none",
    )
    return cost.mean(dim=tuple(range(2, predicted.ndim)))


class VisualGoalPreJEPA(PreJEPA):
    """PreJEPA planning wrapper with a visual-only terminal goal cost."""

    def __init__(self, base_model: PreJEPA) -> None:
        super().__init__(
            encoder=base_model.backbone,
            predictor=base_model.predictor,
            extra_encoders=base_model.extra_encoders,
            decoder=base_model.decoder,
            history_size=base_model.history_size,
            num_pred=base_model.num_pred,
            interpolate_pos_encoding=base_model.interpolate_pos_encoding,
        )
        self.requires_grad_(False)
        self.eval()

    def criterion(
        self,
        info_dict: dict[str, Tensor],
        action_candidates: Tensor,
    ) -> Tensor:
        return visual_terminal_goal_mse(info_dict)
