from __future__ import annotations

import torch
from torch import Tensor, nn

from wm_adapter_freq.adapters.dct import OrthoDCT2d


class SequenceStableAdaptiveDCTAdapter(nn.Module):
    """Sequence-shared adaptive modulation of spatial patch frequencies."""

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        grid_height: int = 16,
        grid_width: int = 16,
        modulation_range: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.rank = rank
        self.modulation_range = modulation_range
        self.eps = eps

        self.dct = OrthoDCT2d(grid_height, grid_width)
        self.router = nn.Sequential(
            nn.Conv2d(embed_dim, rank, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                rank,
                rank,
                kernel_size=3,
                padding=1,
                groups=rank,
            ),
            nn.Conv2d(rank, embed_dim, kernel_size=1),
        )
        final_projection = self.router[-1]
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Adapt patch tokens shaped ``[B, T, P, D]``."""
        batch_size, sequence_length, num_patches, embed_dim = x.shape
        expected_patches = self.grid_height * self.grid_width
        if num_patches != expected_patches or embed_dim != self.embed_dim:
            raise ValueError(
                "Adapter input shape does not match its patch grid and embedding dimension."
            )

        grids = (
            x.permute(0, 1, 3, 2)
            .reshape(
                batch_size * sequence_length,
                embed_dim,
                self.grid_height,
                self.grid_width,
            )
            .contiguous()
        )
        coefficients = self.dct(grids)
        rms = torch.sqrt(
            coefficients.float().square().mean(dim=(-2, -1), keepdim=True)
            + self.eps
        ).to(coefficients.dtype)
        frame_logits = self.router(coefficients / rms)
        frame_logits = frame_logits.reshape(
            batch_size,
            sequence_length,
            embed_dim,
            self.grid_height,
            self.grid_width,
        )
        sequence_logits = frame_logits.mean(dim=1, keepdim=True)
        mask = 1.0 + self.modulation_range * torch.tanh(sequence_logits)

        coefficients = coefficients.reshape(
            batch_size,
            sequence_length,
            embed_dim,
            self.grid_height,
            self.grid_width,
        )
        adapted_coefficients = coefficients * mask
        coefficient_delta = (adapted_coefficients - coefficients).reshape(
            batch_size * sequence_length,
            embed_dim,
            self.grid_height,
            self.grid_width,
        )
        adapted = grids + self.dct.inverse(coefficient_delta)
        return (
            adapted.reshape(
                batch_size,
                sequence_length,
                embed_dim,
                num_patches,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
