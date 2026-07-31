from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from wm_adapter.adapters.base import PEFTMethod


class OrthoDCT2d(nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError(f"DCT grid must be positive, received height={height}, width={width}")
        self.height = height
        self.width = width
        self.register_buffer("dct_height", self._matrix(height), persistent=True)
        self.register_buffer("dct_width", self._matrix(width), persistent=True)

    @staticmethod
    def _matrix(size: int) -> Tensor:
        frequencies = torch.arange(size, dtype=torch.float32).unsqueeze(1)
        positions = torch.arange(size, dtype=torch.float32).unsqueeze(0)
        matrix = torch.cos(math.pi / size * (positions + 0.5) * frequencies)
        scale = torch.full((size,), math.sqrt(2.0 / size), dtype=torch.float32)
        scale[0] = math.sqrt(1.0 / size)
        return matrix * scale.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"OrthoDCT2d expected [B,C,{self.height},{self.width}], received {tuple(x.shape)}"
            )
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = x.float()
            coefficients = torch.matmul(self.dct_height.float(), values)
            coefficients = torch.matmul(coefficients, self.dct_width.float().transpose(0, 1))
        return coefficients.to(dtype=dtype)

    def inverse(self, coefficients: Tensor) -> Tensor:
        if coefficients.ndim != 4 or coefficients.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"OrthoDCT2d.inverse expected [B,C,{self.height},{self.width}], "
                f"received {tuple(coefficients.shape)}"
            )
        dtype = coefficients.dtype
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            values = coefficients.float()
            reconstructed = torch.matmul(self.dct_height.float().transpose(0, 1), values)
            reconstructed = torch.matmul(reconstructed, self.dct_width.float())
        return reconstructed.to(dtype=dtype)


class SequenceStableAdaptiveDCTAdapter(PEFTMethod):
    method_name = "dct_adapter"

    def __init__(
        self,
        embed_dim: int,
        grid_height: int,
        grid_width: int,
        rank: int = 8,
        mask_scale: float = 0.5,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or rank <= 0:
            raise ValueError(f"embed_dim and rank must be positive, received {embed_dim}, {rank}")
        self.embed_dim = embed_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.rank = rank
        self.mask_scale = mask_scale
        self.eps = eps
        self.dct = OrthoDCT2d(grid_height, grid_width)
        self.router = nn.Sequential(
            nn.Conv2d(embed_dim, rank, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank),
            nn.Conv2d(rank, embed_dim, kernel_size=1),
        )
        output = self.router[-1]
        if not isinstance(output, nn.Conv2d):
            raise TypeError(f"Expected final router module to be Conv2d, found {type(output).__name__}")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def apply_patch_tokens(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError(f"DCT adapter expected [B,T,P,D], received {tuple(patch_tokens.shape)}")
        batch, time, patches, dimension = patch_tokens.shape
        expected_patches = self.grid_height * self.grid_width
        if patches != expected_patches or dimension != self.embed_dim:
            raise ValueError(
                f"DCT adapter expected P={expected_patches}, D={self.embed_dim}; "
                f"received {tuple(patch_tokens.shape)}"
            )
        spatial = patch_tokens.reshape(batch * time, patches, dimension).transpose(1, 2)
        spatial = spatial.reshape(batch * time, dimension, self.grid_height, self.grid_width)
        coefficients = self.dct(spatial)
        rms = torch.sqrt(coefficients.float().square().mean(dim=(-2, -1), keepdim=True) + self.eps)
        normalized = coefficients.float() / rms
        frame_logits = self.router(normalized.to(dtype=next(self.router.parameters()).dtype))
        frame_logits = frame_logits.float().reshape(
            batch, time, dimension, self.grid_height, self.grid_width
        )
        sequence_logits = frame_logits.mean(dim=1, keepdim=True)
        mask = 1.0 + self.mask_scale * torch.tanh(sequence_logits)
        adapted = coefficients.float().reshape(
            batch, time, dimension, self.grid_height, self.grid_width
        ) * mask
        reconstructed = self.dct.inverse(
            adapted.reshape(batch * time, dimension, self.grid_height, self.grid_width)
        )
        reconstructed = reconstructed.reshape(batch * time, dimension, patches).transpose(1, 2)
        return reconstructed.reshape(batch, time, patches, dimension).to(dtype=patch_tokens.dtype)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        return self.apply_patch_tokens(patch_tokens)

    def config_dict(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "rank": self.rank,
            "mask_scale": self.mask_scale,
            "eps": self.eps,
        }
