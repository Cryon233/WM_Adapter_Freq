from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class OrthoDCT2d(nn.Module):
    """Fixed orthonormal two-dimensional DCT-II."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.register_buffer("dct_height", self._build_matrix(height))
        self.register_buffer("dct_width", self._build_matrix(width))

    @staticmethod
    def _build_matrix(size: int) -> Tensor:
        n = torch.arange(size, dtype=torch.float32)
        k = torch.arange(size, dtype=torch.float32).unsqueeze(1)
        matrix = torch.cos(math.pi / size * (n + 0.5) * k)
        scale = torch.full((size,), math.sqrt(2.0 / size))
        scale[0] = math.sqrt(1.0 / size)
        return matrix * scale.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``[B, C, H, W]`` spatial tensors to DCT coefficients."""
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = x.float()
            coefficients = torch.matmul(self.dct_height.float(), values)
            coefficients = torch.matmul(
                coefficients, self.dct_width.float().transpose(0, 1)
            )
        return coefficients.to(input_dtype)

    def inverse(self, coefficients: Tensor) -> Tensor:
        """Invert orthonormal coefficients with ``D_H.T @ C @ D_W``."""
        input_dtype = coefficients.dtype
        with torch.autocast(
            device_type=coefficients.device.type,
            enabled=False,
        ):
            values = coefficients.float()
            spatial = torch.matmul(
                self.dct_height.float().transpose(0, 1), values
            )
            spatial = torch.matmul(spatial, self.dct_width.float())
        return spatial.to(input_dtype)
