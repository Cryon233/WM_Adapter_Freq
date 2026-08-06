from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from wm_adapter.appearance.composed_photometric import (
    ComposedPhotometricShift,
)


SUPPORTED_EVALUATION_FAMILIES = (
    "identity",
    "photometric",
    "gaussian_blur",
    "gaussian_noise",
    "dct_compression",
)


class EvaluationCorruption(ABC):
    family: str

    def __init__(self, *, seed: int, strength: float) -> None:
        if not 0.0 <= strength <= 2.0:
            raise ValueError(
                f"Evaluation corruption strength must be in [0,2], "
                f"received {strength}"
            )
        self.seed = int(seed)
        self.strength = float(strength)

    @abstractmethod
    def apply(self, frames: Tensor) -> Tensor:
        """Apply one episode-fixed corruption to [T,3,H,W] RGB frames."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return the complete deterministic evaluation contract."""


class IdentityCorruption(EvaluationCorruption):
    family = "identity"

    def apply(self, frames: Tensor) -> Tensor:
        return frames

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "strength": self.strength,
            "sequence_shared": True,
        }


class PhotometricCorruption(EvaluationCorruption):
    family = "photometric"

    def __init__(self, *, seed: int, strength: float) -> None:
        super().__init__(seed=seed, strength=strength)
        self.transform = ComposedPhotometricShift()
        self.spec = self.transform.sample_spec(seed, strength)

    def apply(self, frames: Tensor) -> Tensor:
        return self.transform.apply(frames, self.spec)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "strength": self.strength,
            "sequence_shared": True,
            "parameters": self.spec.as_dict(),
        }


def _float_rgb(frames: Tensor) -> tuple[Tensor, torch.dtype]:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(
            f"Evaluation corruption expects [T,3,H,W], "
            f"received {tuple(frames.shape)}"
        )
    input_dtype = frames.dtype
    values = frames.float()
    if input_dtype == torch.uint8:
        values = values.div(255.0)
    else:
        minimum = float(values.detach().amin().cpu())
        maximum = float(values.detach().amax().cpu())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "Floating RGB evaluation input must be in [0,1], "
                f"found [{minimum}, {maximum}]"
            )
    return values, input_dtype


def _restore_rgb(values: Tensor, dtype: torch.dtype) -> Tensor:
    values = values.clamp(0.0, 1.0)
    if dtype == torch.uint8:
        return values.mul(255.0).round().to(dtype=torch.uint8)
    return values.to(dtype=dtype)


class GaussianBlurCorruption(EvaluationCorruption):
    family = "gaussian_blur"

    def __init__(self, *, seed: int, strength: float) -> None:
        super().__init__(seed=seed, strength=strength)
        self.sigma = 0.5 + 1.5 * strength
        radius = max(1, int(math.ceil(3.0 * self.sigma)))
        self.kernel_size = 2 * radius + 1

    def apply(self, frames: Tensor) -> Tensor:
        values, dtype = _float_rgb(frames)
        radius = self.kernel_size // 2
        coordinates = torch.arange(
            -radius,
            radius + 1,
            device=values.device,
            dtype=torch.float32,
        )
        kernel_1d = torch.exp(
            -(coordinates.square()) / (2.0 * self.sigma * self.sigma)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.expand(3, 1, -1, -1).contiguous()
        padded = F.pad(
            values,
            (radius, radius, radius, radius),
            mode="reflect",
        )
        blurred = F.conv2d(padded, kernel, groups=3)
        return _restore_rgb(blurred, dtype)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "strength": self.strength,
            "sigma": self.sigma,
            "kernel_size": self.kernel_size,
            "sequence_shared": True,
        }


class GaussianNoiseCorruption(EvaluationCorruption):
    family = "gaussian_noise"

    def __init__(self, *, seed: int, strength: float) -> None:
        super().__init__(seed=seed, strength=strength)
        self.standard_deviation = 0.06 * strength

    def apply(self, frames: Tensor) -> Tensor:
        values, dtype = _float_rgb(frames)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        field = torch.randn(
            (1, values.shape[1], values.shape[2], values.shape[3]),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=values.device)
        noisy = values + self.standard_deviation * field
        return _restore_rgb(noisy, dtype)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "strength": self.strength,
            "standard_deviation": self.standard_deviation,
            "noise_field": "episode_fixed_spatial_gaussian",
            "sequence_shared": True,
        }


def _dct_matrix(device: torch.device) -> Tensor:
    indices = torch.arange(8, device=device, dtype=torch.float32)
    k = indices[:, None]
    n = indices[None, :]
    matrix = torch.cos(math.pi * (2.0 * n + 1.0) * k / 16.0)
    matrix[0] *= 1.0 / math.sqrt(8.0)
    matrix[1:] *= math.sqrt(2.0 / 8.0)
    return matrix


class DCTCompressionCorruption(EvaluationCorruption):
    family = "dct_compression"
    _BASE_QUANTIZATION = (
        (16, 11, 10, 16, 24, 40, 51, 61),
        (12, 12, 14, 19, 26, 58, 60, 55),
        (14, 13, 16, 24, 40, 57, 69, 56),
        (14, 17, 22, 29, 51, 87, 80, 62),
        (18, 22, 37, 56, 68, 109, 103, 77),
        (24, 35, 55, 64, 81, 104, 113, 92),
        (49, 64, 78, 87, 103, 121, 120, 101),
        (72, 92, 95, 98, 112, 100, 103, 99),
    )

    def __init__(self, *, seed: int, strength: float) -> None:
        super().__init__(seed=seed, strength=strength)
        self.quality = int(round(95.0 - 55.0 * strength))
        self.quality = max(10, min(95, self.quality))

    def _quantization_table(self, device: torch.device) -> Tensor:
        base = torch.tensor(
            self._BASE_QUANTIZATION,
            device=device,
            dtype=torch.float32,
        )
        scale = (
            5000.0 / self.quality
            if self.quality < 50
            else 200.0 - 2.0 * self.quality
        )
        return ((base * scale + 50.0) / 100.0).floor().clamp(1, 255)

    def apply(self, frames: Tensor) -> Tensor:
        values, dtype = _float_rgb(frames)
        height, width = values.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        if pad_height or pad_width:
            values = F.pad(
                values,
                (0, pad_width, 0, pad_height),
                mode="reflect",
            )
        centered = values.mul(255.0).sub(128.0)
        blocks = centered.unfold(2, 8, 8).unfold(3, 8, 8)
        dct = _dct_matrix(values.device)
        coefficients = torch.einsum(
            "ab,...bc,cd->...ad", dct, blocks, dct.t()
        )
        table = self._quantization_table(values.device)
        quantized = torch.round(coefficients / table) * table
        reconstructed = torch.einsum(
            "ab,...bc,cd->...ad", dct.t(), quantized, dct
        )
        reconstructed = reconstructed.add(128.0).div(255.0)
        reconstructed = reconstructed.permute(0, 1, 2, 4, 3, 5)
        reconstructed = reconstructed.reshape(
            values.shape[0],
            values.shape[1],
            values.shape[2],
            values.shape[3],
        )
        reconstructed = reconstructed[..., :height, :width]
        return _restore_rgb(reconstructed, dtype)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "strength": self.strength,
            "quality": self.quality,
            "block_size": 8,
            "implementation": "rgb_block_dct_quantization",
            "sequence_shared": True,
        }


def build_evaluation_corruption(
    *, family: str, seed: int, strength: float
) -> EvaluationCorruption:
    normalized = str(family).strip().lower()
    aliases = {
        "clean": "identity",
        "none": "identity",
        "composed_photometric": "photometric",
        "blur": "gaussian_blur",
        "noise": "gaussian_noise",
        "compression": "dct_compression",
        "jpeg": "dct_compression",
    }
    normalized = aliases.get(normalized, normalized)
    classes = {
        "identity": IdentityCorruption,
        "photometric": PhotometricCorruption,
        "gaussian_blur": GaussianBlurCorruption,
        "gaussian_noise": GaussianNoiseCorruption,
        "dct_compression": DCTCompressionCorruption,
    }
    if normalized not in classes:
        raise ValueError(
            f"Unsupported evaluation corruption family {family!r}; "
            f"expected one of {SUPPORTED_EVALUATION_FAMILIES}"
        )
    return classes[normalized](seed=int(seed), strength=float(strength))


def evaluation_corruption_metadata(
    *, family: str, seed: int, strength: float
) -> dict[str, Any]:
    return build_evaluation_corruption(
        family=family,
        seed=seed,
        strength=strength,
    ).as_dict()
