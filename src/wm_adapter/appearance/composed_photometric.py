from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


APPEARANCE_PIPELINE_VERSION = "composed_photometric_v1"


@dataclass(frozen=True)
class AppearanceShiftSpec:
    seed: int
    severity: float
    brightness: float
    contrast: float
    gamma: float
    rgb_gain: tuple[float, float, float]
    illumination_values: tuple[float, ...]
    illumination_height: int
    illumination_width: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComposedPhotometricShift:
    @staticmethod
    def metadata(severity: float, training_seed: int) -> dict[str, Any]:
        return {
            "name": "composed_photometric",
            "pipeline_version": APPEARANCE_PIPELINE_VERSION,
            "severity": severity,
            "training_seed": training_seed,
            "sequence_shared": True,
            "sampling": {
                "brightness_delta": [-0.25, 0.25],
                "contrast_delta": [-0.30, 0.30],
                "gamma_delta": [-0.30, 0.30],
                "rgb_gain_delta": [-0.20, 0.20],
                "illumination_delta": [-0.22, 0.22],
                "illumination_grid": [4, 4],
            },
            "spec_reconstruction": "sample_spec(seed=appearance_seed,severity=severity)",
        }

    def sample_spec(self, seed: int, severity: float) -> AppearanceShiftSpec:
        if not 0.0 <= severity <= 1.5:
            raise ValueError(f"Appearance severity must be in [0,1.5], received {severity}")
        generator = np.random.default_rng(seed)
        brightness = float(1.0 + severity * generator.uniform(-0.25, 0.25))
        contrast = float(1.0 + severity * generator.uniform(-0.30, 0.30))
        gamma = float(1.0 + severity * generator.uniform(-0.30, 0.30))
        rgb_gain = tuple(float(value) for value in 1.0 + severity * generator.uniform(-0.20, 0.20, size=3))
        illumination = severity * generator.uniform(-0.22, 0.22, size=(4, 4))
        return AppearanceShiftSpec(
            seed=seed,
            severity=severity,
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
            rgb_gain=rgb_gain,
            illumination_values=tuple(float(value) for value in illumination.reshape(-1)),
            illumination_height=4,
            illumination_width=4,
        )

    def apply(self, frames: Tensor, spec: AppearanceShiftSpec) -> Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Appearance input must be [T,3,H,W], received {tuple(frames.shape)}")
        input_dtype = frames.dtype
        values = frames.float()
        is_uint8 = input_dtype == torch.uint8
        if is_uint8:
            values = values.div(255.0)
        else:
            minimum = float(values.detach().amin().cpu())
            maximum = float(values.detach().amax().cpu())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(
                    f"Floating appearance input must be in [0,1], found range [{minimum}, {maximum}]"
                )
        values = values * spec.brightness
        mean = values.mean(dim=(-2, -1), keepdim=True)
        values = (values - mean) * spec.contrast + mean
        values = values.clamp(0.0, 1.0).pow(spec.gamma)
        gains = values.new_tensor(spec.rgb_gain).view(1, 3, 1, 1)
        values = values * gains
        illumination = values.new_tensor(spec.illumination_values).view(
            1, 1, spec.illumination_height, spec.illumination_width
        )
        illumination = F.interpolate(
            illumination,
            size=values.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        values = (values * (1.0 + illumination)).clamp(0.0, 1.0)
        if is_uint8:
            return values.mul(255.0).round().to(dtype=torch.uint8)
        return values.to(dtype=input_dtype)
