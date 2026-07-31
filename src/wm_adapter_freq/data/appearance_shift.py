from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


SHIFT_NAMES = (
    "photometric",
    "background_texture",
    "palette_shift",
    "composed",
)
TEXTURE_NAMES = (
    "checkerboard",
    "horizontal_stripes",
    "vertical_stripes",
    "smooth_noise",
)


@dataclass(frozen=True)
class AppearanceShiftSpec:
    shift_type: str
    seed: int
    severity: float
    brightness: float
    contrast: float
    gamma: float
    rgb_gain: tuple[float, float, float]
    texture_type: str
    texture_scale: float
    texture_strength: float
    palette_seed: int


class TwoRoomAppearanceShift:
    """Deterministic appearance-only transformations for TwoRoom frames."""

    @staticmethod
    def _uniform(
        generator: torch.Generator,
        low: float,
        high: float,
    ) -> float:
        value = torch.empty((), dtype=torch.float32)
        return float(value.uniform_(low, high, generator=generator).item())

    def sample_spec(
        self,
        shift_type: str,
        seed: int,
        severity: float,
    ) -> AppearanceShiftSpec:
        if shift_type not in SHIFT_NAMES:
            raise ValueError(f"Unknown appearance shift type: {shift_type}")
        if severity < 0.0:
            raise ValueError("severity must be non-negative")

        generator = torch.Generator().manual_seed(int(seed))
        texture_index = int(
            torch.randint(
                len(TEXTURE_NAMES),
                (),
                generator=generator,
            ).item()
        )
        palette_seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (),
                generator=generator,
            ).item()
        )
        return AppearanceShiftSpec(
            shift_type=shift_type,
            seed=int(seed),
            severity=float(severity),
            brightness=1.0
            + severity * self._uniform(generator, -0.25, 0.25),
            contrast=1.0
            + severity * self._uniform(generator, -0.30, 0.30),
            gamma=1.0
            + severity * self._uniform(generator, -0.30, 0.40),
            rgb_gain=tuple(
                1.0 + severity * self._uniform(generator, -0.25, 0.25)
                for _ in range(3)
            ),
            texture_type=TEXTURE_NAMES[texture_index],
            texture_scale=self._uniform(generator, 10.0, 28.0),
            texture_strength=min(0.45, 0.28 * severity),
            palette_seed=palette_seed,
        )

    @staticmethod
    def _unit_from_seed(seed: int, offset: int) -> float:
        value = (
            int(seed)
            + 0x9E3779B97F4A7C15 * (int(offset) + 1)
        ) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 31
        return float(value & 0xFFFFFF) / float(0xFFFFFF)

    @classmethod
    def _palette_colors(
        cls,
        spec: AppearanceShiftSpec,
        reference: Tensor,
    ) -> Tensor:
        unit = [
            cls._unit_from_seed(spec.palette_seed, index)
            for index in range(12)
        ]
        sampled = reference.new_tensor(
            (
                (
                    0.35 + 0.65 * unit[0],
                    0.10 + 0.75 * unit[1],
                    0.10 + 0.75 * unit[2],
                ),
                (
                    0.10 + 0.75 * unit[3],
                    0.35 + 0.65 * unit[4],
                    0.10 + 0.75 * unit[5],
                ),
                (
                    0.02 + 0.38 * unit[6],
                    0.02 + 0.38 * unit[7],
                    0.02 + 0.38 * unit[8],
                ),
                (
                    0.55 + 0.45 * unit[9],
                    0.55 + 0.45 * unit[10],
                    0.55 + 0.45 * unit[11],
                ),
            )
        )
        return sampled

    @staticmethod
    def _prototype_masks(reference: Tensor) -> Tensor:
        prototypes = reference.new_tensor(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
            )
        )
        colors = reference.permute(0, 2, 3, 1).unsqueeze(-2)
        squared_distance = (colors - prototypes).square().sum(dim=-1)
        return torch.exp(-squared_distance / (2.0 * 0.35**2))

    @classmethod
    def _palette_shift(
        cls,
        pixels: Tensor,
        reference: Tensor,
        spec: AppearanceShiftSpec,
    ) -> Tensor:
        masks = cls._prototype_masks(reference)
        mask_mass = masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = masks / mask_mass
        colors = cls._palette_colors(spec, reference)
        recolored = torch.einsum("thwk,kc->tchw", weights, colors)
        strength = min(1.0, spec.severity)
        return (pixels + strength * (recolored - pixels)).clamp(0.0, 1.0)

    @staticmethod
    def _texture_pattern(
        reference: Tensor,
        spec: AppearanceShiftSpec,
    ) -> Tensor:
        height, width = reference.shape[-2:]
        y = torch.arange(
            height,
            device=reference.device,
            dtype=reference.dtype,
        )
        x = torch.arange(
            width,
            device=reference.device,
            dtype=reference.dtype,
        )
        scale = max(spec.texture_scale, 1.0)
        if spec.texture_type == "checkerboard":
            pattern = torch.remainder(
                torch.floor(y[:, None] / scale)
                + torch.floor(x[None, :] / scale),
                2.0,
            )
        elif spec.texture_type == "horizontal_stripes":
            pattern = torch.floor(y[:, None] / scale).remainder(2.0).expand(
                height,
                width,
            )
        elif spec.texture_type == "vertical_stripes":
            pattern = torch.floor(x[None, :] / scale).remainder(2.0).expand(
                height,
                width,
            )
        elif spec.texture_type == "smooth_noise":
            phase_x = 2.0 * math.pi * TwoRoomAppearanceShift._unit_from_seed(
                spec.seed,
                31,
            )
            phase_y = 2.0 * math.pi * TwoRoomAppearanceShift._unit_from_seed(
                spec.seed,
                37,
            )
            field = (
                torch.sin(
                    2.0 * math.pi * x[None, :] / (3.1 * scale)
                    + phase_x
                )
                + torch.cos(
                    2.0 * math.pi * y[:, None] / (2.3 * scale)
                    + phase_y
                )
                + torch.sin(
                    2.0
                    * math.pi
                    * (x[None, :] + y[:, None])
                    / (4.7 * scale)
                    + phase_x
                    - phase_y
                )
            )
            pattern = (field / 3.0 + 1.0) * 0.5
        else:
            raise ValueError(f"Unknown texture type: {spec.texture_type}")
        return pattern.clamp(0.0, 1.0)

    @classmethod
    def _background_texture(
        cls,
        pixels: Tensor,
        reference: Tensor,
        spec: AppearanceShiftSpec,
    ) -> Tensor:
        background = reference.new_tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
        squared_distance = (reference - background).square().sum(
            dim=1,
            keepdim=True,
        )
        background_mask = torch.exp(
            -squared_distance / (2.0 * 0.25**2)
        )
        pattern = cls._texture_pattern(reference, spec).view(
            1,
            1,
            *reference.shape[-2:],
        )
        tint = reference.new_tensor(
            (
                cls._unit_from_seed(spec.seed, 41),
                cls._unit_from_seed(spec.seed, 43),
                cls._unit_from_seed(spec.seed, 47),
            )
        ).view(1, 3, 1, 1)
        textured = pixels + spec.texture_strength * pattern * (tint - pixels)
        return (
            pixels * (1.0 - background_mask)
            + textured * background_mask
        ).clamp(0.0, 1.0)

    @staticmethod
    def _photometric(
        pixels: Tensor,
        spec: AppearanceShiftSpec,
    ) -> Tensor:
        gains = pixels.new_tensor(spec.rgb_gain).view(1, 3, 1, 1)
        shifted = pixels * spec.brightness
        mean = shifted.mean(dim=(-2, -1), keepdim=True)
        shifted = (shifted - mean) * spec.contrast + mean
        shifted = shifted.clamp(0.0, 1.0).pow(spec.gamma)
        return (shifted * gains).clamp(0.0, 1.0)

    def apply(
        self,
        frames: Tensor,
        spec: AppearanceShiftSpec,
    ) -> Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape [T, 3, H, W]")

        input_dtype = frames.dtype
        is_uint8 = input_dtype == torch.uint8
        pixels = frames.to(dtype=torch.float32)
        if is_uint8:
            pixels = pixels.div(255.0)
        pixels = pixels.clamp(0.0, 1.0)
        reference = pixels

        if spec.shift_type == "photometric":
            shifted = self._photometric(pixels, spec)
        elif spec.shift_type == "background_texture":
            shifted = self._background_texture(pixels, reference, spec)
        elif spec.shift_type == "palette_shift":
            shifted = self._palette_shift(pixels, reference, spec)
        elif spec.shift_type == "composed":
            shifted = self._palette_shift(pixels, reference, spec)
            shifted = self._background_texture(shifted, reference, spec)
            shifted = self._photometric(shifted, spec)
        else:
            raise ValueError(f"Unknown appearance shift type: {spec.shift_type}")

        if is_uint8:
            return shifted.mul(255.0).round().to(dtype=torch.uint8)
        return shifted.to(dtype=input_dtype)
