from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from wm_adapter.adapters.base import PEFTMethod


class HFRASiteAdapter(nn.Module):
    """One fixed Hybrid Fourier Residual Adapter insertion site."""

    def __init__(
        self,
        *,
        embed_dim: int,
        grid_height: int,
        grid_width: int,
        rank: int,
        fourier_enabled: bool,
    ) -> None:
        super().__init__()
        if min(embed_dim, grid_height, grid_width, rank) <= 0:
            raise ValueError(
                "HFRA dimensions and rank must be positive: "
                f"D={embed_dim}, H={grid_height}, W={grid_width}, rank={rank}"
            )
        self.embed_dim = embed_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.rank = rank
        self.fourier_enabled = fourier_enabled
        self.norm = nn.RMSNorm(embed_dim, elementwise_affine=False)
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        if fourier_enabled:
            frequency_shape = (grid_height, grid_width // 2 + 1)
            self.frequency_real = nn.Parameter(torch.empty(frequency_shape))
            self.frequency_imag = nn.Parameter(torch.empty(frequency_shape))
            self.channel_mixer_real = nn.Parameter(torch.eye(rank))
            self.channel_mixer_imag = nn.Parameter(torch.zeros(rank, rank))
            nn.init.normal_(self.frequency_real, mean=0.0, std=1.0e-4)
            nn.init.normal_(self.frequency_imag, mean=0.0, std=1.0e-4)
        self._latest_diagnostics: dict[str, Tensor] = {}

    def _spectral_residual(self, core: Tensor) -> Tensor:
        if not self.fourier_enabled:
            return torch.zeros_like(core)
        batch, time, patches, rank = core.shape
        spatial = core.float().reshape(
            batch * time,
            self.grid_height,
            self.grid_width,
            rank,
        ).permute(0, 3, 1, 2)
        coefficients = torch.fft.rfft2(spatial, norm="ortho")
        frequency = torch.complex(
            self.frequency_real.float(), self.frequency_imag.float()
        )
        mixer = torch.complex(
            self.channel_mixer_real.float(), self.channel_mixer_imag.float()
        )
        mixed = torch.einsum("brhw,rs->bshw", coefficients, mixer)
        delta_coefficients = mixed * frequency.unsqueeze(0).unsqueeze(0)
        residual = torch.fft.irfft2(
            delta_coefficients,
            s=(self.grid_height, self.grid_width),
            norm="ortho",
        )
        return residual.permute(0, 2, 3, 1).reshape(
            batch, time, patches, rank
        ).to(dtype=core.dtype)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        expected_patches = self.grid_height * self.grid_width
        if patch_tokens.ndim != 4 or tuple(patch_tokens.shape[2:]) != (
            expected_patches,
            self.embed_dim,
        ):
            raise ValueError(
                "HFRA site expected "
                f"[B,T,{expected_patches},{self.embed_dim}], "
                f"received {tuple(patch_tokens.shape)}"
            )
        core = self.activation(self.down(self.norm(patch_tokens)))
        spectral = self._spectral_residual(core)
        core_delta = self.up(core)
        spectral_delta = self.up(spectral)
        total_delta = core_delta + spectral_delta
        denominator = patch_tokens.float().square().mean().sqrt().clamp_min(1.0e-12)
        # Diagnostics describe the residual written back to DINO tokens, not the
        # rank-space bottleneck activations that precede the zero-initialized up map.
        self._latest_diagnostics = {
            "core_delta_ratio": (core_delta.float().square().mean().sqrt() / denominator).detach(),
            "spectral_delta_ratio": (spectral_delta.float().square().mean().sqrt() / denominator).detach(),
            "total_delta_ratio": (total_delta.float().square().mean().sqrt() / denominator).detach(),
        }
        if self.fourier_enabled:
            magnitude = torch.complex(
                self.frequency_real.float(), self.frequency_imag.float()
            ).abs()
            self._latest_diagnostics.update(
                frequency_filter_abs_mean=magnitude.mean().detach(),
                frequency_filter_abs_max=magnitude.max().detach(),
            )
        return patch_tokens + total_delta

    def diagnostics(self, patch_tokens: Tensor) -> dict[str, Tensor]:
        core = self.activation(self.down(self.norm(patch_tokens)))
        spectral = self._spectral_residual(core)
        core_delta = self.up(core)
        spectral_delta = self.up(spectral)
        total_delta = core_delta + spectral_delta
        denominator = patch_tokens.float().square().mean().sqrt().clamp_min(1.0e-12)
        values = {
            "core_delta_ratio": core_delta.float().square().mean().sqrt() / denominator,
            "spectral_delta_ratio": spectral_delta.float().square().mean().sqrt()
            / denominator,
            "total_delta_ratio": total_delta.float().square().mean().sqrt()
            / denominator,
        }
        if self.fourier_enabled:
            magnitude = torch.complex(
                self.frequency_real.float(), self.frequency_imag.float()
            ).abs()
            values.update(
                frequency_filter_abs_mean=magnitude.mean(),
                frequency_filter_abs_max=magnitude.max(),
            )
        return values

    def latest_diagnostics(self) -> dict[str, Tensor]:
        return dict(self._latest_diagnostics)


class HybridFourierResidualAdapter(PEFTMethod):
    method_name = "hfra"

    def __init__(
        self,
        *,
        embed_dim: int,
        grid_height: int,
        grid_width: int,
        num_encoder_blocks: int,
        rank: int = 4,
        fourier_enabled: bool = True,
        method_name: str = "hfra",
    ) -> None:
        super().__init__()
        if num_encoder_blocks < 2:
            raise ValueError(
                "HFRA requires at least two visual-transformer blocks, "
                f"received {num_encoder_blocks}"
            )
        if method_name not in {"hfra", "hfra_core_only"}:
            raise ValueError(f"Unsupported HFRA method name: {method_name!r}")
        self.method_name = method_name
        self.embed_dim = embed_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_encoder_blocks = num_encoder_blocks
        self.rank = rank
        self.fourier_enabled = fourier_enabled
        self.middle_site = num_encoder_blocks // 2
        self.late_site = num_encoder_blocks - 1
        self.sites = nn.ModuleDict(
            {
                str(site): HFRASiteAdapter(
                    embed_dim=embed_dim,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    rank=rank,
                    fourier_enabled=fourier_enabled,
                )
                for site in (self.middle_site, self.late_site)
            }
        )

    def adapter_site_indices(self, num_encoder_blocks: int) -> tuple[int, ...]:
        if num_encoder_blocks != self.num_encoder_blocks:
            raise RuntimeError(
                "HFRA encoder depth changed after construction: "
                f"expected={self.num_encoder_blocks}, actual={num_encoder_blocks}"
            )
        return (self.middle_site, self.late_site)

    def apply_at_site(self, site_index: int, patch_tokens: Tensor) -> Tensor:
        key = str(site_index)
        if key not in self.sites:
            raise ValueError(
                f"HFRA has no adapter at block {site_index}; "
                f"sites={self.adapter_site_indices(self.num_encoder_blocks)}"
            )
        return self.sites[key](patch_tokens)

    def apply_patch_tokens(self, patch_tokens: Tensor) -> Tensor:
        return self.apply_at_site(self.late_site, patch_tokens)

    def config_dict(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "num_encoder_blocks": self.num_encoder_blocks,
            "rank": self.rank,
            "fourier_enabled": self.fourier_enabled,
            "sites": [self.middle_site, self.late_site],
        }

    def latest_diagnostics(self) -> dict[str, Tensor]:
        values = [site.latest_diagnostics() for site in self.sites.values()]
        keys = set().union(*(item.keys() for item in values))
        return {
            key: torch.stack([item[key] for item in values if key in item]).mean()
            for key in keys
        }


class HFRACoreOnlyAdapter(HybridFourierResidualAdapter):
    method_name = "hfra_core_only"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            **kwargs,
            fourier_enabled=False,
            method_name="hfra_core_only",
        )
