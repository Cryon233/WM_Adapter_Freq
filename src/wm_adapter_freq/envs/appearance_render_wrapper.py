from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from wm_adapter_freq.data.appearance_shift import (
    AppearanceShiftSpec,
    TwoRoomAppearanceShift,
)


class TwoRoomAppearanceRenderWrapper(gym.Wrapper):
    """Apply one deterministic raw-render appearance shift per episode."""

    def __init__(
        self,
        env: gym.Env,
        shift_type: str,
        severity: float,
        base_seed: int,
        enabled: bool = True,
    ) -> None:
        super().__init__(env)
        self.shift_type = shift_type
        self.severity = float(severity)
        self.base_seed = int(base_seed)
        self.enabled = bool(enabled)
        self.appearance_shift = TwoRoomAppearanceShift()
        self._spec: AppearanceShiftSpec | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        if self.enabled:
            reset_seed = 0 if seed is None else int(seed)
            spec_seed = (
                self.base_seed * 1_000_003 + reset_seed * 97_409
            ) % (2**63 - 1)
            self._spec = self.appearance_shift.sample_spec(
                shift_type=self.shift_type,
                seed=spec_seed,
                severity=self.severity,
            )
        else:
            self._spec = None
        return observation, info

    def render(self) -> np.ndarray:
        image = np.asarray(self.env.render())
        if not self.enabled:
            return image
        if self._spec is None:
            raise RuntimeError("render() called before reset().")
        frames = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        shifted = self.appearance_shift.apply(frames, self._spec)
        return (
            shifted[0]
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
