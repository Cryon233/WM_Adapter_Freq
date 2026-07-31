from __future__ import annotations

from torch import Tensor

from wm_adapter_freq.data.appearance_shift import (
    AppearanceShiftSpec,
    TwoRoomAppearanceShift,
)
from wm_adapter_freq.data.paired_windows import build_image_preprocessor


EVALUATION_PROTOCOL_VERSION = "3.0"


class FixedCurrentObservationTransform:
    """Apply one fixed appearance domain before standard preprocessing."""

    def __init__(
        self,
        enabled: bool,
        shift_type: str,
        severity: float,
        seed: int,
        image_size: int = 224,
    ) -> None:
        self.enabled = bool(enabled)
        self.shift_type = shift_type
        self.severity = float(severity)
        self.seed = int(seed)
        self.appearance_shift = TwoRoomAppearanceShift()
        self.spec: AppearanceShiftSpec | None = (
            self.appearance_shift.sample_spec(
                shift_type=self.shift_type,
                seed=self.seed,
                severity=self.severity,
            )
            if self.enabled
            else None
        )
        self.preprocess = build_image_preprocessor(image_size)

    def __call__(self, image: Tensor) -> Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must have shape [3, H, W]")
        frames = image.unsqueeze(0)
        if self.spec is not None:
            frames = self.appearance_shift.apply(frames, self.spec)
        return self.preprocess(frames[0])
