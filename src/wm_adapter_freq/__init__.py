"""Frequency-domain visual adapters for stable-worldmodel."""

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.backends.lewm import LeWMBackend
from wm_adapter_freq.backends.prejepa import PreJEPABackend

__all__ = [
    "LeWMBackend",
    "PreJEPABackend",
    "SequenceStableAdaptiveDCTAdapter",
]
