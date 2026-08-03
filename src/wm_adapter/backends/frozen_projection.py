from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from torch import nn

from wm_adapter.adapters.lora import FusedQKVLoRA


@contextmanager
def frozen_base_projection(backend: nn.Module) -> Iterator[None]:
    """Temporarily expose the checkpoint QKV when a LoRA wrapper is attached."""

    attention = backend.last_block.attn
    projection = attention.qkv
    if not isinstance(projection, FusedQKVLoRA):
        yield
        return
    attention.qkv = projection.base_projection
    try:
        yield
    finally:
        attention.qkv = projection
