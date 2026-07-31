from __future__ import annotations

import torch
from torch import Tensor, nn


class DINOv2SplitEncoder(nn.Module):
    """Split the installed HuggingFace DINOv2 before its final block."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if getattr(model.config, "model_type", None) != "dinov2":
            raise TypeError("PreJEPA backbone must be a HuggingFace DINOv2 model.")
        self.model = model
        self.embeddings = model.embeddings
        self.prefix_layers = model.encoder.layer[:-1]
        self.last_block = model.encoder.layer[-1]
        self.final_norm = model.layernorm

    def encode_prefix(self, pixels: Tensor) -> Tensor:
        """Encode ``[B,T,3,H,W]`` through the first L-1 blocks."""
        batch_size, sequence_length = pixels.shape[:2]
        flat_pixels = pixels.reshape(
            batch_size * sequence_length, *pixels.shape[2:]
        )
        hidden_states = self.embeddings(flat_pixels)
        for layer in self.prefix_layers:
            hidden_states = layer(hidden_states)
        return hidden_states.reshape(
            batch_size,
            sequence_length,
            hidden_states.shape[-2],
            hidden_states.shape[-1],
        )

    def forward_tail(self, prefix_tokens: Tensor) -> Tensor:
        """Apply the original final DINOv2 block and final LayerNorm."""
        batch_size, sequence_length = prefix_tokens.shape[:2]
        hidden_states = prefix_tokens.reshape(
            batch_size * sequence_length, *prefix_tokens.shape[2:]
        )
        hidden_states = self.last_block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return hidden_states.reshape(
            batch_size,
            sequence_length,
            hidden_states.shape[-2],
            hidden_states.shape[-1],
        )
