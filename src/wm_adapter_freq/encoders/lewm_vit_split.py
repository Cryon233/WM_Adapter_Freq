from __future__ import annotations

from torch import Tensor, nn
from transformers.masking_utils import create_bidirectional_mask


class LeWMViTSplitEncoder(nn.Module):
    """Split stable_pretraining's installed HuggingFace Tiny ViT."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if getattr(model.config, "model_type", None) != "vit":
            raise TypeError("LeWM encoder must be stable_pretraining's HuggingFace ViT.")
        self.model = model
        self.embeddings = model.embeddings
        self.prefix_layers = model.layers[:-1]
        self.last_block = model.layers[-1]
        self.final_norm = model.layernorm

    def encode_prefix(self, pixels: Tensor) -> Tensor:
        """Encode ``[B,T,3,H,W]`` through the first L-1 Tiny ViT blocks."""
        batch_size, sequence_length = pixels.shape[:2]
        flat_pixels = pixels.reshape(
            batch_size * sequence_length, *pixels.shape[2:]
        )
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        hidden_states = self.embeddings(
            flat_pixels.to(expected_dtype),
            interpolate_pos_encoding=True,
        )
        attention_mask = create_bidirectional_mask(
            config=self.model.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
        )
        for layer in self.prefix_layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states.reshape(
            batch_size,
            sequence_length,
            hidden_states.shape[-2],
            hidden_states.shape[-1],
        )

    def forward_tail(self, prefix_tokens: Tensor) -> Tensor:
        """Apply the original final Tiny ViT block and final LayerNorm."""
        batch_size, sequence_length = prefix_tokens.shape[:2]
        hidden_states = prefix_tokens.reshape(
            batch_size * sequence_length, *prefix_tokens.shape[2:]
        )
        attention_mask = create_bidirectional_mask(
            config=self.model.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
        )
        hidden_states = self.last_block(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)
        return hidden_states.reshape(
            batch_size,
            sequence_length,
            hidden_states.shape[-2],
            hidden_states.shape[-1],
        )
