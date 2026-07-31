from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from wm_adapter.adapters.base import PEFTMethod


class TokenMLPAdapter(PEFTMethod):
    method_name = "token_mlp"

    def __init__(self, embed_dim: int, rank: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.down = nn.Linear(embed_dim, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def apply_patch_tokens(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 4 or patch_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Token MLP expected [B,T,P,{self.embed_dim}], received {tuple(patch_tokens.shape)}"
            )
        return patch_tokens + self.up(self.activation(self.down(patch_tokens)))

    def forward(self, patch_tokens: Tensor) -> Tensor:
        return self.apply_patch_tokens(patch_tokens)

    def config_dict(self) -> dict[str, Any]:
        return {"embed_dim": self.embed_dim, "rank": self.rank}
