from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wm_adapter.adapters.base import PEFTMethod


class FusedQKVLoRA(nn.Module):
    def __init__(self, base_projection: nn.Module, embed_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        if not hasattr(base_projection, "weight"):
            raise TypeError(f"Fused QKV projection has no weight: {type(base_projection).__name__}")
        weight = base_projection.weight
        if tuple(weight.shape) != (3 * embed_dim, embed_dim):
            raise ValueError(
                f"Fused QKV weight must have shape {(3 * embed_dim, embed_dim)}, found {tuple(weight.shape)}"
            )
        self.base_projection = base_projection
        self.embed_dim = embed_dim
        self.in_features = embed_dim
        self.out_features = 3 * embed_dim
        self.rank = rank
        self.scaling = alpha / rank
        self.q_a = nn.Parameter(torch.empty(rank, embed_dim, device=weight.device, dtype=weight.dtype))
        self.q_b = nn.Parameter(torch.zeros(embed_dim, rank, device=weight.device, dtype=weight.dtype))
        self.v_a = nn.Parameter(torch.empty(rank, embed_dim, device=weight.device, dtype=weight.dtype))
        self.v_b = nn.Parameter(torch.zeros(embed_dim, rank, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))
        for parameter in self.base_projection.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> Tensor:
        return self.base_projection.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_projection.bias

    def forward(self, x: Tensor) -> Tensor:
        fused = self.base_projection(x)
        query_delta = F.linear(F.linear(x, self.q_a), self.q_b) * self.scaling
        value_delta = F.linear(F.linear(x, self.v_a), self.v_b) * self.scaling
        delta = torch.cat((query_delta, torch.zeros_like(query_delta), value_delta), dim=-1)
        if delta.shape != fused.shape:
            raise RuntimeError(
                f"LoRA fused delta shape {tuple(delta.shape)} does not match QKV output {tuple(fused.shape)}"
            )
        return fused + delta


class LastBlockAttentionLoRA(PEFTMethod):
    method_name = "lora"

    def __init__(self, embed_dim: int, rank: int = 4, alpha: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if dropout != 0.0:
            raise ValueError("The final LoRA method requires dropout=0.0")
        self.embed_dim = embed_dim
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.qkv_lora: FusedQKVLoRA | None = None
        self.attach_identity_max_abs_error: float | None = None

    def attach_backend(self, backend: nn.Module) -> None:
        attention = backend.last_block.attn
        if not hasattr(attention, "qkv"):
            raise TypeError(
                f"DINOv3 last-block attention has no fused qkv projection: {type(attention).__name__}"
            )
        if isinstance(attention.qkv, FusedQKVLoRA):
            if attention.qkv is not self.qkv_lora:
                raise RuntimeError("The DINOv3 last block is already attached to another LoRA method")
            return
        original_qkv = attention.qkv
        qkv_lora = FusedQKVLoRA(original_qkv, self.embed_dim, self.rank, self.alpha)
        weight = original_qkv.weight
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=2 * 3 * self.embed_dim,
            device=weight.device,
            dtype=weight.dtype,
        ).reshape(2, 3, self.embed_dim)
        with torch.no_grad():
            original_output = original_qkv(probe)
            lora_output = qkv_lora(probe)
        if original_output.shape != lora_output.shape:
            raise RuntimeError(
                "LoRA QKV identity shape mismatch before attachment: "
                f"original={tuple(original_output.shape)}, lora={tuple(lora_output.shape)}"
            )
        low_precision = weight.dtype in {torch.float16, torch.bfloat16}
        atol, rtol = (2.0e-3, 2.0e-3) if low_precision else (1.0e-6, 1.0e-5)
        max_error = float((original_output.float() - lora_output.float()).abs().max().cpu())
        if not torch.allclose(
            original_output.float(),
            lora_output.float(),
            atol=atol,
            rtol=rtol,
        ):
            raise RuntimeError(
                "LoRA QKV identity failed before attachment: "
                f"max_abs_error={max_error}, original_shape={tuple(original_output.shape)}, "
                f"lora_shape={tuple(lora_output.shape)}, atol={atol}, rtol={rtol}"
            )
        self.qkv_lora = qkv_lora
        self.attach_identity_max_abs_error = max_error
        attention.qkv = self.qkv_lora

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        if self.qkv_lora is None:
            raise RuntimeError("LoRA must be attached to the JEPA-WM backend before use")
        return (
            parameter
            for name, parameter in self.qkv_lora.named_parameters()
            if not name.startswith("base_projection.") and parameter.requires_grad
        )

    def state_dict_for_checkpoint(self) -> dict[str, Tensor]:
        if self.qkv_lora is None:
            raise RuntimeError("LoRA must be attached before saving")
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.qkv_lora.named_parameters()
            if not name.startswith("base_projection.")
        }

    def load_method_checkpoint(self, state_dict: dict[str, Tensor]) -> None:
        if self.qkv_lora is None:
            raise RuntimeError("LoRA must be attached before loading")
        expected = {"q_a", "q_b", "v_a", "v_b"}
        if set(state_dict) != expected:
            raise RuntimeError(
                f"LoRA state keys must be {sorted(expected)}, received {sorted(state_dict)}"
            )
        with torch.no_grad():
            for name, value in state_dict.items():
                target = getattr(self.qkv_lora, name)
                if target.shape != value.shape:
                    raise RuntimeError(
                        f"LoRA tensor {name} expected shape {tuple(target.shape)}, found {tuple(value.shape)}"
                    )
                target.copy_(value)

    def config_dict(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": ["last_block.attention.q", "last_block.attention.v"],
        }
