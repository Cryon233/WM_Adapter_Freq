from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch import Tensor, nn


class PEFTMethod(nn.Module):
    method_name = "base"

    def attach_backend(self, backend: nn.Module) -> None:
        del backend

    def apply_patch_tokens(self, patch_tokens: Tensor) -> Tensor:
        return patch_tokens

    def adapter_site_indices(self, num_encoder_blocks: int) -> tuple[int, ...]:
        del num_encoder_blocks
        return ()

    def apply_at_site(self, site_index: int, patch_tokens: Tensor) -> Tensor:
        del site_index
        return self.apply_patch_tokens(patch_tokens)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def state_dict_for_checkpoint(self) -> dict[str, Tensor]:
        return {key: value.detach().cpu() for key, value in self.state_dict().items()}

    def load_method_checkpoint(self, state_dict: dict[str, Tensor]) -> None:
        result = self.load_state_dict(state_dict, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"PEFT state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def config_dict(self) -> dict[str, Any]:
        return {}


class BaseMethod(PEFTMethod):
    method_name = "base"

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return iter(())
