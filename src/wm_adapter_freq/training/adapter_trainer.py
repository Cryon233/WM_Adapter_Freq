from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.io.adapter_checkpoint import save_adapter_checkpoint
from wm_adapter_freq.objectives.canonical_dynamics import (
    CanonicalDynamicsObjective,
)


@dataclass(frozen=True)
class AdapterTrainingConfig:
    epochs: int
    gradient_accumulation: int
    precision: str
    lr: float
    weight_decay: float
    gradient_clip_norm: float


class AdapterTrainer:
    """Pure PyTorch trainer that updates and saves only the adapter."""

    def __init__(
        self,
        objective: CanonicalDynamicsObjective,
        adapter: SequenceStableAdaptiveDCTAdapter,
        config: AdapterTrainingConfig,
        device: torch.device,
    ) -> None:
        self.objective = objective
        self.adapter = adapter
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        use_scaler = device.type == "cuda" and config.precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def _autocast(self) -> Any:
        if self.device.type != "cuda":
            return nullcontext()
        dtype = (
            torch.float16
            if self.config.precision == "fp16"
            else torch.bfloat16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def fit(
        self,
        data_loader: DataLoader[dict[str, Tensor]],
        checkpoint_path: str | Path,
        checkpoint_metadata: dict[str, Any],
    ) -> None:
        self.adapter.train()
        num_batches = len(data_loader)
        accumulation = self.config.gradient_accumulation

        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            totals = {"total": 0.0, "canonical": 0.0, "dynamics": 0.0}
            progress = tqdm(
                enumerate(data_loader),
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            )
            for step, batch in progress:
                batch = self._to_device(batch)
                group_start = (step // accumulation) * accumulation
                group_size = min(accumulation, num_batches - group_start)
                with self._autocast():
                    losses = self.objective(batch)
                    scaled_loss = losses.total / group_size
                self.scaler.scale(scaled_loss).backward()

                should_step = (
                    (step + 1) % accumulation == 0
                    or step + 1 == num_batches
                )
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.adapter.parameters(),
                        self.config.gradient_clip_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                totals["total"] += float(losses.total.detach())
                totals["canonical"] += float(losses.canonical.detach())
                totals["dynamics"] += float(losses.dynamics.detach())
                progress.set_postfix(
                    total=f"{totals['total'] / (step + 1):.6f}"
                )

            averages = {
                key: value / num_batches for key, value in totals.items()
            }
            print(
                f"epoch={epoch + 1} "
                f"total={averages['total']:.6f} "
                f"canonical={averages['canonical']:.6f} "
                f"dynamics={averages['dynamics']:.6f}"
            )

        save_adapter_checkpoint(
            checkpoint_path,
            self.adapter,
            **checkpoint_metadata,
        )
