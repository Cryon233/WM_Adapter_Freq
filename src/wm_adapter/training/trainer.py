from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_adapter.adapters.base import BaseMethod, PEFTMethod
from wm_adapter.adapters.lora import LastBlockAttentionLoRA
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.utils.checkpoints import atomic_torch_save


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    gradient_accumulation: int
    lr: float
    weight_decay: float
    gradient_clip_norm: float
    precision: str
    num_workers: int
    seed: int
    canonical_weight: float = 1.0
    dynamics_weight: float = 1.0


class AdapterTrainer:
    def __init__(
        self,
        *,
        backend: JEPAWMDroidBackend,
        method: PEFTMethod,
        config: TrainingConfig,
        device: torch.device | str,
    ) -> None:
        if method.method_name == "base":
            raise ValueError("The base method has no trainable parameters and cannot be trained")
        self.backend = backend
        self.method = method
        self.config = config
        self.device = torch.device(device)
        if config.canonical_weight < 0.0 or config.dynamics_weight < 0.0:
            raise ValueError("canonical_weight and dynamics_weight must be non-negative")
        if config.canonical_weight == 0.0 and config.dynamics_weight == 0.0:
            raise ValueError("canonical_weight and dynamics_weight cannot both be zero")
        self.method.to(self.device)
        self.backend.eval()
        trainable = list(self.method.trainable_parameters())
        if not trainable:
            raise RuntimeError(f"Method {self.method.method_name} has no trainable parameters")
        trainable_ids = {id(parameter) for parameter in trainable}
        leaked = [
            name
            for name, parameter in self.backend.named_parameters()
            if parameter.requires_grad and id(parameter) not in trainable_ids
        ]
        if leaked:
            raise RuntimeError(f"Frozen JEPA-WM contains trainable base parameters: {leaked}")
        self.optimizer = AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
        if config.precision == "bf16":
            if self.device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "precision=bf16 was requested, but the current CUDA/PyTorch combination does not support bf16; "
                    "set training.precision=fp16 explicitly"
                )
            self.autocast_dtype = torch.bfloat16
            self.scaler: torch.amp.GradScaler | None = None
        elif config.precision == "fp16":
            if self.device.type != "cuda":
                raise RuntimeError("precision=fp16 requires a CUDA device")
            self.autocast_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
        elif config.precision == "fp32":
            self.autocast_dtype = torch.float32
            self.scaler = None
        else:
            raise ValueError(f"Unsupported precision {config.precision!r}; expected bf16, fp16, or fp32")

    def _autocast(self) -> torch.amp.autocast_mode.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.config.precision != "fp32",
        )

    def _encode_cached(self, prefix: Tensor, method: PEFTMethod) -> Tensor:
        batch, time = prefix.shape[:2]
        return self.backend.encode_from_prefix(prefix, method, batch, time)

    def _identity_invariant(self, first_batch: dict[str, Tensor]) -> None:
        self.backend.eval()
        self.method.eval()
        if isinstance(self.method, LastBlockAttentionLoRA):
            error = self.method.attach_identity_max_abs_error
            if error is None:
                raise RuntimeError("LoRA QKV identity was not checked before attachment")
            print(f"Identity invariant passed: method={self.method.method_name}, max_abs_error={error}")
            return
        prefix = first_batch["ood_prefix_tokens"][:1].to(self.device, non_blocking=True).float()
        with torch.no_grad(), self._autocast():
            base_latent = self._encode_cached(prefix, BaseMethod().to(self.device))
            method_latent = self._encode_cached(prefix, self.method)
        error = float((base_latent.float() - method_latent.float()).abs().max().cpu())
        if self.config.precision == "fp32":
            atol, rtol = 1.0e-6, 1.0e-5
        else:
            atol, rtol = 2.0e-3, 2.0e-3
        if not torch.allclose(base_latent.float(), method_latent.float(), atol=atol, rtol=rtol):
            raise RuntimeError(
                f"Identity invariant failed for {self.method.method_name}: max_abs_error={error}, "
                f"atol={atol}, rtol={rtol}"
            )
        print(f"Identity invariant passed: method={self.method.method_name}, max_abs_error={error}")

    def _losses(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        prefix = batch["ood_prefix_tokens"].to(self.device, non_blocking=True).float()
        clean_context = batch["clean_context_final_latent"].to(self.device, non_blocking=True).float()
        clean_future = batch["clean_future_latent"].to(self.device, non_blocking=True).float()
        actions = batch["actions"].to(self.device, non_blocking=True).float()
        context = self._encode_cached(prefix[:, :3], self.method)
        future = self._encode_cached(prefix[:, 3:4], self.method)
        canonical = F.mse_loss(torch.cat((context, future), dim=1), torch.cat((clean_context, clean_future), dim=1))
        predicted = self.backend.predict(context, actions[:, :3])
        dynamics_target = torch.cat((clean_context[:, 1:], clean_future), dim=1)
        dynamics = F.mse_loss(predicted, dynamics_target)
        total = (
            self.config.canonical_weight * canonical
            + self.config.dynamics_weight * dynamics
        )
        return total, canonical, dynamics

    def fit(
        self,
        loader: DataLoader[dict[str, Tensor]],
        *,
        checkpoint_path: str | Path,
        cache_metadata: dict[str, Any],
    ) -> dict[str, float]:
        first_batch = next(iter(loader))
        self._identity_invariant(first_batch)
        self.method.train()
        self.backend.eval()
        self.optimizer.zero_grad(set_to_none=True)
        final_losses: dict[str, float] | None = None
        for epoch in range(self.config.epochs):
            total_sum = 0.0
            canonical_sum = 0.0
            dynamics_sum = 0.0
            batches = 0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{self.config.epochs}")
            for batch_index, batch in enumerate(progress):
                with self._autocast():
                    total, canonical, dynamics = self._losses(batch)
                    scaled_loss = total / self.config.gradient_accumulation
                if self.scaler is None:
                    scaled_loss.backward()
                else:
                    self.scaler.scale(scaled_loss).backward()
                should_step = (
                    (batch_index + 1) % self.config.gradient_accumulation == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.method.trainable_parameters()), self.config.gradient_clip_norm
                    )
                    if self.scaler is None:
                        self.optimizer.step()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                total_sum += float(total.detach().cpu())
                canonical_sum += float(canonical.detach().cpu())
                dynamics_sum += float(dynamics.detach().cpu())
                batches += 1
                progress.set_postfix(total=total_sum / batches)
            print(
                f"epoch={epoch + 1} total={total_sum / batches:.6f} "
                f"canonical={canonical_sum / batches:.6f} dynamics={dynamics_sum / batches:.6f}"
            )
            final_losses = {
                "total": total_sum / batches,
                "canonical": canonical_sum / batches,
                "dynamics": dynamics_sum / batches,
            }
        if final_losses is None or not all(
            torch.isfinite(torch.tensor(value)) for value in final_losses.values()
        ):
            raise RuntimeError(f"Training produced invalid final losses: {final_losses}")
        payload = {
            "method_name": self.method.method_name,
            "peft_state_dict": self.method.state_dict_for_checkpoint(),
            "method_config": self.method.config_dict(),
            "trainable_parameter_count": self.method.parameter_count(),
            "base_checkpoint_sha256": self.backend.base_checkpoint_sha256,
            "dinov3_checkpoint_sha256": self.backend.dinov3_checkpoint_sha256,
            "upstream_commits": self.backend.upstream_commits,
            "cache_fingerprint": str(cache_metadata["cache_fingerprint"]),
            "appearance_metadata": cache_metadata["appearance_metadata"],
            "training_config": asdict(self.config),
        }
        atomic_torch_save(payload, checkpoint_path)
        return final_losses
