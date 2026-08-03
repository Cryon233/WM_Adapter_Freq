from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from wm_adapter.adapters.base import BaseMethod, PEFTMethod
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.utils.checkpoints import atomic_torch_save


CHECKPOINT_SCHEMA_V2 = "wm_adapter_checkpoint_v2"


@dataclass(frozen=True)
class TrajectoryTrainingConfig:
    max_optimizer_steps: int
    microbatch_windows: int
    views_per_window: int
    gradient_accumulation: int
    lr: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    precision: str
    num_workers: int
    seed: int
    warmup_steps: int
    minimum_lr: float
    scheduler: str = "cosine"
    loss_name: str = "unified_trajectory_mse"


class TrajectoryAdapterTrainer:
    def __init__(self, *, backend: JEPAWMDroidBackend, method: PEFTMethod, config: TrajectoryTrainingConfig, device: torch.device | str) -> None:
        if method.method_name == "base":
            raise ValueError("Base has no trainable parameters")
        if config.loss_name != "unified_trajectory_mse":
            raise ValueError(f"V2 supports only unified_trajectory_mse, found {config.loss_name}")
        if config.scheduler != "cosine":
            raise ValueError(f"V2 supports only cosine scheduling, found {config.scheduler}")
        self.backend, self.method, self.config = backend, method, config
        self.device = torch.device(device)
        method.to(self.device)
        backend.eval()
        parameters = list(method.trainable_parameters())
        if not parameters:
            raise RuntimeError(f"Method {method.method_name} has no trainable parameters")
        trainable_ids = {id(parameter) for parameter in parameters}
        leaked = [name for name, parameter in backend.named_parameters() if parameter.requires_grad and id(parameter) not in trainable_ids]
        if leaked:
            raise RuntimeError(f"Frozen base contains trainable parameters: {leaked}")
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for name, parameter in method.named_parameters():
            if not parameter.requires_grad or id(parameter) not in trainable_ids:
                continue
            zero_decay = any(marker in name for marker in (
                "frequency_real", "frequency_imag", "channel_mixer_real",
                "channel_mixer_imag", "norm", "bias", "gate",
            ))
            (no_decay if zero_decay else decay).append(parameter)
        self.optimizer = AdamW(
            [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config.lr, betas=config.betas, eps=config.epsilon,
        )
        minimum_ratio = config.minimum_lr / config.lr

        def schedule(step: int) -> float:
            if step < config.warmup_steps:
                return float(step + 1) / max(config.warmup_steps, 1)
            progress = (step - config.warmup_steps) / max(config.max_optimizer_steps - config.warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        self.scheduler = LambdaLR(self.optimizer, schedule)
        if config.precision == "bf16":
            if self.device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise RuntimeError("V2 bf16 training requires CUDA bf16 support")
            self.autocast_dtype, self.scaler = torch.bfloat16, None
        elif config.precision == "fp16":
            if self.device.type != "cuda":
                raise RuntimeError("V2 fp16 training requires CUDA")
            self.autocast_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
        elif config.precision == "fp32":
            self.autocast_dtype, self.scaler = torch.float32, None
        else:
            raise ValueError(f"Unsupported V2 precision {config.precision!r}")

    def _autocast(self) -> Any:
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.config.precision != "fp32")

    def trajectory_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        clean_middle = batch["clean_context_middle_tokens"].to(self.device).float()
        ood_middle = batch["ood_context_middle_tokens"].to(self.device).float()
        clean_target = batch["clean_target_latents"].to(self.device).float()
        rollout_actions = batch["rollout_actions"].to(self.device).float()
        original_batch = clean_middle.shape[0]
        adapted_context = self.backend.encode_from_site(
            torch.cat((clean_middle, ood_middle), dim=0),
            self.backend.num_encoder_blocks // 2,
            self.method,
        )
        predicted_future = self.backend.differentiable_unroll(
            adapted_context, torch.cat((rollout_actions, rollout_actions), dim=0)
        )
        predicted = torch.cat((adapted_context, predicted_future), dim=1)
        target = torch.cat((clean_target, clean_target), dim=0)
        if predicted.shape != target.shape:
            raise RuntimeError(f"Unified trajectory shape mismatch: predicted={tuple(predicted.shape)}, target={tuple(target.shape)}")
        loss = F.mse_loss(predicted, target)
        per_view = (predicted.float() - target.float()).square().mean(dim=(1, 2, 3))
        method_diagnostics = (
            self.method.latest_diagnostics()
            if hasattr(self.method, "latest_diagnostics")
            else {}
        )
        diagnostics = {
            "clean_mse": per_view[:original_batch].mean().detach(),
            "ood_mse": per_view[original_batch:].mean().detach(),
            "context_mse": F.mse_loss(predicted[:, :3].float(), target[:, :3].float()).detach(),
            "future_mse": F.mse_loss(predicted[:, 3:].float(), target[:, 3:].float()).detach(),
            "core_delta_ratio": method_diagnostics.get(
                "core_delta_ratio", torch.zeros((), device=self.device)
            ).detach(),
            "spectral_delta_ratio": method_diagnostics.get(
                "spectral_delta_ratio", torch.zeros((), device=self.device)
            ).detach(),
        }
        return loss, diagnostics

    def _identity_invariant(self, batch: dict[str, Any]) -> float:
        tokens = batch["ood_context_middle_tokens"][:1].to(self.device).float()
        self.method.eval()
        with torch.no_grad(), self._autocast():
            base = self.backend.encode_from_site(tokens, self.backend.num_encoder_blocks // 2, BaseMethod().to(self.device))
            adapted = self.backend.encode_from_site(tokens, self.backend.num_encoder_blocks // 2, self.method)
        error = float((base.float() - adapted.float()).abs().max().cpu())
        atol, rtol = ((1.0e-6, 1.0e-5) if self.config.precision == "fp32" else (2.0e-3, 2.0e-3))
        if not torch.allclose(base.float(), adapted.float(), atol=atol, rtol=rtol):
            raise RuntimeError(f"V2 identity invariant failed for {self.method.method_name}: max_abs_error={error}, atol={atol}, rtol={rtol}")
        print(f"Identity invariant passed: method={self.method.method_name}, max_abs_error={error}", flush=True)
        return error

    def fit(self, loader: DataLoader[dict[str, Any]], *, checkpoint_path: str | Path, cache_metadata: dict[str, Any]) -> dict[str, float]:
        self._identity_invariant(next(iter(loader)))
        self.method.train()
        self.backend.eval()
        iterator = iter(loader)
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_step = micro_step = 0
        started = time.perf_counter()
        final: dict[str, float] = {}
        while optimizer_step < self.config.max_optimizer_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            with self._autocast():
                loss, diagnostics = self.trajectory_loss(batch)
                backward_loss = loss / self.config.gradient_accumulation
            if self.scaler is None:
                backward_loss.backward()
            else:
                self.scaler.scale(backward_loss).backward()
            micro_step += 1
            if micro_step % self.config.gradient_accumulation:
                continue
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(list(self.method.trainable_parameters()), self.config.gradient_clip_norm)
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            optimizer_step += 1
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            final = {"loss": float(loss.detach().cpu()), **{key: float(value.cpu()) for key, value in diagnostics.items()}, "grad_norm": float(torch.as_tensor(grad_norm).cpu())}
            if optimizer_step % 10 == 0 or optimizer_step in {1, self.config.max_optimizer_steps}:
                processed_windows = micro_step * self.config.microbatch_windows
                print(
                    "TRAIN_PROGRESS "
                    f"step={optimizer_step} total={self.config.max_optimizer_steps} loss={final['loss']:.8f} "
                    f"clean_mse={final['clean_mse']:.8f} ood_mse={final['ood_mse']:.8f} "
                    f"context_mse={final['context_mse']:.8f} future_mse={final['future_mse']:.8f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.9f} grad_norm={final['grad_norm']:.6f} "
                    f"samples_per_sec={processed_windows / elapsed:.3f} core_delta_ratio={final['core_delta_ratio']:.8f} "
                    f"spectral_delta_ratio={final['spectral_delta_ratio']:.8f} "
                    f"elapsed_seconds={elapsed:.3f}", flush=True,
                )
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
        data_metadata = {}
        for key, value in cache_metadata.items():
            if isinstance(value, str) and value[:1] in {"{", "["}:
                value = json.loads(value)
            elif hasattr(value, "item"):
                value = value.item()
            data_metadata[key] = value
        serialized_training_config = asdict(self.config)
        serialized_training_config["betas"] = list(self.config.betas)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_V2,
            "method_name": self.method.method_name,
            "peft_state_dict": self.method.state_dict_for_checkpoint(),
            "method_config": self.method.config_dict(),
            "trainable_parameter_count": self.method.parameter_count(),
            "cache_fingerprint": str(cache_metadata["cache_fingerprint"]),
            "base_checkpoint_sha256": self.backend.base_checkpoint_sha256,
            "dinov3_checkpoint_sha256": self.backend.dinov3_checkpoint_sha256,
            "upstream_commits": self.backend.upstream_commits,
            "data_metadata": data_metadata,
            "loss_name": "unified_trajectory_mse",
            "max_optimizer_steps": self.config.max_optimizer_steps,
            "completed_optimizer_steps": optimizer_step,
            "optimizer_config": {"name": "AdamW", "lr": self.config.lr, "betas": list(self.config.betas), "epsilon": self.config.epsilon, "weight_decay": self.config.weight_decay},
            "scheduler_config": {"name": "cosine", "warmup_steps": self.config.warmup_steps, "minimum_lr": self.config.minimum_lr},
            "training_seed": self.config.seed,
            "goal_encoder": "frozen_base",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "training_config": serialized_training_config,
        }
        atomic_torch_save(payload, checkpoint_path)
        print(f"TRAIN_COMPLETE step={optimizer_step} total={self.config.max_optimizer_steps} loss={final['loss']:.8f} checkpoint={Path(checkpoint_path).resolve()}", flush=True)
        return final
