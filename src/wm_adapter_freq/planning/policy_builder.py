from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from wm_adapter_freq.backends.base import build_backend
from wm_adapter_freq.data.paired_windows import build_image_preprocessor
from wm_adapter_freq.io.adapter_checkpoint import load_adapter_checkpoint
from wm_adapter_freq.io.fingerprint import resolve_base_model_identity
from wm_adapter_freq.planning.appearance_transform import (
    FixedCurrentObservationTransform,
)


def _normalizers_from_checkpoint(
    normalization: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from stable_worldmodel.data.normalization import ZScoreScaler

    scalers: dict[str, Any] = {}
    for key in ("action", "proprio"):
        stats = normalization[key]
        if str(stats["method"]) != "zscore":
            raise RuntimeError(
                f"Unsupported checkpoint normalization for {key}."
            )
        feature_dim = int(stats["feature_dim"])
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(
            1,
            feature_dim,
        )
        std = np.asarray(stats["std"], dtype=np.float32).reshape(
            1,
            feature_dim,
        )
        scalers[key] = ZScoreScaler(
            mean=mean,
            std=std,
            eps=float(stats["eps"]),
        )
    scalers["goal_proprio"] = scalers["proprio"]
    return scalers


def build_world_model(
    backend: str,
    base_model_ref: str,
    adapter_checkpoint: str | Path,
    use_adapter: bool,
    device: torch.device | str,
) -> nn.Module:
    """Build a fingerprint-matched base or adapted world model."""
    from stable_worldmodel.wm.utils import load_pretrained

    target_device = torch.device(device)
    identity = resolve_base_model_identity(base_model_ref)
    adapter, metadata = load_adapter_checkpoint(
        Path(adapter_checkpoint).expanduser(),
        device="cpu",
    )
    checkpoint_identity = metadata["base_model_identity"]
    if (
        str(checkpoint_identity["combined_fingerprint"])
        != identity.combined_fingerprint
    ):
        raise RuntimeError(
            "Adapter checkpoint was trained for a different base checkpoint."
        )
    if str(metadata["backend"]) != backend:
        raise RuntimeError(
            "Adapter checkpoint backend does not match the requested model."
        )

    base_model = load_pretrained(identity.resolved_weights_path)
    base_model.eval()
    base_model.requires_grad_(False)
    model_backend = build_backend(backend, base_model)
    if (
        int(metadata["token_dim"]) != model_backend.token_dim
        or int(metadata["latent_dim"]) != model_backend.latent_dim
    ):
        raise RuntimeError(
            "Adapter checkpoint dimensions do not match the base model."
        )

    model = (
        model_backend.build_online_model(adapter)
        if use_adapter
        else base_model
    )
    model.to(target_device)
    model.eval()
    setattr(model, "base_model_fingerprint", identity.combined_fingerprint)
    setattr(model, "adapter_checkpoint_metadata", metadata)
    setattr(model, "model_variant", "adapter" if use_adapter else "base")
    return model


def build_tworoom_mpc_policy(
    backend: str,
    base_model_ref: str,
    adapter_checkpoint: str | Path,
    use_adapter: bool,
    appearance_enabled: bool,
    appearance_shift_type: str,
    appearance_severity: float,
    appearance_seed: int,
    device: torch.device | str,
    horizon: int = 5,
    receding_horizon: int = 1,
    history_len: int = 3,
    action_block: int = 5,
    warm_start: bool = True,
    num_samples: int = 300,
    cem_steps: int = 10,
    topk: int = 30,
    batch_size: int = 4,
    seed: int = 42,
) -> Any:
    """Build the upstream CEM and WorldModelPolicy stack for TwoRoom."""
    from stable_worldmodel.planning import (
        CEMSolver,
        GoalMSE,
        ShootingCostEvaluator,
    )
    from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

    model = build_world_model(
        backend,
        base_model_ref,
        adapter_checkpoint,
        use_adapter,
        device,
    )
    checkpoint_metadata = getattr(model, "adapter_checkpoint_metadata")
    if backend == "prejepa":
        cost: Any = model
        history_keys = ("pixels", "proprio")
    else:
        cost = ShootingCostEvaluator(model, GoalMSE())
        history_keys = ("pixels",)

    solver = CEMSolver(
        cost=cost,
        batch_size=batch_size,
        num_samples=num_samples,
        n_steps=cem_steps,
        topk=topk,
        device=device,
        seed=seed,
    )
    plan_config = PlanConfig(
        horizon=horizon,
        receding_horizon=receding_horizon,
        history_len=history_len,
        action_block=action_block,
        warm_start=warm_start,
    )
    current_transform = FixedCurrentObservationTransform(
        enabled=appearance_enabled,
        shift_type=appearance_shift_type,
        severity=appearance_severity,
        seed=appearance_seed,
        image_size=224,
    )
    goal_transform = build_image_preprocessor(224)
    policy = WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=_normalizers_from_checkpoint(
            checkpoint_metadata["normalization"]
        ),
        transform={
            "pixels": current_transform,
            "goal": goal_transform,
        },
        history_keys=history_keys,
        seed=seed,
    )
    setattr(
        policy,
        "base_model_fingerprint",
        getattr(model, "base_model_fingerprint"),
    )
    setattr(
        policy,
        "adapter_checkpoint_metadata",
        checkpoint_metadata,
    )
    setattr(
        policy,
        "model_variant",
        getattr(model, "model_variant"),
    )
    return policy
