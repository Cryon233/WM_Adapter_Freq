from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from wm_adapter_freq.backends.base import build_backend
from wm_adapter_freq.data.paired_windows import build_image_preprocessor
from wm_adapter_freq.io.adapter_checkpoint import load_adapter_checkpoint


def _model_reference(value: str) -> str:
    expanded = Path(value).expanduser()
    return (
        str(expanded)
        if expanded.exists() or value.startswith(("~", "."))
        else value
    )


def build_adapted_model(
    backend: str,
    base_model_ref: str,
    adapter_checkpoint: str | Path,
    device: torch.device | str,
) -> nn.Module:
    """Load a base checkpoint and attach the matching adapter for inference."""
    from stable_worldmodel.wm.utils import load_pretrained

    device = torch.device(device)
    base_model = load_pretrained(_model_reference(base_model_ref))
    adapter, metadata = load_adapter_checkpoint(
        Path(adapter_checkpoint).expanduser(), device="cpu"
    )
    if metadata["backend"] != backend:
        raise ValueError("Adapter checkpoint backend does not match the requested model.")
    model_backend = build_backend(backend, base_model)
    if (
        metadata["token_dim"] != model_backend.token_dim
        or metadata["latent_dim"] != model_backend.latent_dim
    ):
        raise ValueError("Adapter checkpoint dimensions do not match the base model.")
    model = model_backend.build_online_model(adapter)
    model.to(device)
    model.eval()
    return model


def _fit_normalizers(dataset: Any) -> dict[str, Any]:
    from stable_worldmodel.data.normalization import get_scaler

    process: dict[str, Any] = {}
    for key in ("action", "proprio"):
        if key not in dataset.column_names:
            continue
        values = np.asarray(dataset.get_col_data(key))
        values = values.reshape(-1, values.shape[-1])
        values = values[~np.isnan(values).any(axis=1)]
        scaler = get_scaler("zscore")
        scaler.fit(values)
        process[key] = scaler
        if key != "action":
            process[f"goal_{key}"] = scaler
    return process


def build_tworoom_mpc_policy(
    backend: str,
    base_model_ref: str,
    adapter_checkpoint: str | Path,
    dataset: Any,
    device: torch.device | str,
    horizon: int = 5,
    receding_horizon: int = 5,
    history_len: int = 3,
    action_block: int = 5,
    num_samples: int = 300,
    cem_steps: int = 30,
    topk: int = 30,
    batch_size: int = 1,
    seed: int = 42,
) -> Any:
    """Build the upstream CEM/WorldModelPolicy stack for TwoRoom."""
    from stable_worldmodel.planning import (
        CEMSolver,
        GoalMSE,
        ShootingCostEvaluator,
    )
    from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

    model = build_adapted_model(
        backend,
        base_model_ref,
        adapter_checkpoint,
        device,
    )
    cost: Any
    if backend == "prejepa":
        cost = model
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
        warm_start=True,
    )
    image_preprocessor = build_image_preprocessor(224)
    return WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=_fit_normalizers(dataset),
        transform={
            "pixels": image_preprocessor,
            "goal": image_preprocessor,
        },
        history_keys=history_keys,
        seed=seed,
    )
