from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from wm_adapter_freq.planning.appearance_transform import (
    EVALUATION_PROTOCOL_VERSION,
)
from wm_adapter_freq.planning.policy_builder import build_tworoom_mpc_policy


def _episode_column(dataset: Any) -> str:
    names = set(dataset.column_names)
    names.update(getattr(dataset, "_schema_names", ()))
    return "episode_idx" if "episode_idx" in names else "ep_idx"


def _select_episode_balanced_eval_rows(
    episode_ids: np.ndarray,
    step_ids: np.ndarray,
    goal_offset_steps: int,
    num_eval: int,
    seed: int,
) -> np.ndarray:
    """Select one valid start row from each sampled episode."""
    episode_values = np.asarray(episode_ids)
    step_values = np.asarray(step_ids)
    valid_rows_by_episode: dict[int, np.ndarray] = {}
    for episode_id in np.unique(episode_values):
        episode_rows = np.flatnonzero(episode_values == episode_id)
        max_step = int(step_values[episode_rows].max())
        valid_rows = episode_rows[
            step_values[episode_rows]
            <= max_step - int(goal_offset_steps)
        ]
        if valid_rows.size > 0:
            valid_rows_by_episode[int(episode_id)] = valid_rows

    generator = np.random.default_rng(int(seed))
    eligible_episodes = np.asarray(
        list(valid_rows_by_episode),
        dtype=np.int64,
    )
    selection_count = min(int(num_eval), eligible_episodes.size)
    if selection_count <= 0:
        return np.empty(0, dtype=np.int64)

    selected_episodes = generator.choice(
        eligible_episodes,
        size=selection_count,
        replace=False,
    )
    selected_rows = np.asarray(
        [
            generator.choice(valid_rows_by_episode[int(episode_id)])
            for episode_id in selected_episodes
        ],
        dtype=np.int64,
    )
    order = np.argsort(episode_values[selected_rows], kind="stable")
    return selected_rows[order]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_severity(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("appearance severity must be finite")
    decimal_value = format(Decimal(str(value)), "f")
    if "." not in decimal_value:
        decimal_value += ".0"
    whole, fraction = decimal_value.split(".", maxsplit=1)
    fraction = fraction.rstrip("0") or "0"
    return f"{whole}p{fraction}"


def _planning_run_name(cfg: DictConfig) -> str:
    if not bool(cfg.appearance.enabled):
        return "clean"
    severity = _format_severity(float(cfg.appearance.severity))
    return (
        f"{cfg.appearance.shift_type}_severity{severity}"
        f"_seed{int(cfg.appearance.seed)}"
    )


@hydra.main(
    version_base=None,
    config_path="../configs/plan",
    config_name="prejepa_tworoom",
)
def main(cfg: DictConfig) -> None:
    import stable_worldmodel as swm

    if str(cfg.appearance.protocol) != "fixed":
        raise ValueError("appearance.protocol must be 'fixed'")

    torch.manual_seed(int(cfg.seed))
    dataset = swm.data.load_dataset(
        str(cfg.dataset_name),
        cache_dir=cfg.get("dataset_cache_dir"),
        keys_to_cache=["action", "proprio"],
    )
    policy = build_tworoom_mpc_policy(
        backend=str(cfg.backend),
        base_model_ref=str(cfg.base_model_ref),
        adapter_checkpoint=Path(str(cfg.adapter_checkpoint)).expanduser(),
        use_adapter=bool(cfg.model.use_adapter),
        appearance_enabled=bool(cfg.appearance.enabled),
        appearance_shift_type=str(cfg.appearance.shift_type),
        appearance_severity=float(cfg.appearance.severity),
        appearance_seed=int(cfg.appearance.seed),
        device=str(cfg.device),
        horizon=int(cfg.plan.horizon),
        receding_horizon=int(cfg.plan.receding_horizon),
        history_len=int(cfg.plan.history_len),
        action_block=int(cfg.plan.action_block),
        warm_start=bool(cfg.plan.warm_start),
        num_samples=int(cfg.cem.num_samples),
        cem_steps=int(cfg.cem.n_steps),
        topk=int(cfg.cem.topk),
        batch_size=int(cfg.cem.batch_size),
        seed=int(cfg.seed),
    )

    episode_column = _episode_column(dataset)
    episode_ids = np.asarray(dataset.get_col_data(episode_column))
    step_ids = np.asarray(dataset.get_col_data("step_idx"))
    selected_rows = _select_episode_balanced_eval_rows(
        episode_ids=episode_ids,
        step_ids=step_ids,
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        num_eval=int(cfg.eval.num_eval),
        seed=int(cfg.seed),
    )
    eval_episodes = episode_ids[selected_rows].astype(np.int64)
    eval_steps = step_ids[selected_rows].astype(np.int64)

    output_root = Path(str(cfg.output.root_dir)).expanduser()
    model_variant = (
        "adapter" if bool(cfg.model.use_adapter) else "base"
    )
    run_name = _planning_run_name(cfg)
    evaluation_run_name = f"eval_seed{int(cfg.seed)}"
    run_dir = (
        output_root
        / model_variant
        / run_name
        / evaluation_run_name
    )
    result_path = run_dir / "results.json"
    video_path = run_dir / "videos" if bool(cfg.output.video) else None

    world = swm.World(
        str(cfg.world.env_name),
        num_envs=len(eval_episodes),
        max_episode_steps=2 * int(cfg.eval.eval_budget),
        image_shape=(224, 224),
    )
    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_steps.tolist(),
        goal_offset=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video=video_path,
    )
    world.close()

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = getattr(
        policy,
        "adapter_checkpoint_metadata",
    )
    result = {
        **{
            key: _json_value(value)
            for key, value in metrics.items()
        },
        "backend": str(cfg.backend),
        "base_model_fingerprint": str(
            getattr(policy, "base_model_fingerprint")
        ),
        "adapter_checkpoint": str(
            Path(str(cfg.adapter_checkpoint)).expanduser()
        ),
        "run_name": run_name,
        "evaluation_run_name": evaluation_run_name,
        "output_directory": str(run_dir),
        "model_variant": model_variant,
        "use_adapter": bool(cfg.model.use_adapter),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_seed": int(cfg.seed),
        "evaluation_samples": {
            "episode_ids": eval_episodes.tolist(),
            "start_steps": eval_steps.tolist(),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "eval_budget": int(cfg.eval.eval_budget),
            "num_eval": len(eval_episodes),
        },
        "training_data_selection": checkpoint_metadata[
            "data_selection"
        ],
        "training_appearance": checkpoint_metadata[
            "appearance_training"
        ],
        "evaluation_appearance": OmegaConf.to_container(
            cfg.appearance,
            resolve=True,
        ),
        "planning": OmegaConf.to_container(cfg.plan, resolve=True),
        "cem": OmegaConf.to_container(cfg.cem, resolve=True),
    }
    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
