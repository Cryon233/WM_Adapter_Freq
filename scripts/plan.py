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

from wm_adapter_freq.data.paired_windows import (
    EPISODE_SPLIT_STRATEGY,
    split_episode_indices,
)
from wm_adapter_freq.planning.appearance_transform import (
    EVALUATION_PROTOCOL_VERSION,
)
from wm_adapter_freq.planning.policy_builder import build_tworoom_mpc_policy


def _select_heldout_eval_samples(
    episode_lengths: np.ndarray,
    eval_episode_indices: np.ndarray,
    goal_offset_steps: int,
    num_eval: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one valid start step from each sampled held-out episode."""
    lengths = np.asarray(episode_lengths, dtype=np.int64)
    eval_episodes = np.asarray(eval_episode_indices, dtype=np.int64)
    eligible_episodes = eval_episodes[
        lengths[eval_episodes] >= int(goal_offset_steps) + 1
    ]
    generator = np.random.default_rng(int(seed))
    selection_count = min(int(num_eval), eligible_episodes.size)
    if selection_count <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    selected_episodes = generator.choice(
        eligible_episodes,
        size=selection_count,
        replace=False,
    )
    selected_start_steps = np.asarray(
        [
            generator.integers(
                low=0,
                high=(
                    int(lengths[int(episode_index)])
                    - int(goal_offset_steps)
                ),
            )
            for episode_index in selected_episodes
        ],
        dtype=np.int64,
    )
    order = np.argsort(selected_episodes, kind="stable")
    return selected_episodes[order], selected_start_steps[order]


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

    checkpoint_metadata = getattr(
        policy,
        "adapter_checkpoint_metadata",
    )
    data_selection = checkpoint_metadata["data_selection"]
    if (
        str(data_selection["episode_split_strategy"])
        != EPISODE_SPLIT_STRATEGY
    ):
        raise RuntimeError(
            "Adapter checkpoint episode split strategy is unsupported."
        )
    if int(data_selection["source_episode_count"]) != len(dataset.lengths):
        raise RuntimeError(
            "Planning dataset episode count does not match the Adapter "
            "checkpoint."
        )
    _, eval_episode_indices = split_episode_indices(
        num_episodes=len(dataset.lengths),
        train_fraction=float(
            data_selection["episode_split_train_fraction"]
        ),
        seed=int(data_selection["episode_split_seed"]),
    )
    eval_episodes, eval_steps = _select_heldout_eval_samples(
        episode_lengths=np.asarray(dataset.lengths, dtype=np.int64),
        eval_episode_indices=eval_episode_indices,
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        num_eval=int(cfg.eval.num_eval),
        seed=int(cfg.seed),
    )

    output_root = Path(str(cfg.output.root_dir)).expanduser()
    model_variant = (
        "adapter" if bool(cfg.model.use_adapter) else "base"
    )
    run_name = _planning_run_name(cfg)
    protocol_name = (
        f"protocol_v{EVALUATION_PROTOCOL_VERSION.split('.', maxsplit=1)[0]}"
    )
    evaluation_run_name = f"eval_seed{int(cfg.seed)}"
    run_dir = (
        output_root
        / model_variant
        / run_name
        / protocol_name
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
        "planning_objective": getattr(
            policy,
            "planning_objective_metadata",
        ),
        "planner_profile": getattr(policy, "planner_profile"),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_seed": int(cfg.seed),
        "evaluation_samples": {
            "episode_partition": "eval",
            "episode_indices": eval_episodes.tolist(),
            "start_steps": eval_steps.tolist(),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "eval_budget": int(cfg.eval.eval_budget),
            "num_eval": len(eval_episodes),
        },
        "episode_split": {
            "strategy": data_selection["episode_split_strategy"],
            "seed": data_selection["episode_split_seed"],
            "train_fraction": data_selection[
                "episode_split_train_fraction"
            ],
            "source_episode_count": data_selection[
                "source_episode_count"
            ],
            "train_episode_count": data_selection[
                "train_episode_count"
            ],
            "eval_episode_count": data_selection[
                "eval_episode_count"
            ],
            "train_episode_indices_sha256": data_selection[
                "train_episode_indices_sha256"
            ],
            "eval_episode_indices_sha256": data_selection[
                "eval_episode_indices_sha256"
            ],
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
