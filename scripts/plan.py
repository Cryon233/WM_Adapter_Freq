from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from wm_adapter_freq.envs.appearance_render_wrapper import (
    TwoRoomAppearanceRenderWrapper,
)
from wm_adapter_freq.planning.policy_builder import build_tworoom_mpc_policy


def _episode_column(dataset: Any) -> str:
    names = set(dataset.column_names)
    names.update(getattr(dataset, "_schema_names", ()))
    return "episode_idx" if "episode_idx" in names else "ep_idx"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@hydra.main(
    version_base=None,
    config_path="../configs/plan",
    config_name="prejepa_tworoom",
)
def main(cfg: DictConfig) -> None:
    import stable_worldmodel as swm

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
    episode_ids = dataset.get_col_data(episode_column)
    step_ids = dataset.get_col_data("step_idx")
    unique_episodes = np.unique(episode_ids)
    max_start = {
        episode: int(step_ids[episode_ids == episode].max())
        - int(cfg.eval.goal_offset_steps)
        for episode in unique_episodes
    }
    valid_rows = np.flatnonzero(
        np.asarray(
            [
                step <= max_start[episode]
                for episode, step in zip(episode_ids, step_ids)
            ]
        )
    )
    generator = np.random.default_rng(int(cfg.seed))
    selected_rows = np.sort(
        generator.choice(
            valid_rows,
            size=int(cfg.eval.num_eval),
            replace=False,
        )
    )
    eval_episodes = episode_ids[selected_rows].astype(np.int64)
    eval_steps = step_ids[selected_rows].astype(np.int64)

    world = swm.World(
        str(cfg.world.env_name),
        num_envs=int(cfg.eval.num_eval),
        max_episode_steps=2 * int(cfg.eval.eval_budget),
        image_shape=(224, 224),
        pre_wrappers=[
            functools.partial(
                TwoRoomAppearanceRenderWrapper,
                enabled=bool(cfg.appearance.enabled),
                shift_type=str(cfg.appearance.shift_type),
                severity=float(cfg.appearance.severity),
                base_seed=int(cfg.appearance.seed),
            )
        ],
    )
    world.set_policy(policy)
    video_path = (
        Path(str(cfg.output.video_dir)).expanduser()
        if cfg.output.video
        else None
    )
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

    result_path = Path(str(cfg.output.result_path)).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
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
        "appearance": OmegaConf.to_container(
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
