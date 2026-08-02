from __future__ import annotations

from typing import Any

from wm_adapter.benchmarks.base import BenchmarkAdapter


def build_benchmark(cfg: Any) -> BenchmarkAdapter:
    name = str(cfg.get("benchmark", {}).get("name", "robocasa")).lower()
    if name == "robocasa":
        if "benchmark" not in cfg:
            task_name = str(cfg.data.task_name)
            task_key = str(cfg.planning.get("task_slug", task_name))
            cfg.benchmark = {
                "name": "robocasa",
                "suite": "legacy_single_stage",
                "task_key": task_key,
                "task_id": task_name,
                "task_name": task_name,
                "control_frequency_hz": 20.0,
            }
            if "max_episode_steps" not in cfg.evaluation:
                cfg.evaluation.max_episode_steps = 60
            if "goal_span_steps" not in cfg.evaluation:
                cfg.evaluation.goal_span_steps = 25
            if "episode_cap_basis" not in cfg.evaluation:
                cfg.evaluation.episode_cap_basis = (
                    "legacy pinned JEPA-WM RoboCasa protocol"
                )
        from wm_adapter.benchmarks.robocasa import RoboCasaBenchmark

        return RoboCasaBenchmark(cfg)
    if name == "libero":
        from wm_adapter.benchmarks.libero import LiberoBenchmark

        return LiberoBenchmark(cfg)
    raise ValueError(
        f"Unsupported benchmark {name!r}; expected 'robocasa' or 'libero'"
    )
