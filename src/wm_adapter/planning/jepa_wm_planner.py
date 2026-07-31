from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from wm_adapter.adapters.base import PEFTMethod
from wm_adapter.appearance.composed_photometric import (
    APPEARANCE_PIPELINE_VERSION,
    AppearanceShiftSpec,
    ComposedPhotometricShift,
)
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.data.robocasa_windows import build_robocasa_dataset, split_episode_indices
from wm_adapter.utils.reproducibility import resolve_path, seed_everything


EVALUATION_PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True)
class PlanningResult:
    success_count: int
    total_episodes: int
    success_rate: float
    per_episode_success: list[bool]
    environment_seeds: list[int]
    cem_seed: int
    appearance_spec: dict[str, Any] | None
    elapsed_seconds: float
    peak_cuda_memory_bytes: int


class JEPAWMPlanningModel(nn.Module):
    def __init__(
        self,
        backend: JEPAWMDroidBackend,
        method: PEFTMethod,
        *,
        domain: str,
        appearance_spec: AppearanceShiftSpec,
    ) -> None:
        super().__init__()
        if domain not in {"clean", "ood"}:
            raise ValueError(f"Planning domain must be clean or ood, received {domain!r}")
        self.backend = backend
        self.method = method
        self.domain = domain
        self.appearance = ComposedPhotometricShift()
        self.appearance_spec = appearance_spec
        self._encoding_goal = False
        self.action_dim = backend.official_model.action_dim
        self.tubelet_size_enc = backend.official_model.tubelet_size_enc
        self.action_skip = backend.official_model.action_skip
        self.preprocessor = backend.preprocessor
        self.decode_unroll = None

    @contextmanager
    def goal_encoding(self) -> Iterator[None]:
        previous = self._encoding_goal
        self._encoding_goal = True
        try:
            yield
        finally:
            self._encoding_goal = previous

    def _current_only_shift(self, images: Tensor) -> Tensor:
        if self.domain == "clean" or self._encoding_goal:
            return images
        shifted = [self.appearance.apply(sequence, self.appearance_spec) for sequence in images]
        return torch.stack(shifted, dim=0)

    @torch.no_grad()
    def encode(self, observations: Tensor, act: bool = True) -> Tensor:
        del act
        if observations.ndim != 5:
            raise ValueError(
                f"Official planning observations must be [B,T,3,H,W], received {tuple(observations.shape)}"
            )
        images = self._current_only_shift(observations)
        batch, time_steps = images.shape[:2]
        latents = self.backend.encode_images(images, self.method, batch, time_steps)
        return self.backend.planning_latents(latents)

    @torch.no_grad()
    def unroll(self, z_ctxt: Tensor, act_suffix: Tensor | None = None, debug: bool = False) -> Tensor:
        if act_suffix is None:
            raise ValueError("JEPA-WM planning unroll requires action candidates")
        return self.backend.official_model.unroll(z_ctxt, act_suffix=act_suffix, debug=debug)


def _build_official_agent(
    cfg: DictConfig,
    model: JEPAWMPlanningModel,
    dataset: Any,
    candidate_chunk_size: int,
) -> Any:
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner

    class FixedGoalAgent(GC_Agent):
        @torch.no_grad()
        def set_goal(self, goal_state: Tensor) -> None:
            with self.model.goal_encoding():
                super().set_goal(goal_state)

    class CandidateChunkedCEMPlanner(CEMPlanner):
        def __init__(self, *args: Any, chunk_size: int, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if chunk_size <= 0:
                raise ValueError(f"CEM candidate chunk size must be positive, received {chunk_size}")
            self.chunk_size = chunk_size

        def cost_function(self, actions: Tensor, z_init: Tensor) -> Tensor:
            costs = [
                super(CandidateChunkedCEMPlanner, self).cost_function(
                    actions[:, start : start + self.chunk_size], z_init
                )
                for start in range(0, actions.shape[1], self.chunk_size)
            ]
            result = torch.cat(costs, dim=0)
            if result.shape[0] != actions.shape[1]:
                raise RuntimeError(
                    f"Chunked CEM returned {tuple(result.shape)} costs for {actions.shape[1]} candidates"
                )
            return result

    agent = FixedGoalAgent(cfg, model, dset=dataset, preprocessor=model.preprocessor)
    planner_values = OmegaConf.to_container(cfg.planner, resolve=True)
    if not isinstance(planner_values, dict):
        raise TypeError(f"Planner config must be a mapping, found {type(planner_values).__name__}")
    planner_values.pop("candidate_chunk_size", None)
    agent.planner = CandidateChunkedCEMPlanner(
        unroll=model.unroll,
        action_dim=model.action_dim,
        action_masks=None,
        local_generator=agent.local_gpu_generator,
        decode_unroll=model.decode_unroll,
        chunk_size=candidate_chunk_size,
        **planner_values,
    )
    return agent


def run_robocasa_planning(
    *,
    experiment_config: DictConfig,
    backend: JEPAWMDroidBackend,
    method: PEFTMethod,
    output_directory: str | Path,
) -> PlanningResult:
    from app.plan_common.datasets.traj_dset import TrajSubset
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.common.parser import parse_cfg
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator

    evaluation_seed = int(experiment_config.evaluation.eval_seed)
    if str(experiment_config.appearance.pipeline_version) != APPEARANCE_PIPELINE_VERSION:
        raise ValueError(
            f"Unsupported appearance pipeline version: {experiment_config.appearance.pipeline_version}"
        )
    seed_everything(evaluation_seed)
    output = resolve_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    official_cfg = OmegaConf.create(
        OmegaConf.to_container(backend.official_planning_template, resolve=True)
    )
    official_cfg.work_dir = output
    official_cfg.meta.seed = evaluation_seed
    official_cfg.meta.eval_episodes = int(experiment_config.evaluation.num_episodes)
    official_cfg.meta.quick_debug = False
    official_cfg.logging.optional_plots = False
    official_cfg.logging.save_csv = False
    official_cfg.logging.tqdm_silent = bool(experiment_config.evaluation.tqdm_silent)
    official_cfg.planner.decode_each_iteration = False
    official_cfg.planner.candidate_chunk_size = int(experiment_config.planning.candidate_chunk_size)
    official_cfg = parse_cfg(official_cfg)
    official_cfg.rank = 0
    official_cfg.world_size = 1
    official_cfg.device = str(backend.device)
    official_cfg.num_active_gpus = 1
    official_cfg.active_ranks = [0]
    official_cfg.local_seed = evaluation_seed
    official_cfg.frameskip = int(official_cfg.model_kwargs.data.custom.frameskip)
    official_cfg.action_ratio = 1 if official_cfg.planner.repeat_actskip else official_cfg.frameskip // backend.official_model.action_skip

    source_dataset = build_robocasa_dataset(
        jepa_wms_root=backend.jepa_repo,
        dataset_root=experiment_config.paths.dataset_root,
        hdf5_path=experiment_config.paths.robocasa_hdf5,
        task_name=str(experiment_config.data.task_name),
        camera_view=str(experiment_config.data.camera_view),
        output_environment_info=True,
        transform=backend.preprocessor.transform,
    )
    _, evaluation_episodes = split_episode_indices(
        len(source_dataset),
        float(experiment_config.data.train_fraction),
        int(experiment_config.data.split_seed),
    )
    dataset = TrajSubset(source_dataset, evaluation_episodes.tolist())
    appearance_spec = ComposedPhotometricShift().sample_spec(
        int(experiment_config.appearance.seed), float(experiment_config.appearance.severity)
    )
    planning_model = JEPAWMPlanningModel(
        backend,
        method,
        domain=str(experiment_config.domain),
        appearance_spec=appearance_spec,
    ).eval()
    agent = _build_official_agent(
        official_cfg,
        planning_model,
        dataset,
        int(experiment_config.planning.candidate_chunk_size),
    )
    environment = make_env(official_cfg)
    evaluator = PlanEvaluator(official_cfg, agent)
    successes: list[bool] = []
    environment_seeds: list[int] = []
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    try:
        for episode in range(int(experiment_config.evaluation.num_episodes)):
            episode_seed = (evaluation_seed * evaluation_seed + episode * evaluation_seed) % (2**32 - 2)
            result = evaluator.eval(official_cfg, agent, environment, task_idx=-1, ep=episode)
            successes.append(bool(result[1]))
            environment_seeds.append(int(episode_seed))
    finally:
        environment.close()
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(backend.device)) if backend.device.type == "cuda" else 0
    )
    success_count = sum(successes)
    return PlanningResult(
        success_count=success_count,
        total_episodes=len(successes),
        success_rate=success_count / len(successes),
        per_episode_success=successes,
        environment_seeds=environment_seeds,
        cem_seed=evaluation_seed,
        appearance_spec=appearance_spec.as_dict() if experiment_config.domain == "ood" else None,
        elapsed_seconds=elapsed,
        peak_cuda_memory_bytes=peak_memory,
    )


def save_planning_results(path: str | Path, result: PlanningResult, metadata: dict[str, Any]) -> None:
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, **asdict(result)}
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
