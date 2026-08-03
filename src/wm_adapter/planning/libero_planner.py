from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from wm_adapter.adapters.base import PEFTMethod
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.benchmarks.base import array_sha256
from wm_adapter.benchmarks.libero import LiberoBenchmark, LiberoTrajectoryDataset
from wm_adapter.planning.jepa_wm_planner import (
    JEPAWMPlanningModel,
    PlanningResult,
    _build_official_agent,
    frozen_goal_latent_fingerprint,
)
from wm_adapter.utils.reproducibility import resolve_path, seed_everything


LOGGER = logging.getLogger(__name__)


def _official_config(
    cfg: DictConfig,
    backend: JEPAWMDroidBackend,
    output: Path,
) -> DictConfig:
    from evals.simu_env_planning.planning.common.parser import parse_cfg

    official = OmegaConf.create(
        OmegaConf.to_container(backend.official_planning_template, resolve=False)
    )
    official.folder = str(output)
    OmegaConf.resolve(official)
    official.work_dir = output
    official.meta.seed = int(cfg.evaluation.eval_seed)
    official.meta.eval_episodes = int(cfg.evaluation.num_episodes)
    official.meta.quick_debug = False
    official.logging.optional_plots = False
    official.logging.save_csv = False
    official.logging.tqdm_silent = bool(cfg.evaluation.tqdm_silent)
    official.planner.decode_each_iteration = False
    official.planner.candidate_chunk_size = int(cfg.planning.candidate_chunk_size)
    official.task_specification.obs = "rgb"
    official.task_specification.goal_source = "dset"
    official.task_specification.done_at_succ = True
    official.task_specification.max_episode_steps = int(
        cfg.evaluation.max_episode_steps
    )
    mode = str(cfg.get("suite_mode", "formal"))
    if mode == "self_test":
        self_test = cfg.planning.self_test
        official.planner.iterations = int(self_test.iterations)
        official.planner.num_samples = int(self_test.num_samples)
        official.planner.num_elites = int(self_test.num_elites)
        official.planner.horizon = int(self_test.horizon)
        official.task_specification.max_episode_steps = int(
            self_test.max_episode_steps
        )
    elif mode != "formal":
        raise ValueError(f"Unknown suite_mode {mode!r}")
    if mode == "formal":
        required = {
            "iterations": 15,
            "num_samples": 300,
            "num_elites": 10,
            "horizon": 3,
            "num_act_stepped": 1,
        }
        actual = {key: int(official.planner[key]) for key in required}
        if actual != required:
            raise RuntimeError(
                f"Formal LIBERO CEM budget changed: expected={required}, actual={actual}"
            )
    official = parse_cfg(official)
    official.rank = 0
    official.world_size = 1
    official.device = str(backend.device)
    official.num_active_gpus = 1
    official.active_ranks = [0]
    official.local_seed = int(cfg.evaluation.eval_seed)
    official.frameskip = int(cfg.data.frameskip)
    official.action_ratio = 1
    return official


def _to_chw(image: np.ndarray) -> torch.Tensor:
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise RuntimeError(
            f"LIBERO planning image must be [H,W,3], received {values.shape}"
        )
    tensor = torch.from_numpy(values.copy()).permute(2, 0, 1)
    if tensor.dtype != torch.uint8:
        tensor = tensor.float()
        minimum = float(tensor.min())
        maximum = float(tensor.max())
        if minimum < 0.0 or maximum > 1.0:
            raise RuntimeError(
                f"LIBERO planning float image must be in [0,1], found [{minimum},{maximum}]"
            )
    return tensor


def run_libero_planning(
    *,
    experiment_config: DictConfig,
    backend: JEPAWMDroidBackend,
    method: PEFTMethod,
    output_directory: str | Path,
) -> PlanningResult:
    benchmark = LiberoBenchmark(experiment_config)
    task = benchmark.resolve_task(strict=True)
    manifest_path = benchmark.evaluation_manifest_path()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"LIBERO evaluation manifest does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = manifest.get("instances")
    total_episodes = int(experiment_config.evaluation.num_episodes)
    if not isinstance(instances, list) or len(instances) < total_episodes:
        raise RuntimeError(
            f"LIBERO manifest provides {0 if not isinstance(instances, list) else len(instances)} "
            f"instances, expected at least {total_episodes}"
        )
    selected = instances[:total_episodes]
    method_name = str(experiment_config.method)
    domain_name = str(experiment_config.domain)
    evaluation_seed = int(experiment_config.evaluation.eval_seed)
    seed_everything(evaluation_seed)
    output = resolve_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    official_cfg = _official_config(experiment_config, backend, output)
    source = benchmark.build_source_dataset(output_environment_info=True)
    base_spec = ComposedPhotometricShift().sample_spec(
        int(experiment_config.appearance.seed),
        float(experiment_config.appearance.severity),
    )
    planning_model = JEPAWMPlanningModel(
        backend,
        method,
        domain=domain_name,
        appearance_spec=base_spec,
        inference_precision=str(experiment_config.planning.inference_precision),
    ).eval()
    agent = _build_official_agent(
        official_cfg,
        planning_model,
        None,
        int(experiment_config.planning.candidate_chunk_size),
        int(experiment_config.planning.history_len),
    )
    LOGGER.info(
        "PLANNING_PROGRESS phase=job status=started method=%s domain=%s total_episodes=%d",
        method_name,
        domain_name,
        total_episodes,
    )
    successes: list[bool] = []
    environment_seeds: list[int] = []
    instance_ids: list[str] = []
    source_ids: list[str] = []
    initialization_fingerprints: list[str] = []
    goal_fingerprints: list[str] = []
    goal_base_latent_fingerprints: list[str] = []
    appearance_specs: list[dict[str, Any] | None] = []
    cem_seeds: list[int] = []
    environment: Any | None = None
    job_error: Exception | None = None
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    try:
        LOGGER.info(
            "PLANNING_PROGRESS phase=environment status=started method=%s domain=%s",
            method_name,
            domain_name,
        )
        environment_started = time.perf_counter()
        environment = benchmark._create_environment(task)
        LOGGER.info(
            "PLANNING_PROGRESS phase=environment status=completed method=%s domain=%s elapsed_seconds=%.3f",
            method_name,
            domain_name,
            time.perf_counter() - environment_started,
        )
        for episode, instance in enumerate(selected):
            episode_number = episode + 1
            episode_started = time.perf_counter()
            LOGGER.info(
                "PLANNING_PROGRESS phase=episode status=started method=%s domain=%s "
                "episode=%d total=%d completed=%d success_count=%d",
                method_name,
                domain_name,
                episode_number,
                total_episodes,
                episode,
                sum(successes),
            )
            trajectory = int(instance["source_trajectory_index"])
            start = int(instance["segment_start"])
            end = int(instance["segment_end"])
            observation, _, states, _, info = source.get_frames(
                trajectory, [start, end]
            )
            actual_initialization = array_sha256(states[0].numpy())
            actual_goal = array_sha256(observation["visual"][-1].numpy())
            if actual_initialization != str(instance["initialization_fingerprint"]) or actual_goal != str(instance["goal_fingerprint"]):
                raise RuntimeError(
                    "LIBERO evaluation instance fingerprint changed: "
                    f"instance={instance['instance_id']}, "
                    f"initialization={actual_initialization}, goal={actual_goal}"
                )
            environment_seed = int(instance["environment_seed"])
            environment.seed(environment_seed)
            environment.reset()
            current_observation = environment.set_init_state(states[0].numpy())
            current_image = benchmark._observation_image(current_observation)
            goal = observation["visual"][-1]
            goal_base_latent_fingerprint = frozen_goal_latent_fingerprint(
                backend, goal
            )
            if goal.dtype != torch.uint8:
                goal_tensor = goal.float()
            else:
                goal_tensor = goal
            appearance_spec = ComposedPhotometricShift().sample_spec(
                int(instance["appearance_seed"]),
                float(experiment_config.appearance.severity),
            )
            planning_model.appearance_spec = appearance_spec
            cem_seed = int(instance["cem_seed"])
            agent.local_generator.manual_seed(cem_seed)
            agent.local_gpu_generator.manual_seed(cem_seed)
            agent.set_goal(goal_tensor.unsqueeze(0))
            success = False
            max_environment_steps = int(
                official_cfg.task_specification.max_episode_steps
            )
            action_repeat = max(1, int(experiment_config.data.frameskip))
            max_planning_steps = int(
                np.ceil(max_environment_steps / action_repeat)
            )
            environment_steps = 0
            for step in range(max_planning_steps):
                action_tensor = agent.act(
                    _to_chw(current_image).unsqueeze(0),
                    steps_left=max_planning_steps - step,
                )
                action = benchmark.canonical_to_environment_action(
                    action_tensor.reshape(-1, 7)[0].numpy()
                )
                for _ in range(action_repeat):
                    current_observation, reward, done, step_info = environment.step(action)
                    environment_steps += 1
                    success = bool(
                        reward > 0
                        or step_info.get("success", False)
                        or environment.check_success()
                    )
                    if success or done:
                        break
                current_image = benchmark._observation_image(current_observation)
                LOGGER.info(
                    "LIBERO_STEP_PROGRESS episode=%d executing agent %d/%d success=%s",
                    episode_number,
                    step + 1,
                    max_planning_steps,
                    str(success).lower(),
                )
                if success or done or environment_steps >= max_environment_steps:
                    break
            successes.append(success)
            environment_seeds.append(environment_seed)
            instance_ids.append(str(instance["instance_id"]))
            source_ids.append(str(info["demo_key"]))
            initialization_fingerprints.append(
                str(instance["initialization_fingerprint"])
            )
            goal_fingerprints.append(str(instance["goal_fingerprint"]))
            goal_base_latent_fingerprints.append(goal_base_latent_fingerprint)
            appearance_specs.append(
                appearance_spec.as_dict() if domain_name == "ood" else None
            )
            cem_seeds.append(cem_seed)
            LOGGER.info(
                "PLANNING_PROGRESS phase=episode status=completed method=%s domain=%s "
                "episode=%d total=%d completed=%d success=%s success_count=%d elapsed_seconds=%.3f",
                method_name,
                domain_name,
                episode_number,
                total_episodes,
                len(successes),
                str(success).lower(),
                sum(successes),
                time.perf_counter() - episode_started,
            )
    except Exception as error:
        job_error = error
        LOGGER.exception(
            "PLANNING_PROGRESS phase=job status=failed method=%s domain=%s "
            "completed=%d total=%d error_type=%s",
            method_name,
            domain_name,
            len(successes),
            total_episodes,
            type(error).__name__,
        )
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                if job_error is None:
                    raise
                LOGGER.exception("Secondary LIBERO environment close failure")
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "PLANNING_PROGRESS phase=job status=completed method=%s domain=%s "
        "completed=%d total=%d success_count=%d elapsed_seconds=%.3f",
        method_name,
        domain_name,
        len(successes),
        total_episodes,
        sum(successes),
        elapsed,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(backend.device))
        if backend.device.type == "cuda"
        else 0
    )
    return PlanningResult(
        success_count=sum(successes),
        total_episodes=len(successes),
        success_rate=sum(successes) / len(successes),
        per_episode_success=successes,
        environment_seeds=environment_seeds,
        cem_seed=evaluation_seed,
        appearance_spec=(base_spec.as_dict() if domain_name == "ood" else None),
        elapsed_seconds=elapsed,
        peak_cuda_memory_bytes=peak_memory,
        evaluation_instance_ids=instance_ids,
        source_trajectory_ids=source_ids,
        initialization_fingerprints=initialization_fingerprints,
        goal_fingerprints=goal_fingerprints,
        goal_base_latent_fingerprints=goal_base_latent_fingerprints,
        appearance_specs=appearance_specs,
        cem_seeds=cem_seeds,
    )
