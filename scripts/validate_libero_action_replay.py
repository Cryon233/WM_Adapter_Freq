from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wm_adapter.benchmarks.base import ActionTransform, atomic_json
from wm_adapter.benchmarks.factory import build_benchmark
from wm_adapter.utils.reproducibility import load_experiment_config, resolve_path, seed_everything


def _sim(environment: Any) -> Any:
    for candidate in (environment, getattr(environment, "env", None)):
        sim = getattr(candidate, "sim", None)
        if sim is not None:
            return sim
    raise RuntimeError("LIBERO replay validator cannot locate the MuJoCo simulator")


def _capture_sim_state(sim: Any) -> dict[str, np.ndarray | None]:
    qpos = np.asarray(sim.data.qpos, dtype=np.float64).reshape(-1).copy()
    qvel = np.asarray(sim.data.qvel, dtype=np.float64).reshape(-1).copy()
    flattened = np.asarray(sim.get_state().flatten(), dtype=np.float64).reshape(-1)
    expected_prefix = 1 + qpos.size + qvel.size
    additional = (
        flattened[expected_prefix:].copy()
        if flattened.size > expected_prefix
        else None
    )
    return {"qpos": qpos, "qvel": qvel, "additional": additional}


def _state_metrics(
    sim: Any, target: dict[str, np.ndarray | None]
) -> dict[str, float]:
    current = _capture_sim_state(sim)
    qpos = current["qpos"]
    qvel = current["qvel"]
    target_qpos = target["qpos"]
    target_qvel = target["qvel"]
    if qpos is None or qvel is None or target_qpos is None or target_qvel is None:
        raise RuntimeError("LIBERO simulator did not expose qpos/qvel state")
    if qpos.shape != target_qpos.shape or qvel.shape != target_qvel.shape:
        raise RuntimeError(
            "LIBERO replay simulator-state shape changed: "
            f"qpos={qpos.shape}/{target_qpos.shape}, "
            f"qvel={qvel.shape}/{target_qvel.shape}"
        )
    metrics = {
        "qpos_mae": float(np.mean(np.abs(qpos - target_qpos))),
        "qvel_mae": float(np.mean(np.abs(qvel - target_qvel))),
    }
    current_additional = current["additional"]
    target_additional = target["additional"]
    if current_additional is not None and target_additional is not None:
        if current_additional.shape != target_additional.shape:
            raise RuntimeError(
                "LIBERO additional simulator-state shape changed: "
                f"actual={current_additional.shape}, target={target_additional.shape}"
            )
        metrics["additional_state_mae"] = float(
            np.mean(np.abs(current_additional - target_additional))
        )
    return metrics


def _image_metrics(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    actual_float, target_float = np.asarray(actual, dtype=np.float32), np.asarray(target, dtype=np.float32)
    if actual_float.shape != target_float.shape:
        raise RuntimeError(
            f"LIBERO replay image shape mismatch: actual={actual_float.shape}, target={target_float.shape}"
        )
    mae = float(np.mean(np.abs(actual_float - target_float)))
    mse = float(np.mean((actual_float - target_float) ** 2))
    psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))
    return {"agentview_image_mae": mae, "agentview_psnr": psnr}


def _eef_metrics(actual: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    position_keys = ("robot0_eef_pos", "ee_pos", "eef_pos")
    rotation_keys = ("robot0_eef_quat", "ee_quat", "eef_quat")
    position_key = next((key for key in position_keys if key in actual and key in target), None)
    rotation_key = next((key for key in rotation_keys if key in actual and key in target), None)
    if position_key is None or rotation_key is None:
        raise RuntimeError(
            "LIBERO replay cannot measure EEF parity because observations lack a "
            "shared position/quaternion contract: "
            f"actual_keys={sorted(actual)}, target_keys={sorted(target)}"
        )
    position = np.asarray(actual[position_key], dtype=np.float64).reshape(-1)
    target_position = np.asarray(target[position_key], dtype=np.float64).reshape(-1)
    rotation = np.asarray(actual[rotation_key], dtype=np.float64).reshape(-1)
    target_rotation = np.asarray(target[rotation_key], dtype=np.float64).reshape(-1)
    if position.size != 3 or target_position.size != 3 or rotation.size != 4 or target_rotation.size != 4:
        raise RuntimeError(
            "LIBERO EEF observation shapes are invalid: "
            f"position={position.shape}, target_position={target_position.shape}, "
            f"rotation={rotation.shape}, target_rotation={target_rotation.shape}"
        )
    rotation /= max(np.linalg.norm(rotation), 1.0e-12)
    target_rotation /= max(np.linalg.norm(target_rotation), 1.0e-12)
    return {
        "eef_translation_error": float(np.linalg.norm(position - target_position)),
        "eef_rotation_error": float(1.0 - abs(np.dot(rotation, target_rotation))),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    shared_keys = set.intersection(*(set(row) for row in rows))
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in sorted(shared_keys)
    }


def main() -> None:
    cfg = load_experiment_config()
    if str(cfg.benchmark.name) != "libero":
        output = resolve_path(cfg.protocol_validation.output)
        atomic_json(output, {"status": "not_applicable", "benchmark": str(cfg.benchmark.name)})
        print(f"PROTOCOL_PROGRESS task={cfg.benchmark.task_key} completed=1 total=1")
        return
    seed_everything(int(cfg.evaluation.eval_seed))
    benchmark = build_benchmark(cfg)
    task = benchmark.resolve_task(strict=True)
    source = benchmark.build_source_dataset(output_environment_info=True)
    episodes = min(int(cfg.protocol_validation.get("episodes", 5)), len(source))
    starts_per_episode = int(cfg.protocol_validation.get("starts_per_episode", 20))
    transform = ActionTransform.from_dict(task.action_transform or {})
    environment = benchmark._create_environment(task)
    sequence_rows: list[dict[str, float]] = []
    repeated_rows: list[dict[str, float]] = []
    total = episodes * starts_per_episode
    completed = 0
    try:
        for episode in range(episodes):
            length = int(source.get_seq_length(episode))
            if length < 6 + starts_per_episode:
                raise RuntimeError(
                    f"LIBERO replay episode {episode} is too short: length={length}"
                )
            starts = np.linspace(0, length - 6, starts_per_episode, dtype=np.int64)
            for start in starts.tolist():
                frames = list(range(start, start + 6))
                observation, canonical_actions, states, _, _ = source.get_frames(episode, frames)
                environment_actions = transform.canonical_to_environment_action(canonical_actions.numpy())
                target_image = observation["visual"][-1].permute(1, 2, 0).numpy()
                environment.reset()
                target_environment_observation = environment.set_init_state(
                    states[5].numpy()
                )
                target_sim_state = _capture_sim_state(_sim(environment))
                rows_for_modes: list[dict[str, float]] = []
                for repeated in (False, True):
                    environment.reset()
                    current = environment.set_init_state(states[0].numpy())
                    for offset in range(5):
                        action = environment_actions[0 if repeated else offset]
                        current, _, _, _ = environment.step(action)
                    metrics = _state_metrics(_sim(environment), target_sim_state)
                    metrics.update(
                        _image_metrics(benchmark._observation_image(current), target_image)
                    )
                    metrics.update(_eef_metrics(current, target_environment_observation))
                    rows_for_modes.append(metrics)
                sequence_rows.append(rows_for_modes[0])
                repeated_rows.append(rows_for_modes[1])
                completed += 1
                print(
                    f"PROTOCOL_PROGRESS task={task.task_key} completed={completed} total={total}",
                    flush=True,
                )
    finally:
        environment.close()
    sequence = _aggregate(sequence_rows)
    repeated = _aggregate(repeated_rows)
    stable_state_metrics = ["qpos_mae", "qvel_mae"]
    if "additional_state_mae" in sequence and "additional_state_mae" in repeated:
        stable_state_metrics.append("additional_state_mae")
    sequence_state = sum(sequence[key] for key in stable_state_metrics)
    repeated_state = sum(repeated[key] for key in stable_state_metrics)
    sequence_limit = float(cfg.protocol_validation.get("sequence_state_mae_max", 1.0e-3))
    sequence_passed = bool(np.isfinite(sequence_state) and sequence_state <= sequence_limit)
    repeat_passed = bool(repeated_state <= sequence_state * 1.05 + 1.0e-8)
    payload = {
        "schema_version": "libero_action_replay_contract_v1",
        "task": task.task_key,
        "episodes": episodes,
        "starts_per_episode": starts_per_episode,
        "sequence_replay": sequence,
        "repeated_action_replay": repeated,
        "state_comparison": {
            "metrics": stable_state_metrics,
            "object_state": (
                "included in simulator qpos; no unsupported flattened-state slicing"
            ),
            "additional_state": (
                "supported"
                if "additional_state_mae" in stable_state_metrics
                else "unsupported by the active simulator state binding"
            ),
        },
        "thresholds": {"sequence_state_mae_max": sequence_limit},
        "sequence_replay_contract": "passed" if sequence_passed else "failed",
        "repeat_action_contract": "passed" if repeat_passed else "failed",
        "action_transform": task.action_transform,
        "status": "passed" if sequence_passed and repeat_passed else "failed",
        "failure_reasons": [
            reason
            for condition, reason in (
                (
                    not sequence_passed,
                    f"sequence state error {sequence_state} exceeds {sequence_limit}",
                ),
                (
                    not repeat_passed,
                    "repeated-action replay is not distinguishable from sequence replay",
                ),
            )
            if condition
        ],
    }
    output = resolve_path(cfg.protocol_validation.output)
    atomic_json(output, payload)
    if not sequence_passed:
        raise RuntimeError(
            f"LIBERO sequence replay contract failed for {task.task_key}: "
            f"state_mae={sequence_state}, limit={sequence_limit}"
        )
    if not repeat_passed:
        raise RuntimeError(
            f"LIBERO repeated-action contract failed for {task.task_key}: "
            f"sequence_state_error={sequence_state}, repeated_state_error={repeated_state}; "
            "consider frameskip=1 or a 35-D action-sequence encoder"
        )


if __name__ == "__main__":
    main()
