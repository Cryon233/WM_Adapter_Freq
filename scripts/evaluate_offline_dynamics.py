from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_adapter.adapters.base import BaseMethod, PEFTMethod
from wm_adapter.adapters.factory import build_method
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.backends.factory import build_backend
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.backends.frozen_projection import frozen_base_projection
from wm_adapter.benchmarks.factory import build_benchmark
from wm_adapter.data.feature_cache_v2 import (
    CACHE_SCHEMA_VERSION_V2,
    cache_file_sha256_from_verified_state_v2,
)
from wm_adapter.experiments.cross_benchmark import (
    normalize_metadata_contract,
    training_contract_mismatches_v2,
    training_contract_v2,
)
from wm_adapter.training.trainer_v2 import CHECKPOINT_SCHEMA_V2
from wm_adapter.utils.checkpoints import load_method_checkpoint, sha256_file
from wm_adapter.utils.reproducibility import (
    load_experiment_config,
    resolve_path,
    seed_everything,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _backend(cfg: Any) -> JEPAWMDroidBackend:
    return build_backend(cfg.model, device=cfg.device)


def _load_method(
    cfg: Any,
    backend: JEPAWMDroidBackend,
    resolved_task: Any,
) -> tuple[PEFTMethod, str | None, dict[str, Any], bool]:
    method = build_method(str(cfg.method), backend, cfg.method_config).to(backend.device)
    if method.method_name == "base":
        return method, None, {}, False
    checkpoint_path = resolve_path(cfg.paths.method_checkpoint)
    checkpoint = load_method_checkpoint(checkpoint_path)
    if checkpoint.get("schema_version") == "wm_adapter_checkpoint_v2":
        raise RuntimeError(
            "Offline evaluation rejects obsolete v2 checkpoints without the "
            f"cache-file integrity contract: {checkpoint_path}"
        )
    checkpoint_v2 = checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_V2
    checkpoint_backend = str(checkpoint.get("backend", "jepa_wm_droid"))
    if checkpoint_backend != backend.backend_name:
        raise RuntimeError(
            "Offline checkpoint backend mismatch: "
            f"expected={backend.backend_name}, actual={checkpoint_backend}, "
            f"path={checkpoint_path}"
        )
    checkpoint_encoder = checkpoint.get(
        "encoder_checkpoint_sha256",
        checkpoint.get("dinov3_checkpoint_sha256"),
    )
    if checkpoint_encoder != backend.encoder_checkpoint_sha256:
        raise RuntimeError(
            f"Offline checkpoint visual-encoder fingerprint mismatch: {checkpoint_path}"
        )
    data_metadata = dict(checkpoint.get("data_metadata", {}))
    actual_identity = (
        str(data_metadata.get("benchmark", "")),
        str(data_metadata.get("task_key", "")),
    )
    standard_checkpoint = resolve_path(
        "checkpoints/cross_benchmark_v1/"
        f"{resolved_task.benchmark}/{resolved_task.task_key}/{method.method_name}_final.pt"
    )
    legacy_place = (
        resolved_task.benchmark == "robocasa"
        and resolved_task.task_key == "robocasa_place"
        and checkpoint_path != standard_checkpoint
        and actual_identity == ("", "")
    )
    expected_appearance = ComposedPhotometricShift.metadata(
        float(cfg.appearance.get("training_severity", cfg.appearance.severity)),
        int(cfg.appearance.training_seed),
    )
    expected = {
        "method_name": method.method_name,
        "method_config": method.config_dict(),
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "backend": backend.backend_name,
        "encoder_checkpoint_sha256": backend.encoder_checkpoint_sha256,
        "encoder_name": backend.encoder_name,
        "predictor_depth": backend.predictor_depth,
        "upstream_commits": backend.upstream_commits,
    }
    if not checkpoint_v2:
        expected["appearance_metadata"] = expected_appearance
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Offline method checkpoint does not match the configured run: {mismatches}"
        )
    training_config = dict(checkpoint.get("training_config", {}))
    expected_training = (
        {
            "seed": int(cfg.training.seed),
            "canonical_weight": float(cfg.training.get("canonical_weight", 1.0)),
            "dynamics_weight": float(cfg.training.get("dynamics_weight", 1.0)),
        }
        if not checkpoint_v2
        else {
            "seed": int(cfg.training.seed),
            "max_optimizer_steps": int(cfg.training.max_optimizer_steps),
        }
    )
    actual_training = {
        key: training_config.get(
            key,
            1.0 if legacy_place and key in {"canonical_weight", "dynamics_weight"} else -1,
        )
        for key in expected_training
    }
    training_mismatch = {
        key: {"expected": value, "actual": training_config.get(key)}
        for key, value in expected_training.items()
        if float(actual_training[key]) != float(value)
    }
    if training_mismatch:
        raise RuntimeError(
            f"Offline checkpoint training-contract mismatch: {training_mismatch}"
        )
    if checkpoint_v2:
        training_values = OmegaConf.to_container(cfg.training, resolve=True)
        if not isinstance(training_values, dict):
            raise TypeError("V2 offline training configuration is not a mapping")
        mismatch = training_contract_mismatches_v2(
            checkpoint, training_contract_v2(training_values)
        )
        if mismatch:
            raise RuntimeError(f"Offline V2 checkpoint mismatch: {mismatch}")
    method.load_method_checkpoint(checkpoint["peft_state_dict"])
    return method.eval(), sha256_file(checkpoint_path), data_metadata, legacy_place


def _sample_mse(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise RuntimeError(
            f"Offline latent shape mismatch: predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}"
        )
    return (predicted.float() - target.float()).square().mean(
        dim=tuple(range(1, predicted.ndim))
    )


def _rollout_errors(
    backend: JEPAWMDroidBackend,
    context: Tensor,
    target: Tensor,
    action_suffix: Tensor,
) -> Tensor:
    if action_suffix.ndim != 3 or action_suffix.shape[1] != 3:
        raise ValueError(
            "Offline rollout actions must have shape [B,3,A], "
            f"received {tuple(action_suffix.shape)}"
        )
    predicted_future = backend.differentiable_unroll(context, action_suffix)
    return torch.stack(
        [_sample_mse(predicted_future[:, index], target[:, index]) for index in range(3)],
        dim=1,
    )


def _action_shuffle_permutation(shuffle_seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
    permutation = torch.randperm(3, generator=generator, device="cpu")
    identity = torch.arange(3, dtype=torch.int64)
    if torch.equal(permutation, identity):
        permutation = torch.tensor([1, 2, 0], dtype=torch.int64)
    return permutation


def main() -> None:
    cfg = load_experiment_config()
    metric_profile = str(cfg.offline.get("metric_profile", "legacy_dynamics"))
    standard_cross_backend = metric_profile == "cross_backend_mse_v1"
    offline_seed = int(cfg.offline.get("seed", cfg.evaluation.eval_seed))
    shuffle_seed = int(cfg.offline.get("shuffle_seed", offline_seed + 100003))
    shuffle_permutation = _action_shuffle_permutation(shuffle_seed)
    seed_everything(offline_seed)
    backend = _backend(cfg)
    benchmark = build_benchmark(cfg)
    resolved_task = benchmark.resolve_task(strict=True)
    (
        method,
        method_checkpoint_sha256,
        checkpoint_data_metadata,
        legacy_place,
    ) = _load_method(
        cfg, backend, resolved_task
    )
    if method.method_name != "base":
        actual_identity = (
            str(checkpoint_data_metadata.get("benchmark", "")),
            str(checkpoint_data_metadata.get("task_key", "")),
        )
        expected_identity = (resolved_task.benchmark, resolved_task.task_key)
        if actual_identity != expected_identity and not legacy_place:
            raise RuntimeError(
                "Offline checkpoint benchmark/task mismatch: "
                f"expected={expected_identity}, actual={actual_identity}"
            )
        if not legacy_place:
            expected_contract = {
                "task_manifest_sha256": resolved_task.as_dict()[
                    "task_manifest_sha256"
                ],
                "dataset_sha256": resolved_task.dataset_sha256,
                "camera_key": resolved_task.camera_key,
                "camera_height": resolved_task.camera_height,
                "camera_width": resolved_task.camera_width,
                "camera_channel_order": resolved_task.camera_channel_order,
                "camera_vertical_flip": resolved_task.camera_vertical_flip,
                "action_convention": resolved_task.action_convention,
                "action_transform": resolved_task.action_transform,
                "dataset_format": resolved_task.dataset_format,
                "dataset_source_identifier": resolved_task.dataset_source_identifier,
                "dataset_revision": resolved_task.dataset_revision,
                "robot": resolved_task.robot,
                "gripper": resolved_task.gripper,
                "controller_contract": resolved_task.controller_contract,
                "task_upstream_commits": resolved_task.upstream_commits,
            }
            mismatch = {
                key: {
                    "expected": normalize_metadata_contract(value),
                    "actual": normalize_metadata_contract(
                        checkpoint_data_metadata.get(key)
                    ),
                }
                for key, value in expected_contract.items()
                if normalize_metadata_contract(checkpoint_data_metadata.get(key))
                != normalize_metadata_contract(value)
            }
            if mismatch:
                raise RuntimeError(
                    f"Offline checkpoint task/data contract mismatch: {mismatch}"
                )
    source = benchmark.build_source_dataset(output_environment_info=False)
    _, evaluation_episodes = benchmark.split_trajectory_ids(source)
    num_frames = int(cfg.offline.num_frames)
    if num_frames != 6:
        raise ValueError(f"Offline one/two/three-step evaluation requires num_frames=6, found {num_frames}")
    candidates = benchmark.enumerate_window_candidates(
        source, num_frames, int(cfg.data.frameskip)
    )
    selected = benchmark.select_windows(
        candidates,
        evaluation_episodes,
        int(cfg.offline.num_windows),
        offline_seed,
    )
    unique_window_count = len(set(selected))
    if unique_window_count != len(selected):
        raise RuntimeError(
            "Offline evaluation selected duplicate physical windows: "
            f"selected={len(selected)}, unique={unique_window_count}"
        )
    allow_available_unique = (
        str(cfg.get("suite", {}).get("name", ""))
        == "cross_backend_adapter_v1"
        and resolved_task.benchmark == "robocasa"
        and str(cfg.planning.get("subtask", "")) in {"reach", "place"}
    )
    if len(selected) != int(cfg.offline.num_windows) and not allow_available_unique:
        raise RuntimeError(
            f"Offline evaluation requested {cfg.offline.num_windows} held-out windows, "
            f"but selected {len(selected)}"
        )
    if not selected:
        raise RuntimeError("Offline evaluation has no eligible unique windows")
    negative_count = int(cfg.offline.get("negative_actions", 0))
    action_bank: list[tuple[int, Tensor]] = []
    if negative_count:
        evaluation_set = {int(value) for value in evaluation_episodes.tolist()}
        bank_limit = max(512, negative_count * 8)
        grouped_candidates: dict[int, list[int]] = {}
        for trajectory, start in candidates:
            if int(trajectory) in evaluation_set:
                grouped_candidates.setdefault(int(trajectory), []).append(int(start))
        bank_pairs = [
            (trajectory, starts[offset])
            for offset in range(max((len(values) for values in grouped_candidates.values()), default=0))
            for trajectory, starts in sorted(grouped_candidates.items())
            if offset < len(starts)
        ][:bank_limit]
        for trajectory, start in bank_pairs:
            indices = [
                start + offset * int(cfg.data.frameskip) for offset in range(6)
            ]
            _, bank_actions, _, _, _ = source.get_frames(trajectory, indices)
            action_bank.append((int(trajectory), bank_actions[2:5].float()))
    windows = benchmark.make_window_dataset(
        source,
        selected,
        num_frames=num_frames,
        frameskip=int(cfg.data.frameskip),
        appearance_seed=int(cfg.appearance.seed),
        appearance_severity=float(cfg.appearance.severity),
    )
    loader = DataLoader(
        windows,
        batch_size=int(cfg.offline.batch_size),
        shuffle=False,
        num_workers=int(cfg.offline.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.offline.num_workers) > 0,
    )
    domains = [str(cfg.domain)] if str(cfg.domain) in {"clean", "ood"} else ["clean", "ood"]
    output = resolve_path(cfg.offline.output_directory)
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Offline results will not be overwritten: {output / 'metrics.json'}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    counts = {domain: 0 for domain in domains}
    base_method = BaseMethod().to(backend.device)
    configured_cache = resolve_path(cfg.paths.feature_cache)
    with h5py.File(configured_cache, "r", libver="latest", swmr=True) as cache_handle:
        cache_schema_version = str(cache_handle.attrs.get("schema_version", ""))
        cache_fingerprint = str(cache_handle.attrs.get("cache_fingerprint", ""))
    cache_file_sha256 = (
        cache_file_sha256_from_verified_state_v2(
            configured_cache,
            expected_sha256=cfg.cache.get("expected_file_sha256"),
            expected_size=cfg.cache.get("expected_file_size"),
            expected_mtime_ns=cfg.cache.get("expected_file_mtime_ns"),
        )
        if cache_schema_version == CACHE_SCHEMA_VERSION_V2
        else sha256_file(configured_cache)
    )
    if (
        method.method_name != "base"
        and not legacy_place
        and checkpoint_data_metadata.get("cache_file_sha256")
        != cache_file_sha256
    ):
        raise RuntimeError(
            "Offline checkpoint/cache file fingerprint mismatch: "
            f"checkpoint={checkpoint_data_metadata.get('cache_file_sha256')}, "
            f"cache={cache_file_sha256}, path={configured_cache}"
        )
    precision = str(cfg.planning.inference_precision)
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if backend.device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode():
        for batch in tqdm(loader, desc="offline held-out windows"):
            clean = batch["clean_images"]
            batch_size = clean.shape[0]
            actions = batch["actions"].to(backend.device, non_blocking=True).float()
            rollout_actions = actions[:, 2:5]
            if not standard_cross_backend:
                shuffled_rollout_actions = rollout_actions.index_select(
                    1, shuffle_permutation.to(device=actions.device)
                )
                zero_rollout_actions = torch.zeros_like(rollout_actions)
            with frozen_base_projection(backend), autocast():
                clean_latent = backend.encode_images(
                    clean, base_method, batch_size, num_frames
                )
                clean_target = clean_latent[:, 3:6]
            for domain in domains:
                images = clean if domain == "clean" else batch["ood_images"]
                with autocast():
                    adapted_context = backend.encode_images(
                        images[:, :3], method, batch_size, 3
                    )
                    canonical = _sample_mse(adapted_context, clean_latent[:, :3])
                    predicted_future = backend.differentiable_unroll(
                        adapted_context, rollout_actions
                    )
                    per_step = torch.stack(
                        [
                            _sample_mse(
                                predicted_future[:, index],
                                clean_target[:, index],
                            )
                            for index in range(3)
                        ],
                        dim=1,
                    )
                terminal_h1, terminal_h2, terminal_h3 = per_step.unbind(dim=1)
                mean_h1 = terminal_h1
                mean_h2 = per_step[:, :2].mean(dim=1)
                mean_h3 = per_step.mean(dim=1)
                if standard_cross_backend:
                    predicted_trajectory = torch.cat(
                        (
                            adapted_context.float(),
                            predicted_future.float(),
                        ),
                        dim=1,
                    )
                    trajectory_target = clean_latent[:, :6].float()
                    trajectory_mse = _sample_mse(
                        predicted_trajectory, trajectory_target
                    )
                    tensors = {
                        "h1_autoregressive_latent_mse": terminal_h1,
                        "h2_autoregressive_latent_mse": terminal_h2,
                        "h3_autoregressive_latent_mse": terminal_h3,
                        "future_mean_mse": per_step.mean(dim=1),
                        "terminal_mse": terminal_h3,
                        "unified_6frame_trajectory_mse": trajectory_mse,
                    }
                else:
                    shuffled_steps = _rollout_errors(
                        backend,
                        adapted_context,
                        clean_target,
                        shuffled_rollout_actions,
                    )
                    zero_steps = _rollout_errors(
                        backend,
                        adapted_context,
                        clean_target,
                        zero_rollout_actions,
                    )
                    shuffled = shuffled_steps.mean(dim=1)
                    zero = zero_steps.mean(dim=1)
                    tensors = {
                    "canonical_mse": canonical,
                    "terminal_h1_mse": terminal_h1,
                    "terminal_h2_mse": terminal_h2,
                    "terminal_h3_mse": terminal_h3,
                    "mean_through_h1_mse": mean_h1,
                    "mean_through_h2_mse": mean_h2,
                    "mean_through_h3_mse": mean_h3,
                    "one_step_mse": terminal_h1,
                    "two_step_mse": mean_h2,
                    "three_step_mse": mean_h3,
                    "h1_dynamics_mse": terminal_h1,
                    "h2_dynamics_mse": mean_h2,
                    "h3_dynamics_mse": mean_h3,
                    "shuffled_action_mse": shuffled,
                    "zero_action_mse": zero,
                    "action_shuffle_gap": shuffled - mean_h3,
                    "zero_action_gap": zero - mean_h3,
                    }
                if negative_count:
                    ranks: list[Tensor] = []
                    negative_means: list[Tensor] = []
                    for index in range(batch_size):
                        episode = int(batch["episode_id"][index])
                        candidates = [
                            action
                            for source_episode, action in action_bank
                            if source_episode != episode
                        ]
                        if len(candidates) < negative_count:
                            raise RuntimeError(
                                "Cross-episode action ranking has too few negatives: "
                                f"episode={episode}, available={len(candidates)}, "
                                f"required={negative_count}"
                            )
                        offset = (
                            offline_seed
                            + episode * 1000003
                            + int(batch["window_id"][index])
                        ) % len(candidates)
                        chosen = [
                            candidates[(offset + value) % len(candidates)]
                            for value in range(negative_count)
                        ]
                        negative_actions = torch.stack(chosen).to(
                            backend.device, non_blocking=True
                        )
                        negative_context = adapted_context[index : index + 1].expand(
                            negative_count, -1, -1, -1
                        )
                        negative_target = clean_target[index : index + 1].expand(
                            negative_count, -1, -1, -1
                        )
                        negative_costs = _rollout_errors(
                            backend,
                            negative_context,
                            negative_target,
                            negative_actions,
                        ).mean(dim=1)
                        ranks.append(
                            1
                            + (negative_costs < mean_h3[index]).sum().to(
                                dtype=torch.float32
                            )
                        )
                        negative_means.append(negative_costs.mean())
                    rank_tensor = torch.stack(ranks)
                    negative_mean = torch.stack(negative_means)
                    tensors.update(
                        true_action_rank=rank_tensor,
                        action_top1_accuracy=(rank_tensor == 1).float(),
                        action_mean_reciprocal_rank=rank_tensor.reciprocal(),
                        true_vs_negative_cost_gap=negative_mean - mean_h3,
                    )
                for key, values in tensors.items():
                    totals[domain][key] = totals[domain].get(key, 0.0) + float(values.sum().cpu())
                for index in range(batch_size):
                    rows.append(
                        {
                            "domain": domain,
                            "episode_id": int(batch["episode_id"][index]),
                            "window_id": int(batch["window_id"][index]),
                            **{key: float(value[index].cpu()) for key, value in tensors.items()},
                        }
                    )
                counts[domain] += batch_size
            print(f"OFFLINE_PROGRESS completed={len(rows)} total={len(windows) * len(domains)}")
    domain_metrics = {
        domain: {key: value / counts[domain] for key, value in totals[domain].items()}
        for domain in domains
    }
    if (
        not standard_cross_backend
        and "clean" in domain_metrics
        and "ood" in domain_metrics
    ):
        clean_mse = domain_metrics["clean"]["mean_through_h3_mse"]
        domain_metrics["ood"]["ood_clean_degradation_ratio"] = (
            domain_metrics["ood"]["mean_through_h3_mse"] / clean_mse
            if clean_mse > 0.0
            else None
        )
    metrics = {
        "schema_version": (
            "cross_backend_offline_mse_v1"
            if standard_cross_backend
            else "jepa_wm_offline_metrics_v2"
            if str(cfg.training.get("loss_name", "")) == "unified_trajectory_mse"
            else "jepa_wm_offline_metrics_v1"
        ),
        "benchmark": resolved_task.benchmark,
        "backend": backend.backend_name,
        "benchmark_suite": resolved_task.suite,
        "task_id": resolved_task.task_id,
        "task_name": resolved_task.task_name,
        "task_manifest_sha256": benchmark.task_manifest_sha256(resolved_task),
        "dataset_sha256": resolved_task.dataset_sha256,
        "camera_key": resolved_task.camera_key,
        "camera_height": resolved_task.camera_height,
        "camera_width": resolved_task.camera_width,
        "camera_channel_order": resolved_task.camera_channel_order,
        "camera_vertical_flip": resolved_task.camera_vertical_flip,
        "action_convention": resolved_task.action_convention,
        "action_transform": resolved_task.action_transform,
        "method": method.method_name,
        "training_seed": (
            None if method.method_name == "base" else int(cfg.training.seed)
        ),
        "metric_profile": metric_profile,
        "goal_encoder": "frozen_base",
        "loss_name": str(cfg.training.get("loss_name", "legacy_canonical_dynamics")),
        "task": resolved_task.task_key,
        "requested_window_count": int(cfg.offline.num_windows),
        "window_count": len(selected),
        "unique_window_count": unique_window_count,
        "sampling_with_replacement": False,
        "window_identities": [list(pair) for pair in selected],
        "episode_partition": "eval",
        "action_shuffle": (
            None
            if standard_cross_backend
            else {
                "seed": shuffle_seed,
                "suffix_source_indices": [2, 3, 4],
                "permutation": shuffle_permutation.tolist(),
            }
        ),
        "cross_episode_action_negatives": negative_count,
        "domains": domain_metrics,
        "method_parameter_count": method.parameter_count(),
        "method_checkpoint_sha256": method_checkpoint_sha256,
        "checkpoint_schema_version": (
            "base"
            if method.method_name == "base"
            else CHECKPOINT_SCHEMA_V2
            if str(cfg.training.get("loss_name", "")) == "unified_trajectory_mse"
            else "wm_adapter_checkpoint_v1"
        ),
        "cache_schema_version": cache_schema_version,
        "cache_fingerprint": cache_fingerprint,
        "cache_file_sha256": cache_file_sha256,
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "encoder_checkpoint_sha256": backend.encoder_checkpoint_sha256,
        "encoder_name": backend.encoder_name,
        "predictor_depth": backend.predictor_depth,
        "upstream_commits": backend.upstream_commits,
        "task_upstream_commits": resolved_task.upstream_commits,
        "appearance": ComposedPhotometricShift.metadata(
            float(cfg.appearance.severity), int(cfg.appearance.seed)
        ),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary_rows = output / "per_window.jsonl.tmp"
    temporary_rows.write_text(lines, encoding="utf-8")
    temporary_rows.replace(output / "per_window.jsonl")
    _atomic_json(output / "metrics.json", metrics)
    print(f"Offline metrics written: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
