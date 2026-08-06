from __future__ import annotations

import json
from typing import Any

import h5py
from omegaconf import OmegaConf

from wm_adapter.adapters.factory import build_method
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.appearance.evaluation_corruptions import evaluation_corruption_metadata
from wm_adapter.backends.factory import build_backend
from wm_adapter.benchmarks.factory import build_benchmark
from wm_adapter.data.feature_cache import CACHE_SCHEMA_VERSION
from wm_adapter.data.feature_cache_v2 import (
    CACHE_SCHEMA_VERSION_V2,
    cache_file_sha256_from_verified_state_v2,
)
from wm_adapter.experiments.cross_benchmark import (
    normalize_metadata_contract,
    training_contract_mismatches_v2,
    training_contract_v2,
)
from wm_adapter.planning.jepa_wm_planner import (
    EVALUATION_PROTOCOL_DIRECTORY,
    EVALUATION_PROTOCOL_VERSION,
    save_planning_results,
)
from wm_adapter.training.trainer_v2 import CHECKPOINT_SCHEMA_V2
from wm_adapter.utils.checkpoints import load_method_checkpoint, sha256_file
from wm_adapter.utils.reproducibility import load_experiment_config, resolve_path, seed_everything


def _backend(cfg: Any) -> Any:
    return build_backend(cfg.model, device=cfg.device)


def main() -> None:
    cfg = load_experiment_config()
    method_name = str(cfg.method)
    domain = str(cfg.domain)
    if domain not in {"clean", "ood"}:
        raise ValueError(f"domain must be clean or ood, received {domain!r}")
    seed_everything(int(cfg.evaluation.eval_seed))
    backend = _backend(cfg)
    benchmark = build_benchmark(cfg)
    resolved_task = benchmark.resolve_task(strict=True)
    cfg.benchmark.task_name = resolved_task.task_name
    cfg.data.task_name = resolved_task.task_name
    if resolved_task.benchmark == "robocasa":
        cfg.paths.robocasa_hdf5 = resolved_task.dataset_path
    task_manifest = benchmark.write_task_manifest(resolved_task)
    method = build_method(method_name, backend, cfg.method_config).to(backend.device)
    training_appearance = {
        "family": "identity",
        "strength": 0.0,
        "training_input_domain": "clean",
    }
    checkpoint_fingerprint: str | None = None
    cache_fingerprint: str | None = None
    if method_name != "base":
        checkpoint_path = resolve_path(cfg.paths.method_checkpoint)
        checkpoint = load_method_checkpoint(checkpoint_path)
        if checkpoint.get("schema_version") == "wm_adapter_checkpoint_v2":
            raise RuntimeError(
                "Planning rejects obsolete v2 checkpoints without the cache-file "
                f"integrity contract: {checkpoint_path}"
            )
        checkpoint_v2 = checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_V2
        checkpoint_backend = str(checkpoint.get("backend", "jepa_wm_droid"))
        if checkpoint_backend != backend.backend_name:
            raise RuntimeError(
                "Method checkpoint backend mismatch: "
                f"expected={backend.backend_name}, actual={checkpoint_backend}, "
                f"path={checkpoint_path}"
            )
        checkpoint_encoder = checkpoint.get(
            "encoder_checkpoint_sha256",
            checkpoint.get("dinov3_checkpoint_sha256"),
        )
        if checkpoint_encoder != backend.encoder_checkpoint_sha256:
            raise RuntimeError(
                f"Method checkpoint visual-encoder fingerprint mismatch: {checkpoint_path}"
            )
        if checkpoint["method_name"] != method_name:
            raise RuntimeError(
                f"Method checkpoint is for {checkpoint['method_name']!r}, requested {method_name!r}: {checkpoint_path}"
            )
        if checkpoint["method_config"] != method.config_dict():
            raise RuntimeError(
                f"Method checkpoint config mismatch: checkpoint={checkpoint['method_config']}, "
                f"configured={method.config_dict()}"
            )
        if checkpoint["base_checkpoint_sha256"] != backend.base_checkpoint_sha256:
            raise RuntimeError(f"Method checkpoint base JEPA-WM fingerprint mismatch: {checkpoint_path}")
        if checkpoint.get("dinov3_checkpoint_sha256") != backend.dinov3_checkpoint_sha256:
            raise RuntimeError(f"Method checkpoint DINOv3 fingerprint mismatch: {checkpoint_path}")
        if checkpoint["upstream_commits"] != backend.upstream_commits:
            raise RuntimeError(f"Method checkpoint upstream commit mismatch: {checkpoint_path}")
        if not checkpoint_v2 and checkpoint["appearance_metadata"] != training_appearance:
            raise RuntimeError(f"Method checkpoint training appearance mismatch: {checkpoint_path}")
        data_metadata = dict(checkpoint.get("data_metadata", {}))
        actual_identity = (
            str(data_metadata.get("benchmark", "")),
            str(data_metadata.get("task_key", "")),
        )
        expected_identity = (resolved_task.benchmark, resolved_task.task_key)
        standard_checkpoint = resolve_path(
            "checkpoints/cross_benchmark_v1/"
            f"{resolved_task.benchmark}/{resolved_task.task_key}/{method_name}_final.pt"
        )
        legacy_place = (not checkpoint_v2 and
            resolved_task.benchmark == "robocasa"
            and resolved_task.task_key == "robocasa_place"
            and checkpoint_path != standard_checkpoint
            and actual_identity == ("", "")
        )
        checkpoint_training = dict(checkpoint.get("training_config", {}))
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
            key: checkpoint_training.get(
                key,
                1.0
                if legacy_place and key in {"canonical_weight", "dynamics_weight"}
                else -1,
            )
            for key in expected_training
        }
        training_mismatch = {
            key: {"expected": value, "actual": checkpoint_training.get(key)}
            for key, value in expected_training.items()
            if float(actual_training[key]) != float(value)
        }
        if training_mismatch:
            raise RuntimeError(
                f"Method checkpoint training-contract mismatch: {training_mismatch}"
            )
        method.load_method_checkpoint(checkpoint["peft_state_dict"])
        if int(checkpoint["trainable_parameter_count"]) != method.parameter_count():
            raise RuntimeError(
                f"Method parameter count mismatch: checkpoint={checkpoint['trainable_parameter_count']}, "
                f"model={method.parameter_count()}"
            )
        if actual_identity != expected_identity and not legacy_place:
            raise RuntimeError(
                "Method checkpoint benchmark/task mismatch: "
                f"expected={expected_identity}, actual={actual_identity}, path={checkpoint_path}"
            )
        if not legacy_place:
            checkpoint_contract = {
                "task_manifest_sha256": task_manifest["task_manifest_sha256"],
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
            contract_mismatch = {
                key: {
                    "expected": normalize_metadata_contract(value),
                    "actual": normalize_metadata_contract(data_metadata.get(key)),
                }
                for key, value in checkpoint_contract.items()
                if normalize_metadata_contract(data_metadata.get(key))
                != normalize_metadata_contract(value)
            }
            if contract_mismatch:
                raise RuntimeError(
                    f"Method checkpoint task/data contract mismatch: {contract_mismatch}"
                )
        if checkpoint_v2:
            training_values = OmegaConf.to_container(cfg.training, resolve=True)
            if not isinstance(training_values, dict):
                raise TypeError("V2 planning training configuration is not a mapping")
            mismatch = training_contract_mismatches_v2(
                checkpoint, training_contract_v2(training_values)
            )
            if mismatch:
                raise RuntimeError(f"V2 planning checkpoint contract mismatch: {mismatch}")
        checkpoint_fingerprint = sha256_file(checkpoint_path)
        cache_fingerprint = str(checkpoint["cache_fingerprint"])
    configured_cache = resolve_path(cfg.paths.feature_cache)
    if not configured_cache.is_file():
        raise FileNotFoundError(
            f"Planning requires the task feature cache contract: {configured_cache}"
        )
    with h5py.File(configured_cache, "r", libver="latest", swmr=True) as cache:
        cache_schema = str(cache.attrs.get("schema_version", ""))
        configured_cache_fingerprint = str(cache.attrs["cache_fingerprint"])
        cache_backend = str(cache.attrs.get("backend", "jepa_wm_droid"))
        cache_encoder_sha = str(
            cache.attrs.get(
                "encoder_checkpoint_sha256",
                cache.attrs.get("dinov3_checkpoint_sha256", ""),
            )
        )
    if cache_backend != backend.backend_name:
        raise RuntimeError(
            "Planning feature-cache backend mismatch: "
            f"expected={backend.backend_name}, actual={cache_backend}, path={configured_cache}"
        )
    if cache_encoder_sha != backend.encoder_checkpoint_sha256:
        raise RuntimeError(
            f"Planning feature-cache visual-encoder fingerprint mismatch: {configured_cache}"
        )
    cache_file_sha256 = (
        cache_file_sha256_from_verified_state_v2(
            configured_cache,
            expected_sha256=cfg.cache.get("expected_file_sha256"),
            expected_size=cfg.cache.get("expected_file_size"),
            expected_mtime_ns=cfg.cache.get("expected_file_mtime_ns"),
        )
        if cache_schema == CACHE_SCHEMA_VERSION_V2
        else sha256_file(configured_cache)
    )
    expected_cache_schema = (
        CACHE_SCHEMA_VERSION_V2
        if str(cfg.training.get("loss_name", "")) == "unified_trajectory_mse"
        else CACHE_SCHEMA_VERSION
    )
    if cache_schema != expected_cache_schema:
        raise RuntimeError(
            "Planning feature-cache schema mismatch: "
            f"expected={expected_cache_schema}, actual={cache_schema}, "
            f"path={configured_cache}"
        )
    if cache_fingerprint is not None and cache_fingerprint != configured_cache_fingerprint:
        raise RuntimeError(
            "Planning method checkpoint and feature cache fingerprints differ: "
            f"checkpoint={cache_fingerprint}, cache={configured_cache_fingerprint}"
        )
    if method_name != "base" and checkpoint_v2:
        expected_cache_file_sha256 = str(checkpoint["cache_file_sha256"])
        if expected_cache_file_sha256 != cache_file_sha256:
            raise RuntimeError(
                "Planning method checkpoint and feature cache file fingerprints differ: "
                f"checkpoint={expected_cache_file_sha256}, "
                f"cache={cache_file_sha256}, path={configured_cache}"
            )
        metadata_cache_file_sha256 = str(
            checkpoint.get("data_metadata", {}).get("cache_file_sha256", "")
        )
        if metadata_cache_file_sha256 != cache_file_sha256:
            raise RuntimeError(
                "Planning checkpoint data metadata/cache file fingerprint mismatch: "
                f"checkpoint={metadata_cache_file_sha256}, "
                f"cache={cache_file_sha256}, path={configured_cache}"
            )
    cache_fingerprint = configured_cache_fingerprint
    backend.configure_planning_inference(
        inference_precision=str(cfg.planning.inference_precision),
        allow_tf32=bool(cfg.planning.allow_tf32),
        compile_predictor=bool(cfg.planning.compile_predictor),
    )
    task = resolved_task.task_key
    seed = int(cfg.evaluation.eval_seed)
    configured_run_directory = cfg.output.get("run_directory")
    output_directory = (
        resolve_path(configured_run_directory)
        if configured_run_directory
        else (
            resolve_path(cfg.output.root_dir)
            / backend.backend_name
            / resolved_task.benchmark
            / EVALUATION_PROTOCOL_DIRECTORY
            / task
            / f"seed_{seed}"
            / method_name
            / domain
        )
    )
    result_path = output_directory / "results.json"
    if configured_run_directory and result_path.exists():
        raise FileExistsError(
            f"Suite planning result already exists and will not be overwritten: {result_path}"
        )
    result = benchmark.run_planning(
        backend=backend,
        method=method,
        output_directory=output_directory,
    )
    suite_metadata = (
        OmegaConf.to_container(cfg.suite, resolve=True) if "suite" in cfg else {}
    )
    evaluation_manifest = benchmark.evaluation_manifest_path()
    evaluation_manifest_sha256 = None
    if evaluation_manifest.is_file():
        evaluation_manifest_payload = json.loads(
            evaluation_manifest.read_text(encoding="utf-8")
        )
        evaluation_manifest_sha256 = str(
            evaluation_manifest_payload["evaluation_manifest_sha256"]
        )
    cem_metadata = OmegaConf.to_container(
        backend.official_planning_template.planner,
        resolve=True,
    )
    if not isinstance(cem_metadata, dict):
        raise TypeError("Official CEM configuration must resolve to a mapping")
    if str(cfg.get("suite_mode", "formal")) == "self_test":
        cem_metadata.update(
            iterations=int(cfg.planning.self_test.iterations),
            num_samples=int(cfg.planning.self_test.num_samples),
            num_elites=int(cfg.planning.self_test.num_elites),
            horizon=int(cfg.planning.self_test.horizon),
        )
    cem_metadata.update(
        decode_each_iteration=False,
        candidate_chunk_size=int(cfg.planning.candidate_chunk_size),
    )
    metadata = {
        "backend": backend.backend_name,
        "benchmark": resolved_task.benchmark,
        "benchmark_suite": resolved_task.suite,
        "task_id": resolved_task.task_id,
        "task_name": resolved_task.task_name,
        "language_instruction": resolved_task.language_instruction,
        "task_manifest_sha256": task_manifest["task_manifest_sha256"],
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "dataset_fingerprint": resolved_task.dataset_sha256,
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
        "task": task,
        "domain": domain,
        "number_of_episodes": result.total_episodes,
        "severity": float(cfg.appearance.severity),
        "seeds": {
            "training": int(cfg.training.seed),
            "evaluation": int(cfg.evaluation.eval_seed),
            "appearance": int(cfg.appearance.seed),
            "cem": result.cem_seeds or [int(cfg.evaluation.eval_seed)],
        },
        "evaluation_family": str(
            cfg.appearance.get("evaluation_family", "photometric")
        ),
        "appearance_metadata": evaluation_corruption_metadata(
            family=str(
                cfg.appearance.get(
                    "evaluation_family", "photometric"
                )
            ),
            seed=int(cfg.appearance.seed),
            strength=float(cfg.appearance.severity),
        ),
        "runtime_seconds": result.elapsed_seconds,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "method": method_name,
        "goal_encoder": "frozen_base",
        "current_encoder": "configured_method",
        "goal_base_latent_fingerprint": result.goal_base_latent_fingerprints,
        "method_parameter_count": method.parameter_count(),
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "encoder_checkpoint_sha256": backend.encoder_checkpoint_sha256,
        "encoder_name": backend.encoder_name,
        "predictor_depth": backend.predictor_depth,
        "method_checkpoint_sha256": checkpoint_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "cache_file_sha256": cache_file_sha256,
        "cache_schema_version": cache_schema,
        "checkpoint_schema_version": (
            "base" if method_name == "base" else
            CHECKPOINT_SCHEMA_V2
            if str(cfg.training.get("loss_name", "")) == "unified_trajectory_mse"
            else "wm_adapter_checkpoint_v1"
        ),
        "upstream_commits": backend.upstream_commits,
        "benchmark_upstream_commits": resolved_task.upstream_commits,
        "training_appearance": training_appearance,
        "planning_history_len": int(cfg.planning.history_len),
        "suite": suite_metadata,
        "planning_inference": {
            "precision": str(cfg.planning.inference_precision),
            "allow_tf32": bool(cfg.planning.allow_tf32),
            "compile_predictor": bool(cfg.planning.compile_predictor),
            "candidate_chunk_size": int(cfg.planning.candidate_chunk_size),
        },
        "cem": cem_metadata,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "config_snapshot": OmegaConf.to_container(cfg, resolve=True),
    }
    save_planning_results(result_path, result, metadata)
    print(f"Planning results written: {result_path}")


if __name__ == "__main__":
    main()
