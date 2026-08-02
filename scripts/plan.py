from __future__ import annotations

import json
from typing import Any

import h5py
from omegaconf import OmegaConf

from wm_adapter.adapters.factory import build_method
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.benchmarks.factory import build_benchmark
from wm_adapter.planning.jepa_wm_planner import (
    EVALUATION_PROTOCOL_DIRECTORY,
    EVALUATION_PROTOCOL_VERSION,
    save_planning_results,
)
from wm_adapter.utils.checkpoints import load_method_checkpoint, sha256_file
from wm_adapter.utils.reproducibility import load_experiment_config, resolve_path, seed_everything


def _backend(cfg: Any) -> JEPAWMDroidBackend:
    return JEPAWMDroidBackend(
        third_party_root=cfg.model.third_party_root,
        jepa_checkpoint=cfg.model.jepa_checkpoint,
        dinov3_checkpoint=cfg.model.dinov3_checkpoint,
        official_planning_config=cfg.model.official_planning_config,
        device=cfg.device,
        planning_tag=cfg.model.get("planning_tag"),
        planning_subtask=cfg.model.get("planning_subtask"),
    )


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
    training_appearance = ComposedPhotometricShift.metadata(
        float(cfg.appearance.get("training_severity", cfg.appearance.severity)),
        int(cfg.appearance.training_seed),
    )
    checkpoint_fingerprint: str | None = None
    cache_fingerprint: str | None = None
    if method_name != "base":
        checkpoint_path = resolve_path(cfg.paths.method_checkpoint)
        checkpoint = load_method_checkpoint(checkpoint_path)
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
        if checkpoint["appearance_metadata"] != training_appearance:
            raise RuntimeError(f"Method checkpoint training appearance mismatch: {checkpoint_path}")
        checkpoint_training = dict(checkpoint.get("training_config", {}))
        expected_training = {
            "seed": int(cfg.training.seed),
            "canonical_weight": float(cfg.training.get("canonical_weight", 1.0)),
            "dynamics_weight": float(cfg.training.get("dynamics_weight", 1.0)),
        }
        training_mismatch = {
            key: {"expected": value, "actual": checkpoint_training.get(key)}
            for key, value in expected_training.items()
            if float(checkpoint_training.get(key, -1)) != float(value)
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
        data_metadata = dict(checkpoint.get("data_metadata", {}))
        actual_identity = (
            str(data_metadata.get("benchmark", "")),
            str(data_metadata.get("task_key", "")),
        )
        expected_identity = (resolved_task.benchmark, resolved_task.task_key)
        legacy_place = (
            resolved_task.task_key == "robocasa_place"
            or resolved_task.suite == "legacy_single_stage"
        ) and actual_identity == ("", "")
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
                "action_convention": resolved_task.action_convention,
                "task_upstream_commits": resolved_task.upstream_commits,
            }
            contract_mismatch = {
                key: {"expected": value, "actual": data_metadata.get(key)}
                for key, value in checkpoint_contract.items()
                if data_metadata.get(key) != value
            }
            if contract_mismatch:
                raise RuntimeError(
                    f"Method checkpoint task/data contract mismatch: {contract_mismatch}"
                )
        checkpoint_fingerprint = sha256_file(checkpoint_path)
        cache_fingerprint = str(checkpoint["cache_fingerprint"])
    configured_cache = resolve_path(cfg.paths.feature_cache)
    if not configured_cache.is_file():
        raise FileNotFoundError(
            f"Planning requires the task feature cache contract: {configured_cache}"
        )
    with h5py.File(configured_cache, "r", libver="latest", swmr=True) as cache:
        configured_cache_fingerprint = str(cache.attrs["cache_fingerprint"])
    if cache_fingerprint is not None and cache_fingerprint != configured_cache_fingerprint:
        raise RuntimeError(
            "Planning method checkpoint and feature cache fingerprints differ: "
            f"checkpoint={cache_fingerprint}, cache={configured_cache_fingerprint}"
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
            / "jepa_wm_droid"
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
        "backend": "jepa_wm_droid",
        "benchmark": resolved_task.benchmark,
        "benchmark_suite": resolved_task.suite,
        "task_id": resolved_task.task_id,
        "task_name": resolved_task.task_name,
        "language_instruction": resolved_task.language_instruction,
        "task_manifest_sha256": task_manifest["task_manifest_sha256"],
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "dataset_fingerprint": resolved_task.dataset_sha256,
        "camera_key": resolved_task.camera_key,
        "action_convention": resolved_task.action_convention,
        "task": task,
        "domain": domain,
        "number_of_episodes": result.total_episodes,
        "severity": float(cfg.appearance.severity),
        "seeds": {
            "evaluation": int(cfg.evaluation.eval_seed),
            "appearance": int(cfg.appearance.seed),
            "cem": result.cem_seeds or [int(cfg.evaluation.eval_seed)],
        },
        "appearance_metadata": ComposedPhotometricShift.metadata(
            float(cfg.appearance.severity), int(cfg.appearance.seed)
        ),
        "runtime_seconds": result.elapsed_seconds,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "method": method_name,
        "method_parameter_count": method.parameter_count(),
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "method_checkpoint_sha256": checkpoint_fingerprint,
        "cache_fingerprint": cache_fingerprint,
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
