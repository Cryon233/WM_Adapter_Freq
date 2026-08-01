from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from wm_adapter.adapters.factory import build_method
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.planning.jepa_wm_planner import (
    EVALUATION_PROTOCOL_DIRECTORY,
    EVALUATION_PROTOCOL_VERSION,
    run_robocasa_planning,
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
    )


def main() -> None:
    cfg = load_experiment_config()
    method_name = str(cfg.method)
    domain = str(cfg.domain)
    if domain not in {"clean", "ood"}:
        raise ValueError(f"domain must be clean or ood, received {domain!r}")
    seed_everything(int(cfg.evaluation.eval_seed))
    backend = _backend(cfg)
    method = build_method(method_name, backend, cfg.method_config).to(backend.device)
    training_appearance = ComposedPhotometricShift.metadata(
        float(cfg.appearance.severity), int(cfg.appearance.training_seed)
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
        method.load_method_checkpoint(checkpoint["peft_state_dict"])
        if int(checkpoint["trainable_parameter_count"]) != method.parameter_count():
            raise RuntimeError(
                f"Method parameter count mismatch: checkpoint={checkpoint['trainable_parameter_count']}, "
                f"model={method.parameter_count()}"
            )
        checkpoint_fingerprint = sha256_file(checkpoint_path)
        cache_fingerprint = str(checkpoint["cache_fingerprint"])
    backend.configure_planning_inference(
        inference_precision=str(cfg.planning.inference_precision),
        allow_tf32=bool(cfg.planning.allow_tf32),
        compile_predictor=bool(cfg.planning.compile_predictor),
    )
    task = str(cfg.planning.task_slug)
    seed = int(cfg.evaluation.eval_seed)
    output_directory = (
        resolve_path(cfg.output.root_dir)
        / "jepa_wm_droid"
        / "robocasa"
        / EVALUATION_PROTOCOL_DIRECTORY
        / task
        / f"seed_{seed}"
        / method_name
        / domain
    )
    result = run_robocasa_planning(
        experiment_config=cfg,
        backend=backend,
        method=method,
        output_directory=output_directory,
    )
    metadata = {
        "backend": "jepa_wm_droid",
        "task": task,
        "domain": domain,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "method": method_name,
        "method_parameter_count": method.parameter_count(),
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "method_checkpoint_sha256": checkpoint_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "upstream_commits": backend.upstream_commits,
        "training_appearance": training_appearance,
        "planning_history_len": int(cfg.planning.history_len),
        "planning_inference": {
            "precision": str(cfg.planning.inference_precision),
            "allow_tf32": bool(cfg.planning.allow_tf32),
            "compile_predictor": bool(cfg.planning.compile_predictor),
            "candidate_chunk_size": int(cfg.planning.candidate_chunk_size),
        },
        "cem": {
            **OmegaConf.to_container(
                backend.official_planning_template.planner,
                resolve=True,
            ),
            "decode_each_iteration": False,
            "candidate_chunk_size": int(cfg.planning.candidate_chunk_size),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_planning_results(output_directory / "results.json", result, metadata)
    print(f"Planning results written: {output_directory / 'results.json'}")


if __name__ == "__main__":
    main()
