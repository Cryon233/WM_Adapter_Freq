from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import DataLoader

from wm_adapter.adapters.factory import build_method
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.data.feature_cache import FeatureCacheDataset
from wm_adapter.training.trainer import AdapterTrainer, TrainingConfig
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
    if str(cfg.method) == "base":
        raise ValueError("method=base has no trainable parameters; choose dct_adapter, token_mlp, or lora")
    seed_everything(int(cfg.training.seed))
    backend = _backend(cfg)
    method = build_method(str(cfg.method), backend, cfg.method_config)
    if str(cfg.appearance.pipeline_version) != "composed_photometric_v1":
        raise ValueError(
            f"Unsupported appearance pipeline version: {cfg.appearance.pipeline_version}"
        )
    appearance_metadata = ComposedPhotometricShift.metadata(
        float(cfg.appearance.get("training_severity", cfg.appearance.severity)),
        int(cfg.appearance.training_seed),
    )
    dataset = FeatureCacheDataset(
        resolve_path(cfg.paths.feature_cache),
        expected_base_checkpoint_sha256=backend.base_checkpoint_sha256,
        expected_dinov3_checkpoint_sha256=backend.dinov3_checkpoint_sha256,
        expected_upstream_commits=backend.upstream_commits,
        expected_appearance_metadata=appearance_metadata,
    )
    generator = torch.Generator().manual_seed(int(cfg.training.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.training.num_workers) > 0,
    )
    training_config = TrainingConfig(
        epochs=int(cfg.training.epochs),
        batch_size=int(cfg.training.batch_size),
        gradient_accumulation=int(cfg.training.gradient_accumulation),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        gradient_clip_norm=float(cfg.training.gradient_clip_norm),
        precision=str(cfg.training.precision),
        num_workers=int(cfg.training.num_workers),
        seed=int(cfg.training.seed),
        canonical_weight=float(cfg.training.get("canonical_weight", 1.0)),
        dynamics_weight=float(cfg.training.get("dynamics_weight", 1.0)),
    )
    metadata = dict(dataset.metadata)
    metadata["appearance_metadata"] = json.loads(str(metadata["appearance_metadata"]))
    trainer = AdapterTrainer(
        backend=backend,
        method=method,
        config=training_config,
        device=cfg.device,
    )
    print(f"Trainable parameters: method={method.method_name}, count={method.parameter_count()}")
    final_losses = trainer.fit(
        loader,
        checkpoint_path=resolve_path(cfg.paths.method_checkpoint),
        cache_metadata=metadata,
    )
    print(f"TRAINING_COMPLETE final_losses={json.dumps(final_losses, sort_keys=True)}")


if __name__ == "__main__":
    main()
