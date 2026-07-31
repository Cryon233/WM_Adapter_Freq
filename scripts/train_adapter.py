from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from wm_adapter_freq.adapters.sequence_stable_dct import (
    SequenceStableAdaptiveDCTAdapter,
)
from wm_adapter_freq.backends.base import build_backend
from wm_adapter_freq.data.feature_cache import PairedFeatureDataset
from wm_adapter_freq.objectives.canonical_dynamics import (
    CanonicalDynamicsObjective,
)
from wm_adapter_freq.io.fingerprint import resolve_base_model_identity
from wm_adapter_freq.training.adapter_trainer import (
    AdapterTrainer,
    AdapterTrainingConfig,
)


def _adapter_config(cfg: DictConfig) -> dict[str, object]:
    config_path = Path(get_original_cwd()) / str(cfg.adapter_config_path)
    adapter_cfg = OmegaConf.load(config_path)
    common = OmegaConf.to_container(adapter_cfg.common, resolve=True)
    backend = OmegaConf.to_container(
        adapter_cfg.backends[str(cfg.backend)], resolve=True
    )
    return {**common, **backend}


@hydra.main(
    version_base=None,
    config_path="../configs/train",
    config_name="prejepa_tworoom",
)
def main(cfg: DictConfig) -> None:
    from stable_worldmodel.wm.utils import load_pretrained

    torch.manual_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    dataset = PairedFeatureDataset(
        Path(str(cfg.cache_path)).expanduser(),
        identity_probability=float(cfg.identity_probability),
    )
    identity = resolve_base_model_identity(str(cfg.base_model_ref))
    if (
        str(dataset.metadata["base_model_fingerprint"])
        != identity.combined_fingerprint
    ):
        raise RuntimeError(
            "Feature cache was built from a different base checkpoint."
        )
    if dataset.backend != str(cfg.backend):
        raise RuntimeError(
            "Feature cache backend does not match training backend."
        )

    base_model = load_pretrained(identity.resolved_weights_path)
    base_model.eval()
    base_model.requires_grad_(False)
    backend = build_backend(str(cfg.backend), base_model)
    backend.move_training_modules(device)
    if (
        int(dataset.metadata["token_dim"]) != backend.token_dim
        or int(dataset.metadata["latent_dim"]) != backend.latent_dim
    ):
        raise RuntimeError(
            "Feature cache dimensions do not match the base checkpoint."
        )

    adapter = SequenceStableAdaptiveDCTAdapter(
        **_adapter_config(cfg)
    ).to(device)
    objective = CanonicalDynamicsObjective(
        backend=backend,
        adapter=adapter,
        canonical_weight=float(cfg.canonical_weight),
        dynamics_weight=float(cfg.dynamics_weight),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg.num_workers) > 0,
    )

    trainer = AdapterTrainer(
        objective=objective,
        adapter=adapter,
        config=AdapterTrainingConfig(
            epochs=int(cfg.epochs),
            gradient_accumulation=int(cfg.gradient_accumulation),
            precision=str(cfg.precision),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            gradient_clip_norm=float(cfg.gradient_clip_norm),
        ),
        device=device,
    )
    trainer.fit(
        data_loader,
        Path(str(cfg.output_path)).expanduser(),
        {
            "backend": str(cfg.backend),
            "base_model_ref": str(cfg.base_model_ref),
            "base_model_identity": identity,
            "history_size": 3,
            "image_size": 224,
            "patch_size": 14,
            "token_dim": backend.token_dim,
            "latent_dim": backend.latent_dim,
            "dataset_name": str(dataset.metadata["dataset_name"]),
            "normalization": dataset.normalization,
            "appearance_training": {
                "severity": float(
                    dataset.metadata["appearance_severity"]
                ),
                "shift_names": json.loads(
                    str(dataset.metadata["appearance_shift_names"])
                ),
                "pipeline_version": str(
                    dataset.metadata["appearance_pipeline_version"]
                ),
            },
            "data_selection": {
                "episode_split_strategy": str(
                    dataset.metadata["episode_split_strategy"]
                ),
                "episode_split_seed": int(
                    dataset.metadata["episode_split_seed"]
                ),
                "episode_split_train_fraction": float(
                    dataset.metadata["episode_split_train_fraction"]
                ),
                "source_episode_count": int(
                    dataset.metadata["source_episode_count"]
                ),
                "train_episode_count": int(
                    dataset.metadata["train_episode_count"]
                ),
                "eval_episode_count": int(
                    dataset.metadata["eval_episode_count"]
                ),
                "train_episode_indices_sha256": str(
                    dataset.metadata["train_episode_indices_sha256"]
                ),
                "eval_episode_indices_sha256": str(
                    dataset.metadata["eval_episode_indices_sha256"]
                ),
                "window_selection_strategy": str(
                    dataset.metadata["window_selection_strategy"]
                ),
                "window_selection_seed": int(
                    dataset.metadata["window_selection_seed"]
                ),
                "selected_window_count": int(
                    dataset.metadata["selected_window_count"]
                ),
                "selected_window_indices_sha256": str(
                    dataset.metadata[
                        "selected_window_indices_sha256"
                    ]
                ),
                "selected_window_pairs_sha256": str(
                    dataset.metadata["selected_window_pairs_sha256"]
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
