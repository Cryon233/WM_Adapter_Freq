from __future__ import annotations

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
from wm_adapter_freq.training.adapter_trainer import (
    AdapterTrainer,
    AdapterTrainingConfig,
)


def _model_reference(value: str) -> str:
    path = Path(value).expanduser()
    return str(path) if path.exists() or value.startswith(("~", ".")) else value


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
    base_model = load_pretrained(_model_reference(str(cfg.base_model_ref)))
    base_model.eval()
    base_model.requires_grad_(False)
    backend = build_backend(str(cfg.backend), base_model)
    backend.move_training_modules(device)

    adapter = SequenceStableAdaptiveDCTAdapter(
        **_adapter_config(cfg)
    ).to(device)
    objective = CanonicalDynamicsObjective(
        backend=backend,
        adapter=adapter,
        canonical_weight=float(cfg.canonical_weight),
        dynamics_weight=float(cfg.dynamics_weight),
    )

    dataset = PairedFeatureDataset(
        Path(str(cfg.cache_path)).expanduser(),
        identity_probability=float(cfg.identity_probability),
    )
    if dataset.backend != str(cfg.backend):
        raise ValueError("Feature cache backend does not match training backend.")
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
            "history_size": 3,
            "image_size": 224,
            "patch_size": 14,
            "token_dim": backend.token_dim,
            "latent_dim": backend.latent_dim,
        },
    )


if __name__ == "__main__":
    main()
