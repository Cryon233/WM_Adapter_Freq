from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = (base or project_root()) / expanded
    return expanded.resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_experiment_config(argv: Sequence[str] | None = None) -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    arguments = parser.parse_args(argv)
    config_path = resolve_path(arguments.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
    experiment = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(arguments.overrides))
    root = project_root()
    model_path = resolve_path(str(experiment.model_config), base=root)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_path}")
    method_name = str(experiment.method)
    if method_name not in experiment.method_configs:
        raise ValueError(
            f"Method {method_name!r} has no config; available={list(experiment.method_configs.keys())}"
        )
    method_path = resolve_path(str(experiment.method_configs[method_name]), base=root)
    if not method_path.is_file():
        raise FileNotFoundError(f"Method config does not exist: {method_path}")
    return OmegaConf.merge(
        experiment,
        {"model": OmegaConf.load(model_path), "method_config": OmegaConf.load(method_path)},
    )
