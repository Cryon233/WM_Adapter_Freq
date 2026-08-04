from __future__ import annotations

from typing import Any

from wm_adapter.backends.dino_wm_droid import DinoWMDroidBackend
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend


SUPPORTED_BACKENDS = ("jepa_wm_droid", "dino_wm_droid")


def build_backend(model_config: Any, *, device: Any) -> JEPAWMDroidBackend:
    name = str(model_config.name)
    common = {
        "third_party_root": model_config.third_party_root,
        "official_planning_config": model_config.official_planning_config,
        "device": device,
        "planning_tag": model_config.get("planning_tag"),
        "planning_subtask": model_config.get("planning_subtask"),
    }
    if name == "jepa_wm_droid":
        return JEPAWMDroidBackend(
            **common,
            jepa_checkpoint=model_config.jepa_checkpoint,
            dinov3_checkpoint=model_config.dinov3_checkpoint,
        )
    if name == "dino_wm_droid":
        return DinoWMDroidBackend(
            **common,
            dino_wm_checkpoint=model_config.dino_wm_checkpoint,
            dinov2_checkpoint=model_config.dinov2_checkpoint,
            dinov2_root=model_config.dinov2_root,
        )
    raise ValueError(
        f"Unsupported world-model backend {name!r}; expected one of {SUPPORTED_BACKENDS}"
    )
