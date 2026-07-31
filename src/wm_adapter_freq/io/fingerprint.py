from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


STABLE_WORLDMODEL_COMMIT = "73dade035ff789e007194971ca5a59b3c3f77e6b"


@dataclass(frozen=True)
class BaseModelIdentity:
    reference: str
    resolved_weights_path: str
    resolved_config_path: str
    weights_sha256: str
    config_sha256: str
    combined_fingerprint: str


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_reference(reference: str) -> str:
    return os.path.expandvars(os.path.expanduser(reference))


def _is_explicit_local_reference(reference: str) -> bool:
    expanded = _expand_reference(reference)
    path = Path(expanded)
    return (
        path.is_absolute()
        or reference.startswith("~")
        or reference.startswith(".")
        or path.suffix == ".pt"
    )


def _normalized_reference(reference: str) -> str:
    expanded_reference = _expand_reference(reference)
    expanded_path = Path(expanded_reference)
    if _is_explicit_local_reference(reference):
        resolved_path = expanded_path.resolve()
        if not expanded_path.exists():
            raise FileNotFoundError(
                "Local base-model checkpoint does not exist:\n"
                f"reference: {reference}\n"
                f"resolved path: {resolved_path}"
            )
        return str(resolved_path)
    if expanded_path.exists():
        return str(expanded_path.resolve())
    return expanded_reference


def resolve_base_model_identity(
    base_model_ref: str,
) -> BaseModelIdentity:
    """Resolve and fingerprint a checkpoint using upstream path semantics."""
    from stable_worldmodel.data import get_cache_dir
    from stable_worldmodel.wm.utils import _resolve

    normalized_reference = _normalized_reference(base_model_ref)
    checkpoint_root = get_cache_dir(sub_folder="checkpoints")
    weights_path, _ = _resolve(normalized_reference, checkpoint_root)
    weights_path = weights_path.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"resolved weights file missing: {weights_path}"
        )
    if weights_path.suffix != ".pt":
        raise ValueError(
            f"resolved checkpoint is not a .pt file: {weights_path}"
        )
    config_path = (weights_path.parent / "config.json").resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config.json missing beside checkpoint: {config_path}"
        )

    weights_sha256 = _streaming_sha256(weights_path)
    config_sha256 = _streaming_sha256(config_path)
    combined = hashlib.sha256(
        (
            weights_sha256
            + "\n"
            + config_sha256
            + "\n"
            + STABLE_WORLDMODEL_COMMIT
        ).encode("utf-8")
    ).hexdigest()
    return BaseModelIdentity(
        reference=base_model_ref,
        resolved_weights_path=str(weights_path),
        resolved_config_path=str(config_path),
        weights_sha256=weights_sha256,
        config_sha256=config_sha256,
        combined_fingerprint=combined,
    )
