from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


STABLE_WORLDMODEL_COMMIT = "73dade035ff789e007194971ca5a59b3c3f77e6b"
_UNEXPANDED_ENV_PATTERN = re.compile(
    r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})"
)


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
    if not isinstance(reference, str):
        raise TypeError("Base-model checkpoint reference must be a string.")
    stripped = reference.strip()
    if not stripped:
        raise ValueError(
            "Base-model checkpoint reference is empty.\n"
            "Set base_model_ref to a real local .pt file, a valid checkpoint "
            "cache name, or an owner/repo Hugging Face reference."
        )
    expanded = os.path.expanduser(os.path.expandvars(stripped))
    if _UNEXPANDED_ENV_PATTERN.search(expanded):
        raise ValueError(
            "Base-model checkpoint reference contains an undefined "
            "environment variable:\n"
            f"reference: {reference}"
        )
    if not expanded:
        raise ValueError(
            "Base-model checkpoint reference is empty.\n"
            "Set base_model_ref to a real local .pt file, a valid checkpoint "
            "cache name, or an owner/repo Hugging Face reference."
        )
    return expanded


def _is_explicit_local_reference(
    original_reference: str,
    expanded_reference: str,
) -> bool:
    path = Path(expanded_reference)
    return (
        path.is_absolute()
        or original_reference.startswith("~")
        or original_reference.startswith(".")
        or path.suffix.lower() == ".pt"
        or path.exists()
    )


def _resolve_explicit_local_checkpoint(
    original_reference: str,
    expanded_reference: str,
) -> tuple[Path, Path]:
    path = Path(expanded_reference).resolve()
    if not path.exists():
        raise FileNotFoundError(
            "Local base-model checkpoint does not exist:\n"
            f"reference: {original_reference}\n"
            f"resolved path: {path}"
        )

    if path.is_file():
        if path.suffix.lower() != ".pt":
            raise ValueError(
                "Local base-model checkpoint file is not a .pt file:\n"
                f"{path}"
            )
        weights_path = path
        config_path = path.parent / "config.json"
    elif path.is_dir():
        pt_files = sorted(
            item.resolve()
            for item in path.glob("*.pt")
            if item.is_file()
        )
        if not pt_files:
            raise FileNotFoundError(
                "No .pt checkpoint file found in local checkpoint "
                "directory:\n"
                f"{path}"
            )
        if len(pt_files) > 1:
            candidates = "\n".join(str(item) for item in pt_files)
            raise ValueError(
                "Ambiguous local checkpoint directory: multiple .pt files "
                "found.\n"
                "Specify one .pt file directly:\n"
                f"{candidates}"
            )
        weights_path = pt_files[0]
        config_path = path / "config.json"
    else:
        raise FileNotFoundError(
            "Local base-model checkpoint is neither a file nor a "
            "directory:\n"
            f"{path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"config.json missing beside checkpoint: {config_path.resolve()}"
        )
    return weights_path, config_path


def _validate_resolved_checkpoint(
    weights_path: Path,
    config_path: Path,
) -> tuple[Path, Path]:
    resolved_weights_path = weights_path.resolve()
    resolved_config_path = config_path.resolve()
    if not resolved_weights_path.is_file():
        raise FileNotFoundError(
            f"resolved weights file missing: {resolved_weights_path}"
        )
    if resolved_weights_path.suffix.lower() != ".pt":
        raise ValueError(
            "resolved checkpoint is not a .pt file: "
            f"{resolved_weights_path}"
        )
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            "config.json missing beside checkpoint: "
            f"{resolved_config_path}"
        )
    return resolved_weights_path, resolved_config_path


def resolve_base_model_identity(
    base_model_ref: str,
) -> BaseModelIdentity:
    """Resolve and fingerprint a checkpoint using upstream path semantics."""
    from stable_worldmodel.data import get_cache_dir
    from stable_worldmodel.wm.utils import _resolve

    original_reference = base_model_ref
    expanded_reference = _expand_reference(original_reference)
    if _is_explicit_local_reference(
        original_reference,
        expanded_reference,
    ):
        weights_path, config_path = _resolve_explicit_local_checkpoint(
            original_reference,
            expanded_reference,
        )
    else:
        checkpoint_root = get_cache_dir(sub_folder="checkpoints")
        weights_path, _ = _resolve(expanded_reference, checkpoint_root)
        config_path = weights_path.parent / "config.json"

    weights_path, config_path = _validate_resolved_checkpoint(
        weights_path,
        config_path,
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
