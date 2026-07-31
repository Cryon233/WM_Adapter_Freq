from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch


UPSTREAM_COMMITS: dict[str, str] = {
    "jepa-wms": "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0",
    "dinov3": "6876159a11b4df116f30f667f8c9888617df0751",
    "robosuite": "9548a5a35bde8eabf47f760802045cca447e9c0c",
    "robocasa": "2544dc2e38bb44f5ced80fbc91114a2f7934016a",
}


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: str | Path) -> str:
    resolved = Path(repo).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Upstream source checkout does not exist: {resolved}")
    completed = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cannot resolve upstream Git commit at {resolved}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_upstream_commits(third_party_root: str | Path) -> dict[str, str]:
    root = Path(third_party_root).resolve()
    actual: dict[str, str] = {}
    for name, expected in UPSTREAM_COMMITS.items():
        repo = root / name
        value = git_commit(repo)
        if value != expected:
            raise RuntimeError(
                f"Upstream commit mismatch for {name}: expected {expected}, found {value} at {repo}"
            )
        actual[name] = value
    return actual


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_method_checkpoint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Method checkpoint does not exist: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    required = {
        "method_name",
        "peft_state_dict",
        "method_config",
        "trainable_parameter_count",
        "base_checkpoint_sha256",
        "dinov3_checkpoint_sha256",
        "upstream_commits",
        "cache_fingerprint",
        "appearance_metadata",
        "training_config",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Method checkpoint is missing fields {missing}: {resolved}")
    return payload
