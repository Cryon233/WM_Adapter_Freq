from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info


FEATURE_KEYS = (
    "clean_prefix_tokens",
    "ood_prefix_tokens",
    "clean_context_final_latent",
    "clean_future_latent",
)
ARRAY_KEYS = FEATURE_KEYS + ("actions", "episode_id", "window_id", "appearance_seed")
CACHE_SCHEMA_VERSION = "jepa_wm_robocasa_feature_cache_v2"


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class FeatureCacheWriter:
    def __init__(self, output_path: str | Path, metadata: dict[str, Any]) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.", suffix=".tmp", dir=self.output_path.parent
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        self.file = h5py.File(self.temporary_path, "w", libver="latest")
        self.file.attrs["schema_version"] = CACHE_SCHEMA_VERSION
        self.file.attrs["finalized"] = False
        for key, value in metadata.items():
            self.file.attrs[key] = _json_value(value) if isinstance(value, (dict, list, tuple)) else value
        self.count = 0

    @staticmethod
    def _dtype_for(key: str) -> np.dtype[Any]:
        # Prefix tokens are taken before the final DINOv3 block/norm and can
        # exceed the finite fp16 range. Store them losslessly in float32.
        if key in {"clean_prefix_tokens", "ood_prefix_tokens"}:
            return np.dtype(np.float32)
        if key in {"clean_context_final_latent", "clean_future_latent"}:
            return np.dtype(np.float16)
        if key == "actions":
            return np.dtype(np.float32)
        if key in {"episode_id", "window_id", "appearance_seed"}:
            return np.dtype(np.int64)
        raise ValueError(f"Unknown feature-cache key: {key}")

    def append(self, batch: dict[str, Tensor | np.ndarray]) -> None:
        if set(batch) != set(ARRAY_KEYS):
            raise ValueError(
                f"Feature cache batch keys must be {list(ARRAY_KEYS)}, received {sorted(batch)}"
            )
        arrays = {key: np.asarray(value.detach().cpu() if isinstance(value, Tensor) else value) for key, value in batch.items()}
        batch_size = int(arrays[ARRAY_KEYS[0]].shape[0])
        if any(array.shape[0] != batch_size for array in arrays.values()):
            shapes = {key: tuple(value.shape) for key, value in arrays.items()}
            raise ValueError(f"Feature cache batch dimension mismatch: {shapes}")
        start = self.count
        stop = start + batch_size
        for key in ARRAY_KEYS:
            array = arrays[key].astype(self._dtype_for(key), copy=False)
            if key not in self.file:
                self.file.create_dataset(
                    key,
                    shape=(0, *array.shape[1:]),
                    maxshape=(None, *array.shape[1:]),
                    chunks=(1, *array.shape[1:]),
                    dtype=array.dtype,
                    compression="lzf",
                )
            dataset = self.file[key]
            if tuple(dataset.shape[1:]) != tuple(array.shape[1:]):
                raise ValueError(
                    f"Feature cache shape changed for {key}: expected {dataset.shape[1:]}, "
                    f"received {array.shape[1:]}"
                )
            dataset.resize(stop, axis=0)
            dataset[start:stop] = array
        self.count = stop
        self.file.flush()

    def finalize(self) -> str:
        if self.count == 0:
            raise RuntimeError("Cannot finalize an empty feature cache")
        shapes = {key: list(self.file[key].shape) for key in ARRAY_KEYS}
        fingerprint_payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "count": self.count,
            "shapes": shapes,
            "dataset_sha256": self.file.attrs["dataset_sha256"],
            "base_checkpoint_sha256": self.file.attrs["base_checkpoint_sha256"],
            "dinov3_checkpoint_sha256": self.file.attrs["dinov3_checkpoint_sha256"],
            "upstream_commits": self.file.attrs["upstream_commits"],
            "appearance_metadata": self.file.attrs["appearance_metadata"],
            "selected_window_pairs_sha256": self.file.attrs["selected_window_pairs_sha256"],
            "train_episode_indices_sha256": self.file.attrs["train_episode_indices_sha256"],
            "evaluation_episode_indices_sha256": self.file.attrs[
                "evaluation_episode_indices_sha256"
            ],
            "preprocessing_metadata": self.file.attrs["preprocessing_metadata"],
        }
        for key in (
            "benchmark",
            "benchmark_suite",
            "task_id",
            "task_name",
            "task_key",
            "task_manifest_sha256",
            "camera_key",
            "action_convention",
            "source_trajectory_ids",
            "window_identity",
            "source_trajectory_identity",
            "task_upstream_commits",
        ):
            if key in self.file.attrs:
                fingerprint_payload[key] = self.file.attrs[key]
        fingerprint = hashlib.sha256(_json_value(fingerprint_payload).encode("utf-8")).hexdigest()
        self.file.attrs["window_count"] = self.count
        self.file.attrs["tensor_shapes"] = _json_value(shapes)
        self.file.attrs["cache_fingerprint"] = fingerprint
        self.file.attrs["finalized"] = True
        self.file.flush()
        self.file.close()
        os.replace(self.temporary_path, self.output_path)
        return fingerprint

    def close_unfinalized(self) -> None:
        if self.file.id.valid:
            self.file.close()
        if self.temporary_path.exists():
            self.temporary_path.unlink()

    def __enter__(self) -> FeatureCacheWriter:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None:
            self.close_unfinalized()


class FeatureCacheDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        path: str | Path,
        *,
        expected_base_checkpoint_sha256: str,
        expected_dinov3_checkpoint_sha256: str,
        expected_upstream_commits: dict[str, str],
        expected_appearance_metadata: dict[str, Any],
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Feature cache does not exist: {self.path}")
        self._handles: dict[int, h5py.File] = {}
        with h5py.File(self.path, "r", libver="latest", swmr=True) as cache:
            if not bool(cache.attrs.get("finalized", False)):
                raise RuntimeError(f"Feature cache is not finalized: {self.path}")
            if cache.attrs.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Feature cache schema mismatch: expected {CACHE_SCHEMA_VERSION}, "
                    f"found {cache.attrs.get('schema_version')}"
                )
            if cache.attrs.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256:
                raise RuntimeError(
                    f"Feature cache base checkpoint fingerprint does not match {self.path}"
                )
            if cache.attrs.get("dinov3_checkpoint_sha256") != expected_dinov3_checkpoint_sha256:
                raise RuntimeError(
                    f"Feature cache DINOv3 checkpoint fingerprint does not match {self.path}"
                )
            actual_commits = json.loads(str(cache.attrs["upstream_commits"]))
            if actual_commits != expected_upstream_commits:
                raise RuntimeError(
                    f"Feature cache upstream commits do not match: expected {expected_upstream_commits}, "
                    f"found {actual_commits}"
                )
            actual_appearance = json.loads(str(cache.attrs["appearance_metadata"]))
            if actual_appearance != expected_appearance_metadata:
                raise RuntimeError(
                    f"Feature cache appearance metadata does not match: expected {expected_appearance_metadata}, "
                    f"found {actual_appearance}"
                )
            missing = sorted(set(ARRAY_KEYS).difference(cache.keys()))
            if missing:
                raise RuntimeError(f"Feature cache is missing datasets {missing}: {self.path}")
            lengths = {key: int(cache[key].shape[0]) for key in ARRAY_KEYS}
            if len(set(lengths.values())) != 1:
                raise RuntimeError(f"Feature cache dataset lengths do not match: {lengths}")
            self.length = next(iter(lengths.values()))
            layout = json.loads(str(cache.attrs["token_layout"]))
            expected_shapes = {
                "clean_prefix_tokens": (self.length, 4, layout["total_tokens"], layout["token_dim"]),
                "ood_prefix_tokens": (self.length, 4, layout["total_tokens"], layout["token_dim"]),
                "clean_context_final_latent": (
                    self.length,
                    3,
                    layout["patch_tokens"],
                    layout["token_dim"],
                ),
                "clean_future_latent": (
                    self.length,
                    1,
                    layout["patch_tokens"],
                    layout["token_dim"],
                ),
                "actions": (self.length, 4, 7),
                "episode_id": (self.length,),
                "window_id": (self.length,),
                "appearance_seed": (self.length,),
            }
            actual_shapes = {key: tuple(cache[key].shape) for key in ARRAY_KEYS}
            mismatched_shapes = {
                key: {"expected": expected_shapes[key], "actual": actual_shapes[key]}
                for key in ARRAY_KEYS
                if actual_shapes[key] != expected_shapes[key]
            }
            if mismatched_shapes:
                raise RuntimeError(f"Feature cache tensor shape mismatch: {mismatched_shapes}")
            self.metadata = {key: cache.attrs[key] for key in cache.attrs}

    def __len__(self) -> int:
        return self.length

    def _file(self) -> h5py.File:
        worker = get_worker_info()
        worker_id = -1 if worker is None else worker.id
        handle = self._handles.get(worker_id)
        if handle is None:
            handle = h5py.File(self.path, "r", libver="latest", swmr=True)
            self._handles[worker_id] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        cache = self._file()
        return {key: torch.from_numpy(np.asarray(cache[key][index])) for key in ARRAY_KEYS}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in self._handles.values():
            if handle.id.valid:
                handle.close()
