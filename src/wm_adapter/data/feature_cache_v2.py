from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info


CACHE_SCHEMA_VERSION_V2 = "jepa_wm_feature_cache_v2.1"
V2_ARRAY_KEYS = (
    "clean_context_middle_tokens",
    "ood_context_middle_tokens",
    "clean_target_latents",
    "rollout_actions",
    "episode_id",
    "window_id",
    "source_trajectory_id",
    "appearance_seed",
    "appearance_severity",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_GENERATED_FINGERPRINT_ATTRS = {
    "finalized",
    "cache_fingerprint",
    "window_count",
    "tensor_shapes",
    "array_content_sha256",
    "content_sha256",
}


def _attribute_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _numeric_dtype_v2(key: str) -> np.dtype[Any]:
    if key in {"clean_context_middle_tokens", "ood_context_middle_tokens"}:
        return np.dtype(np.float32)
    if key == "clean_target_latents":
        return np.dtype(np.float16)
    if key == "rollout_actions":
        return np.dtype(np.float32)
    if key in {"episode_id", "window_id", "appearance_seed"}:
        return np.dtype(np.int64)
    if key == "appearance_severity":
        return np.dtype(np.float32)
    raise ValueError(f"Unknown v2 feature-cache key {key!r}")


def _new_array_hasher(key: str) -> Any:
    return hashlib.sha256(f"{key}\0".encode("utf-8"))


def update_array_hasher(
    hasher: Any,
    key: str,
    array_or_strings: Any,
    *,
    initialize_layout: bool,
) -> tuple[str, tuple[int, ...]] | None:
    """Update one canonical content hash for writer and deep verifier alike."""
    if key == "source_trajectory_id":
        values = np.asarray(array_or_strings, dtype=object).reshape(-1).tolist()
        for value in values:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            encoded = str(value).encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "little"))
            hasher.update(encoded)
        return None
    array = np.asarray(array_or_strings).astype(_numeric_dtype_v2(key), copy=False)
    layout = (array.dtype.str, tuple(int(value) for value in array.shape[1:]))
    if initialize_layout:
        hasher.update(_json(layout).encode("utf-8"))
    hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
    return layout


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_file_sha256_from_verified_state_v2(
    path: str | Path,
    *,
    expected_sha256: str | None,
    expected_size: int | None,
    expected_mtime_ns: int | None,
) -> str:
    """Reuse one suite deep-verification result while the physical file is unchanged."""
    resolved = Path(path).expanduser().resolve()
    if expected_sha256 is None:
        return _file_sha256(resolved)
    if len(expected_sha256) != 64:
        raise RuntimeError(
            f"Invalid expected v2 cache file SHA256 {expected_sha256!r}: {resolved}"
        )
    stat = resolved.stat()
    actual_identity = (int(stat.st_size), int(stat.st_mtime_ns))
    expected_identity = (
        int(expected_size) if expected_size is not None else None,
        int(expected_mtime_ns) if expected_mtime_ns is not None else None,
    )
    if actual_identity != expected_identity:
        raise RuntimeError(
            "V2 cache changed after suite deep verification; restart the suite so "
            "the cache is verified before reuse: "
            f"path={resolved}, expected_size_mtime={expected_identity}, "
            f"actual_size_mtime={actual_identity}"
        )
    return expected_sha256


def recompute_array_content_sha256_v2(
    handle: h5py.File,
    *,
    chunk_windows: int = 8,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, str]:
    if chunk_windows <= 0:
        raise ValueError(f"chunk_windows must be positive, received {chunk_windows}")
    missing = sorted(set(V2_ARRAY_KEYS).difference(handle.keys()))
    if missing:
        raise RuntimeError(f"V2 feature cache is missing arrays {missing}")
    hashes: dict[str, str] = {}
    for key in V2_ARRAY_KEYS:
        dataset = handle[key]
        total = int(dataset.shape[0])
        hasher = _new_array_hasher(key)
        initialized = False
        expected_layout: tuple[str, tuple[int, ...]] | None = None
        for start in range(0, total, chunk_windows):
            stop = min(start + chunk_windows, total)
            values = dataset[start:stop]
            layout = update_array_hasher(
                hasher, key, values, initialize_layout=not initialized
            )
            if layout is not None:
                if expected_layout is not None and layout != expected_layout:
                    raise RuntimeError(
                        f"V2 cache layout changed while reading {key}: "
                        f"expected={expected_layout}, actual={layout}"
                    )
                expected_layout = layout
                initialized = True
            if progress is not None:
                progress(key, stop, total)
        hashes[key] = hasher.hexdigest()
    return hashes


def verify_cache_content_v2(
    path: str | Path,
    *,
    chunk_windows: int = 8,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"V2 feature cache does not exist: {resolved}")
    with h5py.File(resolved, "r", libver="latest", swmr=True) as handle:
        schema = str(handle.attrs.get("schema_version", ""))
        if schema != CACHE_SCHEMA_VERSION_V2:
            raise RuntimeError(
                f"V2 feature-cache schema mismatch: expected={CACHE_SCHEMA_VERSION_V2}, "
                f"actual={schema}, path={resolved}"
            )
        if not bool(handle.attrs.get("finalized", False)):
            raise RuntimeError(f"V2 feature cache is not finalized: {resolved}")
        stored_raw = _attribute_value(handle.attrs.get("array_content_sha256", ""))
        stored = json.loads(stored_raw) if isinstance(stored_raw, str) else stored_raw
        if not isinstance(stored, dict) or set(stored) != set(V2_ARRAY_KEYS):
            raise RuntimeError(f"V2 feature cache has invalid stored array hashes: {resolved}")
        recomputed = recompute_array_content_sha256_v2(
            handle, chunk_windows=chunk_windows, progress=progress
        )
        if recomputed != stored:
            mismatch = {
                key: {"stored": stored.get(key), "recomputed": recomputed[key]}
                for key in V2_ARRAY_KEYS
                if stored.get(key) != recomputed[key]
            }
            raise RuntimeError(
                f"V2 feature-cache array content hash mismatch at {resolved}: {mismatch}"
            )
        recomputed_content = hashlib.sha256(
            _json(recomputed).encode("utf-8")
        ).hexdigest()
        stored_content = str(_attribute_value(handle.attrs.get("content_sha256", "")))
        if recomputed_content != stored_content:
            raise RuntimeError(
                "V2 feature-cache unified content hash mismatch: "
                f"stored={stored_content}, recomputed={recomputed_content}, path={resolved}"
            )
        expected_fingerprint = cache_fingerprint_v2(handle)
        stored_fingerprint = str(handle.attrs.get("cache_fingerprint", ""))
        if stored_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "V2 feature-cache fingerprint mismatch: "
                f"stored={stored_fingerprint}, recomputed={expected_fingerprint}, path={resolved}"
            )
    stat = resolved.stat()
    return {
        "content_verified": True,
        "content_verified_at_unix": time.time(),
        "recomputed_array_content_sha256": recomputed,
        "recomputed_content_sha256": recomputed_content,
        "cache_fingerprint": stored_fingerprint,
        "cache_file_sha256": _file_sha256(resolved),
        "cache_file_size": int(stat.st_size),
        "cache_file_mtime_ns": int(stat.st_mtime_ns),
    }


def cache_fingerprint_v2(handle: h5py.File) -> str:
    """Recreate the v2 fingerprint from metadata, shapes and stored content hashes."""
    shapes = {key: list(handle[key].shape) for key in V2_ARRAY_KEYS}
    raw_hashes = _attribute_value(handle.attrs.get("array_content_sha256", ""))
    array_hashes = json.loads(raw_hashes) if isinstance(raw_hashes, str) else raw_hashes
    content_hash = str(_attribute_value(handle.attrs.get("content_sha256", "")))
    fields: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION_V2,
        "window_count": int(handle[V2_ARRAY_KEYS[0]].shape[0]),
        "tensor_shapes": shapes,
        "array_content_sha256": array_hashes,
        "content_sha256": content_hash,
    }
    for key in sorted(handle.attrs):
        if key not in _GENERATED_FINGERPRINT_ATTRS:
            fields[key] = _attribute_value(handle.attrs[key])
    return hashlib.sha256(_json(fields).encode("utf-8")).hexdigest()


class FeatureCacheV2Writer:
    def __init__(self, output_path: str | Path, metadata: dict[str, Any]) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.", suffix=".tmp", dir=self.output_path.parent
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        self.file = h5py.File(self.temporary_path, "w", libver="latest")
        self.file.attrs["schema_version"] = CACHE_SCHEMA_VERSION_V2
        self.file.attrs["finalized"] = False
        for key, value in metadata.items():
            # HDF5 has no native representation for Python None.  Optional task
            # metadata is therefore represented by an absent attribute; readers
            # already normalize an absent attribute to None for contract checks.
            if value is None:
                continue
            self.file.attrs[key] = _json(value) if isinstance(value, (dict, list, tuple)) else value
        self.count = 0
        self._content_hashers = {key: _new_array_hasher(key) for key in V2_ARRAY_KEYS}
        self._content_layouts: dict[str, tuple[str, tuple[int, ...]]] = {}

    def append(self, batch: dict[str, Tensor | np.ndarray | list[str]]) -> None:
        if set(batch) != set(V2_ARRAY_KEYS):
            raise ValueError(
                f"V2 cache keys must be {list(V2_ARRAY_KEYS)}, received {sorted(batch)}"
            )
        arrays: dict[str, np.ndarray] = {}
        for key, value in batch.items():
            if key == "source_trajectory_id":
                arrays[key] = np.asarray(value, dtype=object)
            else:
                arrays[key] = np.asarray(
                    value.detach().cpu() if isinstance(value, Tensor) else value
                )
        batch_size = int(arrays[V2_ARRAY_KEYS[0]].shape[0])
        if batch_size <= 0:
            raise ValueError("V2 feature-cache append batch must not be empty")
        shapes = {key: tuple(value.shape) for key, value in arrays.items()}
        if any(value.shape[0] != batch_size for value in arrays.values()):
            raise ValueError(f"V2 cache batch dimensions differ: {shapes}")
        start, stop = self.count, self.count + batch_size
        for key in V2_ARRAY_KEYS:
            array = arrays[key]
            if key == "source_trajectory_id":
                strings = [str(value) for value in array.tolist()]
                update_array_hasher(
                    self._content_hashers[key], key, strings, initialize_layout=False
                )
                if key not in self.file:
                    self.file.create_dataset(
                        key, shape=(0,), maxshape=(None,), chunks=(1,),
                        dtype=h5py.string_dtype("utf-8"),
                    )
                dataset = self.file[key]
                dataset.resize(stop, axis=0)
                dataset[start:stop] = strings
                continue
            array = array.astype(_numeric_dtype_v2(key), copy=False)
            layout = (array.dtype.str, tuple(int(value) for value in array.shape[1:]))
            if key not in self._content_layouts:
                self._content_layouts[key] = layout
            elif self._content_layouts[key] != layout:
                raise ValueError(
                    f"V2 cache content layout changed for {key}: "
                    f"expected={self._content_layouts[key]}, actual={layout}"
                )
            update_array_hasher(
                self._content_hashers[key], key, array,
                initialize_layout=self._content_layouts[key] == layout and start == 0,
            )
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
                    f"V2 cache shape changed for {key}: expected={dataset.shape[1:]}, "
                    f"actual={array.shape[1:]}"
                )
            dataset.resize(stop, axis=0)
            dataset[start:stop] = array
        self.count = stop
        self.file.flush()

    def finalize(self) -> str:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize an empty v2 feature cache")
        shapes = {key: list(self.file[key].shape) for key in V2_ARRAY_KEYS}
        array_hashes = {
            key: hasher.hexdigest()
            for key, hasher in self._content_hashers.items()
        }
        content_hash = hashlib.sha256(
            _json(array_hashes).encode("utf-8")
        ).hexdigest()
        self.file.attrs["window_count"] = self.count
        self.file.attrs["tensor_shapes"] = _json(shapes)
        self.file.attrs["array_content_sha256"] = _json(array_hashes)
        self.file.attrs["content_sha256"] = content_hash
        fingerprint = cache_fingerprint_v2(self.file)
        self.file.attrs["cache_fingerprint"] = fingerprint
        self.file.attrs["finalized"] = True
        self.file.flush()
        self.file.close()
        os.replace(self.temporary_path, self.output_path)
        return fingerprint

    def close_unfinalized(self) -> None:
        if self.file.id.valid:
            self.file.close()
        self.temporary_path.unlink(missing_ok=True)


class FeatureCacheV2Dataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        path: str | Path,
        *,
        expected_base_checkpoint_sha256: str | None = None,
        expected_dinov3_checkpoint_sha256: str | None = None,
        verify_content: bool = False,
        content_verification_chunk_windows: int = 8,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"V2 feature cache does not exist: {self.path}")
        self._handles: dict[int, h5py.File] = {}
        verification = (
            verify_cache_content_v2(
                self.path, chunk_windows=content_verification_chunk_windows
            )
            if verify_content
            else None
        )
        with h5py.File(self.path, "r", libver="latest", swmr=True) as cache:
            schema = str(cache.attrs.get("schema_version", ""))
            if schema != CACHE_SCHEMA_VERSION_V2:
                raise RuntimeError(
                    "V2 loader rejects incompatible cache schema: "
                    f"expected={CACHE_SCHEMA_VERSION_V2}, actual={schema}, path={self.path}"
                )
            if not bool(cache.attrs.get("finalized", False)):
                raise RuntimeError(f"V2 feature cache is not finalized: {self.path}")
            missing = sorted(set(V2_ARRAY_KEYS).difference(cache.keys()))
            if missing:
                raise RuntimeError(f"V2 feature cache is missing arrays {missing}")
            for attribute in (
                "array_content_sha256",
                "content_sha256",
                "cache_fingerprint",
            ):
                if not str(cache.attrs.get(attribute, "")):
                    raise RuntimeError(
                        f"V2 feature cache lacks required {attribute}: {self.path}"
                    )
            expected_fingerprint = cache_fingerprint_v2(cache)
            if str(cache.attrs["cache_fingerprint"]) != expected_fingerprint:
                raise RuntimeError(
                    "V2 feature-cache fingerprint does not match its content hashes: "
                    f"expected={expected_fingerprint}, "
                    f"actual={cache.attrs['cache_fingerprint']}, path={self.path}"
                )
            lengths = {key: int(cache[key].shape[0]) for key in V2_ARRAY_KEYS}
            if len(set(lengths.values())) != 1:
                raise RuntimeError(f"V2 feature cache lengths differ: {lengths}")
            self.length = next(iter(lengths.values()))
            if self.length <= 0:
                raise RuntimeError(f"V2 feature cache is empty: {self.path}")
            expected_shapes = {
                "clean_context_middle_tokens": (self.length, 3),
                "ood_context_middle_tokens": (self.length, 3),
                "clean_target_latents": (self.length, 6),
                "rollout_actions": (self.length, 3, 7),
            }
            for key, prefix in expected_shapes.items():
                if tuple(cache[key].shape[: len(prefix)]) != prefix:
                    raise RuntimeError(
                        f"V2 cache array {key} has shape {tuple(cache[key].shape)}, "
                        f"expected prefix {prefix}"
                    )
            if expected_base_checkpoint_sha256 is not None and str(
                cache.attrs.get("base_checkpoint_sha256", "")
            ) != expected_base_checkpoint_sha256:
                raise RuntimeError("V2 cache base checkpoint SHA256 mismatch")
            if expected_dinov3_checkpoint_sha256 is not None and str(
                cache.attrs.get("dinov3_checkpoint_sha256", "")
            ) != expected_dinov3_checkpoint_sha256:
                raise RuntimeError("V2 cache DINOv3 checkpoint SHA256 mismatch")
            self.metadata = {key: cache.attrs[key] for key in cache.attrs}
            if verification is not None:
                self.metadata.update(verification)

    def __len__(self) -> int:
        return self.length

    def _file(self) -> h5py.File:
        worker = get_worker_info()
        worker_id = -1 if worker is None else worker.id
        if worker_id not in self._handles:
            self._handles[worker_id] = h5py.File(self.path, "r", libver="latest", swmr=True)
        return self._handles[worker_id]

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        import torch

        cache = self._file()
        result: dict[str, Tensor | str] = {}
        for key in V2_ARRAY_KEYS:
            value = cache[key][index]
            if key == "source_trajectory_id":
                result[key] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            else:
                result[key] = torch.from_numpy(np.asarray(value))
        return result

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in self._handles.values():
            if handle.id.valid:
                handle.close()
