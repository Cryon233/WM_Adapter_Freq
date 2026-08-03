from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

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
        self._content_hashers = {
            key: hashlib.sha256(f"{key}\0".encode("utf-8"))
            for key in V2_ARRAY_KEYS
        }
        self._content_layouts: dict[str, tuple[str, tuple[int, ...]]] = {}

    @staticmethod
    def _numeric_dtype(key: str) -> np.dtype[Any]:
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
        shapes = {key: tuple(value.shape) for key, value in arrays.items()}
        if any(value.shape[0] != batch_size for value in arrays.values()):
            raise ValueError(f"V2 cache batch dimensions differ: {shapes}")
        start, stop = self.count, self.count + batch_size
        for key in V2_ARRAY_KEYS:
            array = arrays[key]
            if key == "source_trajectory_id":
                strings = [str(value) for value in array.tolist()]
                for value in strings:
                    encoded = value.encode("utf-8")
                    self._content_hashers[key].update(
                        len(encoded).to_bytes(8, "little")
                    )
                    self._content_hashers[key].update(encoded)
                if key not in self.file:
                    self.file.create_dataset(
                        key, shape=(0,), maxshape=(None,), chunks=(1,),
                        dtype=h5py.string_dtype("utf-8"),
                    )
                dataset = self.file[key]
                dataset.resize(stop, axis=0)
                dataset[start:stop] = strings
                continue
            array = array.astype(self._numeric_dtype(key), copy=False)
            layout = (array.dtype.str, tuple(int(value) for value in array.shape[1:]))
            if key not in self._content_layouts:
                self._content_layouts[key] = layout
                self._content_hashers[key].update(_json(layout).encode("utf-8"))
            elif self._content_layouts[key] != layout:
                raise ValueError(
                    f"V2 cache content layout changed for {key}: "
                    f"expected={self._content_layouts[key]}, actual={layout}"
                )
            self._content_hashers[key].update(
                np.ascontiguousarray(array).tobytes(order="C")
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
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"V2 feature cache does not exist: {self.path}")
        self._handles: dict[int, h5py.File] = {}
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
