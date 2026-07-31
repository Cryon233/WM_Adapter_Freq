from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor


_FEATURE_KEYS = {
    "clean_prefix_tokens",
    "shifted_prefix_tokens",
    "clean_targets",
}


class FeatureCacheWriter:
    """Append paired feature batches to a chunked LZF HDF5 cache."""

    def __init__(
        self,
        path: str | Path,
        metadata: Mapping[str, str | int | float],
        chunk_size: int,
    ) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.chunk_size = int(chunk_size)
        self.file: h5py.File | None = None
        self.length = 0

    def __enter__(self) -> "FeatureCacheWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w", libver="latest")
        for key, value in self.metadata.items():
            self.file.attrs[key] = value
        return self

    def __exit__(self, *exc: object) -> None:
        if self.file is not None:
            self.file.attrs["length"] = self.length
            self.file.flush()
            self.file.close()
            self.file = None

    @staticmethod
    def _numpy(
        key: str,
        value: Tensor | np.ndarray,
    ) -> np.ndarray:
        if isinstance(value, Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = value
        if key in _FEATURE_KEYS:
            return array.astype(np.float16, copy=False)
        if key in {"action", "proprio"}:
            return array.astype(np.float32, copy=False)
        return array

    def _dataset_chunks(
        self,
        key: str,
        array: np.ndarray,
    ) -> tuple[int, ...]:
        if key in {"clean_prefix_tokens", "clean_targets"}:
            return (1, *array.shape[1:])
        if key == "shifted_prefix_tokens":
            return (1, 1, *array.shape[2:])
        if key in {"action", "proprio", "shift_type", "shift_seed"}:
            return (
                min(self.chunk_size, 64),
                *array.shape[1:],
            )
        raise ValueError(f"Unsupported feature cache key: {key}")

    def append(
        self,
        batch: Mapping[str, Tensor | np.ndarray],
    ) -> None:
        if self.file is None:
            raise RuntimeError(
                "FeatureCacheWriter must be used as a context manager."
            )
        arrays = {
            key: self._numpy(key, value)
            for key, value in batch.items()
        }
        batch_size = len(next(iter(arrays.values())))
        if len(self.file.keys()) == 0:
            for key, array in arrays.items():
                self.file.create_dataset(
                    key,
                    shape=(0, *array.shape[1:]),
                    maxshape=(None, *array.shape[1:]),
                    dtype=array.dtype,
                    chunks=self._dataset_chunks(key, array),
                    compression="lzf",
                )

        start = self.length
        end = start + batch_size
        for key, array in arrays.items():
            dataset = self.file[key]
            dataset.resize(end, axis=0)
            dataset[start:end] = array
        self.length = end


class PairedFeatureDataset(torch.utils.data.Dataset[dict[str, Tensor]]):
    """Lazy per-worker access to all four views of each cached window."""

    def __init__(
        self,
        path: str | Path,
        identity_probability: float = 0.2,
    ) -> None:
        if not 0.0 <= identity_probability <= 1.0:
            raise ValueError("identity_probability must be in [0, 1]")
        self.path = Path(path)
        self.identity_probability = float(identity_probability)
        self._file: h5py.File | None = None

        with h5py.File(self.path, "r") as handle:
            self.num_windows = int(
                handle.attrs.get("length", len(handle["action"]))
            )
            self.variants_per_window = int(
                handle.attrs["variants_per_window"]
            )
            self.metadata = {
                str(key): self._attribute_value(value)
                for key, value in handle.attrs.items()
            }
        self.backend = str(self.metadata["backend"])
        self.normalization = {
            "action": json.loads(
                str(self.metadata["action_normalization"])
            ),
            "proprio": json.loads(
                str(self.metadata["proprio_normalization"])
            ),
        }

    @staticmethod
    def _attribute_value(value: Any) -> str | int | float:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            return value.item()
        return value

    def __len__(self) -> int:
        return self.num_windows * self.variants_per_window

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(
                self.path,
                "r",
                swmr=True,
                rdcc_nbytes=256 * 1024 * 1024,
            )
        return self._file

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def _tensor(dataset: h5py.Dataset, index: Any) -> Tensor:
        return torch.from_numpy(np.asarray(dataset[index]).copy())

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        window_index = int(index) // self.variants_per_window
        view_index = int(index) % self.variants_per_window
        handle = self._open()
        if bool(torch.rand(()) < self.identity_probability):
            prefix = self._tensor(
                handle["clean_prefix_tokens"],
                window_index,
            )
        else:
            prefix = self._tensor(
                handle["shifted_prefix_tokens"],
                (window_index, view_index),
            )

        return {
            "prefix_tokens": prefix,
            "clean_targets": self._tensor(
                handle["clean_targets"],
                window_index,
            ),
            "action": self._tensor(handle["action"], window_index),
            "proprio": self._tensor(handle["proprio"], window_index),
        }
