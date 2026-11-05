"""DatasetAdapter wrapping any torch.utils.data.Dataset."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from trustbench.datasets.base import DatasetAdapter


class TorchDatasetAdapter(DatasetAdapter):
    """Adapter that wraps a ``torch.utils.data.Dataset``.

    ``__getitem__`` must return either:
    - ``(tensor, label)``
    - ``(tensor, label, sample_id)``

    Args:
        dataset: Any ``torch.utils.data.Dataset``.
        sample_ids: Optional explicit sample IDs (length must match dataset).
            If omitted, integer indices ``0…N-1`` are used, unless each item
            already yields a third element.
    """

    def __init__(
        self,
        dataset: Dataset,
        sample_ids: Sequence[Any] | np.ndarray | None = None,
    ) -> None:
        self._dataset = dataset
        n = len(dataset)  # type: ignore[arg-type]

        if sample_ids is not None:
            self._sample_ids: np.ndarray | None = np.asarray(sample_ids)
            if len(self._sample_ids) != n:
                raise ValueError(
                    f"sample_ids length ({len(self._sample_ids)}) does not match "
                    f"dataset length ({n})."
                )
        else:
            self._sample_ids = None  # will be resolved on first access

        self._n = n
        self._labels_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return stacked tensor of shape ``(len(indices), *sample_shape)``."""
        tensors = []
        for idx in indices:
            item = self._dataset[idx]
            if isinstance(item, (tuple, list)):
                x = item[0]
            else:
                x = item
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            tensors.append(x)
        return torch.stack(tensors, dim=0)

    def get_labels(self) -> np.ndarray:
        if self._labels_cache is not None:
            return self._labels_cache

        # Fast path: dataset exposes a .targets attribute (e.g. torchvision datasets)
        if hasattr(self._dataset, "targets"):
            targets = self._dataset.targets  # type: ignore[attr-defined]
            if isinstance(targets, torch.Tensor):
                self._labels_cache = targets.numpy()
            else:
                self._labels_cache = np.asarray(targets)
            return self._labels_cache

        # Slow path: iterate items
        labels = []
        for i in range(self._n):
            item = self._dataset[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                label = item[1]
            else:
                label = -1
            labels.append(int(label) if hasattr(label, "__int__") else label)
        self._labels_cache = np.asarray(labels)
        return self._labels_cache

    def get_sample_ids(self) -> np.ndarray:
        if self._sample_ids is not None:
            return self._sample_ids

        # Try to read sample_id from each item's third element
        item0 = self._dataset[0]
        if isinstance(item0, (tuple, list)) and len(item0) >= 3:
            ids = []
            for i in range(self._n):
                item = self._dataset[i]
                ids.append(item[2])
            self._sample_ids = np.asarray(ids)
        else:
            self._sample_ids = np.arange(self._n)
        return self._sample_ids
