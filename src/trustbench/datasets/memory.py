"""In-memory dataset adapter used for tests and framework-agnostic evaluation."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from trustbench.datasets.base import DatasetAdapter


class InMemoryDatasetAdapter(DatasetAdapter):
    """Dataset adapter over arrays already loaded in memory."""

    def __init__(
        self,
        inputs: Any,
        labels: Sequence[Any] | np.ndarray,
        sample_ids: Sequence[Any] | np.ndarray | None = None,
    ) -> None:
        self._inputs = inputs
        self._labels = np.asarray(labels)
        n = len(self._labels)
        if hasattr(inputs, "__len__") and len(inputs) != n:
            raise ValueError(
                f"inputs length ({len(inputs)}) does not match labels length ({n})."
            )
        if sample_ids is None:
            self._sample_ids = np.arange(n)
        else:
            self._sample_ids = np.asarray(sample_ids)
            if len(self._sample_ids) != n:
                raise ValueError("sample_ids length does not match labels length.")

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def get_batch(self, indices: list[int]) -> Any:
        if hasattr(self._inputs, "__getitem__"):
            try:
                return self._inputs[indices]
            except (TypeError, IndexError):
                return [self._inputs[i] for i in indices]
        raise TypeError("inputs do not support indexing.")

    def get_labels(self) -> np.ndarray:
        return self._labels

    def get_sample_ids(self) -> np.ndarray:
        return self._sample_ids
