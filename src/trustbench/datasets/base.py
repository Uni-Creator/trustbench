"""Abstract base class and contract for dataset adapters."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from trustbench.core.protocols import DatasetAdapter as DatasetAdapterProtocol


class DatasetAdapter(ABC, DatasetAdapterProtocol):
    """Abstract base class implementing the DatasetAdapter protocol.

    Subclasses must implement `__len__`, `get_batch`, `get_labels`, and `get_sample_ids`.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples."""

    @abstractmethod
    def get_batch(self, indices: list[int]) -> Any:
        """Return model inputs for the given sample indices."""

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """Return labels aligned with sample order, shape (N,)."""

    @abstractmethod
    def get_sample_ids(self) -> np.ndarray:
        """Return sample identifiers aligned with sample order, shape (N,)."""
