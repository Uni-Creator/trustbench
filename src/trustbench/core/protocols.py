"""Structural protocols for model and dataset adapters."""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelAdapter(Protocol):
    """Inference contract used by the core evaluator."""

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        ...

    @property
    def supports_predict_proba(self) -> bool:
        """Whether `predict_proba` is implemented."""
        ...

    @property
    def supports_predict_logits(self) -> bool:
        """Whether `predict_logits` is implemented."""
        ...

    def predict(self, inputs: Any) -> np.ndarray:
        """Return predicted class indices of shape (N,)."""
        ...

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Return class probabilities of shape (N, C) if supported."""
        ...

    def predict_logits(self, inputs: Any) -> np.ndarray:
        """Return raw logits of shape (N, C) if supported."""
        ...


@runtime_checkable
class DatasetAdapter(Protocol):
    """Dataset contract used by the core evaluator."""

    def __len__(self) -> int:
        ...

    def get_batch(self, indices: list[int]) -> Any:
        """Return model inputs for the given sample indices."""
        ...

    def get_labels(self) -> np.ndarray:
        """Return labels aligned with sample order, shape (N,)."""
        ...

    def get_sample_ids(self) -> np.ndarray:
        """Return sample identifiers aligned with sample order, shape (N,)."""
        ...
