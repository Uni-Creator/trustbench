"""Abstract base class and contract for model adapters."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from trustbench.core.exceptions import UnsupportedCapabilityError
from trustbench.core.protocols import ModelAdapter as ModelAdapterProtocol


class ModelAdapter(ABC, ModelAdapterProtocol):
    """Abstract base class implementing the ModelAdapter protocol.

    Subclasses must implement `num_classes` and `predict(inputs)`.
    `predict_proba` and `predict_logits` can optionally be overridden.
    """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""

    @property
    def supports_predict_proba(self) -> bool:
        """Whether this adapter can return class probabilities."""
        return False

    @property
    def supports_predict_logits(self) -> bool:
        """Whether this adapter can return raw logits."""
        return False

    @abstractmethod
    def predict(self, inputs: Any) -> np.ndarray:
        """Return predicted class indices of shape (N,)."""

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Return class probabilities of shape (N, C)."""
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support predict_proba()."
        )

    def predict_logits(self, inputs: Any) -> np.ndarray:
        """Return raw logits of shape (N, C)."""
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support predict_logits()."
        )
