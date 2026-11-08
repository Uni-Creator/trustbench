"""TensorFlow and Keras model adapter for deep learning inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from trustbench.core.exceptions import ShapeError, UnsupportedCapabilityError
from trustbench.models.base import ModelAdapter


class KerasModelAdapter(ModelAdapter):
    """Adapter for TensorFlow and Keras models (``tf.keras.Model`` or standalone ``keras.Model``).

    Supports:
      - Classification models emitting logits or probabilities.
      - Regression models emitting continuous outputs.
      - Dynamic batch evaluation with GPU/CPU hardware placement.
    """

    def __init__(
        self,
        model: Any,
        num_classes: int,
        task: str = "classification",
        is_logits: bool = True,
    ) -> None:
        """Initialize Keras model adapter.

        Args:
            model: Keras or tf.keras Model instance.
            num_classes: Number of output classes (or 1 for regression).
            task: Task type ('classification' or 'regression').
            is_logits: Whether the model's raw output represents unnormalized logits.
        """
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            try:
                import keras  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "TensorFlow or Keras is required for KerasModelAdapter. "
                    "Install it with: pip install tensorflow"
                ) from exc

        self._model = model
        self._num_classes = int(num_classes)
        self._task = task.lower()
        self._is_logits = bool(is_logits)

    @property
    def model(self) -> Any:
        """Underlying Keras model."""
        return self._model

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def supports_predict_proba(self) -> bool:
        return self._task == "classification"

    @property
    def supports_predict_logits(self) -> bool:
        return self._is_logits

    def _call_model(self, inputs: Any) -> np.ndarray:
        x = np.asarray(inputs)
        raw = self._model(x, training=False)
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        return np.asarray(raw)

    def predict(self, inputs: Any) -> np.ndarray:
        out = self._call_model(inputs)
        if self._task == "classification":
            if out.ndim >= 2:
                return np.argmax(out, axis=1)
            return (out > 0.5).astype(int) if self._num_classes == 2 else out.astype(int)
        return out.squeeze()

    def predict_logits(self, inputs: Any) -> np.ndarray:
        if not self._is_logits:
            raise UnsupportedCapabilityError(
                "Model outputs probabilities, not raw unnormalized logits."
            )
        out = self._call_model(inputs)
        if out.ndim == 1 and self._num_classes == 2:
            out = np.column_stack([-out, out])
        return out

    def predict_proba(self, inputs: Any) -> np.ndarray:
        if self._task != "classification":
            raise UnsupportedCapabilityError("predict_proba is only available for classification.")
        out = self._call_model(inputs)
        if not self._is_logits:
            # Already probabilities
            return out
        if out.ndim != 2:
            raise ShapeError(f"Expected 2D logits for softmax, got shape {out.shape}.")
        exp = np.exp(out - np.max(out, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
