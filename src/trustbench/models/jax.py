"""JAX and Flax model adapter for functional deep learning inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from trustbench.core.exceptions import ShapeError, UnsupportedCapabilityError
from trustbench.models.base import ModelAdapter


class JAXModelAdapter(ModelAdapter):
    """Adapter for JAX / Flax functional models.

    Accepts an `apply_fn(params, inputs)` function and model parameters.
    """

    def __init__(
        self,
        apply_fn: Callable[[Any, Any], Any],
        params: Any,
        num_classes: int,
        task: str = "classification",
        is_logits: bool = True,
    ) -> None:
        """Initialize JAX model adapter.

        Args:
            apply_fn: Callable taking `(params, inputs)` and returning outputs.
            params: Parameters dictionary/PyTree.
            num_classes: Number of classes (or 1 for regression).
            task: Task type ('classification' or 'regression').
            is_logits: Whether output is logits.
        """
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "JAX is required for JAXModelAdapter. Install it with: pip install jax"
            ) from exc

        self._apply_fn = apply_fn
        self._params = params
        self._num_classes = int(num_classes)
        self._task = task.lower()
        self._is_logits = bool(is_logits)

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
        import jax.numpy as jnp

        x = jnp.asarray(inputs)
        raw = self._apply_fn(self._params, x)
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
            raise UnsupportedCapabilityError("Model outputs probabilities, not raw logits.")
        out = self._call_model(inputs)
        if out.ndim == 1 and self._num_classes == 2:
            out = np.column_stack([-out, out])
        return out

    def predict_proba(self, inputs: Any) -> np.ndarray:
        if self._task != "classification":
            raise UnsupportedCapabilityError("predict_proba is only available for classification.")
        out = self._call_model(inputs)
        if not self._is_logits:
            return out
        if out.ndim != 2:
            raise ShapeError(f"Expected 2D logits for softmax, got shape {out.shape}.")
        exp = np.exp(out - np.max(out, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
