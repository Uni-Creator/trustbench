"""Scikit-learn model adapter for classification and regression."""

from __future__ import annotations

from typing import Any

import numpy as np

from trustbench.core.exceptions import ShapeError, UnsupportedCapabilityError
from trustbench.models.base import ModelAdapter


class SklearnModelAdapter(ModelAdapter):
    """Adapter for Scikit-learn estimators (classifiers and regressors).

    Supports:
      - Estimators implementing ``predict()``
      - Classifiers implementing ``predict_proba()`` or ``decision_function()``
      - Automatic class count inference from ``classes_`` attribute
    """

    def __init__(
        self,
        estimator: Any,
        num_classes: int | None = None,
        task: str = "classification",
    ) -> None:
        """Initialize adapter.

        Args:
            estimator: Scikit-learn estimator instance (e.g. LogisticRegression, RandomForestClassifier, etc.).
            num_classes: Optional number of output classes. If omitted for classification,
                will attempt to infer from ``estimator.classes_``.
            task: Task type, either "classification" or "regression".
        """
        self._estimator = estimator
        self._task = task.lower()

        if self._task == "classification":
            if num_classes is not None:
                self._num_classes = int(num_classes)
            elif hasattr(estimator, "classes_"):
                self._num_classes = len(estimator.classes_)
            else:
                self._num_classes = 2  # default fallback for binary
        else:
            self._num_classes = 1

    @property
    def estimator(self) -> Any:
        """Underlying scikit-learn estimator."""
        return self._estimator

    @property
    def task(self) -> str:
        """Task type ('classification' or 'regression')."""
        return self._task

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def supports_predict_proba(self) -> bool:
        return hasattr(self._estimator, "predict_proba") or hasattr(
            self._estimator, "decision_function"
        )

    @property
    def supports_predict_logits(self) -> bool:
        return hasattr(self._estimator, "decision_function")

    def _convert_inputs(self, inputs: Any) -> np.ndarray:
        """Ensure inputs are in numpy array format."""
        if hasattr(inputs, "numpy"):  # torch tensor or tensorflow tensor
            return inputs.numpy()
        return np.asarray(inputs)

    def predict(self, inputs: Any) -> np.ndarray:
        """Return predicted labels (classification) or target values (regression)."""
        x = self._convert_inputs(inputs)
        preds = self._estimator.predict(x)
        arr = np.asarray(preds)

        if self._task == "classification":
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            # If classes are strings or objects, map to integer indices if classes_ exists
            if hasattr(self._estimator, "classes_") and not np.issubdtype(
                arr.dtype, np.integer
            ):
                classes = list(self._estimator.classes_)
                arr = np.array([classes.index(p) for p in arr], dtype=int)
            return arr.astype(int)

        return arr

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Return class probabilities of shape (N, C)."""
        if not self.supports_predict_proba:
            raise UnsupportedCapabilityError(
                f"{type(self._estimator).__name__} does not support probability estimation."
            )
        x = self._convert_inputs(inputs)

        if hasattr(self._estimator, "predict_proba"):
            probs = np.asarray(self._estimator.predict_proba(x), dtype=float)
            if probs.ndim != 2 or probs.shape[1] != self._num_classes:
                # Handle binary 1D proba case
                if probs.ndim == 1 and self._num_classes == 2:
                    probs = np.column_stack([1.0 - probs, probs])
                elif probs.shape[1] != self._num_classes:
                    raise ShapeError(
                        f"Expected probabilities of shape (N, {self._num_classes}), got {probs.shape}."
                    )
            return probs

        # Fallback to decision_function via softmax or sigmoid
        if hasattr(self._estimator, "decision_function"):
            decision = np.asarray(self._estimator.decision_function(x), dtype=float)
            if decision.ndim == 1:
                # Binary classification
                p1 = 1.0 / (1.0 + np.exp(-decision))
                return np.column_stack([1.0 - p1, p1])
            # Multi-class: apply softmax
            exp = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)

        raise UnsupportedCapabilityError("predict_proba is unavailable.")

    def predict_logits(self, inputs: Any) -> np.ndarray:
        """Return decision function or raw outputs if available."""
        if hasattr(self._estimator, "decision_function"):
            x = self._convert_inputs(inputs)
            logits = np.asarray(self._estimator.decision_function(x), dtype=float)
            if logits.ndim == 1 and self._num_classes == 2:
                logits = np.column_stack([-logits, logits])
            return logits
        raise UnsupportedCapabilityError(
            f"{type(self._estimator).__name__} does not provide raw logits or decision function."
        )
