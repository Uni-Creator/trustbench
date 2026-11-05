"""Model adapter registry, generic wrapper, and contract validation for custom models."""

from collections.abc import Callable
from typing import Any

import numpy as np

from trustbench.core.exceptions import ShapeError, UnsupportedCapabilityError
from trustbench.core.protocols import ModelAdapter as ModelAdapterProtocol


def validate_model_adapter(adapter: Any) -> list[str]:
    """Validate whether an object conforms to the ModelAdapter contract.

    Checks:
      - Has attribute/property `num_classes` (int >= 1).
      - Has attribute/property `supports_predict_proba` (bool).
      - Has attribute/property `supports_predict_logits` (bool).
      - Has callable `predict(inputs)`.
      - Has callable `predict_proba(inputs)`.
      - Has callable `predict_logits(inputs)`.

    Args:
        adapter: Any model adapter instance or class.

    Returns:
        List of validation error strings. Empty if valid.
    """
    errors: list[str] = []

    # 1. num_classes
    if not hasattr(adapter, "num_classes"):
        errors.append("Adapter is missing required attribute or property 'num_classes'.")
    else:
        try:
            nc = adapter.num_classes
            if not isinstance(nc, int) or nc < 1:
                errors.append(f"'num_classes' must be an integer >= 1, got {nc!r}.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Error accessing 'num_classes': {exc}")

    # 2. supports_predict_proba
    s_proba = getattr(adapter, "supports_predict_proba", None)
    if s_proba is None:
        errors.append("Adapter is missing required property 'supports_predict_proba'.")
    elif not isinstance(s_proba, bool):
        errors.append("'supports_predict_proba' must return a boolean.")

    # 3. supports_predict_logits
    s_logits = getattr(adapter, "supports_predict_logits", None)
    if s_logits is None:
        errors.append("Adapter is missing required property 'supports_predict_logits'.")
    elif not isinstance(s_logits, bool):
        errors.append("'supports_predict_logits' must return a boolean.")

    # 4. predict
    if not callable(getattr(adapter, "predict", None)):
        errors.append("Adapter must implement a callable 'predict(inputs)' method.")

    # 5. predict_proba
    if not callable(getattr(adapter, "predict_proba", None)):
        errors.append("Adapter must implement a callable 'predict_proba(inputs)' method.")

    # 6. predict_logits
    if not callable(getattr(adapter, "predict_logits", None)):
        errors.append("Adapter must implement a callable 'predict_logits(inputs)' method.")

    return errors


class GenericModelAdapter(ModelAdapterProtocol):
    """Universal model adapter wrapping any custom prediction callable or ML framework.

    Allows wrapping Scikit-Learn, ONNX, JAX, Hugging Face, or custom Python functions
    without requiring PyTorch or any specific framework.

    Example::

        from trustbench.registry.models import GenericModelAdapter

        adapter = GenericModelAdapter(
            predict_fn=clf.predict,
            num_classes=3,
            predict_proba_fn=clf.predict_proba,
            metadata={"framework": "scikit-learn", "version": "1.0"},
        )
    """

    def __init__(
        self,
        predict_fn: Callable[[Any], np.ndarray],
        num_classes: int,
        predict_proba_fn: Callable[[Any], np.ndarray] | None = None,
        predict_logits_fn: Callable[[Any], np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not callable(predict_fn):
            raise TypeError(f"predict_fn must be callable, got {type(predict_fn).__name__}")

        self._predict_fn = predict_fn
        self._num_classes = int(num_classes)
        self._predict_proba_fn = predict_proba_fn
        self._predict_logits_fn = predict_logits_fn
        self.metadata: dict[str, Any] = metadata or {}

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def supports_predict_proba(self) -> bool:
        return self._predict_proba_fn is not None

    @property
    def supports_predict_logits(self) -> bool:
        return self._predict_logits_fn is not None

    def predict(self, inputs: Any) -> np.ndarray:
        preds = np.asarray(self._predict_fn(inputs))
        if preds.ndim != 1:
            raise ShapeError(f"predict() must return 1D array of shape (N,), got {preds.shape}")
        return preds

    def predict_proba(self, inputs: Any) -> np.ndarray:
        if self._predict_proba_fn is None:
            raise UnsupportedCapabilityError("predict_proba is not configured for this adapter.")
        probs = np.asarray(self._predict_proba_fn(inputs), dtype=float)
        if probs.ndim != 2 or probs.shape[1] != self._num_classes:
            raise ShapeError(
                f"predict_proba() must return 2D array of shape (N, {self._num_classes}), got {probs.shape}"
            )
        return probs

    def predict_logits(self, inputs: Any) -> np.ndarray:
        if self._predict_logits_fn is None:
            raise UnsupportedCapabilityError("predict_logits is not configured for this adapter.")
        logits = np.asarray(self._predict_logits_fn(inputs), dtype=float)
        if logits.ndim != 2 or logits.shape[1] != self._num_classes:
            raise ShapeError(
                f"predict_logits() must return 2D array of shape (N, {self._num_classes}), got {logits.shape}"
            )
        return logits


class ModelAdapterRegistry:
    """Registry for custom model adapter types and factories."""

    def __init__(self) -> None:
        self._registry: dict[str, type | Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        adapter_cls_or_factory: type | Callable[..., Any],
        overwrite: bool = False,
    ) -> None:
        """Register a model adapter class or factory function."""
        if not name or not isinstance(name, str):
            raise ValueError("Adapter name must be a non-empty string.")
        if name in self._registry and not overwrite:
            raise ValueError(f"Model adapter {name!r} already registered. Set overwrite=True to replace.")
        self._registry[name] = adapter_cls_or_factory

    def get(self, name: str) -> type | Callable[..., Any] | None:
        """Retrieve a registered model adapter class or factory."""
        return self._registry.get(name)

    def list_adapters(self) -> list[str]:
        """List registered model adapter names."""
        return list(self._registry.keys())

    def unregister(self, name: str) -> bool:
        """Unregister a model adapter."""
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered model adapters."""
        self._registry.clear()


MODEL_ADAPTER_REGISTRY = ModelAdapterRegistry()


def register_model_adapter(
    name: str,
    adapter_cls_or_factory: type | Callable[..., Any] | None = None,
    overwrite: bool = False,
) -> Any:
    """Register a custom model adapter class or factory function.

    Can be used as a function or decorator:

        @register_model_adapter("sklearn_random_forest")
        class MyRFAdapter(ModelAdapter): ...
    """
    def decorator(cls_or_func: Any) -> Any:
        MODEL_ADAPTER_REGISTRY.register(name, cls_or_func, overwrite=overwrite)
        return cls_or_func

    if adapter_cls_or_factory is not None:
        return decorator(adapter_cls_or_factory)
    return decorator


def get_model_adapter(name: str) -> type | Callable[..., Any] | None:
    """Retrieve a registered model adapter."""
    return MODEL_ADAPTER_REGISTRY.get(name)
