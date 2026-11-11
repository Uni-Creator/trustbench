"""Metric registration and execution interfaces for TrustBench extensibility."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class MetricRegistrationError(ValueError):
    """Raised when metric registration fails (e.g. duplicate name or invalid contract)."""


@dataclass(frozen=True)
class MetricMetadata:
    """Metadata describing a registered evaluation metric."""

    name: str
    description: str = ""
    requires_probabilities: bool = False
    requires_logits: bool = False
    output_type: str = "scalar"  # "scalar" or "dict"
    tags: list[str] = field(default_factory=list)


@dataclass
class RegisteredMetric:
    """Wrapper holding a registered metric function and its metadata."""

    fn: Callable[..., Any]
    metadata: MetricMetadata

    def __call__(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        *,
        probabilities: np.ndarray | None = None,
        logits: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
        class_names: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the metric function, inspecting its signature to supply expected arguments."""
        if self.metadata.requires_probabilities and probabilities is None:
            return {
                "skipped": True,
                "reason": f"Metric {self.metadata.name!r} requires probabilities, but none were provided.",
            }
        if self.metadata.requires_logits and logits is None:
            return {
                "skipped": True,
                "reason": f"Metric {self.metadata.name!r} requires logits, but none were provided.",
            }

        sig = inspect.signature(self.fn)
        params = sig.parameters

        # Check if function accepts variable kwargs (**kwargs)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        available_kwargs: dict[str, Any] = {
            "probabilities": probabilities,
            "logits": logits,
            "sample_ids": sample_ids,
            "class_names": class_names,
            **kwargs,
        }

        # Build positional arguments (first two positional are predictions and labels/targets)
        pos_params = [
            p for p in params.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if len(pos_params) >= 2:
            first_two = [predictions, labels]
            call_kwargs = {}
            for name, val in available_kwargs.items():
                if has_varkw or name in params:
                    call_kwargs[name] = val
            return self.fn(*first_two, **call_kwargs)

        if len(pos_params) == 1:
            # Single positional param: pass predictions
            call_kwargs = {"labels": labels, "targets": labels}
            for name, val in available_kwargs.items():
                if has_varkw or name in params:
                    call_kwargs[name] = val
            return self.fn(predictions, **call_kwargs)

        # Keyword-only
        call_kwargs = {"predictions": predictions, "labels": labels, "targets": labels}
        for name, val in available_kwargs.items():
            if has_varkw or name in params:
                call_kwargs[name] = val
        return self.fn(**call_kwargs)


class MetricRegistry:
    """Thread-safe registry for custom user and plugin metrics."""

    def __init__(self) -> None:
        self._metrics: dict[str, RegisteredMetric] = {}

    def register(
        self,
        name: str | None = None,
        fn: Callable[..., Any] | None = None,
        *,
        description: str = "",
        requires_probabilities: bool = False,
        requires_logits: bool = False,
        output_type: str = "scalar",
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Any:
        """Register a custom evaluation metric function.

        Can be used as a function or as a decorator:

            @register_metric("brier")
            def my_brier(predictions, targets): ...

            @register_metric
            def accuracy_score(predictions, targets): ...

            register_metric("custom", fn=my_fn)
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name is not None:
                if not isinstance(name, str) or not name.strip():
                    raise MetricRegistrationError("Metric name must be a non-empty string.")
                metric_name = name.strip()
            else:
                raw_name = getattr(func, "__name__", None)
                if not raw_name or not isinstance(raw_name, str) or not raw_name.strip():
                    raise MetricRegistrationError("Metric name must be a non-empty string.")
                metric_name = raw_name.strip()

            if not callable(func):
                raise MetricRegistrationError(f"Target metric {func!r} is not callable.")

            if metric_name in self._metrics and not overwrite:
                raise MetricRegistrationError(
                    f"Metric {metric_name!r} is already registered. Set overwrite=True to replace it."
                )

            metadata = MetricMetadata(
                name=metric_name,
                description=description or getattr(func, "__doc__", "") or "",
                requires_probabilities=requires_probabilities,
                requires_logits=requires_logits,
                output_type=output_type,
                tags=tags or [],
            )

            self._metrics[metric_name] = RegisteredMetric(fn=func, metadata=metadata)
            return func

        if fn is not None:
            return decorator(fn)
        return decorator

    def get(self, name: str) -> RegisteredMetric | None:
        """Get a registered metric by name, or None if not registered."""
        return self._metrics.get(name)

    def is_registered(self, name: str) -> bool:
        """Check if a metric name is registered."""
        return name in self._metrics

    def unregister(self, name: str) -> bool:
        """Remove a registered metric by name. Returns True if removed."""
        if name in self._metrics:
            del self._metrics[name]
            return True
        return False

    def list_metrics(self) -> dict[str, MetricMetadata]:
        """Return a mapping of all registered metric names to their metadata."""
        return {k: v.metadata for k, v in self._metrics.items()}

    def clear(self) -> None:
        """Clear all registered custom metrics."""
        self._metrics.clear()


# Global default metric registry
METRIC_REGISTRY = MetricRegistry()


def register_metric(
    name: str | None = None,
    fn: Callable[..., Any] | None = None,
    *,
    description: str = "",
    requires_probabilities: bool = False,
    requires_logits: bool = False,
    output_type: str = "scalar",
    tags: list[str] | None = None,
    overwrite: bool = False,
) -> Any:
    """Register a custom metric function in the global TrustBench metric registry.

    Args:
        name: Name identifier for the metric. Defaults to function __name__.
        fn: Optional callable if called imperatively.
        description: Human-readable description of what the metric computes.
        requires_probabilities: Whether metric requires probabilities from model.
        requires_logits: Whether metric requires raw output logits.
        output_type: 'scalar' (number) or 'dict' (structured dictionary).
        tags: Optional categories or metadata tags.
        overwrite: If True, allows replacing an already registered metric.

    Returns:
        Decorated function or registration decorator.
    """
    return METRIC_REGISTRY.register(
        name=name,
        fn=fn,
        description=description,
        requires_probabilities=requires_probabilities,
        requires_logits=requires_logits,
        output_type=output_type,
        tags=tags,
        overwrite=overwrite,
    )


def get_registered_metric(name: str) -> RegisteredMetric | None:
    """Retrieve a registered metric by name."""
    return METRIC_REGISTRY.get(name)


def list_registered_metrics() -> dict[str, MetricMetadata]:
    """List all registered custom metrics."""
    return METRIC_REGISTRY.list_metrics()


def unregister_metric(name: str) -> bool:
    """Unregister a metric by name."""
    return METRIC_REGISTRY.unregister(name)

