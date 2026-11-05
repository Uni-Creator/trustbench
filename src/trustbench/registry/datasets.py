"""Dataset adapter registry, generic wrapper, and contract validation for custom datasets."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from trustbench.core.protocols import DatasetAdapter as DatasetAdapterProtocol


def validate_dataset_adapter(adapter: Any) -> list[str]:
    """Validate whether an object conforms to the DatasetAdapter contract.

    Checks:
      - Has `__len__()` method returning an integer >= 0.
      - Has callable `get_batch(indices)`.
      - Has callable `get_labels()`.
      - Has callable `get_sample_ids()`.

    Args:
        adapter: Any dataset adapter instance or class.

    Returns:
        List of validation error strings. Empty if valid.
    """
    errors: list[str] = []

    # 1. __len__
    len_fn = getattr(adapter, "__len__", None)
    if len_fn is None or not callable(len_fn):
        errors.append("DatasetAdapter must implement a callable '__len__()' method.")
    else:
        try:
            n = len(adapter)
            if not isinstance(n, int) or n < 0:
                errors.append(f"'__len__()' must return a non-negative integer, got {n!r}.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Error calling '__len__()': {exc}")

    # 2. get_batch
    if not callable(getattr(adapter, "get_batch", None)):
        errors.append("DatasetAdapter must implement a callable 'get_batch(indices)' method.")

    # 3. get_labels
    if not callable(getattr(adapter, "get_labels", None)):
        errors.append("DatasetAdapter must implement a callable 'get_labels()' method.")

    # 4. get_sample_ids
    if not callable(getattr(adapter, "get_sample_ids", None)):
        errors.append("DatasetAdapter must implement a callable 'get_sample_ids()' method.")

    return errors


class GenericDatasetAdapter(DatasetAdapterProtocol):
    """Universal dataset adapter for arbitrary Python data sequences, mappings, or generators.

    Supports:
      - Samples / feature batches
      - Labels (ground truth targets)
      - Custom sample identifiers
      - Class names / label mappings
      - Metadata attributes

    Example::

        from trustbench.registry.datasets import GenericDatasetAdapter

        adapter = GenericDatasetAdapter(
            samples=[{"text": "great"}, {"text": "bad"}],
            labels=[1, 0],
            sample_ids=["doc_1", "doc_2"],
            class_names=["negative", "positive"],
            metadata={"domain": "customer_reviews"},
        )
    """

    def __init__(
        self,
        samples: Sequence[Any],
        labels: Sequence[int] | np.ndarray,
        sample_ids: Sequence[Any] | np.ndarray | None = None,
        class_names: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        batch_collate_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        self.samples = samples
        self._labels = np.asarray(labels)
        n = len(samples)

        if len(self._labels) != n:
            raise ValueError(
                f"Length mismatch: samples ({n}) vs labels ({len(self._labels)})."
            )

        if sample_ids is None:
            self._sample_ids = np.arange(n)
        else:
            self._sample_ids = np.asarray(sample_ids)
            if len(self._sample_ids) != n:
                raise ValueError(
                    f"Length mismatch: samples ({n}) vs sample_ids ({len(self._sample_ids)})."
                )

        self.class_names = list(class_names) if class_names is not None else None
        self.metadata: dict[str, Any] = metadata or {}
        self._batch_collate_fn = batch_collate_fn

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, indices: list[int]) -> Any:
        batch_items = [self.samples[i] for i in indices]
        if self._batch_collate_fn is not None:
            return self._batch_collate_fn(batch_items)
        if isinstance(self.samples, np.ndarray):
            return self.samples[indices]
        return batch_items

    def get_labels(self) -> np.ndarray:
        return self._labels

    def get_sample_ids(self) -> np.ndarray:
        return self._sample_ids


class DatasetAdapterRegistry:
    """Registry for custom dataset adapter types and factories."""

    def __init__(self) -> None:
        self._registry: dict[str, type | Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        adapter_cls_or_factory: type | Callable[..., Any],
        overwrite: bool = False,
    ) -> None:
        """Register a dataset adapter class or factory function."""
        if not name or not isinstance(name, str):
            raise ValueError("Dataset adapter name must be a non-empty string.")
        if name in self._registry and not overwrite:
            raise ValueError(f"Dataset adapter {name!r} already registered. Set overwrite=True to replace.")
        self._registry[name] = adapter_cls_or_factory

    def get(self, name: str) -> type | Callable[..., Any] | None:
        """Retrieve a registered dataset adapter class or factory."""
        return self._registry.get(name)

    def list_adapters(self) -> list[str]:
        """List registered dataset adapter names."""
        return list(self._registry.keys())

    def unregister(self, name: str) -> bool:
        """Unregister a dataset adapter."""
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered dataset adapters."""
        self._registry.clear()


DATASET_ADAPTER_REGISTRY = DatasetAdapterRegistry()


def register_dataset_adapter(
    name: str,
    adapter_cls_or_factory: type | Callable[..., Any] | None = None,
    overwrite: bool = False,
) -> Any:
    """Register a custom dataset adapter class or factory function.

    Can be used as a function or decorator:

        @register_dataset_adapter("huggingface_imdb")
        class IMDBAdapter(DatasetAdapter): ...
    """
    def decorator(cls_or_func: Any) -> Any:
        DATASET_ADAPTER_REGISTRY.register(name, cls_or_func, overwrite=overwrite)
        return cls_or_func

    if adapter_cls_or_factory is not None:
        return decorator(adapter_cls_or_factory)
    return decorator


def get_dataset_adapter(name: str) -> type | Callable[..., Any] | None:
    """Retrieve a registered dataset adapter."""
    return DATASET_ADAPTER_REGISTRY.get(name)
