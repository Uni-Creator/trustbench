"""Adapter-based inference used by the core evaluator."""

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

from trustbench.core.exceptions import ShapeError, UnsupportedCapabilityError
from trustbench.core.protocols import DatasetAdapter, ModelAdapter


@dataclass
class InferenceOutput:
    """Raw inference arrays collected from adapters."""

    predictions: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    logits: np.ndarray | None
    probabilities: np.ndarray | None
    capabilities: dict[str, bool]


def _validate_matrix(
    array: np.ndarray,
    batch_size: int,
    num_classes: int,
    name: str,
) -> np.ndarray:
    if array.ndim != 2 or array.shape[0] != batch_size or array.shape[1] != num_classes:
        raise ShapeError(
            f"{name} must have shape ({batch_size}, {num_classes}); got {array.shape}."
        )
    return array


def run_adapter_inference(
    model: ModelAdapter,
    dataset: DatasetAdapter,
    batch_size: int = 100,
) -> InferenceOutput:
    """Run batched inference through model and dataset adapters."""
    n_samples = len(dataset)
    labels = np.asarray(dataset.get_labels())
    sample_ids = np.asarray(dataset.get_sample_ids())

    if labels.shape[0] != n_samples:
        raise ShapeError(
            f"get_labels() length ({labels.shape[0]}) does not match dataset length ({n_samples})."
        )
    if sample_ids.shape[0] != n_samples:
        raise ShapeError(
            f"get_sample_ids() length ({sample_ids.shape[0]}) does not match dataset length ({n_samples})."
        )

    use_logits = bool(getattr(model, "supports_predict_logits", False))
    use_proba = bool(getattr(model, "supports_predict_proba", False))

    if n_samples == 0:
        return InferenceOutput(
            predictions=np.empty((0,), dtype=int),
            labels=labels,
            sample_ids=sample_ids,
            logits=None,
            probabilities=None,
            capabilities={
                "predict": True,
                "predict_proba": use_proba,
                "predict_logits": use_logits,
            },
        )

    # Pre-allocate arrays to eliminate 2x memory spikes from concatenation
    num_classes = getattr(model, "num_classes", 0)
    predictions = np.empty(n_samples, dtype=int)
    logits_array: np.ndarray | None = (
        np.empty((n_samples, num_classes), dtype=np.float32) if (use_logits and num_classes > 0) else None
    )
    proba_array: np.ndarray | None = (
        np.empty((n_samples, num_classes), dtype=np.float32) if ((use_logits or use_proba) and num_classes > 0) else None
    )

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        indices = list(range(start, end))
        batch = dataset.get_batch(indices)
        expected_n = end - start

        if use_logits:
            try:
                logits = np.asarray(model.predict_logits(batch))
            except UnsupportedCapabilityError:
                use_logits = False
                logits_array = None
            else:
                logits = _validate_matrix(logits, expected_n, model.num_classes, "logits")
                if logits_array is not None:
                    logits_array[start:end] = logits
                predictions[start:end] = np.argmax(logits, axis=1)
                if proba_array is not None:
                    proba_array[start:end] = softmax(logits, axis=1)
                continue

        if use_proba:
            try:
                probabilities = np.asarray(model.predict_proba(batch))
            except UnsupportedCapabilityError:
                use_proba = False
                proba_array = None
            else:
                probabilities = _validate_matrix(
                    probabilities, expected_n, model.num_classes, "probabilities"
                )
                if proba_array is not None:
                    proba_array[start:end] = probabilities
                predictions[start:end] = np.argmax(probabilities, axis=1)
                continue

        batch_preds = np.asarray(model.predict(batch))
        if batch_preds.shape != (expected_n,):
            raise ShapeError(
                f"predict() must return shape ({expected_n},); got {batch_preds.shape}."
            )
        predictions[start:end] = batch_preds

    return InferenceOutput(
        predictions=predictions,
        labels=labels,
        sample_ids=sample_ids,
        logits=logits_array,
        probabilities=proba_array,
        capabilities={
            "predict": True,
            "predict_proba": proba_array is not None,
            "predict_logits": logits_array is not None,
        },
    )
