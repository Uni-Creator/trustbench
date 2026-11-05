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

    pred_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    proba_chunks: list[np.ndarray] = []

    for start in range(0, n_samples, batch_size):
        indices = list(range(start, min(start + batch_size, n_samples)))
        batch = dataset.get_batch(indices)
        expected_n = len(indices)

        if use_logits:
            try:
                logits = np.asarray(model.predict_logits(batch))
            except UnsupportedCapabilityError:
                use_logits = False
            else:
                logits = _validate_matrix(logits, expected_n, model.num_classes, "logits")
                logit_chunks.append(logits)
                pred_chunks.append(np.argmax(logits, axis=1))
                proba_chunks.append(softmax(logits, axis=1))
                continue

        if use_proba:
            try:
                probabilities = np.asarray(model.predict_proba(batch))
            except UnsupportedCapabilityError:
                use_proba = False
            else:
                probabilities = _validate_matrix(
                    probabilities, expected_n, model.num_classes, "probabilities"
                )
                proba_chunks.append(probabilities)
                pred_chunks.append(np.argmax(probabilities, axis=1))
                continue

        predictions = np.asarray(model.predict(batch))
        if predictions.shape != (expected_n,):
            raise ShapeError(
                f"predict() must return shape ({expected_n},); got {predictions.shape}."
            )
        pred_chunks.append(predictions)

    predictions = (
        np.concatenate(pred_chunks, axis=0) if pred_chunks else np.empty((0,), dtype=int)
    )
    logits = np.concatenate(logit_chunks, axis=0) if logit_chunks else None
    probabilities = np.concatenate(proba_chunks, axis=0) if proba_chunks else None

    return InferenceOutput(
        predictions=predictions,
        labels=labels,
        sample_ids=sample_ids,
        logits=logits,
        probabilities=probabilities,
        capabilities={
            "predict": True,
            "predict_proba": probabilities is not None,
            "predict_logits": logits is not None,
        },
    )
