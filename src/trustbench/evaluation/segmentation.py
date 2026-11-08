"""Semantic segmentation task evaluator and structured result definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.metrics.segmentation import compute_segmentation_metrics


@dataclass
class SegmentationResult:
    """Result of a semantic segmentation evaluation."""

    metrics: dict[str, Any]
    predictions: np.ndarray
    ground_truth: np.ndarray
    sample_ids: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        """Convenience accessor for primary segmentation metrics."""
        return {
            "miou": self.metrics.get("miou"),
            "dice": self.metrics.get("dice"),
            "pixel_accuracy": self.metrics.get("pixel_accuracy"),
            "fwiou": self.metrics.get("fwiou"),
        }


def evaluate_segmentation(
    model: ModelAdapter | Any,
    dataset: DatasetAdapter | Any,
    *,
    batch_size: int = 16,
    num_classes: int | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
    ignore_index: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> SegmentationResult:
    """Evaluate a segmentation model on a dataset of images and masks.

    Args:
        model: Model adapter returning pixel class predictions (N, H, W) or probabilities (N, C, H, W).
        dataset: Dataset adapter returning image inputs and ground-truth masks.
        batch_size: Evaluation batch size.
        num_classes: Total number of semantic classes.
        class_names: Optional human-readable class names.
        ignore_index: Optional label index to ignore (e.g. 255).
        extra_metadata: Optional dictionary of evaluation context.

    Returns:
        Structured `SegmentationResult`.
    """
    n_samples = len(dataset)
    if n_samples == 0:
        empty = np.zeros((0, 0, 0), dtype=int)
        return SegmentationResult(
            metrics=compute_segmentation_metrics(empty, empty),
            predictions=empty,
            ground_truth=empty,
            sample_ids=np.array([], dtype=object),
            metadata=extra_metadata or {},
        )

    preds_list: list[np.ndarray] = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        batch_inputs = dataset.get_batch(batch_indices)

        batch_out = model.predict(batch_inputs)
        batch_arr = np.asarray(batch_out)

        # If output is (B, C, H, W), take argmax across channel dim
        if batch_arr.ndim == 4:
            batch_arr = np.argmax(batch_arr, axis=1)

        preds_list.append(batch_arr)

    predictions = np.concatenate(preds_list, axis=0) if len(preds_list) > 1 else preds_list[0]
    ground_truth = np.asarray(dataset.get_labels())
    sample_ids = np.asarray(dataset.get_sample_ids())

    metrics = compute_segmentation_metrics(
        y_true=ground_truth,
        y_pred=predictions,
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=ignore_index,
    )

    return SegmentationResult(
        metrics=metrics,
        predictions=predictions,
        ground_truth=ground_truth,
        sample_ids=sample_ids,
        metadata=extra_metadata or {},
    )
