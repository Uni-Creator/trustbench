"""Regression task evaluator and structured result definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.core.reproducibility import set_seed
from trustbench.metrics.regression import compute_regression_metrics


@dataclass
class RegressionResult:
    """Result of a regression model evaluation."""

    metrics: dict[str, Any]
    predictions: np.ndarray
    targets: np.ndarray
    sample_ids: np.ndarray
    residuals: np.ndarray
    uncertainties: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        """Convenience property for primary regression metrics."""
        return {
            "mae": self.metrics.get("mae"),
            "mse": self.metrics.get("mse"),
            "rmse": self.metrics.get("rmse"),
            "r2": self.metrics.get("r2"),
            "medae": self.metrics.get("medae"),
            "mape": self.metrics.get("mape"),
        }


def evaluate_regression(
    model: ModelAdapter | Any,
    dataset: DatasetAdapter | Any,
    *,
    batch_size: int = 100,
    seed: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> RegressionResult:
    """Evaluate a model on a regression dataset.

    Args:
        model: Model adapter or estimator with `predict()` returning continuous values.
        dataset: Dataset adapter with `get_batch`, `get_labels` (as continuous targets),
            and `get_sample_ids`.
        batch_size: Batch size for chunked evaluation.
        seed: Random seed for reproducibility.
        extra_metadata: Optional dictionary of additional evaluation context.

    Returns:
        Structured `RegressionResult` with metrics, predictions, targets, and residuals.
    """
    if seed is not None:
        set_seed(seed)

    n_samples = len(dataset)
    if n_samples == 0:
        empty = np.array([], dtype=float)
        return RegressionResult(
            metrics=compute_regression_metrics(empty, empty),
            predictions=empty,
            targets=empty,
            sample_ids=np.array([], dtype=object),
            residuals=empty,
            metadata=extra_metadata or {},
        )

    # Collect continuous predictions batch by batch
    preds_list: list[np.ndarray] = []
    uncertainties_list: list[np.ndarray] = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = list(range(start_idx, end_idx))
        batch_inputs = dataset.get_batch(batch_indices)

        batch_preds = model.predict(batch_inputs)
        preds_list.append(np.asarray(batch_preds, dtype=float).squeeze())

        # Check if model optionally outputs standard deviation / uncertainty
        if hasattr(model, "predict_uncertainty"):
            std_batch = model.predict_uncertainty(batch_inputs)
            uncertainties_list.append(np.asarray(std_batch, dtype=float).squeeze())

    predictions = np.concatenate(preds_list) if len(preds_list) > 1 else preds_list[0]
    targets = np.asarray(dataset.get_labels(), dtype=float).squeeze()
    sample_ids = np.asarray(dataset.get_sample_ids())

    uncertainties = (
        np.concatenate(uncertainties_list)
        if uncertainties_list
        else None
    )

    metrics = compute_regression_metrics(
        y_true=targets,
        y_pred=predictions,
        y_std=uncertainties,
    )

    residuals = targets - predictions

    return RegressionResult(
        metrics=metrics,
        predictions=predictions,
        targets=targets,
        sample_ids=sample_ids,
        residuals=residuals,
        uncertainties=uncertainties,
        metadata=extra_metadata or {},
    )
