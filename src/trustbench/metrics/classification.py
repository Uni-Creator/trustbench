"""Compatibility shim for classification metrics."""

from trustbench.evaluation.classification import (
    compute_balanced_accuracy,
    compute_classification_metrics,
)

__all__ = [
    "compute_balanced_accuracy",
    "compute_classification_metrics",
]
