"""Compatibility shim for error detection metrics."""

from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
    compute_detection_error,
    compute_error_detection_metrics,
    compute_fpr_at_tpr,
    compute_slice_error_rates,
)

__all__ = [
    "compute_binary_detection_curve",
    "compute_detection_error",
    "compute_error_detection_metrics",
    "compute_fpr_at_tpr",
    "compute_slice_error_rates",
]
