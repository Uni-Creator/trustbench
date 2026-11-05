"""Compatibility shim for error detection metrics."""

from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
    compute_error_detection_metrics,
    compute_slice_error_rates,
)

__all__ = [
    "compute_binary_detection_curve",
    "compute_error_detection_metrics",
    "compute_slice_error_rates",
]
