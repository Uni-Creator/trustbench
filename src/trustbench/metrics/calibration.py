"""Compatibility shim for calibration metrics."""

from trustbench.evaluation.calibration import (
    compute_adaptive_ece,
    compute_calibration_metrics,
    compute_nll,
    compute_reliability_diagram_data,
)

__all__ = [
    "compute_adaptive_ece",
    "compute_calibration_metrics",
    "compute_nll",
    "compute_reliability_diagram_data",
]
