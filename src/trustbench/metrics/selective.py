"""Compatibility shim for selective prediction metrics."""

from trustbench.evaluation.selective import (
    compute_risk_coverage,
    compute_selective_prediction_metrics,
)

__all__ = [
    "compute_risk_coverage",
    "compute_selective_prediction_metrics",
]
