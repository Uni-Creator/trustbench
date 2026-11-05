"""Compatibility shim for selective prediction metrics."""

from trustbench.evaluation.selective import (
    compute_optimal_aurc,
    compute_risk_coverage,
    compute_selective_prediction_metrics,
    get_coverage_at_risk,
    get_risk_at_coverage,
)

__all__ = [
    "compute_optimal_aurc",
    "compute_risk_coverage",
    "compute_selective_prediction_metrics",
    "get_coverage_at_risk",
    "get_risk_at_coverage",
]
