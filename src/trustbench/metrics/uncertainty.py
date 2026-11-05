"""Compatibility shim for uncertainty metrics."""

from trustbench.evaluation.uncertainty import (
    apply_temperature_scaling,
    compute_energy_scores,
    compute_predictive_entropy,
    compute_uncertainty_metrics,
    fit_temperature,
)

__all__ = [
    "apply_temperature_scaling",
    "compute_energy_scores",
    "compute_predictive_entropy",
    "compute_uncertainty_metrics",
    "fit_temperature",
]
