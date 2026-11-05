"""Compatibility shim for uncertainty metrics."""

from trustbench.evaluation.uncertainty import (
    apply_temperature_scaling,
    compute_energy_scores,
    compute_ensemble_uncertainties,
    compute_predictive_entropy,
    compute_uncertainty_metrics,
    evaluate_deep_ensemble,
    evaluate_mc_dropout,
    fit_temperature,
)

__all__ = [
    "apply_temperature_scaling",
    "compute_energy_scores",
    "compute_ensemble_uncertainties",
    "compute_predictive_entropy",
    "compute_uncertainty_metrics",
    "evaluate_deep_ensemble",
    "evaluate_mc_dropout",
    "fit_temperature",
]
