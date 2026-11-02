"""Metrics package for TrustBench."""

from trustbench.metrics.calibration import compute_calibration_metrics
from trustbench.metrics.classification import compute_classification_metrics
from trustbench.metrics.error_detection import (
    compute_binary_detection_curve,
    compute_error_detection_metrics,
    compute_slice_error_rates,
)
from trustbench.metrics.selective import (
    compute_risk_coverage,
    compute_selective_prediction_metrics,
)
from trustbench.metrics.uncertainty import (
    apply_temperature_scaling,
    compute_energy_scores,
    compute_predictive_entropy,
    compute_uncertainty_metrics,
    fit_temperature,
)

__all__ = [
    "apply_temperature_scaling",
    "compute_binary_detection_curve",
    "compute_calibration_metrics",
    "compute_classification_metrics",
    "compute_energy_scores",
    "compute_error_detection_metrics",
    "compute_predictive_entropy",
    "compute_risk_coverage",
    "compute_selective_prediction_metrics",
    "compute_slice_error_rates",
    "compute_uncertainty_metrics",
    "fit_temperature",
]
