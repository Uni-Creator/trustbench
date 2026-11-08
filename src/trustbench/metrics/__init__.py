"""Compatibility package re-exporting evaluation metrics."""

from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.distribution_shift import compute_ood_detection_metrics
from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
    compute_error_detection_metrics,
    compute_slice_error_rates,
)
from trustbench.evaluation.fairness import compute_fairness_metrics
from trustbench.evaluation.selective import (
    compute_risk_coverage,
    compute_selective_prediction_metrics,
)
from trustbench.evaluation.uncertainty import (
    apply_temperature_scaling,
    compute_energy_scores,
    compute_predictive_entropy,
    compute_uncertainty_metrics,
    fit_temperature,
)
from trustbench.metrics.regression import compute_regression_metrics
from trustbench.metrics.retrieval import compute_retrieval_metrics
from trustbench.metrics.segmentation import compute_segmentation_metrics

__all__ = [
    "apply_temperature_scaling",
    "compute_binary_detection_curve",
    "compute_calibration_metrics",
    "compute_classification_metrics",
    "compute_energy_scores",
    "compute_error_detection_metrics",
    "compute_fairness_metrics",
    "compute_ood_detection_metrics",
    "compute_predictive_entropy",
    "compute_regression_metrics",
    "compute_retrieval_metrics",
    "compute_risk_coverage",
    "compute_segmentation_metrics",
    "compute_selective_prediction_metrics",
    "compute_slice_error_rates",
    "compute_uncertainty_metrics",
    "fit_temperature",
]
