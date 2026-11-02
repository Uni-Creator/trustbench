"""Selective prediction and risk-coverage curve metrics."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_risk_coverage(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> dict[str, Any]:
    """Compute selective classification error rates at specified coverage levels.

    Args:
        labels: 1D ground truth labels (N,).
        predictions: 1D predicted labels (N,).
        confidences: 1D confidence scores where higher means more confident.
        coverages: Sequence of coverage fractions in (0, 1] (e.g. 1.0, 0.9, 0.8, 0.7, 0.5).

    Returns:
        Dictionary containing:
            - curve: List of dicts with 'coverage', 'threshold', 'samples_retained', 'error_rate', 'accuracy'
            - aurc: Area Under the Risk-Coverage curve (lower is better)
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)

    n_samples = len(labels)
    if n_samples == 0:
        return {"curve": [], "aurc": 0.0}

    errors = (predictions != labels).astype(int)

    # Sort samples by confidence descending (highest confidence first)
    sorted_indices = np.argsort(-confidences)
    sorted_errors = errors[sorted_indices]
    sorted_confidences = confidences[sorted_indices]

    curve = []
    for cov in coverages:
        k = max(1, min(n_samples, int(np.round(n_samples * cov))))
        retained_errors = sorted_errors[:k]
        threshold = float(sorted_confidences[k - 1])
        err_rate = float(np.mean(retained_errors))

        curve.append({
            "coverage": float(cov),
            "retained_fraction": float(k / n_samples),
            "samples_retained": k,
            "threshold": threshold,
            "error_rate": err_rate,
            "accuracy": float(1.0 - err_rate),
        })

    # Continuous risk-coverage curve for Area Under Risk-Coverage (AURC)
    # Compute cumulative error rates from k=1 to n_samples
    cumulative_errors = np.cumsum(sorted_errors)
    sample_indices = np.arange(1, n_samples + 1)
    risks = cumulative_errors / sample_indices
    # Trapezoidal approximation or simple mean of risks
    aurc = float(np.mean(risks))

    return {
        "curve": curve,
        "aurc": aurc,
    }


def compute_selective_prediction_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> dict[str, Any]:
    """Compute selective prediction performance using max softmax probability and negative entropy.

    Args:
        labels: 1D ground truth labels.
        predictions: 1D predicted labels.
        probabilities: 2D probability matrix (N, C).
        coverages: List of coverage fractions to report.

    Returns:
        Dict comparing risk-coverage metrics for MSP confidence vs. negative entropy.
    """
    conf_msp = np.max(probabilities, axis=1)

    eps = 1e-12
    clipped = np.clip(probabilities, eps, 1.0)
    entropies = -np.sum(clipped * np.log(clipped), axis=1)
    conf_entropy = -entropies  # higher is more confident

    msp_res = compute_risk_coverage(labels, predictions, conf_msp, coverages=coverages)
    entropy_res = compute_risk_coverage(labels, predictions, conf_entropy, coverages=coverages)

    return {
        "max_softmax_probability": msp_res,
        "predictive_entropy": entropy_res,
    }
