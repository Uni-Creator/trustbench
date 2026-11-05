"""Selective prediction, risk-coverage curve, and Excess AURC metrics."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_optimal_aurc(n_samples: int, n_errors: int) -> float:
    """Compute theoretical minimum AURC achieved under perfect uncertainty ranking.

    Under an optimal ranking, all correct samples are assigned higher confidence
    than any error.

    Mathematical formula:
      If E = 0: AURC* = 0
      If E = N: AURC* = 1
      Otherwise: (1 / N) * sum_{k = N - E + 1}^N (1 - (N - E) / k)

    Args:
        n_samples: Total number of samples N.
        n_errors: Total number of incorrect predictions E.

    Returns:
        Optimal AURC in [0.0, 1.0].
    """
    if n_samples == 0 or n_errors == 0:
        return 0.0
    if n_errors >= n_samples:
        return 1.0

    n_correct = n_samples - n_errors
    k_err = np.arange(n_correct + 1, n_samples + 1)
    opt_risks = (k_err - n_correct) / k_err
    return float(np.sum(opt_risks) / n_samples)


def get_risk_at_coverage(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    coverage: float,
) -> float:
    """Compute selective classification error rate (risk) at an exact coverage fraction.

    Args:
        labels: 1D ground truth labels.
        predictions: 1D model predictions.
        confidences: 1D confidence scores.
        coverage: Target coverage fraction in (0.0, 1.0].

    Returns:
        Selective classification error rate on the retained fraction of samples.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)

    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    errors = (predictions != labels).astype(int)
    sorted_indices = np.argsort(-confidences)
    sorted_errors = errors[sorted_indices]

    k = max(1, min(n_samples, int(np.round(n_samples * coverage))))
    return float(np.mean(sorted_errors[:k]))


def get_coverage_at_risk(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    target_risk: float,
) -> float:
    """Find the maximum coverage fraction in [0.0, 1.0] that maintains risk <= target_risk.

    Args:
        labels: 1D ground truth labels.
        predictions: 1D model predictions.
        confidences: 1D confidence scores.
        target_risk: Maximum acceptable error rate threshold (e.g. 0.01 for 1% error).

    Returns:
        Maximum coverage in [0.0, 1.0].
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)

    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    errors = (predictions != labels).astype(int)
    sorted_indices = np.argsort(-confidences)
    sorted_errors = errors[sorted_indices]

    cumulative_errors = np.cumsum(sorted_errors)
    sample_indices = np.arange(1, n_samples + 1)
    risks = cumulative_errors / sample_indices

    valid_k = np.where(risks <= target_risk)[0]
    if len(valid_k) == 0:
        return 0.0
    best_k = int(valid_k[-1]) + 1
    return float(best_k / n_samples)


def compute_risk_coverage(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> dict[str, Any]:
    """Compute selective classification error rates across coverage levels, AURC, and Excess AURC.

    Mathematical concepts:
      - Risk at coverage c: (1 / |S_c|) * sum_{i in S_c} I(y_i != ŷ_i)
      - AURC: Area Under the Risk-Coverage curve from c = 1/N to 1.0
      - Optimal AURC: Theoretical lower bound achieved by perfect uncertainty ranking
      - Excess AURC (E-AURC): AURC - Optimal AURC (lower is better, 0.0 is perfect)

    Args:
        labels: 1D ground truth labels (N,).
        predictions: 1D predicted labels (N,).
        confidences: 1D confidence scores where higher means more confident.
        coverages: Sequence of coverage fractions in (0, 1] (e.g. 1.0, 0.9, 0.8, 0.7, 0.5).

    Returns:
        Dictionary containing:
            - curve: List of dicts with 'coverage', 'threshold', 'samples_retained', 'error_rate', 'accuracy'
            - aurc: Area Under the Risk-Coverage curve
            - optimal_aurc: Theoretical minimum AURC
            - excess_aurc: E-AURC = AURC - Optimal AURC
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)

    n_samples = len(labels)
    if n_samples == 0:
        return {
            "curve": [],
            "aurc": 0.0,
            "optimal_aurc": 0.0,
            "excess_aurc": 0.0,
        }

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

    cumulative_errors = np.cumsum(sorted_errors)
    sample_indices = np.arange(1, n_samples + 1)
    risks = cumulative_errors / sample_indices
    aurc = float(np.mean(risks))

    n_errors = int(np.sum(errors))
    opt_aurc = compute_optimal_aurc(n_samples, n_errors)
    excess_aurc = max(0.0, float(aurc - opt_aurc))

    return {
        "curve": curve,
        "aurc": aurc,
        "optimal_aurc": opt_aurc,
        "excess_aurc": excess_aurc,
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
