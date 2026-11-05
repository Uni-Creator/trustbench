"""Calibration metrics including Expected Calibration Error (ECE) and Brier Score."""

from typing import Any

import numpy as np


def compute_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute calibration metrics: Brier score, ECE, and bin statistics.

    Args:
        labels: 1D array of ground truth labels of shape (N,).
        probabilities: 2D array of predicted class probabilities of shape (N, C).
        n_bins: Number of equal-width confidence bins (default: 15).

    Returns:
        Dictionary containing:
            - brier_score: Multi-class Brier score (mean squared error of probability vectors)
            - ece: Expected Calibration Error (weighted absolute difference between accuracy and confidence)
            - bin_confidences: Mean confidence per bin
            - bin_accuracies: Mean accuracy per bin
            - bin_counts: Number of samples per bin
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (N, C).")
    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Length mismatch between labels and probabilities.")

    n_samples, n_classes = probabilities.shape
    if n_samples == 0:
        return {
            "brier_score": 0.0,
            "ece": 0.0,
            "bin_confidences": [],
            "bin_accuracies": [],
            "bin_counts": [],
        }

    # 1. Multi-class Brier Score: (1/N) * sum_i sum_c (p_ic - y_ic)^2
    one_hot = np.zeros_like(probabilities)
    valid_idx = (labels >= 0) & (labels < n_classes)
    one_hot[np.arange(n_samples)[valid_idx], labels[valid_idx]] = 1.0
    brier_score = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    # 2. Expected Calibration Error (ECE)
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            ece += (count / n_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return {
        "brier_score": brier_score,
        "ece": float(ece),
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }
