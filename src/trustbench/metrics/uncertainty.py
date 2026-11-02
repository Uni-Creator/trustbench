"""Uncertainty quantification metrics including predictive entropy."""

from typing import Any

import numpy as np


def compute_predictive_entropy(
    probabilities: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute per-sample Shannon entropy: H(p) = -sum(p_c * log(p_c)).

    Args:
        probabilities: 2D array of class probabilities of shape (N, C).
        epsilon: Small epsilon to prevent log(0).

    Returns:
        1D array of entropy values of shape (N,).
    """
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (N, C).")

    clipped_probs = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)
    return entropy


def compute_uncertainty_metrics(
    probabilities: np.ndarray,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute summary metrics for prediction uncertainty.

    Args:
        probabilities: 2D array of class probabilities of shape (N, C).
        epsilon: Small epsilon to avoid log(0).

    Returns:
        Dictionary containing:
            - mean_entropy: Mean predictive entropy across all samples
            - std_entropy: Standard deviation of predictive entropy
            - min_entropy: Minimum predictive entropy
            - max_entropy: Maximum predictive entropy
            - median_entropy: Median predictive entropy
            - mean_max_probability: Mean confidence (max probability)
    """
    probabilities = np.asarray(probabilities)
    if len(probabilities) == 0:
        return {
            "mean_entropy": 0.0,
            "std_entropy": 0.0,
            "min_entropy": 0.0,
            "max_entropy": 0.0,
            "median_entropy": 0.0,
            "mean_max_probability": 0.0,
        }

    entropies = compute_predictive_entropy(probabilities, epsilon=epsilon)
    max_probs = np.max(probabilities, axis=1)

    return {
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "min_entropy": float(np.min(entropies)),
        "max_entropy": float(np.max(entropies)),
        "median_entropy": float(np.median(entropies)),
        "mean_max_probability": float(np.mean(max_probs)),
    }
