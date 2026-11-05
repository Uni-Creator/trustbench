"""Uncertainty quantification methods: MSP, Entropy, Temperature Scaling, and Energy scores."""

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


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


def compute_energy_scores(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute negative free energy score: E(x; T) = -T * logsumexp(logits / T).

    In energy-based OOD/uncertainty detection, lower energy represents higher model certainty,
    meaning uncertainty is directly measured by positive free energy: -E(x) = T * logsumexp(logits / T).

    Args:
        logits: 2D array of raw output logits of shape (N, C).
        temperature: Temperature scaling factor.

    Returns:
        1D array of energy scores of shape (N,).
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array of shape (N, C).")
    # Free energy: -T * logsumexp(logits / T)
    energy = -temperature * logsumexp(logits / temperature, axis=1)
    return energy


def fit_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Fit optimal temperature scalar T on a held-out validation/calibration split using NLL.

    Args:
        val_logits: 2D array of validation logits of shape (N_val, C).
        val_labels: 1D array of validation labels of shape (N_val,).

    Returns:
        Optimal temperature T > 0.
    """
    val_logits = np.asarray(val_logits, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)
    n_samples, _n_classes = val_logits.shape

    def nll_objective(t: np.ndarray) -> float:
        temp = float(t[0])
        scaled_logits = val_logits / temp
        # Log-softmax
        log_probs = scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
        # NLL
        loss = -np.mean(log_probs[np.arange(n_samples), val_labels])
        return float(loss)

    res = minimize(
        nll_objective,
        x0=np.array([1.5]),
        bounds=[(0.01, 10.0)],
        method="L-BFGS-B",
    )
    return float(res.x[0])


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply temperature scaling to logits and compute calibrated probabilities and max confidence.

    Args:
        logits: 2D array of shape (N, C).
        temperature: Scaling temperature T.

    Returns:
        Tuple of (calibrated_probabilities of shape (N, C), calibrated_max_probabilities of shape (N,)).
    """
    logits = np.asarray(logits, dtype=float)
    scaled_logits = logits / temperature
    cal_probs = softmax(scaled_logits, axis=1)
    cal_msp = np.max(cal_probs, axis=1)
    return cal_probs, cal_msp


def compute_uncertainty_metrics(
    probabilities: np.ndarray,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute summary metrics for prediction uncertainty."""
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
