"""Error detection metrics evaluating how well uncertainty scores identify incorrect predictions."""

from typing import Any

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def compute_binary_detection_curve(
    y_true_binary: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> dict[str, float]:
    """Calculate AUROC and AUPRC for binary error detection.

    Args:
        y_true_binary: 1 for error/incorrect prediction, 0 for correct.
        uncertainty_scores: Higher values indicate higher probability of error.

    Returns:
        Dict with 'auroc' and 'auprc' (and auprc_baseline, i.e. prevalence).
    """
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=float)

    n_samples = len(y_true_binary)
    n_errors = int(np.sum(y_true_binary))

    if n_samples == 0 or n_errors == 0 or n_errors == n_samples:
        # Edge cases where AUROC is mathematically undefined or trivial
        prevalence = float(n_errors / n_samples) if n_samples > 0 else 0.0
        return {
            "auroc": 0.5,
            "auprc": prevalence,
            "prevalence": prevalence,
        }

    try:
        auroc = float(roc_auc_score(y_true_binary, uncertainty_scores))
    except ValueError:
        auroc = 0.5

    try:
        precision, recall, _ = precision_recall_curve(y_true_binary, uncertainty_scores)
        auprc = float(auc(recall, precision))
    except ValueError:
        auprc = float(n_errors / n_samples)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "prevalence": float(n_errors / n_samples),
    }


def compute_slice_error_rates(
    errors: np.ndarray,
    confidences: np.ndarray,
    top_k_pct: float = 0.1,
) -> dict[str, float]:
    """Compute error rates in highest-confidence and lowest-confidence slices.

    Args:
        errors: 1D binary array where 1 = incorrect prediction, 0 = correct.
        confidences: 1D array of confidence scores (e.g. max softmax prob, or -entropy).
        top_k_pct: Fraction for top/bottom slices (default: 0.1 for 10%).

    Returns:
        Dict containing error rates for highest and lowest confidence slices.
    """
    errors = np.asarray(errors, dtype=int)
    confidences = np.asarray(confidences, dtype=float)

    n = len(errors)
    if n == 0:
        return {"top_confidence_error_rate": 0.0, "bottom_confidence_error_rate": 0.0}

    k = max(1, int(np.round(n * top_k_pct)))

    sorted_indices = np.argsort(confidences)  # ascending: lowest first, highest last
    bottom_indices = sorted_indices[:k]
    top_indices = sorted_indices[-k:]

    bottom_error_rate = float(np.mean(errors[bottom_indices]))
    top_error_rate = float(np.mean(errors[top_indices]))

    return {
        "top_confidence_error_rate": top_error_rate,
        "bottom_confidence_error_rate": bottom_error_rate,
        "slice_sample_count": k,
    }


def compute_error_detection_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    entropies: np.ndarray | None = None,
    slice_fraction: float = 0.1,
) -> dict[str, Any]:
    """Comprehensive error detection evaluation across uncertainty scoring methods.

    Evaluates:
      1. Maximum softmax probability (MSP) where uncertainty = -MSP
      2. Predictive entropy where uncertainty = +Entropy

    Args:
        labels: 1D array of ground truth labels (N,).
        predictions: 1D array of predicted class indices (N,).
        probabilities: 2D array of class probabilities (N, C).
        entropies: Optional 1D array of predictive entropy (N,). Computed if not provided.
        slice_fraction: Fraction for top and bottom confidence slices (default 0.1 = 10%).

    Returns:
        Dict with detection performance for both MSP and Predictive Entropy.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    errors = (predictions != labels).astype(int)
    max_confidences = np.max(probabilities, axis=1)

    if entropies is None:
        eps = 1e-12
        clipped = np.clip(probabilities, eps, 1.0)
        entropies = -np.sum(clipped * np.log(clipped), axis=1)
    else:
        entropies = np.asarray(entropies)

    # Uncertainty scores (higher = more likely to be an error)
    uncertainty_from_msp = -max_confidences
    uncertainty_from_entropy = entropies

    # 1. Error detection via MSP
    msp_detection = compute_binary_detection_curve(errors, uncertainty_from_msp)
    msp_slices = compute_slice_error_rates(errors, max_confidences, top_k_pct=slice_fraction)

    # 2. Error detection via Predictive Entropy
    # Note: for confidence slices using entropy, confidence is -entropy
    entropy_detection = compute_binary_detection_curve(errors, uncertainty_from_entropy)
    entropy_slices = compute_slice_error_rates(errors, -entropies, top_k_pct=slice_fraction)

    return {
        "error_count": int(np.sum(errors)),
        "total_samples": len(errors),
        "error_rate": float(np.mean(errors)) if len(errors) > 0 else 0.0,
        "max_softmax_probability": {
            "auroc": msp_detection["auroc"],
            "auprc": msp_detection["auprc"],
            "highest_confidence_error_rate": msp_slices["top_confidence_error_rate"],
            "lowest_confidence_error_rate": msp_slices["bottom_confidence_error_rate"],
        },
        "predictive_entropy": {
            "auroc": entropy_detection["auroc"],
            "auprc": entropy_detection["auprc"],
            "highest_confidence_error_rate": entropy_slices["top_confidence_error_rate"],
            "lowest_confidence_error_rate": entropy_slices["bottom_confidence_error_rate"],
        },
    }
