"""Calibration metrics including Expected Calibration Error (ECE), Adaptive ECE, and Brier Score."""

from typing import Any

import numpy as np


def compute_nll(
    labels: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """Compute Negative Log-Likelihood (cross-entropy loss): - (1/N) * sum_i log(p_{i, y_i}).

    Args:
        labels: 1D array of ground truth labels of shape (N,).
        probabilities: 2D array of class probabilities of shape (N, C).
        epsilon: Minimum probability floor to avoid log(0).

    Returns:
        Negative log-likelihood as a positive float.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0
    n_classes = probabilities.shape[1]
    valid = (labels >= 0) & (labels < n_classes)
    if not np.any(valid):
        return 0.0
    clipped = np.clip(probabilities, epsilon, 1.0)
    true_probs = clipped[np.arange(n_samples)[valid], labels[valid]]
    return float(-np.mean(np.log(true_probs)))


def compute_adaptive_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Adaptive Expected Calibration Error (equal-frequency binning).

    Samples are partitioned into M quantile bins of equal sample count,
    mitigating empty-bin and confidence-clustering artifacts.

    Mathematical formulation:
      AdaECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        labels: 1D array of ground truth labels (N,).
        probabilities: 2D array of probabilities (N, C).
        n_bins: Number of equal-frequency bins (default: 15).

    Returns:
        Adaptive ECE in [0.0, 1.0].
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    order = np.argsort(confidences)
    sorted_conf = confidences[order]
    sorted_acc = accuracies[order]

    num_bins = min(max(1, n_bins), n_samples)
    bin_splits = np.array_split(np.arange(n_samples), num_bins)

    adaptive_ece = 0.0
    for split_indices in bin_splits:
        if len(split_indices) == 0:
            continue
        bin_acc = float(np.mean(sorted_acc[split_indices]))
        bin_conf = float(np.mean(sorted_conf[split_indices]))
        weight = len(split_indices) / n_samples
        adaptive_ece += weight * abs(bin_acc - bin_conf)

    return float(adaptive_ece)


def compute_reliability_diagram_data(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute structured data for plotting or rendering a reliability diagram.

    Args:
        labels: 1D array of ground truth labels (N,).
        probabilities: 2D array of predicted probabilities (N, C).
        n_bins: Number of equal-width bins.

    Returns:
        Dictionary containing bin boundaries, centers, accuracies, confidences,
        counts, and calibration gaps.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    n_samples = len(labels)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = [float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2) for i in range(n_bins)]

    if n_samples == 0:
        return {
            "bin_edges": bin_boundaries.tolist(),
            "bin_centers": bin_centers,
            "bin_confidences": [0.0] * n_bins,
            "bin_accuracies": [0.0] * n_bins,
            "bin_counts": [0] * n_bins,
            "calibration_gaps": [0.0] * n_bins,
        }

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []
    calibration_gaps: list[float] = []

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
            b_acc = float(np.mean(accuracies[in_bin]))
            b_conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(b_acc)
            bin_confs.append(b_conf)
            calibration_gaps.append(float(abs(b_acc - b_conf)))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            calibration_gaps.append(0.0)

    return {
        "bin_edges": bin_boundaries.tolist(),
        "bin_centers": bin_centers,
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
        "calibration_gaps": calibration_gaps,
    }


def compute_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute calibration metrics: Brier score, ECE, Adaptive ECE, MCE, NLL, and bin statistics.

    Mathematical definitions:
      - Brier Score: (1 / N) * sum_i sum_c (p_ic - y_ic)^2
      - ECE (Equal-width): sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
      - Adaptive ECE (Equal-frequency): quantile-based partition
      - MCE: max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|
      - NLL: - (1 / N) * sum_i log(p_{i, y_i})

    Args:
        labels: 1D array of ground truth labels of shape (N,).
        probabilities: 2D array of predicted class probabilities of shape (N, C).
        n_bins: Number of equal-width confidence bins (default: 15).

    Returns:
        Dictionary containing:
            - brier_score: Multi-class Brier score
            - ece: Expected Calibration Error
            - adaptive_ece: Adaptive Expected Calibration Error
            - mce: Maximum Calibration Error
            - nll: Negative Log-Likelihood
            - bin_confidences: Mean confidence per bin
            - bin_accuracies: Mean accuracy per bin
            - bin_counts: Number of samples per bin
            - reliability_diagram: Structured coordinates and gaps for plotting
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
            "adaptive_ece": 0.0,
            "mce": 0.0,
            "nll": 0.0,
            "bin_confidences": [],
            "bin_accuracies": [],
            "bin_counts": [],
            "reliability_diagram": compute_reliability_diagram_data(labels, probabilities, n_bins=n_bins),
        }

    # 1. Multi-class Brier Score: (1/N) * sum_i sum_c (p_ic - y_ic)^2
    one_hot = np.zeros_like(probabilities)
    valid_idx = (labels >= 0) & (labels < n_classes)
    one_hot[np.arange(n_samples)[valid_idx], labels[valid_idx]] = 1.0
    brier_score = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    # 2. Equal-width binning for ECE and MCE
    diagram = compute_reliability_diagram_data(labels, probabilities, n_bins=n_bins)
    bin_counts = diagram["bin_counts"]
    bin_accs = diagram["bin_accuracies"]
    bin_confs = diagram["bin_confidences"]

    ece = 0.0
    mce = 0.0
    for b_acc, b_conf, b_cnt in zip(bin_accs, bin_confs, bin_counts, strict=False):
        if b_cnt > 0:
            gap = abs(b_acc - b_conf)
            ece += (b_cnt / n_samples) * gap
            mce = max(mce, gap)

    # 3. Adaptive ECE
    ada_ece = compute_adaptive_ece(labels, probabilities, n_bins=n_bins)

    # 4. NLL
    nll = compute_nll(labels, probabilities)

    return {
        "brier_score": brier_score,
        "ece": float(ece),
        "adaptive_ece": float(ada_ece),
        "mce": float(mce),
        "nll": float(nll),
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
        "reliability_diagram": diagram,
    }
