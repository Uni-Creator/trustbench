"""Unit tests for Phase 3 calibration metrics: Adaptive ECE, MCE, NLL, and Reliability Diagram."""

import numpy as np
import pytest

from trustbench.evaluation.calibration import (
    compute_adaptive_ece,
    compute_calibration_metrics,
    compute_nll,
    compute_reliability_diagram_data,
)


def test_nll_calculation():
    # 2 samples, 2 classes
    labels = np.array([0, 1])
    probs = np.array([[0.8, 0.2], [0.1, 0.9]])
    # Expected NLL = - (log(0.8) + log(0.9)) / 2
    expected_nll = -float(np.mean([np.log(0.8), np.log(0.9)]))

    nll = compute_nll(labels, probs)
    assert pytest.approx(nll, abs=1e-6) == expected_nll


def test_nll_empty():
    assert compute_nll(np.array([]), np.empty((0, 2))) == 0.0


def test_adaptive_ece_perfect():
    # Perfectly calibrated predictions: accuracy equals confidence in each quantile bin
    labels = np.array([0, 0, 1, 1])
    probs = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    ada_ece = compute_adaptive_ece(labels, probs, n_bins=2)
    assert pytest.approx(ada_ece, abs=1e-6) == 0.0


def test_mce_calculation():
    labels = np.array([0, 1])
    # Model predicts class 0 with 0.9 confidence on sample 0 (correct, acc=1, conf=0.9 -> gap=0.1)
    # Model predicts class 0 with 0.8 confidence on sample 1 (incorrect, acc=0, conf=0.8 -> gap=0.8)
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
    ])
    metrics = compute_calibration_metrics(labels, probs, n_bins=10)
    assert "mce" in metrics
    assert "adaptive_ece" in metrics
    assert "nll" in metrics
    assert metrics["mce"] > 0.0
    assert metrics["mce"] >= metrics["ece"]  # MCE is always >= ECE


def test_reliability_diagram_structure():
    labels = np.array([0, 1, 0, 1])
    probs = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.3, 0.7],
    ])
    diagram = compute_reliability_diagram_data(labels, probs, n_bins=5)

    assert len(diagram["bin_edges"]) == 6
    assert len(diagram["bin_centers"]) == 5
    assert len(diagram["bin_confidences"]) == 5
    assert len(diagram["bin_accuracies"]) == 5
    assert len(diagram["bin_counts"]) == 5
    assert len(diagram["calibration_gaps"]) == 5
    assert sum(diagram["bin_counts"]) == 4


def test_calibration_numerical_extremes():
    labels = np.array([0, 1])
    # Confidences at absolute extremes (0.0 and 1.0)
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    metrics = compute_calibration_metrics(labels, probs)
    assert pytest.approx(metrics["ece"]) == 0.0
    assert pytest.approx(metrics["adaptive_ece"]) == 0.0
    assert pytest.approx(metrics["mce"]) == 0.0
    assert pytest.approx(metrics["brier_score"]) == 0.0
    assert pytest.approx(metrics["nll"], abs=1e-5) == 0.0

