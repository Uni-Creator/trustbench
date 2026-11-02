"""Unit tests for trustbench.metrics."""

import numpy as np
import pytest

from trustbench.metrics.calibration import compute_calibration_metrics
from trustbench.metrics.classification import compute_classification_metrics
from trustbench.metrics.uncertainty import (
    compute_predictive_entropy,
    compute_uncertainty_metrics,
)


def test_classification_metrics_perfect() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    predictions = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_classification_metrics(labels, predictions, class_names=["a", "b", "c"])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert metrics["per_class"]["a"]["precision"] == 1.0
    assert metrics["per_class"]["a"]["recall"] == 1.0


def test_classification_metrics_mismatch() -> None:
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 0, 1, 1])
    metrics = compute_classification_metrics(labels, predictions, class_names=["c0", "c1"])

    assert metrics["accuracy"] == 0.5
    assert len(metrics["per_class"]) == 2


def test_calibration_metrics() -> None:
    # 2 samples, 2 classes
    labels = np.array([0, 1])
    # perfectly confident and correct
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    metrics = compute_calibration_metrics(labels, probabilities, n_bins=5)

    assert pytest.approx(metrics["brier_score"]) == 0.0
    assert pytest.approx(metrics["ece"]) == 0.0


def test_uncertainty_metrics() -> None:
    # Uniform distribution has max entropy
    probabilities = np.array([[0.5, 0.5], [1.0, 0.0]])
    entropies = compute_predictive_entropy(probabilities)

    assert entropies[0] > entropies[1]
    assert pytest.approx(entropies[1], abs=1e-5) == 0.0

    summary = compute_uncertainty_metrics(probabilities)
    assert summary["mean_entropy"] > 0.0
    assert summary["min_entropy"] >= 0.0
