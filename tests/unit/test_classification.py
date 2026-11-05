"""Unit tests for Phase 3 classification metrics."""

import numpy as np
import pytest

from trustbench.evaluation.classification import (
    compute_balanced_accuracy,
    compute_classification_metrics,
)


def test_balanced_accuracy_balanced():
    # 2 classes, equal representation: 2/2 correct in class 0, 1/2 correct in class 1
    # Recall_0 = 1.0, Recall_1 = 0.5 -> Balanced Acc = 0.75
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 0])
    b_acc = compute_balanced_accuracy(labels, predictions)
    assert pytest.approx(b_acc) == 0.75


def test_balanced_accuracy_imbalanced():
    # Class 0 has 10 samples (all correct), Class 1 has 2 samples (0 correct)
    # Standard accuracy = 10 / 12 = 83.3%
    # Balanced accuracy = (1.0 + 0.0) / 2 = 50.0%
    labels = np.array([0] * 10 + [1] * 2)
    predictions = np.array([0] * 12)
    metrics = compute_classification_metrics(labels, predictions)
    assert pytest.approx(metrics["accuracy"]) == 10 / 12
    assert pytest.approx(metrics["balanced_accuracy"]) == 0.5


def test_classification_metrics_f1_variants():
    labels = np.array([0, 1, 2, 0, 1, 2])
    predictions = np.array([0, 1, 1, 0, 1, 2])
    metrics = compute_classification_metrics(labels, predictions, class_names=["c0", "c1", "c2"])

    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "macro_precision" in metrics
    assert "micro_precision" in metrics
    assert "weighted_precision" in metrics
    assert "macro_recall" in metrics
    assert "micro_recall" in metrics
    assert "weighted_recall" in metrics

    # Bounds
    for k in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "macro_precision",
        "micro_precision",
        "weighted_precision",
        "macro_recall",
        "micro_recall",
        "weighted_recall",
    ]:
        assert 0.0 <= metrics[k] <= 1.0


def test_classification_empty_input():
    metrics = compute_classification_metrics(np.array([]), np.array([]))
    assert metrics["accuracy"] == 0.0
    assert metrics["balanced_accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["per_class"] == {}
    assert metrics["confusion_matrix"] == []


def test_classification_single_class():
    labels = np.array([0, 0, 0])
    predictions = np.array([0, 0, 0])
    metrics = compute_classification_metrics(labels, predictions)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[3]]


def test_classification_shape_validation():
    with pytest.raises(ValueError, match="1D arrays"):
        compute_classification_metrics(np.zeros((3, 2)), np.zeros((3, 2)))

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_classification_metrics(np.array([0, 1]), np.array([0]))

