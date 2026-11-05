"""Unit tests for Phase 3 distribution shift evaluation interfaces."""

import numpy as np
import pytest

from trustbench.evaluation.distribution_shift import (
    ShiftType,
    compute_label_distribution_shift,
    compute_ood_detection_metrics,
    compute_tv_distance,
    evaluate_distribution_shift,
)


def test_tv_distance():
    # Identical distributions -> TV = 0
    p1 = np.array([0.5, 0.5])
    assert compute_tv_distance(p1, p1) == 0.0

    # Orthogonal distributions -> TV = 1.0
    p_a = np.array([1.0, 0.0])
    p_b = np.array([0.0, 1.0])
    assert pytest.approx(compute_tv_distance(p_a, p_b)) == 1.0


def test_label_distribution_shift():
    # Source is balanced [5, 5]
    # Target is imbalanced [9, 1]
    y_source = np.array([0] * 5 + [1] * 5)
    y_target = np.array([0] * 9 + [1] * 1)

    res = compute_label_distribution_shift(y_source, y_target, num_classes=2)
    assert res["tv_distance"] > 0.0
    assert res["jensen_shannon_distance"] > 0.0
    assert len(res["source_priors"]) == 2


def test_ood_detection():
    # In-distribution has low entropy / high confidence
    id_scores = np.array([0.1, 0.2, 0.15, 0.05])
    # OOD has high entropy / high uncertainty score
    ood_scores = np.array([0.8, 0.9, 0.85, 0.95])

    res = compute_ood_detection_metrics(id_scores, ood_scores)
    assert pytest.approx(res["auroc"]) == 1.0
    assert pytest.approx(res["fpr_at_95_tpr"]) == 0.0
    assert pytest.approx(res["detection_accuracy"]) == 1.0


def test_evaluate_distribution_shift_pipeline():
    clean_preds = np.array([0, 1, 0, 1])
    clean_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.85, 0.15], [0.15, 0.85]])
    clean_labels = np.array([0, 1, 0, 1])

    # Shifted: degraded accuracy and higher entropy
    shifted_preds = np.array([0, 0, 1, 1])
    shifted_probs = np.array([[0.55, 0.45], [0.52, 0.48], [0.48, 0.52], [0.45, 0.55]])
    shifted_labels = np.array([0, 1, 0, 1])

    report = evaluate_distribution_shift(
        clean_predictions=clean_preds,
        clean_probabilities=clean_probs,
        clean_labels=clean_labels,
        shifted_predictions=shifted_preds,
        shifted_probabilities=shifted_probs,
        shifted_labels=shifted_labels,
        shift_type=ShiftType.DOMAIN_SHIFT,
    )

    assert report["shift_type"] == "domain_shift"
    assert report["degradation"]["clean_accuracy"] == 1.0
    assert report["degradation"]["shifted_accuracy"] == 0.5
    assert report["degradation"]["accuracy_drop"] == 0.5
    assert report["degradation"]["delta_ece"] > 0.0  # ECE got worse
    assert report["confidence_shift"]["wasserstein_distance"] > 0.0
    assert report["entropy_shift"]["wasserstein_distance"] > 0.0

