"""Unit tests for Phase 3 error detection: FPR at 95% TPR, detection accuracy, and imbalance."""

import numpy as np
import pytest

from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
    compute_detection_error,
    compute_error_detection_metrics,
    compute_fpr_at_tpr,
)


def test_fpr_at_95_tpr_perfect():
    # 20 samples: 2 errors (indices 18, 19), 18 correct (indices 0..17)
    # Errors have strictly higher uncertainty scores
    y_true = np.array([0] * 18 + [1] * 2)
    scores = np.linspace(0.0, 1.0, 20)

    fpr95 = compute_fpr_at_tpr(y_true, scores, target_tpr=0.95)
    # Perfect ranking separates errors completely -> 0 false positives needed to catch >=95% errors
    assert pytest.approx(fpr95) == 0.0


def test_detection_error_bounds():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    det_err = compute_detection_error(y_true, scores)
    assert pytest.approx(det_err) == 0.0


def test_error_detection_all_correct():
    # No errors at all (all y_true == 0)
    y_true = np.zeros(10, dtype=int)
    scores = np.random.default_rng(42).random(10)
    res = compute_binary_detection_curve(y_true, scores)
    assert res["prevalence"] == 0.0
    assert res["auroc"] == 0.5
    assert res["fpr_at_95_tpr"] == 0.0


def test_error_detection_all_errors():
    # All errors (all y_true == 1)
    y_true = np.ones(10, dtype=int)
    scores = np.random.default_rng(42).random(10)
    res = compute_binary_detection_curve(y_true, scores)
    assert res["prevalence"] == 1.0
    assert res["auroc"] == 0.5


def test_error_detection_metrics_comprehensive():
    labels = np.array([0, 1, 0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1, 0, 0])  # samples 2 and 5 are errors
    probs = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.49, 0.51],  # error, uncertain
        [0.1, 0.9],
        [0.85, 0.15],
        [0.55, 0.45],  # error, uncertain
    ])
    logits = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
        [0.1, 0.2],
        [0.0, 2.5],
        [1.8, 0.0],
        [0.2, 0.1],
    ])

    results = compute_error_detection_metrics(
        labels=labels,
        predictions=predictions,
        probabilities=probs,
        logits=logits,
        temperature=1.2,
    )

    assert results["error_count"] == 2
    for method in ["max_softmax_probability", "predictive_entropy", "temperature_scaled_probability", "energy_score"]:
        assert method in results
        det = results[method]
        assert "auroc" in det
        assert "auprc" in det
        assert "fpr_at_95_tpr" in det
        assert "detection_error" in det
        assert "detection_accuracy" in det
        assert 0.0 <= det["auroc"] <= 1.0
        assert 0.0 <= det["fpr_at_95_tpr"] <= 1.0

