"""Unit tests for error detection and selective prediction metrics."""

import numpy as np
import pytest

from trustbench.metrics.error_detection import (
    compute_binary_detection_curve,
    compute_error_detection_metrics,
    compute_slice_error_rates,
)
from trustbench.metrics.selective import (
    compute_risk_coverage,
    compute_selective_prediction_metrics,
)


def test_binary_detection_curve() -> None:
    # Errors at indices 2 and 3 have higher uncertainty
    y_true_binary = np.array([0, 0, 1, 1])
    uncertainty_scores = np.array([0.1, 0.2, 0.8, 0.9])
    res = compute_binary_detection_curve(y_true_binary, uncertainty_scores)

    assert pytest.approx(res["auroc"]) == 1.0
    assert pytest.approx(res["auprc"]) == 1.0
    assert pytest.approx(res["prevalence"]) == 0.5


def test_slice_error_rates() -> None:
    # 10 samples: first 5 have low confidence (indices 0..4), last 5 have high confidence (indices 5..9)
    # let errors be concentrated in low confidence
    errors = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    confidences = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

    slices = compute_slice_error_rates(errors, confidences, top_k_pct=0.3)
    # top 3 confidence samples: all correct (error_rate = 0.0)
    assert slices["top_confidence_error_rate"] == 0.0
    # bottom 3 confidence samples: all errors (error_rate = 1.0)
    assert slices["bottom_confidence_error_rate"] == 1.0


def test_error_detection_metrics_integration() -> None:
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])  # sample 2 is an error
    probabilities = np.array([
        [0.9, 0.1],  # correct, confident
        [0.2, 0.8],  # correct, confident
        [0.45, 0.55],  # incorrect, uncertain
        [0.1, 0.9],  # correct, confident
    ])

    results = compute_error_detection_metrics(labels, predictions, probabilities)
    assert results["error_count"] == 1
    assert "max_softmax_probability" in results
    assert "predictive_entropy" in results
    # The incorrect sample has lower confidence (0.55 vs 0.8/0.9)
    assert results["max_softmax_probability"]["auroc"] == 1.0
    assert results["predictive_entropy"]["auroc"] == 1.0


def test_risk_coverage() -> None:
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])  # sample 2 is error
    confidences = np.array([0.95, 0.90, 0.55, 0.85])  # error has lowest confidence

    res = compute_risk_coverage(labels, predictions, confidences, coverages=[1.0, 0.75])
    curve = res["curve"]

    # At 100% coverage: 1 error out of 4 -> 25% error rate
    assert pytest.approx(curve[0]["error_rate"]) == 0.25
    # At 75% coverage: bottom 25% (sample 2, the error) is rejected -> 0% error rate!
    assert pytest.approx(curve[1]["error_rate"]) == 0.0
    assert pytest.approx(curve[1]["accuracy"]) == 1.0


def test_selective_prediction_metrics() -> None:
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

    res = compute_selective_prediction_metrics(labels, predictions, probabilities)
    assert "max_softmax_probability" in res
    assert "predictive_entropy" in res
