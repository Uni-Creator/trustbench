"""Unit tests for Phase 3 robustness aggregation."""

import pytest

from trustbench.robustness.aggregation import compute_robustness_aggregation


def test_robustness_aggregation():
    clean_baseline = {
        "condition": "Clean",
        "accuracy": 0.95,
        "ece": 0.02,
        "mean_entropy": 0.05,
    }

    conditions = [
        clean_baseline,
        {
            "condition": "Gaussian Noise S1",
            "corruption": "gaussian_noise",
            "severity": 1,
            "accuracy": 0.90,
            "ece": 0.04,
            "mean_entropy": 0.08,
            "brier_score": 0.08,
        },
        {
            "condition": "Gaussian Noise S2",
            "corruption": "gaussian_noise",
            "severity": 2,
            "accuracy": 0.80,
            "ece": 0.06,
            "mean_entropy": 0.12,
            "brier_score": 0.12,
        },
        {
            "condition": "Brightness S1",
            "corruption": "brightness",
            "severity": 1,
            "accuracy": 0.92,
            "ece": 0.03,
            "mean_entropy": 0.06,
            "brier_score": 0.07,
        },
    ]

    agg = compute_robustness_aggregation(conditions, clean_baseline=clean_baseline)

    assert "corruption_wise" in agg
    assert "severity_wise" in agg
    assert "mean_corruption_accuracy" in agg
    assert "mean_accuracy_drop" in agg
    assert "worst_case_condition" in agg

    # Corruption-wise
    gn = agg["corruption_wise"]["gaussian_noise"]
    assert pytest.approx(gn["mean_accuracy"]) == 0.85
    assert pytest.approx(gn["accuracy_drop"]) == 0.10
    assert pytest.approx(gn["delta_ece"]) == 0.03

    # Severity-wise
    s1 = agg["severity_wise"][1]
    assert pytest.approx(s1["mean_accuracy"]) == 0.91

    # Worst case
    assert agg["worst_case_condition"]["corruption"] == "gaussian_noise"
    assert agg["worst_case_condition"]["severity"] == 2
    assert agg["worst_case_condition"]["accuracy"] == 0.80


def test_robustness_aggregation_empty():
    agg = compute_robustness_aggregation([])
    assert agg["corruption_wise"] == {}
    assert agg["severity_wise"] == {}
    assert agg["mean_corruption_accuracy"] == 0.0

