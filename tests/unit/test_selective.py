"""Unit tests for Phase 3 selective prediction: Excess AURC and coverage/risk queries."""

import numpy as np
import pytest

from trustbench.evaluation.selective import (
    compute_optimal_aurc,
    compute_risk_coverage,
    get_coverage_at_risk,
    get_risk_at_coverage,
)


def test_optimal_aurc_bounds():
    # If 0 errors: optimal AURC = 0.0
    assert compute_optimal_aurc(10, 0) == 0.0
    # If all errors: optimal AURC = 1.0
    assert compute_optimal_aurc(10, 10) == 1.0
    # 2 errors out of 10
    opt = compute_optimal_aurc(10, 2)
    assert 0.0 < opt < 1.0


def test_excess_aurc():
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])  # sample 2 is error
    confidences = np.array([0.9, 0.8, 0.4, 0.7])  # error has lowest confidence

    res = compute_risk_coverage(labels, predictions, confidences)
    assert "aurc" in res
    assert "optimal_aurc" in res
    assert "excess_aurc" in res

    # Because ranking placed error at the very end, ranking is optimal!
    assert pytest.approx(res["excess_aurc"], abs=1e-6) == 0.0
    assert pytest.approx(res["aurc"]) == res["optimal_aurc"]


def test_get_risk_at_coverage():
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])  # sample 2 is error
    confidences = np.array([0.9, 0.8, 0.4, 0.7])  # error has lowest confidence

    # At 100% coverage: 1 error in 4 = 0.25 risk
    risk_full = get_risk_at_coverage(labels, predictions, confidences, coverage=1.0)
    assert pytest.approx(risk_full) == 0.25

    # At 75% coverage: bottom 25% (sample 2) rejected -> 0 errors in 3 = 0.0 risk
    risk_75 = get_risk_at_coverage(labels, predictions, confidences, coverage=0.75)
    assert pytest.approx(risk_75) == 0.0


def test_get_coverage_at_risk():
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])  # sample 2 is error
    confidences = np.array([0.9, 0.8, 0.4, 0.7])

    # If target risk is 0.0, we can achieve 75% coverage
    cov_0 = get_coverage_at_risk(labels, predictions, confidences, target_risk=0.0)
    assert pytest.approx(cov_0) == 0.75

    # If target risk is 0.3, we can achieve 100% coverage
    cov_30 = get_coverage_at_risk(labels, predictions, confidences, target_risk=0.3)
    assert pytest.approx(cov_30) == 1.0


def test_selective_empty_input():
    res = compute_risk_coverage(np.array([]), np.array([]), np.array([]))
    assert res["curve"] == []
    assert res["aurc"] == 0.0
    assert res["excess_aurc"] == 0.0

