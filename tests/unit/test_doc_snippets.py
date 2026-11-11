"""Tests verifying that code examples and snippets in documentation remain runnable and accurate.

Prevents documentation drift across releases for:
- docs/metrics.md (Examples A, B, C, D)
- docs/quickstart.md (Classification & Regression)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from trustbench import InMemoryDatasetAdapter, TrustBench, evaluate_regression
from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.error_detection import compute_binary_detection_curve
from trustbench.evaluation.fairness import compute_fairness_metrics
from trustbench.evaluation.selective import compute_risk_coverage
from trustbench.evaluation.uncertainty import compute_predictive_entropy


class TestDocumentationSnippets:
    """Verifies that all Python code snippets in user-facing documentation execute without error."""

    def test_metrics_doc_example_a_classification_and_calibration(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_prob = np.array([
            [0.90, 0.10],
            [0.20, 0.80],
            [0.45, 0.55],
            [0.70, 0.30],
            [0.30, 0.70],
            [0.85, 0.15],
            [0.40, 0.60],  # Misclassification
            [0.10, 0.90],
        ])
        y_pred = np.argmax(y_prob, axis=1)

        clf_metrics = compute_classification_metrics(y_true, y_pred)
        cal_metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)

        assert clf_metrics["accuracy"] == pytest.approx(7.0 / 8.0, abs=1e-2)
        assert cal_metrics["ece"] >= 0.0
        assert cal_metrics["brier_score"] >= 0.0

    def test_metrics_doc_example_b_uncertainty_and_error_detection(self) -> None:
        probs = np.array([[0.99, 0.01], [0.55, 0.45], [0.10, 0.90]])
        entropy = compute_predictive_entropy(probs)
        assert len(entropy) == 3
        assert np.all(entropy >= 0.0)

        errors = np.array([0, 1, 0])
        curve = compute_binary_detection_curve(errors, entropy)
        assert curve["auroc"] == pytest.approx(1.0, abs=1e-5)

    def test_metrics_doc_example_c_selective_prediction(self) -> None:
        labels = np.array([0, 0, 0, 1, 1])
        predictions = np.array([0, 0, 0, 0, 0])  # 2 errors
        confidences = np.array([0.95, 0.90, 0.85, 0.30, 0.20])

        res = compute_risk_coverage(labels, predictions, confidences)
        assert res["excess_aurc"] == pytest.approx(0.0, abs=1e-5)
        assert res["aurc"] == pytest.approx(res["optimal_aurc"], abs=1e-5)

    def test_metrics_doc_example_d_fairness_teaching_case(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        fair = compute_fairness_metrics(y_true, y_pred, groups)

        # Equal selection rates (0.50 each) -> zero demographic disparity
        assert fair["demographic_parity_difference"] == pytest.approx(0.0, abs=1e-5)
        assert fair["disparate_impact_ratio"] == pytest.approx(1.0, abs=1e-5)

        # Disparate error rates across groups: TPR_A=0.67 vs TPR_B=1.00, FPR_A=0.00 vs FPR_B=0.33
        assert fair["equalized_odds_difference"] == pytest.approx(1.0 / 3.0, abs=1e-2)

    def test_quickstart_doc_classification_snippet(self) -> None:
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 2, size=100)
        clf = LogisticRegression().fit(X_train, y_train)

        X_test = np.random.randn(50, 4)
        y_test = np.random.randint(0, 2, size=50)
        dataset = InMemoryDatasetAdapter(inputs=X_test, labels=y_test)

        bench = TrustBench(model=clf, dataset=dataset)
        result = bench.evaluate()

        assert "classification" in result.metrics
        assert "accuracy" in result.metrics["classification"]
        assert "calibration" in result.metrics
        assert "brier_score" in result.metrics["calibration"]

    def test_quickstart_doc_regression_snippet(self) -> None:
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = 2.5 * X.squeeze() + np.random.normal(0, 0.5, size=100)
        reg = Ridge().fit(X, y)

        dataset = InMemoryDatasetAdapter(inputs=X, labels=y)
        result = evaluate_regression(reg, dataset)

        assert "mae" in result.metrics
        assert "r2" in result.metrics
