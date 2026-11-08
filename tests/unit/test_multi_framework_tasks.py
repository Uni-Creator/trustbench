"""Unit tests for Phase 8: Multi-Framework and Multi-Task Support.

Covers:
  - Framework adapters: Scikit-learn, ONNX, Keras, JAX
  - Task-specific evaluators: Regression, Retrieval, Semantic Segmentation
  - Advanced evaluation: Fairness, OOD Detection
  - TrustBench wrapping of scikit-learn estimators
  - Dependency isolation & error handling
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from trustbench import (
    RegressionResult,
    RetrievalResult,
    SegmentationResult,
    SklearnModelAdapter,
    TrustBench,
    compute_fairness_metrics,
    compute_ood_detection_metrics,
    compute_regression_metrics,
    compute_retrieval_metrics,
    compute_segmentation_metrics,
    evaluate_regression,
    evaluate_retrieval,
    evaluate_segmentation,
)
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.models.jax import JAXModelAdapter
from trustbench.models.keras import KerasModelAdapter
from trustbench.models.onnx import ONNXModelAdapter


# ==============================================================================
# 1. Framework Adapters: Scikit-Learn
# ==============================================================================
class TestSklearnModelAdapter:
    """Test suite for Scikit-learn estimator adapter."""

    def test_classifier_probabilities_and_predictions(self):
        # Train simple logistic regression on binary data
        x = np.array([[1.0, 2.0], [2.0, 3.0], [-1.0, -2.0], [-2.0, -1.0]])
        y = np.array([1, 1, 0, 0])
        clf = LogisticRegression()
        clf.fit(x, y)

        adapter = SklearnModelAdapter(clf)
        assert adapter.num_classes == 2
        assert adapter.supports_predict_proba is True
        assert adapter.supports_predict_logits is True

        preds = adapter.predict(x)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 4
        np.testing.assert_array_equal(preds, clf.predict(x))

        probs = adapter.predict_proba(x)
        assert probs.shape == (4, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)

        logits = adapter.predict_logits(x)
        assert logits.shape == (4, 2)

    def test_multiclass_tree_classifier(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1, 2, 2])
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(x, y)

        adapter = SklearnModelAdapter(clf)
        assert adapter.num_classes == 3
        probs = adapter.predict_proba(x)
        assert probs.shape == (6, 3)

    def test_regressor_adapter(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        reg = LinearRegression()
        reg.fit(x, y)

        adapter = SklearnModelAdapter(reg, task="regression")
        assert adapter.num_classes == 1
        assert adapter.task == "regression"
        preds = adapter.predict(x)
        assert preds.shape == (4,)
        np.testing.assert_allclose(preds, y, atol=1e-5)


# ==============================================================================
# 2. Framework Adapters: ONNX, Keras, JAX
# ==============================================================================
class TestOptionalFrameworkAdapters:
    """Test suite for optional framework adapters and error reporting."""

    def test_onnx_adapter_mock_session(self):
        mock_input = MagicMock()
        mock_input.name = "input_node"
        mock_input.type = "tensor(float)"

        mock_output = MagicMock()
        mock_output.name = "output_node"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        # Simulate returning 3-class logits
        mock_session.run.return_value = [np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])]

        # Patch onnxruntime import
        with patch.dict(sys.modules, {"onnxruntime": MagicMock()}):
            adapter = ONNXModelAdapter(mock_session, num_classes=3)
            assert adapter.num_classes == 3
            assert adapter.supports_predict_proba is True

            preds = adapter.predict([[1.0], [2.0]])
            np.testing.assert_array_equal(preds, [0, 1])

            probs = adapter.predict_proba([[1.0], [2.0]])
            assert probs.shape == (2, 3)
            np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)

    def test_onnx_missing_dependency_raises(self):
        with (
            patch.dict(sys.modules, {"onnxruntime": None}),
            pytest.raises(ImportError, match="ONNX Runtime is required"),
        ):
            ONNXModelAdapter("model.onnx", num_classes=2)

    def test_keras_missing_dependency_raises(self):
        with (
            patch.dict(sys.modules, {"tensorflow": None, "keras": None}),
            pytest.raises(ImportError, match="TensorFlow or Keras is required"),
        ):
            KerasModelAdapter(MagicMock(), num_classes=2)

    def test_keras_adapter_mock_inference(self):
        mock_model = MagicMock()
        mock_model.return_value = np.array([[1.5, -0.5], [-1.0, 2.0]])

        with patch.dict(sys.modules, {"tensorflow": MagicMock()}):
            adapter = KerasModelAdapter(mock_model, num_classes=2, is_logits=True)
            assert adapter.num_classes == 2
            preds = adapter.predict([[1.0], [2.0]])
            np.testing.assert_array_equal(preds, [0, 1])
            probs = adapter.predict_proba([[1.0], [2.0]])
            assert probs.shape == (2, 2)

    def test_jax_missing_dependency_raises(self):
        with (
            patch.dict(sys.modules, {"jax": None, "jax.numpy": None}),
            pytest.raises(ImportError, match="JAX is required"),
        ):
            JAXModelAdapter(lambda p, x: x, params={}, num_classes=2)


# ==============================================================================
# 3. Task Support: Regression Evaluator & Metrics
# ==============================================================================
class TestRegressionTask:
    """Test suite for regression evaluation metrics and evaluator."""

    def test_compute_regression_metrics_known_values(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        metrics = compute_regression_metrics(y_true, y_pred)
        # residuals = [0.5, -0.5, 0.0, -1.0]
        # abs = [0.5, 0.5, 0.0, 1.0] -> mean = 2.0 / 4 = 0.5
        assert metrics["mae"] == pytest.approx(0.5)
        # sq = [0.25, 0.25, 0.0, 1.0] -> mean = 1.5 / 4 = 0.375
        assert metrics["mse"] == pytest.approx(0.375)
        assert metrics["rmse"] == pytest.approx(np.sqrt(0.375))
        assert metrics["max_error"] == pytest.approx(1.0)
        assert metrics["medae"] == pytest.approx(0.5)
        assert metrics["n_samples"] == 4
        assert "r2" in metrics

    def test_regression_uncertainty_coverage(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.2, 19.8, 30.1])
        # Uncertainty std is 1.0, so 95% interval is +/- 1.96
        y_std = np.array([1.0, 1.0, 1.0])

        metrics = compute_regression_metrics(y_true, y_pred, y_std=y_std)
        assert "uncertainty" in metrics
        assert metrics["uncertainty"]["picp_95"] == 1.0
        assert metrics["uncertainty"]["mpiw_95"] == pytest.approx(2 * 1.95996, rel=1e-3)

    def test_evaluate_regression_pipeline(self):
        x = np.linspace(-5, 5, 50).reshape(-1, 1)
        y = 3.0 * x.squeeze() + 2.0

        model = LinearRegression()
        model.fit(x, y)
        adapter = SklearnModelAdapter(model, task="regression")

        dataset = InMemoryDatasetAdapter(
            inputs=x,
            labels=y,
            sample_ids=[f"s_{i}" for i in range(len(y))],
        )

        result = evaluate_regression(adapter, dataset, batch_size=16)
        assert isinstance(result, RegressionResult)
        assert result.metrics["r2"] == pytest.approx(1.0, rel=1e-4)
        assert result.metrics["mae"] < 1e-4
        assert result.summary["mae"] < 1e-4
        assert len(result.residuals) == 50


# ==============================================================================
# 4. Task Support: Information Retrieval Evaluator & Metrics
# ==============================================================================
class TestRetrievalTask:
    """Test suite for retrieval and ranking evaluation."""

    def test_compute_retrieval_metrics(self):
        # Query 1: relevant = {d1, d2}, retrieved = [d1, d3, d2, d4] -> hits: 1, 0, 1, 0
        # Query 2: relevant = {d5}, retrieved = [d6, d7, d5] -> hits: 0, 0, 1
        ground_truth = {
            "q1": ["d1", "d2"],
            "q2": ["d5"],
        }
        ranked_predictions = {
            "q1": ["d1", "d3", "d2", "d4"],
            "q2": ["d6", "d7", "d5"],
        }

        metrics = compute_retrieval_metrics(ground_truth, ranked_predictions, k_list=(1, 3))

        # MRR: q1 first hit at rank 1 (1.0), q2 first hit at rank 3 (1/3)
        # mean MRR = (1.0 + 0.3333) / 2 = ~0.6667
        assert metrics["mrr"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)

        # Recall@1: q1 = 1/2, q2 = 0/1 -> mean = 0.25
        assert metrics["recall@1"] == pytest.approx(0.25)
        # Recall@3: q1 = 2/2 = 1.0, q2 = 1/1 = 1.0 -> mean = 1.0
        assert metrics["recall@3"] == pytest.approx(1.0)

        # Precision@1: q1 = 1/1 = 1.0, q2 = 0/1 = 0.0 -> mean = 0.5
        assert metrics["precision@1"] == pytest.approx(0.5)

    def test_evaluate_retrieval_pipeline(self):
        gt = [["docA"], ["docB", "docC"]]
        preds = [["docA", "docX"], ["docY", "docB"]]

        res = evaluate_retrieval(gt, preds, k_list=(1, 2))
        assert isinstance(res, RetrievalResult)
        assert res.summary["recall@1"] == pytest.approx(0.5)
        assert "mrr" in res.summary


# ==============================================================================
# 5. Task Support: Semantic Segmentation Evaluator & Metrics
# ==============================================================================
class TestSegmentationTask:
    """Test suite for semantic segmentation evaluation."""

    def test_compute_segmentation_metrics(self):
        # 2 samples of (2, 2) masks, 2 classes (0 and 1)
        # True:  [[0, 1], [1, 0]]
        # Pred:  [[0, 1], [0, 0]]  -> pixel at (1, 0) was 1, predicted 0
        y_true = np.array([[[0, 1], [1, 0]]])
        y_pred = np.array([[[0, 1], [0, 0]]])

        metrics = compute_segmentation_metrics(
            y_true, y_pred, num_classes=2, class_names=["bg", "fg"]
        )

        assert metrics["pixel_accuracy"] == pytest.approx(0.75)  # 3 of 4 correct
        # Class 0: TP=2, FP=1, FN=0 -> IoU = 2 / 3
        # Class 1: TP=1, FP=0, FN=1 -> IoU = 1 / 2
        # mIoU = (2/3 + 1/2) / 2 = 7/12 ~ 0.5833
        assert metrics["per_class_iou"]["bg"] == pytest.approx(2.0 / 3.0)
        assert metrics["per_class_iou"]["fg"] == pytest.approx(0.5)
        assert metrics["miou"] == pytest.approx(7.0 / 12.0)
        assert "dice" in metrics

    def test_evaluate_segmentation_pipeline(self):
        # Mock segmentation model
        mock_model = MagicMock()
        # Returns (B, H, W)
        mock_model.predict.return_value = np.zeros((2, 4, 4), dtype=int)

        dataset = InMemoryDatasetAdapter(
            inputs=np.zeros((2, 3, 4, 4)),
            labels=np.zeros((2, 4, 4), dtype=int),
            sample_ids=["img1", "img2"],
        )

        result = evaluate_segmentation(mock_model, dataset, num_classes=1)
        assert isinstance(result, SegmentationResult)
        assert result.metrics["pixel_accuracy"] == pytest.approx(1.0)
        assert result.summary["miou"] == pytest.approx(1.0)


# ==============================================================================
# 6. Advanced Evaluation: Fairness & OOD Detection
# ==============================================================================
class TestAdvancedEvaluation:
    """Test suite for algorithmic fairness and out-of-distribution evaluation."""

    def test_fairness_disparity_metrics(self):
        # 10 samples: group 'A' has 4/5 positive rate, group 'B' has 1/5 positive rate
        y_true = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

        fairness = compute_fairness_metrics(y_true, y_pred, sensitive_features=groups)

        # Sel rate A = 4/5 = 0.8, Sel rate B = 1/5 = 0.2
        assert fairness["demographic_parity_difference"] == pytest.approx(0.6)
        assert fairness["disparate_impact_ratio"] == pytest.approx(0.25)
        assert "equalized_odds_difference" in fairness
        assert "A" in fairness["groups"]
        assert "B" in fairness["groups"]

    def test_ood_detection_metrics(self):
        # In-distribution scores are low, OOD scores are high
        id_scores = np.array([0.1, 0.2, 0.15, 0.05])
        ood_scores = np.array([0.8, 0.9, 0.95, 0.85])

        metrics = compute_ood_detection_metrics(id_scores, ood_scores)
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["detection_accuracy"] == pytest.approx(1.0)


# ==============================================================================
# 7. TrustBench Integration: Auto-wrapping Scikit-Learn Estimators
# ==============================================================================
class TestTrustBenchWrapping:
    """Test TrustBench high-level entrypoint with scikit-learn estimators."""

    def test_trustbench_wrap_sklearn(self):
        x = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
        y = np.array([1, 1, 0, 0])
        clf = LogisticRegression()
        clf.fit(x, y)

        dataset = InMemoryDatasetAdapter(inputs=x, labels=y, sample_ids=["1", "2", "3", "4"])

        # Auto-wrap scikit-learn estimator
        bench = TrustBench(model=clf, dataset=dataset)
        result = bench.run()

        assert result.metrics["classification"]["accuracy"] == pytest.approx(1.0)
        assert result.probabilities is not None
