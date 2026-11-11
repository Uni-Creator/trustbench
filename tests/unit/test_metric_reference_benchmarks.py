"""Reference benchmark tests comparing TrustBench metrics against reference implementations.

Validates TrustBench implementations against:
- scikit-learn (classification, regression, Brier score, log-loss, AUROC, AUPRC)
- Fairlearn mathematical definitions (Demographic Parity, Equalized Odds, Disparate Impact)
- Published calibration literature references (Guo et al. 2017, Naeini et al. 2015, Ding et al. 2020)
- Analytical selective prediction bounds (Geifman & El-Yaniv 2017)
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.metrics as skm
from scipy.special import softmax

from trustbench.evaluation.calibration import (
    compute_adaptive_ece,
    compute_calibration_metrics,
    compute_nll,
)
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
    compute_fpr_at_tpr,
)
from trustbench.evaluation.fairness import compute_fairness_metrics
from trustbench.evaluation.selective import (
    compute_optimal_aurc,
    compute_risk_coverage,
)
from trustbench.metrics.regression import compute_regression_metrics

# 1. Classification: Comparison against scikit-learn

class TestClassificationReferenceBenchmarks:
    def test_multiclass_metrics_match_sklearn(self):
        rng = np.random.default_rng(42)
        n_samples = 200
        n_classes = 5
        y_true = rng.integers(0, n_classes, size=n_samples)
        y_pred = rng.integers(0, n_classes, size=n_samples)
        class_names = [f"class_{i}" for i in range(n_classes)]

        tb_metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)

        # Accuracy
        sk_acc = skm.accuracy_score(y_true, y_pred)
        assert tb_metrics["accuracy"] == pytest.approx(sk_acc, abs=1e-9)

        # Balanced accuracy
        sk_bal_acc = skm.balanced_accuracy_score(y_true, y_pred)
        assert tb_metrics["balanced_accuracy"] == pytest.approx(sk_bal_acc, abs=1e-9)

        # Macro metrics
        sk_prec, sk_rec, sk_f1, _ = skm.precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        assert tb_metrics["macro_precision"] == pytest.approx(sk_prec, abs=1e-9)
        assert tb_metrics["macro_recall"] == pytest.approx(sk_rec, abs=1e-9)
        assert tb_metrics["macro_f1"] == pytest.approx(sk_f1, abs=1e-9)

        # Confusion matrix
        sk_cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        np.testing.assert_array_equal(tb_metrics["confusion_matrix"], sk_cm)



# 2. Regression: Comparison against scikit-learn

class TestRegressionReferenceBenchmarks:
    def test_regression_metrics_match_sklearn(self):
        rng = np.random.default_rng(123)
        y_true = rng.normal(10.0, 3.0, size=150)
        y_pred = y_true + rng.normal(0.0, 0.8, size=150)

        tb_reg = compute_regression_metrics(y_true, y_pred)

        sk_mae = skm.mean_absolute_error(y_true, y_pred)
        sk_mse = skm.mean_squared_error(y_true, y_pred)
        sk_rmse = np.sqrt(sk_mse)
        sk_r2 = skm.r2_score(y_true, y_pred)
        sk_medae = skm.median_absolute_error(y_true, y_pred)
        sk_max_err = skm.max_error(y_true, y_pred)

        assert tb_reg["mae"] == pytest.approx(sk_mae, abs=1e-9)
        assert tb_reg["mse"] == pytest.approx(sk_mse, abs=1e-9)
        assert tb_reg["rmse"] == pytest.approx(sk_rmse, abs=1e-9)
        assert tb_reg["r2"] == pytest.approx(sk_r2, abs=1e-9)
        assert tb_reg["medae"] == pytest.approx(sk_medae, abs=1e-9)
        assert tb_reg["max_error"] == pytest.approx(sk_max_err, abs=1e-9)



# 3. Calibration: Brier score, NLL, and Independent ECE Reference

class TestCalibrationReferenceBenchmarks:
    def test_brier_score_binary_matches_sklearn_convention(self):
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 2, size=100)
        logits = rng.normal(0.0, 1.0, size=(100, 2))
        probs = softmax(logits, axis=1)

        cal = compute_calibration_metrics(y_true, probs)

        # In binary classification, multi-class Brier sum_c (p_c - y_c)^2 = 2 * (p_1 - y_1)^2
        sk_brier = skm.brier_score_loss(y_true, probs[:, 1])
        assert cal["brier_score"] == pytest.approx(2.0 * sk_brier, abs=1e-9)

    def test_nll_matches_sklearn_log_loss(self):
        rng = np.random.default_rng(88)
        y_true = rng.integers(0, 3, size=120)
        probs = rng.dirichlet(np.ones(3), size=120)

        nll = compute_nll(y_true, probs)
        sk_log_loss = skm.log_loss(y_true, probs)
        assert nll == pytest.approx(sk_log_loss, rel=1e-4)

    def test_ece_matches_independent_reference_calculation(self):
        """Independent reference calculation of ECE (equal-width binning)."""
        rng = np.random.default_rng(77)
        n_samples = 300
        n_classes = 4
        y_true = rng.integers(0, n_classes, size=n_samples)
        probs = rng.dirichlet(np.ones(n_classes), size=n_samples)

        n_bins = 10
        cal = compute_calibration_metrics(y_true, probs, n_bins=n_bins)

        # Independent textbook reference implementation of ECE
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ref_ece = 0.0
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            if i == n_bins - 1:
                mask = (confidences >= low) & (confidences <= high)
            else:
                mask = (confidences >= low) & (confidences < high)
            count = int(np.sum(mask))
            if count > 0:
                bin_acc = float(np.mean(accuracies[mask]))
                bin_conf = float(np.mean(confidences[mask]))
                ref_ece += (count / n_samples) * abs(bin_acc - bin_conf)

        assert cal["ece"] == pytest.approx(ref_ece, abs=1e-9)

    def test_adaptive_ece_matches_quantile_reference(self):
        """Independent reference calculation of Adaptive ECE (equal-frequency binning)."""
        rng = np.random.default_rng(55)
        n_samples = 250
        y_true = rng.integers(0, 2, size=n_samples)
        probs = rng.dirichlet(np.ones(2), size=n_samples)

        n_bins = 15
        ada_ece = compute_adaptive_ece(y_true, probs, n_bins=n_bins)

        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true).astype(float)

        order = np.argsort(confidences)
        splits = np.array_split(np.arange(n_samples), n_bins)
        ref_ada_ece = sum(
            (len(idx) / n_samples) * abs(np.mean(accuracies[order][idx]) - np.mean(confidences[order][idx]))
            for idx in splits if len(idx) > 0
        )
        assert ada_ece == pytest.approx(ref_ada_ece, abs=1e-9)



# 4. Error Detection: Comparison against scikit-learn

class TestErrorDetectionReferenceBenchmarks:
    def test_auroc_and_auprc_match_sklearn(self):
        rng = np.random.default_rng(44)
        errors = rng.integers(0, 2, size=150)
        uncertainties = rng.uniform(0.0, 1.0, size=150)

        metrics = compute_binary_detection_curve(errors, uncertainties)

        sk_auroc = skm.roc_auc_score(errors, uncertainties)
        assert metrics["auroc"] == pytest.approx(sk_auroc, abs=1e-9)

        prec, rec, _ = skm.precision_recall_curve(errors, uncertainties)
        sk_auprc = skm.auc(rec, prec)
        assert metrics["auprc"] == pytest.approx(sk_auprc, abs=1e-9)

    def test_fpr_at_95_tpr_reference_definition(self):
        errors = np.array([1] * 20 + [0] * 80)
        uncertainties = np.linspace(0.0, 1.0, 100)

        fpr95 = compute_fpr_at_tpr(errors, uncertainties, target_tpr=0.95)
        fpr, tpr, _ = skm.roc_curve(errors, uncertainties)
        expected_fpr = float(np.min(fpr[tpr >= 0.95]))
        assert fpr95 == pytest.approx(expected_fpr, abs=1e-9)



# 5. Algorithmic Fairness: Comparison against Fairlearn Mathematical Definitions

class TestFairnessReferenceBenchmarks:
    def test_fairness_definitions_against_exact_group_disparities(self):
        """Verify demographic parity, disparate impact, and equalized odds."""
        # 100 individuals in group A, 100 in group B
        rng = np.random.default_rng(101)
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        groups = np.array(["group_A"] * 100 + ["group_B"] * 100)

        fair = compute_fairness_metrics(y_true, y_pred, groups, favorable_label=1)

        # Independent computation following Fairlearn formulas:
        mask_a = groups == "group_A"
        mask_b = groups == "group_B"

        sel_rate_a = float(np.mean(y_pred[mask_a] == 1))
        sel_rate_b = float(np.mean(y_pred[mask_b] == 1))

        ref_dpd = abs(sel_rate_a - sel_rate_b)
        ref_dir = min(sel_rate_a, sel_rate_b) / max(sel_rate_a, sel_rate_b)

        tpr_a = float(np.mean(y_pred[mask_a & (y_true == 1)] == 1))
        tpr_b = float(np.mean(y_pred[mask_b & (y_true == 1)] == 1))
        fpr_a = float(np.mean(y_pred[mask_a & (y_true == 0)] == 1))
        fpr_b = float(np.mean(y_pred[mask_b & (y_true == 0)] == 1))

        ref_eod = max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))

        assert fair["demographic_parity_difference"] == pytest.approx(ref_dpd, abs=1e-9)
        assert fair["disparate_impact_ratio"] == pytest.approx(ref_dir, abs=1e-9)
        assert fair["equalized_odds_difference"] == pytest.approx(ref_eod, abs=1e-9)



# 6. Selective Prediction & AURC: Comparison against Analytical Theory

class TestSelectivePredictionReferenceBenchmarks:
    def test_aurc_and_optimal_aurc_bounds(self):
        """Verify AURC properties per Geifman & El-Yaniv (2017)."""
        # When all predictions are correct (0 errors), AURC and Optimal AURC are 0.0
        assert compute_optimal_aurc(n_samples=50, n_errors=0) == 0.0

        # When all predictions are incorrect (50 errors), AURC and Optimal AURC are 1.0
        assert compute_optimal_aurc(n_samples=50, n_errors=50) == 1.0

        # Hand calculation for N=4, E=2:
        # k_err = [3, 4]
        # opt_risks = [(3-2)/3, (4-2)/4] = [1/3, 2/4] = [1/3, 1/2] = [2/6, 3/6] = 5/6
        # AURC* = (5/6) / 4 = 5/24 ≈ 0.208333...
        assert compute_optimal_aurc(n_samples=4, n_errors=2) == pytest.approx(5.0 / 24.0, abs=1e-9)

    def test_perfect_uncertainty_ranking_achieves_optimal_aurc(self):
        """When uncertainty perfectly separates correct from error, AURC == Optimal AURC."""
        labels = np.array([0, 0, 0, 1, 1])
        predictions = np.array([0, 0, 0, 0, 0])  # last 2 are errors
        confidences = np.array([0.95, 0.90, 0.85, 0.20, 0.10])  # perfect ranking

        metrics = compute_risk_coverage(labels, predictions, confidences)
        assert metrics["excess_aurc"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["aurc"] == pytest.approx(metrics["optimal_aurc"], abs=1e-6)
