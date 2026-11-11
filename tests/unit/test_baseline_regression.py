"""Regression checks against existing baseline experiment artifacts."""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.uncertainty import compute_uncertainty_metrics
from trustbench.reporting.artifacts import load_experiment_artifacts

BASELINE_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "experiments" / "baseline_001"
)


def test_baseline_metrics_match_stored_predictions() -> None:
    if not BASELINE_DIR.exists():
        pytest.skip(f"Baseline artifacts directory not found: {BASELINE_DIR}")
    artifacts = load_experiment_artifacts(BASELINE_DIR)
    predictions = artifacts["predictions"]
    stored = artifacts["metrics"]
    class_names = artifacts["config"]["classes"]

    labels = np.asarray(predictions["labels"])
    pred_idx = np.asarray(predictions["predictions"])
    logits = np.asarray(predictions["logits"])
    if "probabilities" in predictions:
        probabilities = np.asarray(predictions["probabilities"])
    else:
        probabilities = softmax(logits, axis=1)

    classification = compute_classification_metrics(labels, pred_idx, class_names=class_names)
    calibration = compute_calibration_metrics(labels, probabilities)
    uncertainty = compute_uncertainty_metrics(probabilities)

    assert classification["accuracy"] == pytest.approx(stored["classification"]["accuracy"])
    assert classification["macro_f1"] == pytest.approx(stored["classification"]["macro_f1"])
    assert classification["confusion_matrix"] == stored["classification"]["confusion_matrix"]
    assert calibration["brier_score"] == pytest.approx(stored["calibration"]["brier_score"])
    assert calibration["ece"] == pytest.approx(stored["calibration"]["ece"])
    assert uncertainty["mean_entropy"] == pytest.approx(stored["uncertainty"]["mean_entropy"])
    assert stored["summary"]["total_samples"] == 10000
