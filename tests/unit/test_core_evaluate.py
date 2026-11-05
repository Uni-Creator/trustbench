"""Tests for the persistence-free core evaluator."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trustbench.core import EvaluationResult, evaluate
from trustbench.core.exceptions import UnsupportedCapabilityError
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.models.base import ModelAdapter


class LogitModel(ModelAdapter):
    def __init__(self, logits: np.ndarray) -> None:
        self._logits = logits

    @property
    def num_classes(self) -> int:
        return int(self._logits.shape[1])

    @property
    def supports_predict_proba(self) -> bool:
        return True

    @property
    def supports_predict_logits(self) -> bool:
        return True

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        indices = np.asarray(inputs, dtype=int)
        return np.argmax(self._logits[indices], axis=1)

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        indices = np.asarray(inputs, dtype=int)
        return self._logits[indices]

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        raise UnsupportedCapabilityError("unused")


class PredictOnlyModel(ModelAdapter):
    def __init__(self, predictions: np.ndarray, num_classes: int = 2) -> None:
        self._predictions = predictions
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        indices = np.asarray(inputs, dtype=int)
        return self._predictions[indices]


def test_evaluation_result_structure() -> None:
    logits = np.array([[4.0, 0.1], [0.2, 3.0], [2.5, 0.1], [0.1, 2.0]], dtype=float)
    labels = np.array([0, 1, 0, 1])
    inputs = np.arange(len(labels))
    dataset = InMemoryDatasetAdapter(inputs, labels, sample_ids=["s0", "s1", "s2", "s3"])
    result = evaluate(LogitModel(logits), dataset, batch_size=2, class_names=["a", "b"], seed=42)

    assert isinstance(result, EvaluationResult)
    assert set(result.metrics) >= {"classification", "calibration", "uncertainty", "summary"}
    assert result.predictions.shape == (4,)
    assert result.labels.shape == (4,)
    assert result.sample_ids.shape == (4,)
    assert result.logits is not None and result.logits.shape == (4, 2)
    assert result.probabilities is not None and result.probabilities.shape == (4, 2)
    assert result.capabilities["predict_logits"] is True
    assert result.summary["total_samples"] == 4
    assert result.summary["accuracy"] == 1.0
    payload = result.to_predictions_dict()
    assert "logits" in payload and "probabilities" in payload


def test_evaluate_without_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRUSTBENCH_DB_PATH", str(tmp_path / "should_not_exist.db"))

    labels = np.array([0, 1, 0, 1])
    inputs = np.arange(len(labels))
    logits = np.array([[3.0, 0.1], [0.1, 2.0], [2.0, 0.2], [0.3, 2.5]])
    evaluate(LogitModel(logits), InMemoryDatasetAdapter(inputs, labels), batch_size=2)

    assert not (tmp_path / "should_not_exist.db").exists()
    assert list(tmp_path.glob("**/*.db")) == []


def test_evaluate_predict_only_skips_probability_metrics() -> None:
    labels = np.array([0, 1, 0])
    preds = np.array([0, 1, 1])
    result = evaluate(
        PredictOnlyModel(preds),
        InMemoryDatasetAdapter(np.arange(3), labels),
        metrics=("classification", "calibration", "uncertainty"),
    )
    assert result.logits is None
    assert result.probabilities is None
    assert result.metrics["calibration"]["skipped"] is True
    assert result.metrics["uncertainty"]["skipped"] is True
    assert result.metrics["classification"]["accuracy"] == pytest.approx(2 / 3)


def test_core_import_isolation() -> None:
    script = """
import sys
import trustbench.core

forbidden = (
    "fastapi",
    "sqlalchemy",
    "streamlit",
    "torch",
    "trustbench.api",
    "trustbench.storage",
    "trustbench.experiments",
)
loaded = [name for name in forbidden if name in sys.modules]
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_evaluate_does_not_import_fastapi() -> None:
    script = """
import sys
from trustbench.core import evaluate
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.models.base import ModelAdapter
import numpy as np

class M(ModelAdapter):
    @property
    def num_classes(self):
        return 2
    def predict(self, inputs):
        return np.zeros(len(inputs), dtype=int)

evaluate(M(), InMemoryDatasetAdapter(np.arange(3), np.array([0, 1, 0])))
assert "fastapi" not in sys.modules
assert "trustbench.api" not in sys.modules
assert "sqlalchemy" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_evaluate_batch_size_independence() -> None:
    # Evaluating with batch_size=1 vs batch_size=4 gives identical predictions and metrics
    np.random.seed(42)
    logits = np.random.randn(8, 3)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    sample_ids = np.array([f"sample_{i}" for i in range(8)])
    inputs = np.arange(8)

    model1 = LogitModel(logits)
    dataset1 = InMemoryDatasetAdapter(inputs, labels, sample_ids=sample_ids)
    res_b1 = evaluate(model1, dataset1, batch_size=1, seed=123)

    model2 = LogitModel(logits)
    dataset2 = InMemoryDatasetAdapter(inputs, labels, sample_ids=sample_ids)
    res_b4 = evaluate(model2, dataset2, batch_size=4, seed=123)

    np.testing.assert_array_equal(res_b1.predictions, res_b4.predictions)
    np.testing.assert_array_equal(res_b1.sample_ids, res_b4.sample_ids)
    np.testing.assert_allclose(res_b1.probabilities, res_b4.probabilities)
    assert res_b1.summary["accuracy"] == res_b4.summary["accuracy"]
    assert res_b1.summary["ece"] == pytest.approx(res_b4.summary["ece"])


def test_evaluate_deterministic_seed() -> None:
    # Evaluating with the same seed produces exact same result
    logits = np.array([[2.0, 1.0], [0.5, 1.5], [1.2, 0.8]])
    labels = np.array([0, 1, 0])
    inputs = np.arange(3)
    dataset = InMemoryDatasetAdapter(inputs, labels)

    res1 = evaluate(LogitModel(logits), dataset, seed=999)
    res2 = evaluate(LogitModel(logits), dataset, seed=999)

    np.testing.assert_array_equal(res1.predictions, res2.predictions)
    np.testing.assert_allclose(res1.probabilities, res2.probabilities)


def test_evaluate_empty_dataset() -> None:
    model = LogitModel(np.empty((0, 2)))
    dataset = InMemoryDatasetAdapter(
        np.empty((0,)), np.empty((0,), dtype=int), sample_ids=np.empty((0,), dtype=str)
    )
    result = evaluate(model, dataset)
    assert len(result.predictions) == 0
    assert len(result.labels) == 0
    assert result.summary["total_samples"] == 0


def test_checkpoint_sha256_computation(tmp_path: Path) -> None:
    import hashlib

    from trustbench.reporting.hashing import compute_file_sha256

    test_file = tmp_path / "model.pt"
    content = b"trustbench test checkpoint bytes 12345"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    computed_hash = compute_file_sha256(test_file)
    assert computed_hash == expected_hash
