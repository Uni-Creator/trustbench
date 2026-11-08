"""Tests for TrustBench Phase 7 — Performance and Production Readiness.

Coverage:
- Streaming & incremental metric aggregation (StreamingMetricTracker).
- Equivalence of streaming metrics vs full batch calculation.
- Metric retention classification (METRIC_RETENTION_REQUIREMENTS).
- Atomic artifact writes and failure cleanup.
- Checkpoint loading security (weights_only=True).
- Archive import security (path traversal prevention).
- Interrupted evaluation handling (ExperimentState.FAILED on exception / KeyboardInterrupt).
- Memory-efficient pre-allocated inference output shapes.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from trustbench.core.exceptions import SecurityError
from trustbench.core.inference import run_adapter_inference
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.evaluation.streaming import (
    METRIC_RETENTION_REQUIREMENTS,
    StreamingMetricTracker,
)
from trustbench.experiments.manager import ExperimentManager
from trustbench.models.checkpoint import load_model_from_checkpoint
from trustbench.models.pytorch import PyTorchModelAdapter
from trustbench.reporting.artifacts import _atomic_write_file, save_experiment_artifacts
from trustbench.storage.base import ExperimentState
from trustbench.storage.filesystem import FilesystemExperimentStore

# 1. Streaming & Incremental Metric Aggregation



class TestStreamingMetrics:
    def test_streaming_retention_classification_complete(self):
        """All core metrics must be classified as either 'streaming' or 'full_retention'."""
        assert "accuracy" in METRIC_RETENTION_REQUIREMENTS
        assert METRIC_RETENTION_REQUIREMENTS["accuracy"] == "streaming"
        assert METRIC_RETENTION_REQUIREMENTS["brier_score"] == "streaming"
        assert METRIC_RETENTION_REQUIREMENTS["auroc"] == "full_retention"
        assert METRIC_RETENTION_REQUIREMENTS["adaptive_ece"] == "full_retention"
        assert METRIC_RETENTION_REQUIREMENTS["aurc"] == "full_retention"

    def test_streaming_tracker_equivalence_to_full_batch(self):
        """Streaming aggregation in 4 chunks must match single-batch calculation."""
        num_classes = 5
        n_samples = 200
        rng = np.random.default_rng(42)

        preds = rng.integers(0, num_classes, size=n_samples)
        lbls = rng.integers(0, num_classes, size=n_samples)
        logits = rng.standard_normal((n_samples, num_classes))
        # Compute softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Full-batch tracker
        full_tracker = StreamingMetricTracker(num_classes=num_classes)
        full_tracker.update(preds, lbls, probs)
        full_results = full_tracker.compute()

        # Chunked streaming tracker (50 samples per chunk)
        chunk_tracker = StreamingMetricTracker(num_classes=num_classes)
        chunk_size = 50
        for i in range(0, n_samples, chunk_size):
            chunk_tracker.update(
                preds[i : i + chunk_size],
                lbls[i : i + chunk_size],
                probs[i : i + chunk_size],
            )
        chunk_results = chunk_tracker.compute()

        assert chunk_results["total_samples"] == full_results["total_samples"]
        assert chunk_results["accuracy"] == pytest.approx(full_results["accuracy"], rel=1e-6)
        assert chunk_results["brier_score"] == pytest.approx(full_results["brier_score"], rel=1e-6)
        assert chunk_results["mean_entropy"] == pytest.approx(full_results["mean_entropy"], rel=1e-6)
        assert chunk_results["mean_max_probability"] == pytest.approx(
            full_results["mean_max_probability"], rel=1e-6
        )
        assert chunk_results["confusion_matrix"] == full_results["confusion_matrix"]

    def test_streaming_tracker_empty_input(self):
        """Empty streaming tracker must return zero metrics without error."""
        tracker = StreamingMetricTracker(num_classes=3)
        res = tracker.compute()
        assert res["total_samples"] == 0
        assert res["accuracy"] == 0.0



# 2. Atomic Writes & Reliability



class TestAtomicWrites:
    def test_atomic_write_file_success(self):
        """Atomic write creates the target file cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "output.txt"
            _atomic_write_file(target, lambda p: p.write_text("success", encoding="utf-8"))
            assert target.is_file()
            assert target.read_text(encoding="utf-8") == "success"
            # Ensure no .tmp leftover files
            leftovers = list(Path(tmp_dir).glob(".tmp_*"))
            assert len(leftovers) == 0

    def test_atomic_write_file_failure_cleans_up_temp(self):
        """If write fails mid-way, temporary file is deleted and target is untouched."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "corrupt_test.txt"

            def failing_write(path: Path):
                path.write_text("partial data", encoding="utf-8")
                raise RuntimeError("Simulated crash")

            with pytest.raises(RuntimeError, match="Simulated crash"):
                _atomic_write_file(target, failing_write)

            assert not target.exists()
            # Ensure no partial .tmp files remain on disk
            leftovers = list(Path(tmp_dir).glob(".tmp_*"))
            assert len(leftovers) == 0

    def test_save_experiment_artifacts_atomic(self):
        """save_experiment_artifacts creates all four artifact files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "run_01"
            save_experiment_artifacts(
                output_dir=out,
                metadata={"test": True},
                metrics={"accuracy": 0.95},
                config={"batch_size": 32},
                predictions={
                    "predictions": np.array([0, 1]),
                    "labels": np.array([0, 1]),
                    "sample_ids": np.array([0, 1]),
                },
            )
            assert (out / "metadata.json").is_file()
            assert (out / "metrics.json").is_file()
            assert (out / "config.json").is_file()
            assert (out / "predictions.npz").is_file()
            leftovers = list(out.glob(".tmp_*"))
            assert len(leftovers) == 0



# 3. Security: Checkpoint & Archive Verification



class TestSecurity:
    def test_checkpoint_loading_weights_only(self):
        """Trained checkpoint must load successfully with weights_only=True."""
        ckpt_path = Path("checkpoint/ckpt.pth")
        if not ckpt_path.is_file():
            pytest.skip("Checkpoint file not present.")

        model, raw = load_model_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_name="ResNet18",
            device="cpu",
            weights_only=True,
        )
        assert isinstance(model, torch.nn.Module)
        assert "net" in raw

    def test_archive_path_traversal_rejected(self):
        """Importing an archive with path traversal ('../') must raise SecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            evil_tar_path = Path(tmp_dir) / "evil.tar.gz"

            # Create an in-memory tarball containing an entry with path traversal
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
                # Malicious entry that attempts to write outside extraction dir
                evil_info = tarfile.TarInfo(name="../../evil.txt")
                evil_info.size = 12
                tar.addfile(evil_info, io.BytesIO(b"malicious!!!"))

            evil_tar_path.write_bytes(tar_stream.getvalue())

            store = FilesystemExperimentStore(base_dir=tmp_dir)
            with pytest.raises(SecurityError, match="path traversal"):
                store.import_experiment(evil_tar_path)



# 4. Reliability: Interrupted Evaluation Handling



class TestFailureHandling:
    def test_interrupted_evaluation_marks_experiment_failed(self):
        """KeyboardInterrupt during run_experiment must set status to FAILED."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ExperimentManager(backend="filesystem", base_dir=tmp_dir)

            model = PyTorchModelAdapter(torch.nn.Linear(2, 2), num_classes=2)
            dataset = InMemoryDatasetAdapter(inputs=torch.randn(10, 2), labels=np.zeros(10, dtype=int))

            with (
                patch("trustbench.core.evaluator.evaluate", side_effect=KeyboardInterrupt("User stopped")),
                pytest.raises(KeyboardInterrupt),
            ):
                mgr.run_experiment(
                    model=model,
                    dataset=dataset,
                    name="interrupted_run",
                    raise_on_error=True,
                )

            # Retrieve experiment record from store
            record = mgr.get_experiment("interrupted_run")
            assert record is not None
            assert record.status == ExperimentState.FAILED
            assert "KeyboardInterrupt" in str(record.error)

    def test_exception_during_evaluation_marks_experiment_failed(self):
        """Unhandled exception during evaluation must mark experiment as FAILED."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ExperimentManager(backend="filesystem", base_dir=tmp_dir)

            model = PyTorchModelAdapter(torch.nn.Linear(2, 2), num_classes=2)
            dataset = InMemoryDatasetAdapter(inputs=torch.randn(10, 2), labels=np.zeros(10, dtype=int))

            with (
                patch("trustbench.core.evaluator.evaluate", side_effect=ValueError("CUDA out of memory")),
                pytest.raises(ValueError, match="CUDA out of memory"),
            ):
                mgr.run_experiment(
                    model=model,
                    dataset=dataset,
                    name="failing_run",
                    raise_on_error=True,
                )

            record = mgr.get_experiment("failing_run")
            assert record is not None
            assert record.status == ExperimentState.FAILED
            assert "CUDA out of memory" in str(record.error)



# 5. Pre-allocated Inference Output



class TestInferencePreallocation:
    def test_run_adapter_inference_preallocation(self):
        """Inference with pre-allocation correctly shapes output arrays."""
        model = MagicMock()
        model.supports_predict_logits = True
        model.supports_predict_proba = True
        model.num_classes = 3
        model.predict_logits.side_effect = lambda b: np.ones((len(b), 3))

        samples = torch.zeros(15, 3)
        labels = np.zeros(15, dtype=int)
        adapter = InMemoryDatasetAdapter(inputs=samples, labels=labels)

        output = run_adapter_inference(model, adapter, batch_size=4)
        assert output.predictions.shape == (15,)
        assert output.logits is not None
        assert output.logits.shape == (15, 3)
        assert output.probabilities is not None
        assert output.probabilities.shape == (15, 3)
