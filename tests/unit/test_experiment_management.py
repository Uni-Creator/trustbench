"""Comprehensive unit tests for experiment lifecycle, storage backends, and reproducibility."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from trustbench.core.results import EvaluationResult
from trustbench.experiments.comparison import (
    ExperimentComparator,
    validate_compatibility,
)
from trustbench.experiments.lineage import (
    MISSING,
    UNAVAILABLE,
    build_lineage_record,
    compute_checkpoint_sha256,
    compute_dataset_fingerprint,
)
from trustbench.experiments.manager import ExperimentManager
from trustbench.registry.datasets import GenericDatasetAdapter
from trustbench.registry.models import GenericModelAdapter
from trustbench.reporting.hashing import compute_file_sha256
from trustbench.storage.base import (
    CorruptedArtifactError,
    ExperimentAlreadyExistsError,
    ExperimentRecord,
    ExperimentState,
    IncompatibleExperimentsError,
)
from trustbench.storage.filesystem import (
    FilesystemExperimentStore,
    _atomic_write_json,
    _atomic_write_npz,
)

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_fs_store(tmp_path: Path) -> FilesystemExperimentStore:
    return FilesystemExperimentStore(base_dir=tmp_path / "experiments")


@pytest.fixture
def dummy_result() -> EvaluationResult:
    preds = np.array([0, 1, 0, 1])
    labels = np.array([0, 1, 0, 1])
    sample_ids = np.array([0, 1, 2, 3])
    metrics = {
        "classification": {"accuracy": 1.0, "macro_f1": 1.0},
        "calibration": {"ece": 0.05, "brier_score": 0.02},
        "summary": {"accuracy": 1.0, "macro_f1": 1.0, "ece": 0.05, "brier_score": 0.02},
    }
    return EvaluationResult(
        metrics=metrics,
        predictions=preds,
        labels=labels,
        sample_ids=sample_ids,
        metadata={"model": {"num_classes": 2}, "seed": 42},
    )


# ---------------------------------------------------------------------------
# 1. Persistence across restarts
# ---------------------------------------------------------------------------


def test_filesystem_store_persistence_across_restarts(
    tmp_path: Path, dummy_result: EvaluationResult
):
    base_dir = tmp_path / "persistence_store"
    store1 = FilesystemExperimentStore(base_dir=base_dir)

    exp1 = store1.create("run_alpha", experiment_type="evaluation", config={"batch_size": 32})
    store1.save_result(exp1.id, dummy_result)

    # Simulate restart by creating a completely new store instance pointing to same dir
    store2 = FilesystemExperimentStore(base_dir=base_dir)
    recovered = store2.get(exp1.id)

    assert recovered is not None
    assert recovered.id == exp1.id
    assert recovered.name == "run_alpha"
    assert recovered.status == ExperimentState.COMPLETED.value
    assert recovered.metrics["summary"]["accuracy"] == 1.0
    assert recovered.config["batch_size"] == 32


# ---------------------------------------------------------------------------
# 2. Duplicate ID handling
# ---------------------------------------------------------------------------


def test_duplicate_experiment_id_handling(temp_fs_store: FilesystemExperimentStore):
    exp_id = "custom-uuid-001"
    temp_fs_store.create("first_run", experiment_id=exp_id)

    # Creating another experiment with the same ID must raise without overwrite
    with pytest.raises(ExperimentAlreadyExistsError, match="already exists"):
        temp_fs_store.create("second_run", experiment_id=exp_id, overwrite=False)

    # With overwrite=True, it succeeds
    overwritten = temp_fs_store.create("second_run", experiment_id=exp_id, overwrite=True)
    assert overwritten.id == exp_id
    assert overwritten.name == "second_run"


# ---------------------------------------------------------------------------
# 3. Interrupted writes & atomic safety
# ---------------------------------------------------------------------------


def test_atomic_write_and_interrupted_write_safety(tmp_path: Path):
    target_json = tmp_path / "data.json"
    _atomic_write_json(target_json, {"state": "initial"})
    assert target_json.read_text().startswith("{\n  \"state\": \"initial\"")

    # Overwrite atomically
    _atomic_write_json(target_json, {"state": "updated"})
    assert "updated" in target_json.read_text()

    # Verify no dangling temp files remain
    temp_files = list(tmp_path.glob("*.tmp*"))
    assert len(temp_files) == 0

    # Atomic npz
    target_npz = tmp_path / "arrays.npz"
    _atomic_write_npz(target_npz, {"a": np.array([1, 2, 3])})
    assert target_npz.is_file()
    loaded = np.load(target_npz)
    assert (loaded["a"] == np.array([1, 2, 3])).all()


# ---------------------------------------------------------------------------
# 4. Corrupted artifacts & checksum verification
# ---------------------------------------------------------------------------


def test_corrupted_artifacts_detected(
    temp_fs_store: FilesystemExperimentStore, dummy_result: EvaluationResult
):
    exp = temp_fs_store.create("tamper_test")
    temp_fs_store.save_result(exp.id, dummy_result)

    # All checksums should initially be valid
    checks = temp_fs_store.verify_checksums(exp.id)
    assert all(checks.values()) is True

    # Tamper with metrics.json
    metrics_file = temp_fs_store._exp_dir(exp.id) / "metrics.json"
    with open(metrics_file, "a") as f:
        f.write("\n/* tampered */")

    # verify_checksums must now flag metrics.json as corrupted
    checks_after = temp_fs_store.verify_checksums(exp.id)
    assert checks_after["metrics.json"] is False
    assert checks_after["metadata.json"] is True


# ---------------------------------------------------------------------------
# 5. Experiment retrieval and filtering
# ---------------------------------------------------------------------------


def test_experiment_retrieval_and_filtering(temp_fs_store: FilesystemExperimentStore):
    e1 = temp_fs_store.create("b1", experiment_type="baseline", metadata={"checkpoint_sha256": "sha_a"})
    temp_fs_store.update_status(e1.id, ExperimentState.COMPLETED)

    e2 = temp_fs_store.create("b2", experiment_type="baseline", metadata={"checkpoint_sha256": "sha_b"})
    temp_fs_store.update_status(e2.id, ExperimentState.FAILED, error="OOM")

    e3 = temp_fs_store.create("r1", experiment_type="robustness", metadata={"checkpoint_sha256": "sha_a"})
    temp_fs_store.update_status(e3.id, ExperimentState.COMPLETED)

    # Filter by type
    baselines = temp_fs_store.list(experiment_type="baseline")
    assert len(baselines) == 2

    # Filter by status
    failed = temp_fs_store.list(status="failed")
    assert len(failed) == 1
    assert failed[0].name == "b2"

    # Filter by checkpoint sha
    sha_a_runs = temp_fs_store.list(checkpoint_sha256="sha_a")
    assert len(sha_a_runs) == 2


# ---------------------------------------------------------------------------
# 6. Compatibility validation
# ---------------------------------------------------------------------------


def test_compatibility_validation():
    # Incompatible number of classes
    exp_10 = ExperimentRecord(
        id="1",
        name="exp_cifar10",
        experiment_type="evaluation",
        status="completed",
        created_at="",
        metadata={"model": {"num_classes": 10}},
        metrics={"summary": {"accuracy": 0.9}},
    )
    exp_100 = ExperimentRecord(
        id="2",
        name="exp_cifar100",
        experiment_type="evaluation",
        status="completed",
        created_at="",
        metadata={"model": {"num_classes": 100}},
        metrics={"summary": {"accuracy": 0.7}},
    )

    with pytest.raises(IncompatibleExperimentsError, match="classes"):
        validate_compatibility(exp_10, exp_100)

    # Disjoint metrics
    exp_a = ExperimentRecord(
        id="a",
        name="a",
        experiment_type="evaluation",
        status="completed",
        created_at="",
        metadata={"model": {"num_classes": 2}},
        metrics={"summary": {"accuracy": 0.9}},
    )
    exp_b = ExperimentRecord(
        id="b",
        name="b",
        experiment_type="evaluation",
        status="completed",
        created_at="",
        metadata={"model": {"num_classes": 2}},
        metrics={"summary": {"brier_score": 0.1}},
    )
    with pytest.raises(IncompatibleExperimentsError, match="overlapping evaluation metrics"):
        validate_compatibility(exp_a, exp_b)

    # Compatible
    exp_c = ExperimentRecord(
        id="c",
        name="c",
        experiment_type="evaluation",
        status="completed",
        created_at="",
        metadata={"model": {"num_classes": 2}},
        metrics={"summary": {"accuracy": 0.85, "brier_score": 0.15}},
    )
    res = validate_compatibility(exp_a, exp_c)
    assert res["compatible"] is True
    assert "accuracy" in res["shared_metrics"]


# ---------------------------------------------------------------------------
# 7. Hash verification (checkpoints, dataset fingerprint, artifacts)
# ---------------------------------------------------------------------------


def test_hash_verification(tmp_path: Path):
    ckpt_file = tmp_path / "model.pt"
    ckpt_file.write_bytes(b"dummy_weights_12345")

    expected_sha = compute_file_sha256(ckpt_file)
    assert compute_checkpoint_sha256(ckpt_file) == expected_sha
    assert compute_checkpoint_sha256(tmp_path / "missing.pt") == MISSING
    assert compute_checkpoint_sha256(None) == UNAVAILABLE

    # Dataset fingerprint
    ds1 = GenericDatasetAdapter(samples=[1, 2, 3], labels=[0, 1, 0], sample_ids=[10, 11, 12])
    ds2 = GenericDatasetAdapter(samples=[1, 2, 3], labels=[0, 1, 0], sample_ids=[10, 11, 12])
    ds3 = GenericDatasetAdapter(samples=[1, 2, 3], labels=[1, 1, 0], sample_ids=[10, 11, 12])

    fp1 = compute_dataset_fingerprint(ds1)
    fp2 = compute_dataset_fingerprint(ds2)
    fp3 = compute_dataset_fingerprint(ds3)

    assert fp1 != UNAVAILABLE
    assert fp1 == fp2  # Deterministic equality
    assert fp1 != fp3  # Different labels produce different fingerprint


# ---------------------------------------------------------------------------
# 8. Database-free evaluation & import isolation
# ---------------------------------------------------------------------------


def test_database_free_evaluation(tmp_path: Path):
    script = f"""
import sys
from trustbench.experiments.manager import ExperimentManager
from trustbench.registry.models import GenericModelAdapter
from trustbench.registry.datasets import GenericDatasetAdapter
import numpy as np

# Verify forbidden modules not loaded
forbidden = ('sqlalchemy', 'fastapi', 'torch', 'streamlit')
assert not [n for n in forbidden if n in sys.modules], "Forbidden module loaded before run"

manager = ExperimentManager(backend="filesystem", base_dir="{tmp_path}/db_free")

model = GenericModelAdapter(predict_fn=lambda x: np.array([0, 1]), num_classes=2)
dataset = GenericDatasetAdapter(samples=[[0], [1]], labels=[0, 1])

exp = manager.run_experiment(model, dataset, name="pure_fs_eval", metrics=["classification"])
assert exp.status == "completed"
assert exp.metrics["summary"]["accuracy"] == 1.0

# Verify forbidden modules still not loaded after evaluation
assert not [n for n in forbidden if n in sys.modules], "Forbidden module loaded during run"
print("SUCCESS")
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SUCCESS" in proc.stdout


# ---------------------------------------------------------------------------
# 9. Experiment lifecycle failure handling
# ---------------------------------------------------------------------------


def test_experiment_lifecycle_failure_transition(temp_fs_store: FilesystemExperimentStore):
    manager = ExperimentManager(store=temp_fs_store)

    failing_model = GenericModelAdapter(
        predict_fn=lambda x: (_ for _ in ()).throw(RuntimeError("Inference engine crashed!")),
        num_classes=2,
    )
    dataset = GenericDatasetAdapter(samples=[[1], [2]], labels=[0, 1])

    with pytest.raises(RuntimeError, match="Inference engine crashed"):
        manager.run_experiment(failing_model, dataset, name="failing_run", raise_on_error=True)

    failed_rec = manager.get_experiment("failing_run")
    assert failed_rec is not None
    assert failed_rec.status == ExperimentState.FAILED.value
    assert "Inference engine crashed" in str(failed_rec.error)
    assert failed_rec.completed_at is not None


# ---------------------------------------------------------------------------
# 10. Portable export and import roundtrip
# ---------------------------------------------------------------------------


def test_export_and_import_roundtrip(tmp_path: Path, dummy_result: EvaluationResult):
    store_a = FilesystemExperimentStore(base_dir=tmp_path / "store_a")
    store_b = FilesystemExperimentStore(base_dir=tmp_path / "store_b")

    exp = store_a.create("portable_exp", config={"alpha": 123}, metadata={"seed": 42})
    store_a.save_result(exp.id, dummy_result)

    archive_path = tmp_path / "portable_exp.tar.gz"
    store_a.export(exp.id, archive_path)
    assert archive_path.is_file()

    # Import into fresh store_b
    imported = store_b.import_experiment(archive_path)
    assert imported.id == exp.id
    assert imported.name == "portable_exp"
    assert imported.metrics["summary"]["accuracy"] == 1.0
    assert imported.config["alpha"] == 123
    assert store_b.verify_checksums(imported.id)["metrics.json"] is True

    # Tamper with archive and verify import fails
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=tmp_path / "tamper_extract", filter="data")
    (tmp_path / "tamper_extract" / "metrics.json").write_text('{"hacked": true}')

    tampered_archive = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered_archive, "w:gz") as tar:
        for f in (tmp_path / "tamper_extract").iterdir():
            tar.add(f, arcname=f.name)

    with pytest.raises(CorruptedArtifactError, match="checksum mismatch"):
        store_b.import_experiment(tampered_archive, overwrite=True)


# ---------------------------------------------------------------------------
# 11. Experiment deletion
# ---------------------------------------------------------------------------


def test_experiment_deletion(temp_fs_store: FilesystemExperimentStore, dummy_result: EvaluationResult):
    exp = temp_fs_store.create("to_delete")
    temp_fs_store.save_result(exp.id, dummy_result)

    assert temp_fs_store.get(exp.id) is not None
    exp_dir = temp_fs_store._exp_dir(exp.id)
    assert exp_dir.is_dir()

    deleted = temp_fs_store.delete(exp.id, delete_files=True)
    assert deleted is True
    assert temp_fs_store.get(exp.id) is None
    assert not exp_dir.exists()


# ---------------------------------------------------------------------------
# 12. Lineage and metadata completeness
# ---------------------------------------------------------------------------


def test_lineage_and_metadata_completeness():
    model = GenericModelAdapter(predict_fn=lambda x: np.array([0, 1]), num_classes=2)
    dataset = GenericDatasetAdapter(samples=[[0], [1]], labels=[0, 1])

    record = build_lineage_record(
        model=model,
        dataset=dataset,
        config={"lr": 0.001},
        checkpoint_path=None,
        seed=123,
    )

    assert record["schema_version"] == "1.0"
    assert record["random_seeds"]["seed"] == 123
    assert record["checkpoint_sha256"] == UNAVAILABLE
    assert record["dataset"]["num_samples"] == 2
    assert record["dataset"]["dataset_fingerprint"] != UNAVAILABLE
    assert record["environment"]["python"] == sys.version.split()[0]
    assert "hardware" in record
    assert record["hardware"]["platform"] is not None


# ---------------------------------------------------------------------------
# 13. Comparison with deltas
# ---------------------------------------------------------------------------


def test_experiment_comparator_deltas(temp_fs_store: FilesystemExperimentStore):
    res_a = EvaluationResult(
        metrics={"summary": {"accuracy": 0.80, "ece": 0.15}},
        predictions=np.array([0]),
        labels=np.array([0]),
        sample_ids=np.array([0]),
        metadata={"model": {"num_classes": 2}},
    )
    res_b = EvaluationResult(
        metrics={"summary": {"accuracy": 0.90, "ece": 0.05}},
        predictions=np.array([0]),
        labels=np.array([0]),
        sample_ids=np.array([0]),
        metadata={"model": {"num_classes": 2}},
    )

    exp_a = temp_fs_store.create("model_v1")
    temp_fs_store.save_result(exp_a.id, res_a)

    exp_b = temp_fs_store.create("model_v2")
    temp_fs_store.save_result(exp_b.id, res_b)

    comparator = ExperimentComparator(store=temp_fs_store)
    comparison = comparator.compare([exp_a.id, exp_b.id])

    assert len(comparison["experiment_ids"]) == 2
    assert "Clean" in comparison["comparison_table"]

    deltas = comparison["deltas"]["Clean"]
    # Accuracy higher is better -> model_v2 won
    assert deltas["accuracy"]["absolute_difference"] == pytest.approx(0.10)
    assert deltas["accuracy"]["preferred"] == "model_v2"

    # ECE lower is better -> model_v2 won
    assert deltas["ece"]["absolute_difference"] == pytest.approx(-0.10)
    assert deltas["ece"]["preferred"] == "model_v2"
