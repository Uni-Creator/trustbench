"""End-to-end integration tests for TrustBench Phase 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from trustbench import EvaluationResult, TrustBench
from trustbench.datasets.torch_dataset import TorchDatasetAdapter
from trustbench.reporting.artifacts import load_experiment_artifacts
from trustbench.reporting.result_io import save_result

# Helper: synthetic Dataset (inlined so test file is self-contained)



class _LinearDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 50, n_features: int = 4, n_classes: int = 3) -> None:
        torch.manual_seed(42)
        self.data = torch.randn(n, n_features)
        self.targets = torch.randint(0, n_classes, (n,)).tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


@pytest.fixture()
def tiny_linear_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(4, 3)


@pytest.fixture()
def synthetic_dataset() -> _LinearDataset:
    return _LinearDataset(n=50, n_features=4, n_classes=3)



# Test: TorchDatasetAdapter



class TestTorchDatasetAdapter:
    def test_len(self, synthetic_dataset: _LinearDataset) -> None:
        adapter = TorchDatasetAdapter(synthetic_dataset)
        assert len(adapter) == 50

    def test_get_labels_shape(self, synthetic_dataset: _LinearDataset) -> None:
        adapter = TorchDatasetAdapter(synthetic_dataset)
        labels = adapter.get_labels()
        assert labels.shape == (50,)
        assert labels.dtype in (np.int32, np.int64, np.intp, int, np.dtype("int64"), np.dtype("int32"))

    def test_get_sample_ids_default(self, synthetic_dataset: _LinearDataset) -> None:
        adapter = TorchDatasetAdapter(synthetic_dataset)
        ids = adapter.get_sample_ids()
        assert ids.shape == (50,)
        assert ids[0] == 0
        assert ids[-1] == 49

    def test_get_sample_ids_explicit(self, synthetic_dataset: _LinearDataset) -> None:
        custom_ids = list(range(100, 150))
        adapter = TorchDatasetAdapter(synthetic_dataset, sample_ids=custom_ids)
        ids = adapter.get_sample_ids()
        assert list(ids) == custom_ids

    def test_get_batch_shape(self, synthetic_dataset: _LinearDataset) -> None:
        adapter = TorchDatasetAdapter(synthetic_dataset)
        batch = adapter.get_batch([0, 1, 2])
        assert batch.shape == (3, 4)

    def test_mismatched_sample_ids_raises(self, synthetic_dataset: _LinearDataset) -> None:
        with pytest.raises(ValueError, match="sample_ids length"):
            TorchDatasetAdapter(synthetic_dataset, sample_ids=[1, 2, 3])

    def test_targets_attribute(self, synthetic_dataset: _LinearDataset) -> None:
        """When dataset has .targets, labels come from it without iteration."""
        adapter = TorchDatasetAdapter(synthetic_dataset)
        labels = adapter.get_labels()
        expected = np.asarray(synthetic_dataset.targets)
        np.testing.assert_array_equal(labels, expected)



# Test: TrustBench with nn.Module



class TestTrustBenchWithModule:
    def test_evaluate_returns_evaluation_result(
        self, tiny_linear_model: nn.Module, synthetic_dataset: _LinearDataset
    ) -> None:
        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        assert isinstance(result, EvaluationResult)

    def test_result_has_summary(
        self, tiny_linear_model: nn.Module, synthetic_dataset: _LinearDataset
    ) -> None:
        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        summary = result.summary
        assert "accuracy" in summary
        assert "macro_f1" in summary
        assert isinstance(summary["accuracy"], float)

    def test_result_predictions_shape(
        self, tiny_linear_model: nn.Module, synthetic_dataset: _LinearDataset
    ) -> None:
        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        assert result.predictions.shape == (50,)
        assert result.labels.shape == (50,)

    def test_result_has_probabilities(
        self, tiny_linear_model: nn.Module, synthetic_dataset: _LinearDataset
    ) -> None:
        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        assert result.probabilities is not None
        assert result.probabilities.shape == (50, 3)

    def test_with_seed_is_reproducible(
        self, tiny_linear_model: nn.Module, synthetic_dataset: _LinearDataset
    ) -> None:
        tb1 = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3, seed=7)
        tb2 = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3, seed=7)
        r1 = tb1.evaluate()
        r2 = tb2.evaluate()
        np.testing.assert_array_equal(r1.predictions, r2.predictions)

    def test_with_dataloader(
        self, tiny_linear_model: nn.Module, synthetic_dataset: _LinearDataset
    ) -> None:
        from torch.utils.data import DataLoader

        loader = DataLoader(synthetic_dataset, batch_size=16)
        tb = TrustBench(tiny_linear_model, loader, num_classes=3)
        result = tb.evaluate()
        assert result.predictions.shape == (50,)

    def test_no_num_classes_raises(
        self, tiny_linear_model: nn.Module, synthetic_dataset: _LinearDataset
    ) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            TrustBench(tiny_linear_model, synthetic_dataset)

    def test_bad_model_type_raises(self, synthetic_dataset: _LinearDataset) -> None:
        with pytest.raises(TypeError):
            TrustBench("not_a_model", synthetic_dataset)



# Test: save_result + load_experiment_artifacts



class TestSaveResult:
    def test_artifacts_created(
        self,
        tmp_path: Path,
        tiny_linear_model: nn.Module,
        synthetic_dataset: _LinearDataset,
    ) -> None:
        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        save_result(result, tmp_path)

        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "predictions.npz").exists()

    def test_artifacts_round_trip(
        self,
        tmp_path: Path,
        tiny_linear_model: nn.Module,
        synthetic_dataset: _LinearDataset,
    ) -> None:
        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        save_result(result, tmp_path)

        artifacts = load_experiment_artifacts(tmp_path)
        loaded_metrics = artifacts["metrics"]
        assert "summary" in loaded_metrics
        # accuracy should round-trip exactly
        orig_acc = result.summary.get("accuracy")
        loaded_acc = loaded_metrics["summary"].get("accuracy")
        if orig_acc is not None and loaded_acc is not None:
            assert abs(orig_acc - loaded_acc) < 1e-6

    def test_predictions_npz_content(
        self,
        tmp_path: Path,
        tiny_linear_model: nn.Module,
        synthetic_dataset: _LinearDataset,
    ) -> None:
        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        save_result(result, tmp_path)

        artifacts = load_experiment_artifacts(tmp_path)
        preds = artifacts["predictions"]
        np.testing.assert_array_equal(preds["predictions"], result.predictions)
        np.testing.assert_array_equal(preds["labels"], result.labels)

    def test_output_dir_auto_save(
        self,
        tmp_path: Path,
        tiny_linear_model: nn.Module,
        synthetic_dataset: _LinearDataset,
    ) -> None:
        """When output_dir is passed to TrustBench, artifacts are saved automatically."""
        out = tmp_path / "auto_save"
        tb = TrustBench(
            tiny_linear_model, synthetic_dataset, num_classes=3, output_dir=out
        )
        tb.evaluate()
        assert (out / "metrics.json").exists()



# Test: Config system



class TestConfigSystem:
    def test_load_valid_config(self, tmp_path: Path, minimal_config_yaml: str) -> None:
        from trustbench.config.loader import load_config

        cfg_path = tmp_path / "trustbench.yaml"
        cfg_path.write_text(minimal_config_yaml, encoding="utf-8")
        cfg = load_config(cfg_path)
        assert cfg.experiment_name == "test_run"
        assert cfg.seed == 42
        assert cfg.model.num_classes == 3

    def test_validate_valid_config(
        self, tmp_path: Path, minimal_config_yaml: str
    ) -> None:
        from trustbench.config.loader import validate_config

        cfg_path = tmp_path / "trustbench.yaml"
        cfg_path.write_text(minimal_config_yaml, encoding="utf-8")
        errors = validate_config(cfg_path)
        assert errors == []

    def test_validate_invalid_config(self, tmp_path: Path) -> None:
        from trustbench.config.loader import validate_config

        bad_yaml = "config_version: 1\nmodel:\n  num_classes: not_an_int\n"
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(bad_yaml, encoding="utf-8")
        errors = validate_config(cfg_path)
        assert len(errors) > 0

    def test_missing_config_file(self, tmp_path: Path) -> None:
        from trustbench.config.loader import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_extra_fields_forbidden(self, tmp_path: Path) -> None:
        from trustbench.config.loader import validate_config

        yaml_with_extra = (
            "config_version: 1\nmodel:\n  num_classes: 3\n  unknown_field: oops\n"
        )
        cfg_path = tmp_path / "extra.yaml"
        cfg_path.write_text(yaml_with_extra, encoding="utf-8")
        errors = validate_config(cfg_path)
        assert len(errors) > 0

    def test_config_version_int_coerced(
        self, tmp_path: Path
    ) -> None:
        """config_version: 1 (int) should be coerced to '1' (str) silently."""
        from trustbench.config.loader import load_config

        yaml_text = "config_version: 1\nexperiment_name: coerce_test\n"
        cfg_path = tmp_path / "coerce.yaml"
        cfg_path.write_text(yaml_text, encoding="utf-8")
        cfg = load_config(cfg_path)
        assert cfg.config_version == "1"



# Test: HTML report



class TestHtmlReport:
    def test_generates_report_html(
        self,
        tmp_path: Path,
        tiny_linear_model: nn.Module,
        synthetic_dataset: _LinearDataset,
    ) -> None:
        from trustbench.reporting.html_report import generate_html_report

        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        report_path = generate_html_report(result, tmp_path)

        assert report_path.exists()
        assert report_path.suffix == ".html"
        content = report_path.read_text(encoding="utf-8")
        assert "TrustBench" in content
        assert "Accuracy" in content

    def test_report_contains_summary_values(
        self,
        tmp_path: Path,
        tiny_linear_model: nn.Module,
        synthetic_dataset: _LinearDataset,
    ) -> None:
        from trustbench.reporting.html_report import generate_html_report

        tb = TrustBench(tiny_linear_model, synthetic_dataset, num_classes=3)
        result = tb.evaluate()
        report_path = generate_html_report(result, tmp_path)

        content = report_path.read_text(encoding="utf-8")
        # Accuracy value should appear somewhere in the HTML
        acc = result.summary.get("accuracy")
        if acc is not None:
            assert f"{acc:.4f}" in content



# Fixture (re-exposed from conftest but also here as a local fallback)



@pytest.fixture()
def minimal_config_yaml() -> str:
    return """\
config_version: 1
experiment_name: test_run
seed: 42

model:
  num_classes: 3

dataset:
  dataset_type: cifar10
  batch_size: 50

metrics:
  groups:
    - classification
    - calibration
    - uncertainty

reporting:
  output_dir: trustbench_output
  save_html: true
  save_json: true
"""
