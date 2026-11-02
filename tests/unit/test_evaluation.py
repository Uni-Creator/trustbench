"""Unit tests for trustbench.evaluation."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trustbench.evaluation.artifacts import (
    load_experiment_artifacts,
    save_experiment_artifacts,
)
from trustbench.evaluation.inference import (
    IndexedDataset,
    clean_state_dict,
    load_model_from_checkpoint,
    run_inference,
)
from trustbench.evaluation.reproducibility import get_environment_info, set_seed


class SimpleNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_clean_state_dict() -> None:
    dp_dict = {"module.linear.weight": torch.ones(2, 2), "module.linear.bias": torch.zeros(2)}
    cleaned = clean_state_dict(dp_dict)
    assert "linear.weight" in cleaned
    assert "linear.bias" in cleaned
    assert "module.linear.weight" not in cleaned


def test_indexed_dataset() -> None:
    x = torch.randn(5, 4)
    y = torch.tensor([0, 1, 0, 1, 0])
    ds = TensorDataset(x, y)
    indexed = IndexedDataset(ds, sample_ids=[f"id_{i}" for i in range(5)])

    assert len(indexed) == 5
    sample_id, feat, target = indexed[2]
    assert sample_id == "id_2"
    assert torch.equal(feat, x[2])
    assert target == y[2]


def test_run_inference() -> None:
    model = SimpleNet()
    x = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    results = run_inference(model, loader, device="cpu")
    assert "logits" in results
    assert "predictions" in results
    assert "labels" in results
    assert "sample_ids" in results
    assert len(results["predictions"]) == 10
    assert len(results["sample_ids"]) == 10
    assert results["logits"].shape == (10, 2)


def test_artifacts_save_and_load(tmp_path: pytest.TempPathFactory) -> None:
    temp_dir = tmp_path / "baseline_test"
    metadata = {"experiment_id": "test_exp", "epoch": 10}
    metrics = {"accuracy": 0.95}
    config = {"lr": 0.01}
    predictions = {
        "logits": np.zeros((5, 2)),
        "predictions": np.array([0, 1, 0, 1, 0]),
        "labels": np.array([0, 1, 0, 1, 0]),
        "sample_ids": np.arange(5),
    }

    save_path = save_experiment_artifacts(temp_dir, metadata, metrics, config, predictions)
    assert (save_path / "metadata.json").is_file()
    assert (save_path / "metrics.json").is_file()
    assert (save_path / "config.json").is_file()
    assert (save_path / "predictions.npz").is_file()

    loaded = load_experiment_artifacts(save_path)
    assert loaded["metadata"]["experiment_id"] == "test_exp"
    assert loaded["metrics"]["accuracy"] == 0.95
    assert len(loaded["predictions"]["predictions"]) == 5


def test_reproducibility() -> None:
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    assert val1 == val2

    env = get_environment_info()
    assert "python_version" in env
    assert "torch_version" in env


def test_load_checkpoint_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_model_from_checkpoint("non_existent_ckpt.pth")
