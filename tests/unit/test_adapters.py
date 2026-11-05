"""Contract tests for model and dataset adapters."""

import numpy as np
import pytest
import torch
from torch import nn

from trustbench.core.exceptions import ShapeError, UnsupportedCapabilityError
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.models.base import ModelAdapter
from trustbench.models.pytorch import PyTorchModelAdapter


class PredictOnlyModel(ModelAdapter):
    def __init__(self, num_classes: int = 2) -> None:
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.zeros(len(inputs), dtype=int)


class TinyNet(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class BadShapeNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)


def test_predict_only_model_marks_optional_capabilities() -> None:
    model = PredictOnlyModel()
    assert model.supports_predict_proba is False
    assert model.supports_predict_logits is False
    with pytest.raises(UnsupportedCapabilityError):
        model.predict_proba(np.zeros((2, 4)))
    with pytest.raises(UnsupportedCapabilityError):
        model.predict_logits(np.zeros((2, 4)))
    assert list(model.predict(np.zeros((3, 4)))) == [0, 0, 0]


def test_in_memory_dataset_adapter_alignment() -> None:
    inputs = np.arange(12, dtype=float).reshape(4, 3)
    labels = np.array([0, 1, 0, 1])
    sample_ids = np.array(["a", "b", "c", "d"])
    dataset = InMemoryDatasetAdapter(inputs, labels, sample_ids=sample_ids)

    assert len(dataset) == 4
    np.testing.assert_array_equal(dataset.get_labels(), labels)
    np.testing.assert_array_equal(dataset.get_sample_ids(), sample_ids)
    batch = dataset.get_batch([1, 3])
    np.testing.assert_array_equal(batch, inputs[[1, 3]])


def test_pytorch_adapter_logits_proba_and_predict() -> None:
    torch.manual_seed(0)
    net = TinyNet(num_classes=3)
    adapter = PyTorchModelAdapter(net, num_classes=3, device="cpu")
    inputs = torch.randn(5, 4)

    logits = adapter.predict_logits(inputs)
    proba = adapter.predict_proba(inputs)
    preds = adapter.predict(inputs)

    assert adapter.supports_predict_logits is True
    assert adapter.supports_predict_proba is True
    assert logits.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert preds.shape == (5,)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)
    np.testing.assert_array_equal(preds, np.argmax(logits, axis=1))


def test_pytorch_adapter_rejects_wrong_output_shape() -> None:
    adapter = PyTorchModelAdapter(BadShapeNet(), num_classes=3, device="cpu")
    with pytest.raises(ShapeError):
        adapter.predict_logits(torch.randn(4, 8))


def test_pytorch_adapter_rejects_class_mismatch() -> None:
    adapter = PyTorchModelAdapter(TinyNet(num_classes=3), num_classes=2, device="cpu")
    with pytest.raises(ShapeError, match="2 classes"):
        adapter.predict_logits(torch.randn(4, 4))

def test_dataset_adapter_empty_and_batch_indexing() -> None:
    empty_dataset = InMemoryDatasetAdapter(
        np.empty((0, 4)), np.empty((0,), dtype=int), sample_ids=np.empty((0,), dtype=str)
    )
    assert len(empty_dataset) == 0
    assert len(empty_dataset.get_labels()) == 0
    assert len(empty_dataset.get_sample_ids()) == 0
    assert empty_dataset.get_batch([]).shape == (0, 4)


def test_pytorch_adapter_preserves_sample_ids() -> None:
    class DummyDataset:
        def __len__(self) -> int:
            return 3

        def get_batch(self, indices: list[int]) -> np.ndarray:
            return np.ones((len(indices), 4), dtype=float)

        def get_labels(self) -> np.ndarray:
            return np.array([0, 1, 2])

        def get_sample_ids(self) -> np.ndarray:
            return np.array(["id_101", "id_102", "id_103"])

    ds = DummyDataset()
    assert list(ds.get_sample_ids()) == ["id_101", "id_102", "id_103"]
