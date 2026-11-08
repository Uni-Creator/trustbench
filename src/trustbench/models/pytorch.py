"""PyTorch model adapter and legacy DataLoader inference."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from trustbench.core.exceptions import ShapeError
from trustbench.models.base import ModelAdapter


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class PyTorchModelAdapter(ModelAdapter):
    """Adapter for a PyTorch ``nn.Module`` that emits classification logits."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: str | torch.device | None = None,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        self._model = model
        self._num_classes = int(num_classes)
        self._device = _as_device(device)
        self._model.to(self._device)
        self._model.eval()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def supports_predict_proba(self) -> bool:
        return True

    @property
    def supports_predict_logits(self) -> bool:
        return True

    @property
    def module(self) -> nn.Module:
        """Underlying PyTorch module."""
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def _to_tensor(self, inputs: Any) -> torch.Tensor:
        if torch.is_tensor(inputs):
            tensor = inputs
        else:
            tensor = torch.as_tensor(inputs)
        if tensor.device != self._device:
            tensor = tensor.to(self._device, non_blocking=True)
        return tensor

    def _validate_logits(self, outputs: torch.Tensor, batch_size: int) -> torch.Tensor:
        if outputs.ndim != 2:
            raise ShapeError(
                "PyTorch model outputs must be logits of shape (N, C); "
                f"got shape {tuple(outputs.shape)}."
            )
        if outputs.shape[0] != batch_size:
            raise ShapeError(
                f"Expected logits batch size {batch_size}, got {outputs.shape[0]}."
            )
        if outputs.shape[1] != self._num_classes:
            raise ShapeError(
                f"Expected logits with {self._num_classes} classes, got {outputs.shape[1]}."
            )
        return outputs

    def predict_logits(self, inputs: Any) -> np.ndarray:
        tensor = self._to_tensor(inputs)
        batch_size = int(tensor.shape[0])
        self._model.eval()
        with torch.inference_mode():
            outputs = self._model(tensor)
        logits = self._validate_logits(outputs, batch_size)
        return logits.detach().cpu().float().numpy()

    def predict_proba(self, inputs: Any) -> np.ndarray:
        tensor = self._to_tensor(inputs)
        batch_size = int(tensor.shape[0])
        self._model.eval()
        with torch.inference_mode():
            outputs = self._model(tensor)
        logits = self._validate_logits(outputs, batch_size)
        probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().float().numpy()

    def predict(self, inputs: Any) -> np.ndarray:
        tensor = self._to_tensor(inputs)
        batch_size = int(tensor.shape[0])
        self._model.eval()
        with torch.inference_mode():
            outputs = self._model(tensor)
        logits = self._validate_logits(outputs, batch_size)
        preds = torch.argmax(logits, dim=1)
        return preds.detach().cpu().numpy()


class IndexedDataset(Dataset):
    """Wrapper that yields (sample_id, data, target) preserving sample IDs."""

    def __init__(self, dataset: Dataset, sample_ids: Sequence[Any] | None = None):
        self.dataset = dataset
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        item = self.dataset[idx]
        sample_id = self.sample_ids[idx] if self.sample_ids is not None else idx
        if isinstance(item, (tuple, list)) and len(item) == 2:
            data, target = item
            return sample_id, data, target
        return sample_id, item, None


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run model inference with sample ID tracking.

    Args:
        model: Evaluated PyTorch model.
        dataloader: PyTorch DataLoader yielding either:
                    (inputs, targets) or (sample_ids, inputs, targets)
        device: Device to run inference on.

    Returns:
        Dictionary containing:
            - "logits": np.ndarray of shape (N, num_classes)
            - "predictions": np.ndarray of shape (N,)
            - "labels": np.ndarray of shape (N,)
            - "sample_ids": np.ndarray of shape (N,)
    """
    device = _as_device(device)

    model.to(device)
    model.eval()

    all_logits = []
    all_predictions = []
    all_labels = []
    all_sample_ids = []

    current_idx = 0

    with torch.inference_mode():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                if len(batch) == 3:
                    sample_ids, inputs, targets = batch
                    if torch.is_tensor(sample_ids):
                        ids = sample_ids.cpu().numpy()
                    elif isinstance(sample_ids, (list, tuple)):
                        ids = np.array(sample_ids)
                    else:
                        ids = np.array(list(sample_ids))
                elif len(batch) == 2:
                    inputs, targets = batch
                    batch_len = inputs.size(0)
                    ids = np.arange(current_idx, current_idx + batch_len)
                    current_idx += batch_len
                else:
                    raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
            else:
                inputs = batch
                targets = torch.full((inputs.size(0),), -1)
                batch_len = inputs.size(0)
                ids = np.arange(current_idx, current_idx + batch_len)
                current_idx += batch_len

            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_logits.append(outputs.cpu().float().numpy())
            all_predictions.append(preds.cpu().numpy())
            if targets is not None:
                if torch.is_tensor(targets):
                    all_labels.append(targets.cpu().numpy())
                else:
                    all_labels.append(np.array(targets))
            all_sample_ids.append(ids)

    return {
        "logits": np.concatenate(all_logits, axis=0) if all_logits else np.empty((0,)),
        "predictions": np.concatenate(all_predictions, axis=0) if all_predictions else np.empty((0,), dtype=int),
        "labels": np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=int),
        "sample_ids": np.concatenate(all_sample_ids, axis=0) if all_sample_ids else np.empty((0,)),
    }
