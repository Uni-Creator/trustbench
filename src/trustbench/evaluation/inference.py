"""Model loading and inference pipeline."""

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from trustbench import models


def resolve_model_architecture(model_name: str) -> nn.Module:
    """Instantiate a model architecture by name.

    Args:
        model_name: Name of model (e.g. 'ResNet18', 'VGG19').

    Returns:
        Instantiated nn.Module.
    """
    if model_name.startswith("VGG") and hasattr(models, "VGG"):
        return models.VGG(model_name)
    if hasattr(models, model_name):
        model_cls = getattr(models, model_name)
        return model_cls()
    raise ValueError(f"Unknown or unsupported model architecture: {model_name}")


def clean_state_dict(state_dict: dict[str, Any]) -> OrderedDict:
    """Strip 'module.' prefix from DataParallel checkpoints if present."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.removeprefix("module.")
        new_state_dict[name] = v
    return new_state_dict


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str = "ResNet18",
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model architecture and weights from checkpoint file independently of training script.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        model_name: Model architecture name (default: ResNet18).
        device: Target device (cpu, cuda). Defaults to cuda if available.

    Returns:
        Tuple of (loaded nn.Module in eval mode, raw checkpoint dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    cleaned_dict = clean_state_dict(state_dict)

    model = resolve_model_architecture(model_name)
    model.load_state_dict(cleaned_dict)
    model.to(device)
    model.eval()

    return model, checkpoint if isinstance(checkpoint, dict) else {}


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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

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
                    # convert sample_ids to list/array
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
