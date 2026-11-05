"""Checkpoint loading for PyTorch CIFAR architectures."""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

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
