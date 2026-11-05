"""Compatibility shim for model loading and DataLoader inference."""

from trustbench.models.checkpoint import (
    clean_state_dict,
    load_model_from_checkpoint,
    resolve_model_architecture,
)
from trustbench.models.pytorch import IndexedDataset, run_inference

__all__ = [
    "IndexedDataset",
    "clean_state_dict",
    "load_model_from_checkpoint",
    "resolve_model_architecture",
    "run_inference",
]
