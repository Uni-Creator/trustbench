"""Shared fixtures for TrustBench integration tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

# Tiny synthetic nn.Module



@pytest.fixture()
def tiny_model() -> nn.Module:
    """A tiny linear model: 4 inputs → 3 classes."""
    torch.manual_seed(0)
    return nn.Linear(4, 3)



# Synthetic torch Dataset



class _SyntheticDataset(Dataset):
    """50 random samples with float features and integer labels."""

    def __init__(self, n: int = 50, n_features: int = 4, n_classes: int = 3) -> None:
        torch.manual_seed(42)
        self.data = torch.randn(n, n_features)
        self.targets = torch.randint(0, n_classes, (n,)).tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


@pytest.fixture()
def synthetic_torch_dataset() -> _SyntheticDataset:
    return _SyntheticDataset(n=50, n_features=4, n_classes=3)



# Config YAML content helper



@pytest.fixture()
def minimal_config_yaml() -> str:
    """Minimal valid trustbench.yaml content (no checkpoint — for schema tests)."""
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
