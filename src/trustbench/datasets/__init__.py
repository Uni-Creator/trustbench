"""Dataset adapters for TrustBench."""

from trustbench.datasets.base import DatasetAdapter
from trustbench.datasets.memory import InMemoryDatasetAdapter

__all__ = [
    "CIFAR10DatasetAdapter",
    "DatasetAdapter",
    "InMemoryDatasetAdapter",
    "TorchDatasetAdapter",
]


def __getattr__(name: str):
    if name == "TorchDatasetAdapter":
        from trustbench.datasets.torch_dataset import TorchDatasetAdapter

        return TorchDatasetAdapter
    if name == "CIFAR10DatasetAdapter":
        from trustbench.datasets.cifar10 import CIFAR10DatasetAdapter

        return CIFAR10DatasetAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
