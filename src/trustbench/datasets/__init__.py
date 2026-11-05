"""Dataset adapters for TrustBench."""

from trustbench.datasets.base import DatasetAdapter
from trustbench.datasets.memory import InMemoryDatasetAdapter

__all__ = [
    "DatasetAdapter",
    "InMemoryDatasetAdapter",
]
