# Dataset Adapters Reference

Dataset adapters provide a uniform slicing and indexing interface for datasets across various formats (PyTorch Datasets, NumPy memory arrays, pandas dataframes, and disk loaders).

---

## The `DatasetAdapter` Protocol

Defined in `trustbench.core.protocols.DatasetAdapter`:

```python
class DatasetAdapter(Protocol):
    def __len__(self) -> int:
        """Total number of samples."""
        ...

    def get_batch(self, indices: list[int]) -> Any:
        """Return model inputs for the given sample indices."""
        ...

    def get_labels(self) -> np.ndarray:
        """Return labels aligned with sample order, shape (N,)."""
        ...

    def get_sample_ids(self) -> np.ndarray:
        """Return unique sample identifiers aligned with sample order, shape (N,)."""
        ...
```

---

## Built-In Dataset Adapters

### 1. In-Memory Adapter (`InMemoryDatasetAdapter`)
For arrays already in RAM (NumPy arrays, lists, tensors):

```python
from trustbench.datasets.memory import InMemoryDatasetAdapter

dataset = InMemoryDatasetAdapter(
    inputs=X_test,               # Shape (N, ...)
    labels=y_test,               # Shape (N,)
    sample_ids=sample_ids,       # Optional list of identifiers
)
```

### 2. PyTorch Dataset Adapter (`TorchDatasetAdapter`)
Wraps any `torch.utils.data.Dataset` or `torchvision.datasets.*`:

```python
from torchvision.datasets import CIFAR10
from trustbench.datasets.torch_dataset import TorchDatasetAdapter

torch_ds = CIFAR10(root="data/", train=False, download=True)
adapter = TorchDatasetAdapter(torch_ds)
```

### 3. CIFAR-10 Adapter (`CIFAR10DatasetAdapter`)
Provides standardized normalization, tensor formatting, and corruption support for CIFAR-10 evaluation:

```python
from trustbench.datasets.cifar10 import CIFAR10DatasetAdapter, get_cifar10_datasets

trainset, testset = get_cifar10_datasets(data_dir="./data")
adapter = CIFAR10DatasetAdapter(testset)
```
