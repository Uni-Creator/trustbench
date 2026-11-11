"""Dataset utilities for CIFAR-10 including validation split generation."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from trustbench.datasets.base import DatasetAdapter

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return standard training and evaluation transforms for CIFAR-10."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transform_train, transform_test


def get_cifar10_datasets(
    data_dir: str = "./data", download: bool = True
) -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Download and return CIFAR-10 train and test datasets."""
    transform_train, transform_test = get_cifar10_transforms()
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform_test
    )
    return trainset, testset


def get_cifar10_validation_split(
    data_dir: str = "./data",
    val_size: int = 5000,
    seed: int = 42,
    download: bool = True,
) -> tuple[Dataset, Dataset]:
    """Split CIFAR-10 train set (50k) into train (50k - val_size) and validation (val_size).

    The validation split uses the standard test evaluation transform (no data augmentation).
    """
    transform_train, transform_eval = get_cifar10_transforms()

    # Base dataset for evaluation transform
    val_base = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform_eval
    )
    # Base dataset for training transform
    train_base = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform_train
    )

    total_len = len(val_base)
    train_len = total_len - val_size

    # Deterministic split indices
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(total_len), [train_len, val_size], generator=generator)  # type: ignore[arg-type,var-annotated]

    train_subset = Subset(train_base, train_indices.indices)
    val_subset = Subset(val_base, val_indices.indices)

    return train_subset, val_subset


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size_train: int = 128,
    batch_size_test: int = 100,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return DataLoader instances for CIFAR-10 training and testing."""
    trainset, testset = get_cifar10_datasets(data_dir=data_dir, download=download)

    trainloader = DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers
    )
    return trainloader, testloader


class CIFAR10DatasetAdapter(DatasetAdapter):
    """DatasetAdapter over a torchvision CIFAR-10-style dataset."""

    def __init__(
        self,
        dataset: Dataset,
        sample_ids: Sequence[Any] | None = None,
    ) -> None:
        self.dataset = dataset
        n = len(dataset)  # type: ignore[arg-type]
        self._sample_ids = np.arange(n) if sample_ids is None else np.asarray(sample_ids)
        if len(self._sample_ids) != n:
            raise ValueError("sample_ids length does not match dataset length.")

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        tensors = []
        for idx in indices:
            item = self.dataset[idx]
            image = item[0] if isinstance(item, (tuple, list)) else item
            tensors.append(image)
        return torch.stack(tensors)

    def get_labels(self) -> np.ndarray:
        dataset = self.dataset
        if hasattr(dataset, "targets"):
            return np.asarray(dataset.targets)
        if hasattr(dataset, "labels"):
            return np.asarray(dataset.labels)
        labels = []
        for idx in range(len(self)):
            item = dataset[idx]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                labels.append(item[1])
            else:
                labels.append(-1)
        return np.asarray(labels)

    def get_sample_ids(self) -> np.ndarray:
        return self._sample_ids


def get_cifar10_test_adapter(
    data_dir: str = "./data",
    download: bool = True,
) -> CIFAR10DatasetAdapter:
    """Return a DatasetAdapter for the CIFAR-10 test split."""
    _, testset = get_cifar10_datasets(data_dir=data_dir, download=download)
    return CIFAR10DatasetAdapter(testset)
