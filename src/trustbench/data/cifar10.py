"""Compatibility shim for CIFAR-10 dataset helpers."""

from trustbench.datasets.cifar10 import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR10DatasetAdapter,
    get_cifar10_datasets,
    get_cifar10_loaders,
    get_cifar10_test_adapter,
    get_cifar10_transforms,
    get_cifar10_validation_split,
)

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR10DatasetAdapter",
    "get_cifar10_datasets",
    "get_cifar10_loaders",
    "get_cifar10_test_adapter",
    "get_cifar10_transforms",
    "get_cifar10_validation_split",
]
