"""CIFAR-10 dataset loading and data loaders."""


import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

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
