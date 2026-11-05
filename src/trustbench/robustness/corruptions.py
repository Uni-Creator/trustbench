"""Image corruption transformations applied before normalization."""

import math

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from trustbench.datasets.cifar10 import CIFAR10_MEAN, CIFAR10_STD
from trustbench.robustness.severity import get_severity_config


def apply_gaussian_noise(
    image: Image.Image,
    sigma: float,
    seed: int | None = None,
) -> Image.Image:
    """Apply additive zero-mean Gaussian noise to PIL image.

    Args:
        image: Input PIL Image (RGB).
        sigma: Standard deviation in [0, 1] scale.
        seed: Deterministic seed for reproducible noise generation.

    Returns:
        Corrupted PIL Image.
    """
    arr = np.asarray(image, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy_arr * 255.0).astype(np.uint8))


def apply_gaussian_blur(
    image: Image.Image,
    sigma: float,
) -> Image.Image:
    """Apply Gaussian blur to PIL image.

    Args:
        image: Input PIL Image.
        sigma: Standard deviation (radius) of Gaussian filter.

    Returns:
        Corrupted PIL Image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_brightness(
    image: Image.Image,
    factor: float,
) -> Image.Image:
    """Scale image brightness before normalization.

    Args:
        image: Input PIL Image.
        factor: Multiplicative factor (e.g. 0.8, 0.6, 0.4).

    Returns:
        Corrupted PIL Image.
    """
    arr = np.asarray(image, dtype=np.float32) * factor
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def apply_occlusion(
    image: Image.Image,
    area_fraction: float,
    seed: int | None = None,
) -> Image.Image:
    """Apply centered square occlusion patch masking area_fraction of the image.

    Args:
        image: Input PIL Image.
        area_fraction: Fraction of total image area to occlude (e.g. 0.10, 0.20, 0.30).
        seed: Optional seed (for future randomized occlusion locations; currently centered).

    Returns:
        Corrupted PIL Image.
    """
    w, h = image.size
    total_area = w * h
    occlusion_area = total_area * area_fraction
    patch_size = round(math.sqrt(occlusion_area))
    patch_size = max(1, min(min(w, h), patch_size))

    arr = np.array(image).copy()
    y_start = (h - patch_size) // 2
    x_start = (w - patch_size) // 2
    arr[y_start : y_start + patch_size, x_start : x_start + patch_size] = 0
    return Image.fromarray(arr)


def corrupt_pil_image(
    image: Image.Image,
    corruption_type: str,
    severity: int,
    sample_seed: int | None = None,
) -> Image.Image:
    """Apply corruption to a single PIL image based on corruption type and severity.

    Args:
        image: Clean PIL Image.
        corruption_type: 'gaussian_noise', 'gaussian_blur', 'brightness', or 'occlusion'.
        severity: Level 1, 2, or 3.
        sample_seed: Sample-specific deterministic seed for stochastic corruptions.

    Returns:
        Corrupted PIL Image.
    """
    cfg = get_severity_config(corruption_type, severity)
    param = cfg.param_value

    if corruption_type == "gaussian_noise":
        return apply_gaussian_noise(image, sigma=float(param), seed=sample_seed)
    if corruption_type == "gaussian_blur":
        return apply_gaussian_blur(image, sigma=float(param))
    if corruption_type == "brightness":
        return apply_brightness(image, factor=float(param))
    if corruption_type == "occlusion":
        return apply_occlusion(image, area_fraction=float(param), seed=sample_seed)

    raise ValueError(f"Unsupported corruption type: {corruption_type}")


class CorruptedDataset(Dataset):
    """Wraps an existing dataset to apply image corruptions deterministically before normalization.

    Maintains identical sample IDs and original targets.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        corruption_type: str,
        severity: int,
        base_seed: int = 42,
        mean: tuple[float, float, float] = CIFAR10_MEAN,
        std: tuple[float, float, float] = CIFAR10_STD,
    ):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        self.base_seed = base_seed
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, int]:
        # base_dataset could be CIFAR10 or similar dataset yielding (img, target)
        # Note: torchvision CIFAR10 with transform=None returns (PIL.Image, target)
        raw_item = self.base_dataset[idx]
        if isinstance(raw_item, tuple):
            img, target = raw_item[0], raw_item[1]
        else:
            img, target = raw_item, -1

        # Derive a unique deterministic seed per sample for reproducible stochastic corruption
        sample_seed = (self.base_seed * 1000003 + idx) % (2**31 - 1)

        if not isinstance(img, Image.Image):
            # If tensor, convert to PIL Image in [0, 255]
            if torch.is_tensor(img):
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                img = Image.fromarray(img_np)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))

        # 1. Apply corruption BEFORE normalization
        corrupted_img = corrupt_pil_image(
            image=img,
            corruption_type=self.corruption_type,
            severity=self.severity,
            sample_seed=sample_seed,
        )

        # 2. Transform to tensor and normalize
        tensor = TF.to_tensor(corrupted_img)
        norm_tensor = TF.normalize(tensor, self.mean, self.std)

        # Return (sample_id, tensor, target)
        return idx, norm_tensor, target
