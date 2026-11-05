"""Image corruption transformations applied before normalization."""

import io
import math
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import convolve

from trustbench.robustness.severity import get_severity_config

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


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


def apply_shot_noise(
    image: Image.Image,
    scale: float,
    seed: int | None = None,
) -> Image.Image:
    """Apply Poisson (shot) noise representing photon counting statistics.

    Args:
        image: Input PIL Image.
        scale: Poisson intensity scale (lower scale = higher noise).
        seed: Deterministic seed.

    Returns:
        Corrupted PIL Image.
    """
    arr = np.asarray(image, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(np.clip(arr, 0.0, 1.0) * scale) / float(scale)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


def apply_impulse_noise(
    image: Image.Image,
    amount: float,
    seed: int | None = None,
) -> Image.Image:
    """Apply salt-and-pepper (impulse/spike) noise to PIL image.

    Args:
        image: Input PIL Image.
        amount: Fraction of image pixels replaced by salt (255) or pepper (0).
        seed: Deterministic seed.

    Returns:
        Corrupted PIL Image.
    """
    arr = np.asarray(image, dtype=np.uint8).copy()
    rng = np.random.default_rng(seed)
    mask = rng.uniform(0.0, 1.0, size=arr.shape[:2])

    pepper = mask < (amount / 2.0)
    salt = (mask >= (amount / 2.0)) & (mask < amount)

    if arr.ndim == 3:
        for c in range(arr.shape[2]):
            arr[pepper, c] = 0
            arr[salt, c] = 255
    else:
        arr[pepper] = 0
        arr[salt] = 255

    return Image.fromarray(arr)


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


def _get_motion_blur_kernel(kernel_size: int, angle_degrees: float = 45.0) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    radian = math.radians(angle_degrees)
    cos_a = math.cos(radian)
    sin_a = math.sin(radian)
    for i in range(kernel_size):
        offset = i - center
        r = round(center - offset * sin_a)
        c = round(center + offset * cos_a)
        if 0 <= r < kernel_size and 0 <= c < kernel_size:
            kernel[r, c] = 1.0
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        kernel[center, center] = 1.0
    return kernel


def apply_motion_blur(
    image: Image.Image,
    kernel_size: int,
) -> Image.Image:
    """Apply linear motion blur to PIL image.

    Args:
        image: Input PIL Image.
        kernel_size: Length of the linear motion blur kernel in pixels.

    Returns:
        Corrupted PIL Image.
    """
    kernel = _get_motion_blur_kernel(kernel_size)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        blurred = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            blurred[:, :, c] = convolve(arr[:, :, c], kernel, mode="nearest")
    else:
        blurred = convolve(arr, kernel, mode="nearest")
    return Image.fromarray(np.clip(blurred, 0.0, 255.0).astype(np.uint8))


def _get_disk_kernel(radius: int) -> np.ndarray:
    d = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y) <= (radius * radius)
    kernel = np.zeros((d, d), dtype=np.float32)
    kernel[mask] = 1.0
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def apply_defocus_blur(
    image: Image.Image,
    radius: int,
) -> Image.Image:
    """Apply disk defocus blur to PIL image.

    Args:
        image: Input PIL Image.
        radius: Radius of circular aperture in pixels.

    Returns:
        Corrupted PIL Image.
    """
    kernel = _get_disk_kernel(radius)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        blurred = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            blurred[:, :, c] = convolve(arr[:, :, c], kernel, mode="nearest")
    else:
        blurred = convolve(arr, kernel, mode="nearest")
    return Image.fromarray(np.clip(blurred, 0.0, 255.0).astype(np.uint8))


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


def apply_contrast(
    image: Image.Image,
    factor: float,
) -> Image.Image:
    """Adjust image contrast.

    Args:
        image: Input PIL Image.
        factor: Contrast scale (1.0 = original, <1.0 = lower contrast).

    Returns:
        Corrupted PIL Image.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_saturation(
    image: Image.Image,
    factor: float,
) -> Image.Image:
    """Adjust image color saturation.

    Args:
        image: Input PIL Image.
        factor: Saturation scale (1.0 = original, 0.0 = complete grayscale).

    Returns:
        Corrupted PIL Image.
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def apply_occlusion(
    image: Image.Image,
    area_fraction: float,
    seed: int | None = None,
) -> Image.Image:
    """Apply centered square occlusion patch masking area_fraction of the image.

    Args:
        image: Input PIL Image.
        area_fraction: Fraction of total image area to occlude.
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


def apply_jpeg_compression(
    image: Image.Image,
    quality: int,
) -> Image.Image:
    """Apply JPEG compression artifacts to PIL image.

    Args:
        image: Input PIL Image.
        quality: JPEG compression quality factor in [1, 100].

    Returns:
        Corrupted PIL Image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_pixelation(
    image: Image.Image,
    factor: float,
) -> Image.Image:
    """Apply pixelation (downsampling resolution followed by nearest-neighbor upsampling).

    Args:
        image: Input PIL Image.
        factor: Downsampling scale factor in (0.0, 1.0].

    Returns:
        Corrupted PIL Image.
    """
    w, h = image.size
    down_w = max(1, round(w * factor))
    down_h = max(1, round(h * factor))
    small = image.resize((down_w, down_h), resample=Image.Resampling.BOX)
    return small.resize((w, h), resample=Image.Resampling.NEAREST)


def corrupt_pil_image(
    image: Image.Image,
    corruption_type: str,
    severity: int,
    sample_seed: int | None = None,
) -> Image.Image:
    """Apply corruption to a single PIL image based on corruption type and severity.

    Supports 12 corruption types:
      - Noise: 'gaussian_noise', 'shot_noise', 'impulse_noise'
      - Blur: 'gaussian_blur', 'motion_blur', 'defocus_blur'
      - Weather/Digital: 'brightness', 'contrast', 'saturation'
      - Artifacts: 'occlusion', 'jpeg_compression', 'pixelation'

    Args:
        image: Clean PIL Image.
        corruption_type: Name of corruption.
        severity: Level 1 to 5.
        sample_seed: Deterministic seed for reproducible stochastic corruptions.

    Returns:
        Corrupted PIL Image.
    """
    from trustbench.registry.corruptions import CORRUPTION_REGISTRY

    custom = CORRUPTION_REGISTRY.get(corruption_type)
    if custom is not None:
        return custom.apply(image, severity, seed=sample_seed)

    cfg = get_severity_config(corruption_type, severity)
    param = cfg.param_value

    if corruption_type == "gaussian_noise":
        return apply_gaussian_noise(image, sigma=float(param), seed=sample_seed)
    if corruption_type == "shot_noise":
        return apply_shot_noise(image, scale=float(param), seed=sample_seed)
    if corruption_type == "impulse_noise":
        return apply_impulse_noise(image, amount=float(param), seed=sample_seed)
    if corruption_type == "gaussian_blur":
        return apply_gaussian_blur(image, sigma=float(param))
    if corruption_type == "motion_blur":
        return apply_motion_blur(image, kernel_size=int(param))
    if corruption_type == "defocus_blur":
        return apply_defocus_blur(image, radius=int(param))
    if corruption_type == "brightness":
        return apply_brightness(image, factor=float(param))
    if corruption_type == "contrast":
        return apply_contrast(image, factor=float(param))
    if corruption_type == "saturation":
        return apply_saturation(image, factor=float(param))
    if corruption_type == "occlusion":
        return apply_occlusion(image, area_fraction=float(param), seed=sample_seed)
    if corruption_type == "jpeg_compression":
        return apply_jpeg_compression(image, quality=int(param))
    if corruption_type == "pixelation":
        return apply_pixelation(image, factor=float(param))

    raise ValueError(f"Unsupported corruption type: {corruption_type}")


class CorruptedDataset:
    """Wraps an existing dataset to apply image corruptions deterministically before normalization.

    Maintains identical sample IDs and original targets.
    """

    def __init__(
        self,
        base_dataset: Any,
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

    def __getitem__(self, idx: int) -> tuple[int, Any, int]:
        import torch
        import torchvision.transforms.functional as TF

        raw_item = self.base_dataset[idx]
        if isinstance(raw_item, tuple):
            img, target = raw_item[0], raw_item[1]
        else:
            img, target = raw_item, -1

        # Derive a unique deterministic seed per sample for reproducible stochastic corruption
        sample_seed = (self.base_seed * 1000003 + idx) % (2**31 - 1)

        if not isinstance(img, Image.Image):
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

        return idx, norm_tensor, target
