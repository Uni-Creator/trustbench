"""Unit tests for image corruptions and severity configs."""

import numpy as np
from PIL import Image

from trustbench.robustness.corruptions import (
    apply_brightness,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
    corrupt_pil_image,
)
from trustbench.robustness.severity import get_severity_config


def test_severity_configs() -> None:
    cfg = get_severity_config("gaussian_noise", 1)
    assert cfg.param_name == "sigma"
    assert cfg.param_value == 0.03

    cfg_blur = get_severity_config("gaussian_blur", 2)
    assert cfg_blur.param_value == 1.0


def test_corruptions_transform() -> None:
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    # Brightness
    darkened = apply_brightness(img, factor=0.5)
    assert np.all(np.array(darkened) <= 65)

    # Blur
    blurred = apply_gaussian_blur(img, sigma=1.0)
    assert blurred.size == (32, 32)

    # Noise reproducibility
    noisy1 = apply_gaussian_noise(img, sigma=0.08, seed=123)
    noisy2 = apply_gaussian_noise(img, sigma=0.08, seed=123)
    assert np.array_equal(np.array(noisy1), np.array(noisy2))

    # Occlusion
    occluded = apply_occlusion(img, area_fraction=0.25)
    arr = np.array(occluded)
    # Check that center has zeros
    assert arr[16, 16, 0] == 0


def test_corrupt_pil_image() -> None:
    img = Image.new("RGB", (32, 32), color=(200, 200, 200))
    res = corrupt_pil_image(img, "gaussian_noise", 1, sample_seed=42)
    assert res.size == (32, 32)
