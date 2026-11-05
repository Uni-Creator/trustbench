"""Unit tests for image corruptions, severity configs, and dataset transformations."""

import numpy as np
from PIL import Image

from trustbench.robustness.corruptions import (
    CorruptedDataset,
    apply_brightness,
    apply_contrast,
    apply_defocus_blur,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_impulse_noise,
    apply_jpeg_compression,
    apply_motion_blur,
    apply_occlusion,
    apply_pixelation,
    apply_saturation,
    apply_shot_noise,
    corrupt_pil_image,
)
from trustbench.robustness.severity import (
    ALL_CORRUPTION_TYPES,
    CORRUPTION_SEVERITIES,
    get_severity_config,
)


def test_severity_configs() -> None:
    """Verify severity configuration parameters and backward compatibility."""
    cfg = get_severity_config("gaussian_noise", 1)
    assert cfg.param_name == "sigma"
    assert cfg.param_value == 0.03

    cfg_blur = get_severity_config("gaussian_blur", 2)
    assert cfg_blur.param_value == 1.0


def test_all_12_corruptions_registered() -> None:
    """Verify that all 12 corruptions across severities 1 to 5 are registered."""
    assert len(ALL_CORRUPTION_TYPES) == 12
    for c_name in ALL_CORRUPTION_TYPES:
        assert c_name in CORRUPTION_SEVERITIES
        for s in [1, 2, 3, 4, 5]:
            cfg = get_severity_config(c_name, s)
            assert cfg.severity == s
            assert cfg.param_value is not None


def test_corruptions_transform() -> None:
    """Verify original basic corruption transformations."""
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
    assert arr[16, 16, 0] == 0


def test_corrupt_pil_image() -> None:
    """Verify corrupt_pil_image interface."""
    img = Image.new("RGB", (32, 32), color=(200, 200, 200))
    res = corrupt_pil_image(img, "gaussian_noise", 1, sample_seed=42)
    assert res.size == (32, 32)


def test_all_corruptions_execute_severities() -> None:
    """Verify that all 12 corruptions execute across multiple severity levels."""
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    for c_type in ALL_CORRUPTION_TYPES:
        for sev in [1, 3, 5]:
            corrupted = corrupt_pil_image(img, c_type, sev, sample_seed=42)
            assert isinstance(corrupted, Image.Image)
            assert corrupted.size == (32, 32)
            assert corrupted.mode == "RGB"


def test_deterministic_stochastic_corruptions() -> None:
    """Verify deterministic repeatability for stochastic noise corruptions."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))

    # 1. Gaussian noise
    g1 = apply_gaussian_noise(img, sigma=0.08, seed=999)
    g2 = apply_gaussian_noise(img, sigma=0.08, seed=999)
    assert np.array_equal(np.array(g1), np.array(g2))

    # 2. Shot noise
    s1 = apply_shot_noise(img, scale=30.0, seed=999)
    s2 = apply_shot_noise(img, scale=30.0, seed=999)
    assert np.array_equal(np.array(s1), np.array(s2))

    # 3. Impulse noise
    i1 = apply_impulse_noise(img, amount=0.1, seed=999)
    i2 = apply_impulse_noise(img, amount=0.1, seed=999)
    assert np.array_equal(np.array(i1), np.array(i2))


def test_corruptions_specific_transforms() -> None:
    """Verify all individual corruption transformations and properties."""
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))

    # Gaussian Blur
    gb = apply_gaussian_blur(img, sigma=1.0)
    assert gb.size == (32, 32)

    # Motion blur
    mb = apply_motion_blur(img, kernel_size=5)
    assert mb.size == (32, 32)

    # Defocus blur
    db = apply_defocus_blur(img, radius=2)
    assert db.size == (32, 32)

    # Contrast
    contrast = apply_contrast(img, factor=0.5)
    assert contrast.size == (32, 32)

    # Saturation
    sat = apply_saturation(img, factor=0.0)
    assert sat.size == (32, 32)

    # JPEG compression
    jpeg = apply_jpeg_compression(img, quality=20)
    assert jpeg.size == (32, 32)

    # Pixelation
    pix = apply_pixelation(img, factor=0.25)
    assert pix.size == (32, 32)

    # Brightness & Occlusion
    br = apply_brightness(img, factor=0.5)
    assert np.all(np.array(br) <= 65)

    occ = apply_occlusion(img, area_fraction=0.25)
    assert np.array(occ)[16, 16, 0] == 0


def test_corrupted_dataset_deterministic() -> None:
    """Verify CorruptedDataset reproducibility and sample indexing."""
    class DummyDS:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int):
            return Image.new("RGB", (32, 32), color=(idx * 40, idx * 40, idx * 40)), idx

    base_ds = DummyDS()
    corr_ds = CorruptedDataset(base_ds, corruption_type="shot_noise", severity=2, base_seed=123)

    idx1, tensor1, target1 = corr_ds[1]
    _idx2, tensor2, _target2 = corr_ds[1]

    assert idx1 == 1
    assert target1 == 1
    assert (tensor1 == tensor2).all()
