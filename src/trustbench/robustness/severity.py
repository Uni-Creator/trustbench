"""Severity levels and parameter configurations for image corruptions."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeverityConfig:
    """Configuration parameter for a corruption severity level."""

    severity: int
    param_name: str
    param_value: float | int
    unit: str
    description: str


CORRUPTION_SEVERITIES: dict[str, dict[int, SeverityConfig]] = {
    # 1. Gaussian Noise
    "gaussian_noise": {
        1: SeverityConfig(
            severity=1,
            param_name="sigma",
            param_value=0.03,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.03.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="sigma",
            param_value=0.08,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.08.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="sigma",
            param_value=0.15,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.15.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="sigma",
            param_value=0.25,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.25.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="sigma",
            param_value=0.38,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.38.",
        ),
    },
    # 2. Shot Noise (Poisson)
    "shot_noise": {
        1: SeverityConfig(
            severity=1,
            param_name="scale",
            param_value=60.0,
            unit="photon count scale factor",
            description="Poisson shot noise with photon scale 60.0.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="scale",
            param_value=30.0,
            unit="photon count scale factor",
            description="Poisson shot noise with photon scale 30.0.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="scale",
            param_value=15.0,
            unit="photon count scale factor",
            description="Poisson shot noise with photon scale 15.0.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="scale",
            param_value=8.0,
            unit="photon count scale factor",
            description="Poisson shot noise with photon scale 8.0.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="scale",
            param_value=4.0,
            unit="photon count scale factor",
            description="Poisson shot noise with photon scale 4.0.",
        ),
    },
    # 3. Impulse Noise (Salt and Pepper)
    "impulse_noise": {
        1: SeverityConfig(
            severity=1,
            param_name="amount",
            param_value=0.02,
            unit="fraction of corrupted pixels",
            description="Salt-and-pepper impulse noise affecting 2% of pixels.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="amount",
            param_value=0.05,
            unit="fraction of corrupted pixels",
            description="Salt-and-pepper impulse noise affecting 5% of pixels.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="amount",
            param_value=0.10,
            unit="fraction of corrupted pixels",
            description="Salt-and-pepper impulse noise affecting 10% of pixels.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="amount",
            param_value=0.17,
            unit="fraction of corrupted pixels",
            description="Salt-and-pepper impulse noise affecting 17% of pixels.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="amount",
            param_value=0.27,
            unit="fraction of corrupted pixels",
            description="Salt-and-pepper impulse noise affecting 27% of pixels.",
        ),
    },
    # 4. Gaussian Blur
    "gaussian_blur": {
        1: SeverityConfig(
            severity=1,
            param_name="sigma",
            param_value=0.5,
            unit="pixels (Gaussian kernel std)",
            description="Gaussian filter blur with kernel std sigma=0.5.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="sigma",
            param_value=1.0,
            unit="pixels (Gaussian kernel std)",
            description="Gaussian filter blur with kernel std sigma=1.0.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="sigma",
            param_value=1.5,
            unit="pixels (Gaussian kernel std)",
            description="Gaussian filter blur with kernel std sigma=1.5.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="sigma",
            param_value=2.0,
            unit="pixels (Gaussian kernel std)",
            description="Gaussian filter blur with kernel std sigma=2.0.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="sigma",
            param_value=2.5,
            unit="pixels (Gaussian kernel std)",
            description="Gaussian filter blur with kernel std sigma=2.5.",
        ),
    },
    # 5. Motion Blur
    "motion_blur": {
        1: SeverityConfig(
            severity=1,
            param_name="kernel_size",
            param_value=3,
            unit="pixels (kernel length)",
            description="Linear motion blur with kernel length 3.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="kernel_size",
            param_value=5,
            unit="pixels (kernel length)",
            description="Linear motion blur with kernel length 5.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="kernel_size",
            param_value=7,
            unit="pixels (kernel length)",
            description="Linear motion blur with kernel length 7.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="kernel_size",
            param_value=9,
            unit="pixels (kernel length)",
            description="Linear motion blur with kernel length 9.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="kernel_size",
            param_value=11,
            unit="pixels (kernel length)",
            description="Linear motion blur with kernel length 11.",
        ),
    },
    # 6. Defocus Blur
    "defocus_blur": {
        1: SeverityConfig(
            severity=1,
            param_name="radius",
            param_value=1,
            unit="pixels (disk aperture radius)",
            description="Disk defocus blur with aperture radius 1.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="radius",
            param_value=2,
            unit="pixels (disk aperture radius)",
            description="Disk defocus blur with aperture radius 2.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="radius",
            param_value=3,
            unit="pixels (disk aperture radius)",
            description="Disk defocus blur with aperture radius 3.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="radius",
            param_value=4,
            unit="pixels (disk aperture radius)",
            description="Disk defocus blur with aperture radius 4.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="radius",
            param_value=5,
            unit="pixels (disk aperture radius)",
            description="Disk defocus blur with aperture radius 5.",
        ),
    },
    # 7. Brightness
    "brightness": {
        1: SeverityConfig(
            severity=1,
            param_name="factor",
            param_value=0.8,
            unit="multiplier (relative luminance scale)",
            description="Image luminance scaled by 0.8x.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="factor",
            param_value=0.6,
            unit="multiplier (relative luminance scale)",
            description="Image luminance scaled by 0.6x.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="factor",
            param_value=0.4,
            unit="multiplier (relative luminance scale)",
            description="Image luminance scaled by 0.4x.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="factor",
            param_value=0.3,
            unit="multiplier (relative luminance scale)",
            description="Image luminance scaled by 0.3x.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="factor",
            param_value=0.2,
            unit="multiplier (relative luminance scale)",
            description="Image luminance scaled by 0.2x.",
        ),
    },
    # 8. Contrast
    "contrast": {
        1: SeverityConfig(
            severity=1,
            param_name="factor",
            param_value=0.8,
            unit="contrast scale factor",
            description="Image contrast scaled to 0.8x.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="factor",
            param_value=0.6,
            unit="contrast scale factor",
            description="Image contrast scaled to 0.6x.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="factor",
            param_value=0.4,
            unit="contrast scale factor",
            description="Image contrast scaled to 0.4x.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="factor",
            param_value=0.25,
            unit="contrast scale factor",
            description="Image contrast scaled to 0.25x.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="factor",
            param_value=0.1,
            unit="contrast scale factor",
            description="Image contrast scaled to 0.1x.",
        ),
    },
    # 9. Saturation
    "saturation": {
        1: SeverityConfig(
            severity=1,
            param_name="factor",
            param_value=0.8,
            unit="saturation scale factor",
            description="Color saturation scaled to 0.8x.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="factor",
            param_value=0.5,
            unit="saturation scale factor",
            description="Color saturation scaled to 0.5x.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="factor",
            param_value=0.3,
            unit="saturation scale factor",
            description="Color saturation scaled to 0.3x.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="factor",
            param_value=0.15,
            unit="saturation scale factor",
            description="Color saturation scaled to 0.15x.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="factor",
            param_value=0.0,
            unit="saturation scale factor",
            description="Color saturation reduced to complete grayscale (0.0x).",
        ),
    },
    # 10. Occlusion
    "occlusion": {
        1: SeverityConfig(
            severity=1,
            param_name="area_fraction",
            param_value=0.10,
            unit="area percentage",
            description="Centered rectangular patch masking 10% of image pixels.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="area_fraction",
            param_value=0.20,
            unit="area percentage",
            description="Centered rectangular patch masking 20% of image pixels.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="area_fraction",
            param_value=0.30,
            unit="area percentage",
            description="Centered rectangular patch masking 30% of image pixels.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="area_fraction",
            param_value=0.40,
            unit="area percentage",
            description="Centered rectangular patch masking 40% of image pixels.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="area_fraction",
            param_value=0.50,
            unit="area percentage",
            description="Centered rectangular patch masking 50% of image pixels.",
        ),
    },
    # 11. JPEG Compression
    "jpeg_compression": {
        1: SeverityConfig(
            severity=1,
            param_name="quality",
            param_value=75,
            unit="JPEG quality factor (1-100)",
            description="JPEG compression artifact with quality factor 75.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="quality",
            param_value=50,
            unit="JPEG quality factor (1-100)",
            description="JPEG compression artifact with quality factor 50.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="quality",
            param_value=30,
            unit="JPEG quality factor (1-100)",
            description="JPEG compression artifact with quality factor 30.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="quality",
            param_value=15,
            unit="JPEG quality factor (1-100)",
            description="JPEG compression artifact with quality factor 15.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="quality",
            param_value=5,
            unit="JPEG quality factor (1-100)",
            description="JPEG compression artifact with quality factor 5.",
        ),
    },
    # 12. Pixelation
    "pixelation": {
        1: SeverityConfig(
            severity=1,
            param_name="factor",
            param_value=0.8,
            unit="resolution downsampling factor",
            description="Downsampled by 0.8x then upsampled via nearest neighbor.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="factor",
            param_value=0.6,
            unit="resolution downsampling factor",
            description="Downsampled by 0.6x then upsampled via nearest neighbor.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="factor",
            param_value=0.4,
            unit="resolution downsampling factor",
            description="Downsampled by 0.4x then upsampled via nearest neighbor.",
        ),
        4: SeverityConfig(
            severity=4,
            param_name="factor",
            param_value=0.25,
            unit="resolution downsampling factor",
            description="Downsampled by 0.25x then upsampled via nearest neighbor.",
        ),
        5: SeverityConfig(
            severity=5,
            param_name="factor",
            param_value=0.15,
            unit="resolution downsampling factor",
            description="Downsampled by 0.15x then upsampled via nearest neighbor.",
        ),
    },
}

ALL_CORRUPTION_TYPES = list(CORRUPTION_SEVERITIES.keys())


def get_severity_config(corruption_type: str, severity: int) -> SeverityConfig:
    """Retrieve severity configuration for a corruption type and level.

    Args:
        corruption_type: Name of corruption.
        severity: Severity level (1 to 5).

    Returns:
        SeverityConfig instance.
    """
    if corruption_type not in CORRUPTION_SEVERITIES:
        raise ValueError(
            f"Unknown corruption type: {corruption_type}. Available: {list(CORRUPTION_SEVERITIES.keys())}"
        )
    if severity not in CORRUPTION_SEVERITIES[corruption_type]:
        raise ValueError(
            f"Invalid severity {severity} for {corruption_type}. Available: {list(CORRUPTION_SEVERITIES[corruption_type].keys())}"
        )
    return CORRUPTION_SEVERITIES[corruption_type][severity]


def serialize_severities() -> dict[str, Any]:
    """Return dictionary representation of all corruption parameters."""
    return {
        c_type: {
            s: {
                "param_name": cfg.param_name,
                "param_value": cfg.param_value,
                "unit": cfg.unit,
                "description": cfg.description,
            }
            for s, cfg in s_dict.items()
        }
        for c_type, s_dict in CORRUPTION_SEVERITIES.items()
    }
