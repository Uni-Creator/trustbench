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
    "gaussian_noise": {
        1: SeverityConfig(
            severity=1,
            param_name="sigma",
            param_value=0.03,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.03 added to normalized pixel values.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="sigma",
            param_value=0.08,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.08 added to normalized pixel values.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="sigma",
            param_value=0.15,
            unit="fractional pixel intensity (standard deviation in [0, 1])",
            description="Additive zero-mean Gaussian noise with sigma=0.15 added to normalized pixel values.",
        ),
    },
    "gaussian_blur": {
        1: SeverityConfig(
            severity=1,
            param_name="sigma",
            param_value=0.5,
            unit="pixels (Gaussian kernel standard deviation)",
            description="Gaussian filter blur with kernel std sigma=0.5.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="sigma",
            param_value=1.0,
            unit="pixels (Gaussian kernel standard deviation)",
            description="Gaussian filter blur with kernel std sigma=1.0.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="sigma",
            param_value=1.5,
            unit="pixels (Gaussian kernel standard deviation)",
            description="Gaussian filter blur with kernel std sigma=1.5.",
        ),
    },
    "brightness": {
        1: SeverityConfig(
            severity=1,
            param_name="factor",
            param_value=0.8,
            unit="multiplier (relative luminance scale)",
            description="Image pixel values scaled by 0.8x before normalization.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="factor",
            param_value=0.6,
            unit="multiplier (relative luminance scale)",
            description="Image pixel values scaled by 0.6x before normalization.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="factor",
            param_value=0.4,
            unit="multiplier (relative luminance scale)",
            description="Image pixel values scaled by 0.4x before normalization.",
        ),
    },
    "occlusion": {
        1: SeverityConfig(
            severity=1,
            param_name="area_fraction",
            param_value=0.10,
            unit="area percentage (0.10 = 10% bounding box area)",
            description="Centered rectangular patch masking 10% of the image pixels with zero/mean value.",
        ),
        2: SeverityConfig(
            severity=2,
            param_name="area_fraction",
            param_value=0.20,
            unit="area percentage (0.20 = 20% bounding box area)",
            description="Centered rectangular patch masking 20% of the image pixels with zero/mean value.",
        ),
        3: SeverityConfig(
            severity=3,
            param_name="area_fraction",
            param_value=0.30,
            unit="area percentage (0.30 = 30% bounding box area)",
            description="Centered rectangular patch masking 30% of the image pixels with zero/mean value.",
        ),
    },
}


def get_severity_config(corruption_type: str, severity: int) -> SeverityConfig:
    """Retrieve severity configuration for a corruption type and level.

    Args:
        corruption_type: Name of corruption ('gaussian_noise', 'gaussian_blur', etc.).
        severity: Severity level (1, 2, 3).

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
