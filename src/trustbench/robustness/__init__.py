"""Robustness evaluation module for TrustBench."""

from trustbench.robustness.aggregation import compute_robustness_aggregation
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
from trustbench.robustness.evaluator import RobustnessEvaluator
from trustbench.robustness.report import (
    format_robustness_markdown_table,
    format_selective_markdown_table,
)
from trustbench.robustness.severity import (
    ALL_CORRUPTION_TYPES,
    CORRUPTION_SEVERITIES,
    SeverityConfig,
    get_severity_config,
    serialize_severities,
)

__all__ = [
    "ALL_CORRUPTION_TYPES",
    "CORRUPTION_SEVERITIES",
    "CorruptedDataset",
    "RobustnessEvaluator",
    "SeverityConfig",
    "apply_brightness",
    "apply_contrast",
    "apply_defocus_blur",
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_impulse_noise",
    "apply_jpeg_compression",
    "apply_motion_blur",
    "apply_occlusion",
    "apply_pixelation",
    "apply_saturation",
    "apply_shot_noise",
    "compute_file_sha256",
    "compute_robustness_aggregation",
    "corrupt_pil_image",
    "format_robustness_markdown_table",
    "format_selective_markdown_table",
    "get_severity_config",
    "serialize_severities",
]
