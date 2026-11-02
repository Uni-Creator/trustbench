"""Robustness evaluation module for TrustBench."""

from trustbench.robustness.corruptions import (
    CorruptedDataset,
    apply_brightness,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
    corrupt_pil_image,
)
from trustbench.robustness.evaluator import RobustnessEvaluator, compute_file_sha256
from trustbench.robustness.report import (
    format_robustness_markdown_table,
    format_selective_markdown_table,
)
from trustbench.robustness.severity import (
    CORRUPTION_SEVERITIES,
    SeverityConfig,
    get_severity_config,
    serialize_severities,
)

__all__ = [
    "CORRUPTION_SEVERITIES",
    "CorruptedDataset",
    "RobustnessEvaluator",
    "SeverityConfig",
    "apply_brightness",
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_occlusion",
    "compute_file_sha256",
    "corrupt_pil_image",
    "format_robustness_markdown_table",
    "format_selective_markdown_table",
    "get_severity_config",
    "serialize_severities",
]
