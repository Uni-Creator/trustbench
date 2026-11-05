"""Artifact persistence and report formatting."""

from trustbench.reporting.artifacts import (
    NpEncoder,
    load_experiment_artifacts,
    save_experiment_artifacts,
)
from trustbench.reporting.hashing import (
    compute_file_sha256,
    compute_file_sha256_or_none,
)
from trustbench.reporting.html_report import generate_html_report
from trustbench.reporting.plots import (
    ReportingDependencyError,
    auroc_curve_data,
    confidence_histogram_data,
    confusion_matrix_data,
    experiment_comparison_data,
    reliability_diagram_data,
    risk_coverage_data,
    robustness_degradation_data,
    severity_comparison_data,
)
from trustbench.reporting.result_io import save_result
from trustbench.reporting.schema import (
    REPORT_SCHEMA_VERSION,
    build_report_dict,
    generate_json_report,
)
from trustbench.reporting.tables import (
    format_robustness_markdown_table,
    format_selective_markdown_table,
)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "NpEncoder",
    "ReportingDependencyError",
    "auroc_curve_data",
    "build_report_dict",
    "compute_file_sha256",
    "compute_file_sha256_or_none",
    "confidence_histogram_data",
    "confusion_matrix_data",
    "experiment_comparison_data",
    "format_robustness_markdown_table",
    "format_selective_markdown_table",
    "generate_html_report",
    "generate_json_report",
    "load_experiment_artifacts",
    "reliability_diagram_data",
    "risk_coverage_data",
    "robustness_degradation_data",
    "save_experiment_artifacts",
    "save_result",
    "severity_comparison_data",
]
