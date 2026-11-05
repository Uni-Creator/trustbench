"""Versioned JSON report schema for TrustBench evaluation results.

Schema version: 1.0

The schema is stable and machine-readable.  Every generated report contains a
``schema_version`` field so consumers can detect breaking changes without
inspecting field names.

No metric calculations are performed here — values are taken directly from
:class:`~trustbench.core.results.EvaluationResult` and optional lineage/
experiment data passed by the caller.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trustbench
from trustbench.reporting.artifacts import NpEncoder

if TYPE_CHECKING:
    from trustbench.config.schema import TrustBenchConfig
    from trustbench.core.results import EvaluationResult

#: Current stable schema version.  Increment the minor version for backward-
#: compatible additions; increment the major version for breaking changes.
REPORT_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------


def build_report_dict(
    result: EvaluationResult,
    *,
    config: TrustBenchConfig | None = None,
    experiment_name: str = "unnamed",
    experiment_id: str | None = None,
    lineage: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Construct a fully structured, schema-versioned report dictionary.

    The returned dictionary is JSON-serialisable via :class:`~trustbench.reporting.artifacts.NpEncoder`
    and can be passed directly to :func:`generate_json_report`.

    Args:
        result: Completed evaluation result.  Metrics are read as-is — no
            recalculation is performed.
        config: Optional parsed configuration.  Serialised under ``configuration``.
        experiment_name: Human-readable label for the experiment.
        experiment_id: Optional unique identifier (e.g. UUID).
        lineage: Optional lineage/provenance dict produced by
            :func:`~trustbench.experiments.lineage.build_lineage_record`.
        extra_metadata: Arbitrary caller-supplied key-value pairs merged into
            the ``execution_metadata`` block.
        include_predictions: When ``True`` the ``predictions`` block is
            populated from :meth:`~trustbench.core.results.EvaluationResult.to_predictions_dict`.
            Defaults to ``False`` because per-sample arrays can be large.

    Returns:
        A dict that conforms to the TrustBench report schema v1.0.
    """
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()

    # -- Execution metadata
    execution_metadata: dict[str, Any] = {
        "generated_at": generated_at,
        "trustbench_version": trustbench.__version__,
        "experiment_name": experiment_name,
    }
    if experiment_id is not None:
        execution_metadata["experiment_id"] = experiment_id
    if extra_metadata:
        execution_metadata.update(extra_metadata)

    # -- Configuration block
    configuration: dict[str, Any] | None = None
    if config is not None:
        try:
            configuration = config.model_dump(mode="json")
        except (AttributeError, ValueError):
            configuration = None

    # -- Model & dataset metadata (from result.metadata)
    model_metadata: dict[str, Any] = {}
    dataset_metadata: dict[str, Any] = {}
    for key, val in result.metadata.items():
        if key.startswith("model_") or key in ("model_class", "num_parameters", "device"):
            model_metadata[key] = val
        elif key.startswith("dataset_") or key in ("dataset_name", "num_samples", "class_names", "num_classes"):
            dataset_metadata[key] = val

    # -- Metrics — copied verbatim from result.metrics, grouped by domain
    classification = result.metrics.get("classification", {})
    calibration = result.metrics.get("calibration", {})
    uncertainty = result.metrics.get("uncertainty", {})
    error_detection = result.metrics.get("error_detection", {})
    selective = result.metrics.get("selective", {})
    robustness = result.metrics.get("robustness", {})
    summary = result.summary

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "execution_metadata": execution_metadata,
        "model_metadata": model_metadata or result.metadata.get("model", {}),
        "dataset_metadata": dataset_metadata or result.metadata.get("dataset", {}),
        "summary": summary,
        "metrics": {
            "classification": classification,
            "calibration": _strip_array_fields(calibration),
            "uncertainty": uncertainty,
            "error_detection": error_detection,
            "selective_prediction": selective,
            "robustness": robustness,
        },
        "lineage": lineage,
        "configuration": configuration,
    }

    if include_predictions:
        try:
            pred_dict = result.to_predictions_dict()
            report["predictions"] = {k: v.tolist() for k, v in pred_dict.items()}
        except Exception:  # noqa: BLE001
            report["predictions"] = None

    return report


def _strip_array_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Remove large per-bin array fields that are better read from plot data."""
    skip = {"bin_confidences", "bin_accuracies", "bin_counts"}
    return {k: v for k, v in d.items() if k not in skip}


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------


def generate_json_report(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    config: TrustBenchConfig | None = None,
    experiment_name: str = "unnamed",
    experiment_id: str | None = None,
    lineage: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    include_predictions: bool = False,
    filename: str = "report.json",
) -> Path:
    """Write a versioned JSON report to *output_dir/filename*.

    The file is written atomically: data is serialised to a string first,
    then written in a single ``write_text`` call to minimise partial-write risk.

    Args:
        result: Completed evaluation result.
        output_dir: Directory where the report will be written (created if needed).
        config: Optional evaluation configuration.
        experiment_name: Label embedded in ``execution_metadata``.
        experiment_id: Optional unique experiment identifier.
        lineage: Optional provenance dict.
        extra_metadata: Additional key-value pairs for ``execution_metadata``.
        include_predictions: Embed per-sample prediction arrays (may be large).
        filename: Output filename.  Defaults to ``report.json``.

    Returns:
        Path to the written JSON report file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_dict = build_report_dict(
        result,
        config=config,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        lineage=lineage,
        extra_metadata=extra_metadata,
        include_predictions=include_predictions,
    )

    content = json.dumps(report_dict, indent=2, cls=NpEncoder)
    report_path = out / filename
    report_path.write_text(content, encoding="utf-8")
    return report_path
