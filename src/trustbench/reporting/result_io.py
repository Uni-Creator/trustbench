"""Standalone helpers for persisting EvaluationResult to disk.

``save_result`` writes the standard artifact bundle (metadata, metrics, config,
predictions) and also generates a JSON report and an HTML report.  Callers can
opt out of either report by passing the respective flag.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trustbench.core.reproducibility import get_environment_info
from trustbench.reporting.artifacts import save_experiment_artifacts

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from trustbench.config.schema import TrustBenchConfig
    from trustbench.core.results import EvaluationResult


def save_result(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    experiment_name: str = "unnamed",
    extra_metadata: dict[str, Any] | None = None,
    config: TrustBenchConfig | None = None,
    lineage: dict[str, Any] | None = None,
    generate_json: bool = True,
    generate_html: bool = True,
) -> Path:
    """Persist an :class:`EvaluationResult` to *output_dir*.

    Writes:
    - ``metadata.json``
    - ``metrics.json``
    - ``config.json``
    - ``predictions.npz``
    - ``report.json``  (versioned JSON report, unless *generate_json* is False)
    - ``report.html``  (standalone HTML report, unless *generate_html* is False)

    Args:
        result: The evaluation result to save.
        output_dir: Destination directory (created if necessary).
        experiment_name: Optional label embedded in metadata.
        extra_metadata: Additional key-value pairs merged into metadata.
        config: Optional parsed :class:`~trustbench.config.schema.TrustBenchConfig`.
            When provided, embedded in both JSON and HTML reports.
        lineage: Optional provenance dict from
            :func:`~trustbench.experiments.lineage.build_lineage_record`.
        generate_json: Whether to write ``report.json``.  Default ``True``.
        generate_html: Whether to write ``report.html``.  Default ``True``.

    Returns:
        The resolved output directory path.
    """
    out = Path(output_dir)

    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "environment": get_environment_info(),
        **result.metadata,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    save_experiment_artifacts(
        output_dir=out,
        metadata=metadata,
        metrics=result.metrics,
        config=result.metadata,
        predictions=result.to_predictions_dict(),
    )

    if generate_json:
        try:
            from trustbench.reporting.schema import generate_json_report

            generate_json_report(
                result,
                out,
                config=config,
                experiment_name=experiment_name,
                lineage=lineage,
                extra_metadata=extra_metadata,
            )
        except Exception:
            logger.debug("Failed to generate JSON report", exc_info=True)

    if generate_html:
        try:
            from trustbench.reporting.html_report import generate_html_report

            generate_html_report(
                result,
                out,
                config=config,
                experiment_name=experiment_name,
                lineage=lineage,
            )
        except Exception:
            logger.debug("Failed to generate HTML report", exc_info=True)

    return out
