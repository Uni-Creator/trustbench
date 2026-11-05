"""Standalone helpers for persisting EvaluationResult to disk."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trustbench.core.reproducibility import get_environment_info
from trustbench.reporting.artifacts import save_experiment_artifacts

if TYPE_CHECKING:
    from trustbench.core.results import EvaluationResult


def save_result(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    experiment_name: str = "unnamed",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist an :class:`EvaluationResult` to *output_dir*.

    Writes four files:
    - ``metadata.json``
    - ``metrics.json``
    - ``config.json``
    - ``predictions.npz``

    Args:
        result: The evaluation result to save.
        output_dir: Destination directory (created if necessary).
        experiment_name: Optional label embedded in metadata.
        extra_metadata: Additional key-value pairs merged into metadata.

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

    return save_experiment_artifacts(
        output_dir=out,
        metadata=metadata,
        metrics=result.metrics,
        config=result.metadata,
        predictions=result.to_predictions_dict(),
    )
