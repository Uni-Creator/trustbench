"""Structured evaluation outputs."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationResult:
    """Result of a core evaluation run without persistence."""

    metrics: dict[str, Any]
    predictions: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    logits: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    capabilities: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        """Convenience accessor for the metrics summary block."""
        summary = self.metrics.get("summary")
        return summary if isinstance(summary, dict) else {}

    def to_predictions_dict(self) -> dict[str, np.ndarray]:
        """Serialize arrays in the on-disk predictions.npz layout."""
        payload: dict[str, np.ndarray] = {
            "predictions": self.predictions,
            "labels": self.labels,
            "sample_ids": self.sample_ids,
        }
        if self.logits is not None:
            payload["logits"] = self.logits
        if self.probabilities is not None:
            payload["probabilities"] = self.probabilities
        return payload

    def save(
        self,
        output_dir: str | Any,
        *,
        experiment_name: str = "unnamed",
        extra_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Persist evaluation result artifacts to output_dir."""
        from trustbench.reporting.result_io import save_result

        return save_result(
            self,
            output_dir=output_dir,
            experiment_name=experiment_name,
            extra_metadata=extra_metadata,
        )
