"""Experiments package for TrustBench."""

from trustbench.experiments.comparison import (
    ExperimentComparator,
    validate_compatibility,
)
from trustbench.experiments.lineage import (
    MISSING,
    UNAVAILABLE,
    build_lineage_record,
    compute_checkpoint_sha256,
    compute_dataset_fingerprint,
)
from trustbench.experiments.manager import ExperimentManager
from trustbench.experiments.schemas import (
    ArtifactEntry,
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
    ExperimentDetail,
    ExperimentResponse,
    MetricEntry,
)

__all__ = [
    "MISSING",
    "UNAVAILABLE",
    "ArtifactEntry",
    "ExperimentComparator",
    "ExperimentComparisonRequest",
    "ExperimentComparisonResponse",
    "ExperimentDetail",
    "ExperimentManager",
    "ExperimentRegistry",
    "ExperimentResponse",
    "MetricEntry",
    "build_lineage_record",
    "compute_checkpoint_sha256",
    "compute_dataset_fingerprint",
    "validate_compatibility",
]


def __getattr__(name: str):
    if name == "ExperimentRegistry":
        from trustbench.experiments.registry import ExperimentRegistry

        return ExperimentRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
