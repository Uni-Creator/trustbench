"""TrustBench: Evaluation framework for machine learning model reliability."""

__version__ = "0.2.0"

from trustbench.benchmark import TrustBench
from trustbench.core.evaluator import evaluate
from trustbench.core.exceptions import (
    ShapeError,
    TrustBenchError,
    UnsupportedCapabilityError,
)
from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.core.results import EvaluationResult
from trustbench.evaluation.distribution_shift import (
    ShiftType,
    evaluate_distribution_shift,
)
from trustbench.registry import (
    GenericDatasetAdapter,
    GenericModelAdapter,
    discover_plugins,
    list_registered_corruptions,
    list_registered_metrics,
    load_plugin,
    register_corruption,
    register_dataset_adapter,
    register_metric,
    register_model_adapter,
    validate_dataset_adapter,
    validate_model_adapter,
)

__all__ = [
    "DatasetAdapter",
    "EvaluationResult",
    "GenericDatasetAdapter",
    "GenericModelAdapter",
    "ModelAdapter",
    "ShapeError",
    "ShiftType",
    "TrustBench",
    "TrustBenchError",
    "UnsupportedCapabilityError",
    "__version__",
    "discover_plugins",
    "evaluate",
    "evaluate_distribution_shift",
    "list_registered_corruptions",
    "list_registered_metrics",
    "load_plugin",
    "register_corruption",
    "register_dataset_adapter",
    "register_metric",
    "register_model_adapter",
    "validate_dataset_adapter",
    "validate_model_adapter",
]


def __getattr__(name: str):
    if name == "ExperimentManager":
        from trustbench.experiments.manager import ExperimentManager

        return ExperimentManager
    if name in ("FilesystemExperimentStore", "SQLiteExperimentStore"):
        from trustbench import storage

        return getattr(storage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

