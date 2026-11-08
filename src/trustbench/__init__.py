"""TrustBench: Evaluation framework for machine learning model reliability."""

__version__ = "0.2.0"

from trustbench.benchmark import TrustBench
from trustbench.core.evaluator import evaluate
from trustbench.core.exceptions import (
    SecurityError,
    ShapeError,
    TrustBenchError,
    UnsupportedCapabilityError,
)
from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.core.results import EvaluationResult
from trustbench.evaluation.distribution_shift import (
    ShiftType,
    compute_label_distribution_shift,
    compute_ood_detection_metrics,
    evaluate_distribution_shift,
)
from trustbench.evaluation.fairness import compute_fairness_metrics
from trustbench.evaluation.regression import RegressionResult, evaluate_regression
from trustbench.evaluation.retrieval import RetrievalResult, evaluate_retrieval
from trustbench.evaluation.segmentation import (
    SegmentationResult,
    evaluate_segmentation,
)
from trustbench.metrics.regression import compute_regression_metrics
from trustbench.metrics.retrieval import compute_retrieval_metrics
from trustbench.metrics.segmentation import compute_segmentation_metrics
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
    "JAXModelAdapter",
    "KerasModelAdapter",
    "ModelAdapter",
    "ONNXModelAdapter",
    "PyTorchModelAdapter",
    "RegressionResult",
    "RetrievalResult",
    "SecurityError",
    "SegmentationResult",
    "ShapeError",
    "ShiftType",
    "SklearnModelAdapter",
    "TrustBench",
    "TrustBenchError",
    "UnsupportedCapabilityError",
    "__version__",
    "compute_fairness_metrics",
    "compute_label_distribution_shift",
    "compute_ood_detection_metrics",
    "compute_regression_metrics",
    "compute_retrieval_metrics",
    "compute_segmentation_metrics",
    "discover_plugins",
    "evaluate",
    "evaluate_distribution_shift",
    "evaluate_regression",
    "evaluate_retrieval",
    "evaluate_segmentation",
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
    if name == "SklearnModelAdapter":
        from trustbench.models.sklearn import SklearnModelAdapter

        return SklearnModelAdapter
    if name == "PyTorchModelAdapter":
        from trustbench.models.pytorch import PyTorchModelAdapter

        return PyTorchModelAdapter
    if name == "ONNXModelAdapter":
        from trustbench.models.onnx import ONNXModelAdapter

        return ONNXModelAdapter
    if name == "KerasModelAdapter":
        from trustbench.models.keras import KerasModelAdapter

        return KerasModelAdapter
    if name == "JAXModelAdapter":
        from trustbench.models.jax import JAXModelAdapter

        return JAXModelAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

