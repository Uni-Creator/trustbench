"""TrustBench: Evaluation framework for machine learning model reliability."""

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

__all__ = [
    "DatasetAdapter",
    "EvaluationResult",
    "ModelAdapter",
    "ShapeError",
    "ShiftType",
    "TrustBench",
    "TrustBenchError",
    "UnsupportedCapabilityError",
    "evaluate",
    "evaluate_distribution_shift",
]
