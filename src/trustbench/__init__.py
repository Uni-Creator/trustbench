"""TrustBench: Evaluation framework for machine learning model reliability."""

from trustbench.core.evaluator import evaluate
from trustbench.core.exceptions import (
    ShapeError,
    TrustBenchError,
    UnsupportedCapabilityError,
)
from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.core.results import EvaluationResult

__all__ = [
    "DatasetAdapter",
    "EvaluationResult",
    "ModelAdapter",
    "ShapeError",
    "TrustBenchError",
    "UnsupportedCapabilityError",
    "evaluate",
]
