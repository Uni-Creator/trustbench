"""Core evaluation engine: adapters, inference, and structured results."""

from trustbench.core.evaluator import evaluate
from trustbench.core.exceptions import (
    ShapeError,
    TrustBenchError,
    UnsupportedCapabilityError,
)
from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.core.reproducibility import (
    get_environment_info,
    get_git_commit,
    set_seed,
)
from trustbench.core.results import EvaluationResult

__all__ = [
    "DatasetAdapter",
    "EvaluationResult",
    "ModelAdapter",
    "ShapeError",
    "TrustBenchError",
    "UnsupportedCapabilityError",
    "evaluate",
    "get_environment_info",
    "get_git_commit",
    "set_seed",
]
