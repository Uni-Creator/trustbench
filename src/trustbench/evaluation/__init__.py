"""Evaluation metrics and compatibility exports for the evaluation pipeline."""

from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
    compute_error_detection_metrics,
    compute_slice_error_rates,
)
from trustbench.evaluation.selective import (
    compute_risk_coverage,
    compute_selective_prediction_metrics,
)
from trustbench.evaluation.uncertainty import (
    apply_temperature_scaling,
    compute_energy_scores,
    compute_predictive_entropy,
    compute_uncertainty_metrics,
    fit_temperature,
)

__all__ = [
    "IndexedDataset",
    "apply_temperature_scaling",
    "compute_binary_detection_curve",
    "compute_calibration_metrics",
    "compute_classification_metrics",
    "compute_energy_scores",
    "compute_error_detection_metrics",
    "compute_predictive_entropy",
    "compute_risk_coverage",
    "compute_selective_prediction_metrics",
    "compute_slice_error_rates",
    "compute_uncertainty_metrics",
    "fit_temperature",
    "get_environment_info",
    "get_git_commit",
    "load_experiment_artifacts",
    "load_model_from_checkpoint",
    "resolve_model_architecture",
    "run_evaluation",
    "run_inference",
    "save_experiment_artifacts",
    "set_seed",
]


def __getattr__(name: str):
    """Lazily load inference, artifact, and runner helpers (PyTorch/IO)."""
    if name in {"load_experiment_artifacts", "save_experiment_artifacts"}:
        from trustbench.reporting.artifacts import (
            load_experiment_artifacts,
            save_experiment_artifacts,
        )

        exports = {
            "load_experiment_artifacts": load_experiment_artifacts,
            "save_experiment_artifacts": save_experiment_artifacts,
        }
        return exports[name]
    if name in {
        "IndexedDataset",
        "load_model_from_checkpoint",
        "resolve_model_architecture",
        "run_inference",
    }:
        from trustbench.evaluation.inference import (
            IndexedDataset,
            load_model_from_checkpoint,
            resolve_model_architecture,
            run_inference,
        )

        exports = {
            "IndexedDataset": IndexedDataset,
            "load_model_from_checkpoint": load_model_from_checkpoint,
            "resolve_model_architecture": resolve_model_architecture,
            "run_inference": run_inference,
        }
        return exports[name]
    if name in {"get_environment_info", "get_git_commit", "set_seed"}:
        from trustbench.core.reproducibility import (
            get_environment_info,
            get_git_commit,
            set_seed,
        )

        exports = {
            "get_environment_info": get_environment_info,
            "get_git_commit": get_git_commit,
            "set_seed": set_seed,
        }
        return exports[name]
    if name == "run_evaluation":
        from trustbench.evaluation.runner import run_evaluation

        return run_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
