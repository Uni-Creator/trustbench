"""Evaluation metrics and compatibility exports for the evaluation pipeline."""

from trustbench.evaluation.calibration import (
    compute_adaptive_ece,
    compute_calibration_metrics,
    compute_nll,
    compute_reliability_diagram_data,
)
from trustbench.evaluation.classification import (
    compute_balanced_accuracy,
    compute_classification_metrics,
)
from trustbench.evaluation.distribution_shift import (
    ShiftType,
    compute_label_distribution_shift,
    compute_ood_detection_metrics,
    compute_tv_distance,
    evaluate_distribution_shift,
)
from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
    compute_detection_error,
    compute_error_detection_metrics,
    compute_fpr_at_tpr,
    compute_slice_error_rates,
)
from trustbench.evaluation.selective import (
    compute_optimal_aurc,
    compute_risk_coverage,
    compute_selective_prediction_metrics,
    get_coverage_at_risk,
    get_risk_at_coverage,
)
from trustbench.evaluation.uncertainty import (
    apply_temperature_scaling,
    compute_energy_scores,
    compute_ensemble_uncertainties,
    compute_predictive_entropy,
    compute_uncertainty_metrics,
    evaluate_deep_ensemble,
    evaluate_mc_dropout,
    fit_temperature,
)

__all__ = [
    "IndexedDataset",
    "ShiftType",
    "apply_temperature_scaling",
    "compute_adaptive_ece",
    "compute_balanced_accuracy",
    "compute_binary_detection_curve",
    "compute_calibration_metrics",
    "compute_classification_metrics",
    "compute_detection_error",
    "compute_energy_scores",
    "compute_ensemble_uncertainties",
    "compute_error_detection_metrics",
    "compute_fpr_at_tpr",
    "compute_label_distribution_shift",
    "compute_nll",
    "compute_ood_detection_metrics",
    "compute_optimal_aurc",
    "compute_predictive_entropy",
    "compute_reliability_diagram_data",
    "compute_risk_coverage",
    "compute_selective_prediction_metrics",
    "compute_slice_error_rates",
    "compute_tv_distance",
    "compute_uncertainty_metrics",
    "evaluate_deep_ensemble",
    "evaluate_distribution_shift",
    "evaluate_mc_dropout",
    "fit_temperature",
    "get_coverage_at_risk",
    "get_environment_info",
    "get_git_commit",
    "get_risk_at_coverage",
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
