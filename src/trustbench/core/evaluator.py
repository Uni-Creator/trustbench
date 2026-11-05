"""Persistence-free evaluation engine."""

from collections.abc import Iterable
from typing import Any

from trustbench.core.inference import run_adapter_inference
from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.core.reproducibility import set_seed
from trustbench.core.results import EvaluationResult
from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.error_detection import compute_error_detection_metrics
from trustbench.evaluation.selective import compute_selective_prediction_metrics
from trustbench.evaluation.uncertainty import compute_uncertainty_metrics
from trustbench.registry.metrics import METRIC_REGISTRY

DEFAULT_METRICS = ("classification", "calibration", "uncertainty")


def evaluate(
    model: ModelAdapter,
    dataset: DatasetAdapter,
    *,
    metrics: Iterable[str] | None = None,
    batch_size: int = 100,
    class_names: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
    temperature: float | None = None,
) -> EvaluationResult:
    """Evaluate a model adapter on a dataset adapter.

    Does not initialize a database, register an experiment, or write artifacts.
    """
    if seed is not None:
        set_seed(seed)

    requested = tuple(DEFAULT_METRICS if metrics is None else metrics)
    inference = run_adapter_inference(model, dataset, batch_size=batch_size)

    combined: dict[str, Any] = {}

    if "classification" in requested:
        combined["classification"] = compute_classification_metrics(
            labels=inference.labels,
            predictions=inference.predictions,
            class_names=class_names,
        )

    if "calibration" in requested:
        if inference.probabilities is None:
            combined["calibration"] = {
                "skipped": True,
                "reason": "probabilities unavailable; model does not expose logits or predict_proba",
            }
        else:
            combined["calibration"] = compute_calibration_metrics(
                labels=inference.labels,
                probabilities=inference.probabilities,
            )

    if "uncertainty" in requested:
        if inference.probabilities is None:
            combined["uncertainty"] = {
                "skipped": True,
                "reason": "probabilities unavailable; model does not expose logits or predict_proba",
            }
        else:
            combined["uncertainty"] = compute_uncertainty_metrics(
                probabilities=inference.probabilities,
            )

    if "error_detection" in requested:
        if inference.probabilities is None:
            combined["error_detection"] = {
                "skipped": True,
                "reason": "probabilities unavailable",
            }
        else:
            combined["error_detection"] = compute_error_detection_metrics(
                labels=inference.labels,
                predictions=inference.predictions,
                probabilities=inference.probabilities,
                logits=inference.logits,
                temperature=temperature,
            )

    if "selective_prediction" in requested:
        if inference.probabilities is None:
            combined["selective_prediction"] = {
                "skipped": True,
                "reason": "probabilities unavailable",
            }
        else:
            combined["selective_prediction"] = compute_selective_prediction_metrics(
                labels=inference.labels,
                predictions=inference.predictions,
                probabilities=inference.probabilities,
            )

    # Custom registered metrics
    for m_name in requested:
        if m_name in ("classification", "calibration", "uncertainty", "error_detection", "selective_prediction"):
            continue
        custom_metric = METRIC_REGISTRY.get(m_name)
        if custom_metric is not None:
            try:
                res = custom_metric(
                    predictions=inference.predictions,
                    labels=inference.labels,
                    probabilities=inference.probabilities,
                    logits=inference.logits,
                    sample_ids=inference.sample_ids,
                    class_names=class_names,
                )
                combined[m_name] = res
            except Exception as exc:  # noqa: BLE001
                combined[m_name] = {"error": str(exc), "status": "failed"}
        else:
            builtins = list(DEFAULT_METRICS) + ["error_detection", "selective_prediction"]
            registered = list(METRIC_REGISTRY.list_metrics().keys())
            raise ValueError(
                f"Unknown metric {m_name!r}. "
                f"Available built-in metrics: {builtins}. "
                f"Registered custom metrics: {registered}."
            )

    classification = combined.get("classification", {})
    calibration = combined.get("calibration", {})
    uncertainty = combined.get("uncertainty", {})
    summary_data = {
        "accuracy": classification.get("accuracy"),
        "balanced_accuracy": classification.get("balanced_accuracy"),
        "macro_f1": classification.get("macro_f1"),
        "brier_score": calibration.get("brier_score"),
        "ece": calibration.get("ece"),
        "adaptive_ece": calibration.get("adaptive_ece"),
        "mean_entropy": uncertainty.get("mean_entropy"),
        "total_samples": len(inference.labels),
    }

    # Include scalar custom metrics into summary
    for m_name in requested:
        val = combined.get(m_name)
        if isinstance(val, (int, float)):
            summary_data[m_name] = float(val)

    combined["summary"] = summary_data

    return EvaluationResult(
        metrics=combined,
        predictions=inference.predictions,
        labels=inference.labels,
        sample_ids=inference.sample_ids,
        logits=inference.logits,
        probabilities=inference.probabilities,
        capabilities=inference.capabilities,
        metadata={
            "batch_size": batch_size,
            "metrics": list(requested),
            "class_names": list(class_names) if class_names is not None else None,
            "seed": seed,
        },
    )
