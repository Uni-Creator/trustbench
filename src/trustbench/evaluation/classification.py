"""Classification performance metrics for model evaluation."""

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        labels: 1D array of ground truth labels of shape (N,).
        predictions: 1D array of predicted class indices of shape (N,).
        class_names: Optional sequence of class names.

    Returns:
        Dictionary containing:
            - accuracy: Overall classification accuracy (float)
            - macro_f1: Unweighted average F1 score across classes (float)
            - per_class: Dict with precision, recall, f1, and support per class
            - confusion_matrix: 2D confusion matrix as nested lists [[int, ...], ...]
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be 1D arrays.")
    if len(labels) != len(predictions):
        raise ValueError(f"Length mismatch: labels ({len(labels)}) vs predictions ({len(predictions)})")

    total_samples = len(labels)
    if total_samples == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }

    unique_labels = np.unique(np.concatenate([labels, predictions]))
    num_classes = len(class_names) if class_names is not None else int(max(unique_labels.max() + 1, 1))
    all_classes = list(range(num_classes))

    accuracy = float(np.mean(labels == predictions))
    macro_f1 = float(f1_score(labels, predictions, labels=all_classes, average="macro", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=all_classes, average=None, zero_division=0
    )

    per_class_metrics: dict[str, dict[str, float | int]] = {}
    for idx, c_idx in enumerate(all_classes):
        name = class_names[c_idx] if class_names is not None and c_idx < len(class_names) else f"class_{c_idx}"
        per_class_metrics[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(labels, predictions, labels=all_classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }
