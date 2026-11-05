"""Classification performance metrics for model evaluation."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute balanced classification accuracy (unweighted mean of recalls per class).

    Args:
        labels: 1D ground truth labels.
        predictions: 1D predicted class indices.

    Returns:
        Balanced accuracy score in [0.0, 1.0].
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if len(labels) == 0:
        return 0.0
    return float(balanced_accuracy_score(labels, predictions))


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Mathematical definitions:
      - Accuracy: (1 / N) * sum_i I(y_i == ŷ_i)
      - Balanced Accuracy: (1 / C) * sum_{c=1}^C Recall_c
      - Precision (macro): (1 / C) * sum_{c=1}^C Precision_c
      - Recall (macro): (1 / C) * sum_{c=1}^C Recall_c
      - Macro F1: (1 / C) * sum_{c=1}^C F1_c
      - Micro F1: 2 * (TP_total) / (2 * TP_total + FP_total + FN_total)
      - Weighted F1: sum_{c=1}^C (N_c / N) * F1_c

    Args:
        labels: 1D array of ground truth labels of shape (N,).
        predictions: 1D array of predicted class indices of shape (N,).
        class_names: Optional sequence of class names.

    Returns:
        Dictionary containing:
            - accuracy: Overall classification accuracy (float)
            - balanced_accuracy: Balanced accuracy across classes (float)
            - macro_f1: Unweighted average F1 score across classes (float)
            - micro_f1: Micro-averaged F1 score (float)
            - weighted_f1: Support-weighted average F1 score (float)
            - macro_precision: Unweighted average precision (float)
            - micro_precision: Micro-averaged precision (float)
            - weighted_precision: Support-weighted average precision (float)
            - macro_recall: Unweighted average recall (float)
            - micro_recall: Micro-averaged recall (float)
            - weighted_recall: Support-weighted average recall (float)
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
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "micro_precision": 0.0,
            "weighted_precision": 0.0,
            "macro_recall": 0.0,
            "micro_recall": 0.0,
            "weighted_recall": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }

    unique_labels = np.unique(np.concatenate([labels, predictions]))
    num_classes = len(class_names) if class_names is not None else int(max(unique_labels.max() + 1, 1))
    all_classes = list(range(num_classes))

    accuracy = float(np.mean(labels == predictions))
    balanced_acc = compute_balanced_accuracy(labels, predictions)

    macro_f1 = float(f1_score(labels, predictions, labels=all_classes, average="macro", zero_division=0))
    micro_f1 = float(f1_score(labels, predictions, labels=all_classes, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(labels, predictions, labels=all_classes, average="weighted", zero_division=0))

    prec_macro, rec_macro, _, _ = precision_recall_fscore_support(
        labels, predictions, labels=all_classes, average="macro", zero_division=0
    )
    prec_micro, rec_micro, _, _ = precision_recall_fscore_support(
        labels, predictions, labels=all_classes, average="micro", zero_division=0
    )
    prec_weighted, rec_weighted, _, _ = precision_recall_fscore_support(
        labels, predictions, labels=all_classes, average="weighted", zero_division=0
    )

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
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": float(prec_macro),
        "micro_precision": float(prec_micro),
        "weighted_precision": float(prec_weighted),
        "macro_recall": float(rec_macro),
        "micro_recall": float(rec_micro),
        "weighted_recall": float(rec_weighted),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }
