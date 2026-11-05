"""Distribution shift evaluation interfaces and metrics.

Covers:
  - Covariate Shift: P(X) changes while P(Y|X) remains constant.
  - Domain Shift: Changes in input representation, style, or environmental factors.
  - Label Distribution Shift: P(Y) changes, modifying class priors.
  - Corrupted Distributions: Systematic physical/digital corruptions applied to inputs.
  - Out-of-Distribution (OOD): Inputs belonging to classes unseen during training.
"""

from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.error_detection import compute_binary_detection_curve
from trustbench.evaluation.uncertainty import compute_predictive_entropy


class ShiftType(str, Enum):
    """Categorization of distribution shift phenomena."""

    COVARIATE_SHIFT = "covariate_shift"
    DOMAIN_SHIFT = "domain_shift"
    LABEL_SHIFT = "label_shift"
    CORRUPTED_DISTRIBUTION = "corrupted_distribution"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


def compute_tv_distance(p_source: np.ndarray, p_target: np.ndarray) -> float:
    """Compute Total Variation (TV) distance between two discrete probability distributions.

    Mathematical definition:
      TV(P, Q) = 0.5 * sum_c |P(c) - Q(c)| in [0.0, 1.0]

    Args:
        p_source: 1D normalized probability vector over classes.
        p_target: 1D normalized probability vector over classes.

    Returns:
        TV distance in [0.0, 1.0].
    """
    p = np.asarray(p_source, dtype=float)
    q = np.asarray(p_target, dtype=float)
    if len(p) == 0 or len(q) == 0 or len(p) != len(q):
        return 0.0
    return float(0.5 * np.sum(np.abs(p - q)))


def compute_label_distribution_shift(
    labels_source: np.ndarray,
    labels_target: np.ndarray,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Quantify prior distribution shift between source and target label sets.

    Args:
        labels_source: 1D array of source labels (N_s,).
        labels_target: 1D array of target labels (N_t,).
        num_classes: Optional total number of classes.

    Returns:
        Dictionary with prior distributions, TV distance, and Jensen-Shannon distance.
    """
    labels_s = np.asarray(labels_source, dtype=int)
    labels_t = np.asarray(labels_target, dtype=int)

    if len(labels_s) == 0 or len(labels_t) == 0:
        return {
            "tv_distance": 0.0,
            "jensen_shannon_distance": 0.0,
            "source_priors": [],
            "target_priors": [],
        }

    max_c = max(int(np.max(labels_s)), int(np.max(labels_t)))
    n_c = num_classes if num_classes is not None else max_c + 1

    counts_s = np.bincount(labels_s[labels_s >= 0], minlength=n_c)[:n_c]
    counts_t = np.bincount(labels_t[labels_t >= 0], minlength=n_c)[:n_c]

    priors_s = counts_s / max(1, np.sum(counts_s))
    priors_t = counts_t / max(1, np.sum(counts_t))

    tv = compute_tv_distance(priors_s, priors_t)
    js = float(jensenshannon(priors_s, priors_t))

    return {
        "tv_distance": tv,
        "jensen_shannon_distance": js,
        "source_priors": priors_s.tolist(),
        "target_priors": priors_t.tolist(),
    }


def compute_ood_detection_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict[str, float]:
    """Evaluate how well an uncertainty or score distinguishes ID samples from OOD samples.

    Higher scores indicate higher likelihood of being OOD.

    Args:
        id_scores: 1D array of scores for In-Distribution samples (negative class 0).
        ood_scores: 1D array of scores for Out-of-Distribution samples (positive class 1).

    Returns:
        Dict with 'auroc', 'auprc', 'fpr_at_95_tpr', and 'detection_accuracy'.
    """
    id_arr = np.asarray(id_scores, dtype=float)
    ood_arr = np.asarray(ood_scores, dtype=float)

    if len(id_arr) == 0 or len(ood_arr) == 0:
        return {
            "auroc": 0.5,
            "auprc": 0.5,
            "fpr_at_95_tpr": 0.0,
            "detection_accuracy": 0.5,
        }

    y_true = np.concatenate([np.zeros(len(id_arr), dtype=int), np.ones(len(ood_arr), dtype=int)])
    scores = np.concatenate([id_arr, ood_arr])

    curve = compute_binary_detection_curve(y_true, scores)
    return {
        "auroc": curve["auroc"],
        "auprc": curve["auprc"],
        "fpr_at_95_tpr": curve["fpr_at_95_tpr"],
        "detection_accuracy": curve["detection_accuracy"],
    }


def evaluate_distribution_shift(
    clean_predictions: np.ndarray,
    clean_probabilities: np.ndarray,
    clean_labels: np.ndarray | None,
    shifted_predictions: np.ndarray,
    shifted_probabilities: np.ndarray,
    shifted_labels: np.ndarray | None = None,
    shift_type: ShiftType | str = ShiftType.COVARIATE_SHIFT,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Comprehensive evaluation of model behavior under distribution shift.

    Evaluates beyond classification accuracy:
      - Total Variation shift on prediction marginals P(Y_pred).
      - Wasserstein distance and KS drift on predictive entropy and confidence.
      - Calibration degradation (Delta ECE, Delta Brier Score).
      - OOD detection capabilities when target data is out-of-distribution.

    Args:
        clean_predictions: 1D array of model predictions on clean/reference data.
        clean_probabilities: 2D array of predicted probabilities on clean data.
        clean_labels: Optional 1D array of ground truth labels for clean data.
        shifted_predictions: 1D array of model predictions on shifted data.
        shifted_probabilities: 2D array of predicted probabilities on shifted data.
        shifted_labels: Optional 1D array of labels on shifted data (if available).
        shift_type: Category of distribution shift.
        class_names: Optional sequence of class names.

    Returns:
        Structured dictionary detailing degradation and distribution shift metrics.
    """
    clean_preds = np.asarray(clean_predictions)
    shifted_preds = np.asarray(shifted_predictions)
    clean_probs = np.asarray(clean_probabilities)
    shifted_probs = np.asarray(shifted_probabilities)

    st = ShiftType(shift_type) if isinstance(shift_type, str) else shift_type

    # 1. Prediction Marginals & Total Variation Distance
    n_classes = clean_probs.shape[1] if clean_probs.ndim == 2 else int(max(np.max(clean_preds), np.max(shifted_preds)) + 1)
    counts_c = np.bincount(clean_preds[clean_preds >= 0], minlength=n_classes)[:n_classes]
    counts_s = np.bincount(shifted_preds[shifted_preds >= 0], minlength=n_classes)[:n_classes]
    marg_clean = counts_c / max(1, np.sum(counts_c))
    marg_shifted = counts_s / max(1, np.sum(counts_s))
    pred_tv = compute_tv_distance(marg_clean, marg_shifted)

    # 2. Confidence & Entropy Shifts
    clean_conf = np.max(clean_probs, axis=1) if clean_probs.ndim == 2 else np.ones(len(clean_preds))
    shifted_conf = np.max(shifted_probs, axis=1) if shifted_probs.ndim == 2 else np.ones(len(shifted_preds))

    clean_ent = compute_predictive_entropy(clean_probs) if clean_probs.ndim == 2 else np.zeros(len(clean_preds))
    shifted_ent = compute_predictive_entropy(shifted_probs) if shifted_probs.ndim == 2 else np.zeros(len(shifted_preds))

    conf_wass = float(wasserstein_distance(clean_conf, shifted_conf)) if len(clean_conf) and len(shifted_conf) else 0.0
    ent_wass = float(wasserstein_distance(clean_ent, shifted_ent)) if len(clean_ent) and len(shifted_ent) else 0.0

    ks_conf = ks_2samp(clean_conf, shifted_conf) if len(clean_conf) and len(shifted_conf) else None
    ks_ent = ks_2samp(clean_ent, shifted_ent) if len(clean_ent) and len(shifted_ent) else None

    # 3. Accuracy and Calibration Degradation (if labels available)
    degradation: dict[str, Any] = {}
    if clean_labels is not None and shifted_labels is not None:
        clean_cls = compute_classification_metrics(clean_labels, clean_preds, class_names=class_names)
        shift_cls = compute_classification_metrics(shifted_labels, shifted_preds, class_names=class_names)

        acc_clean = clean_cls["accuracy"]
        acc_shift = shift_cls["accuracy"]
        acc_drop = float(acc_clean - acc_shift)
        rel_drop = float(acc_drop / acc_clean) if acc_clean > 0 else 0.0

        clean_cal = compute_calibration_metrics(clean_labels, clean_probs)
        shift_cal = compute_calibration_metrics(shifted_labels, shifted_probs)

        delta_ece = float(shift_cal["ece"] - clean_cal["ece"])
        delta_brier = float(shift_cal["brier_score"] - clean_cal["brier_score"])

        degradation = {
            "clean_accuracy": acc_clean,
            "shifted_accuracy": acc_shift,
            "accuracy_drop": acc_drop,
            "relative_accuracy_drop": rel_drop,
            "clean_ece": clean_cal["ece"],
            "shifted_ece": shift_cal["ece"],
            "delta_ece": delta_ece,
            "clean_brier_score": clean_cal["brier_score"],
            "shifted_brier_score": shift_cal["brier_score"],
            "delta_brier_score": delta_brier,
        }

    # 4. Out-of-Distribution Detection (if OOD shift)
    ood_metrics: dict[str, Any] = {}
    if st == ShiftType.OUT_OF_DISTRIBUTION:
        # For OOD: uncertainty is higher for OOD -> use entropy
        ood_metrics = compute_ood_detection_metrics(clean_ent, shifted_ent)

    # 5. Label Shift Analysis (if ground truth available)
    label_shift: dict[str, Any] = {}
    if clean_labels is not None and shifted_labels is not None:
        label_shift = compute_label_distribution_shift(clean_labels, shifted_labels, num_classes=n_classes)

    return {
        "shift_type": st.value,
        "samples_clean": len(clean_preds),
        "samples_shifted": len(shifted_preds),
        "prediction_total_variation": pred_tv,
        "confidence_shift": {
            "mean_clean": float(np.mean(clean_conf)) if len(clean_conf) else 0.0,
            "mean_shifted": float(np.mean(shifted_conf)) if len(shifted_conf) else 0.0,
            "wasserstein_distance": conf_wass,
            "ks_statistic": float(ks_conf.statistic) if ks_conf else 0.0,
            "ks_pvalue": float(ks_conf.pvalue) if ks_conf else 1.0,
        },
        "entropy_shift": {
            "mean_clean": float(np.mean(clean_ent)) if len(clean_ent) else 0.0,
            "mean_shifted": float(np.mean(shifted_ent)) if len(shifted_ent) else 0.0,
            "wasserstein_distance": ent_wass,
            "ks_statistic": float(ks_ent.statistic) if ks_ent else 0.0,
            "ks_pvalue": float(ks_ent.pvalue) if ks_ent else 1.0,
        },
        "degradation": degradation,
        "label_shift": label_shift,
        "ood_detection": ood_metrics,
        "summary": {
            "shift_type": st.value,
            "prediction_tv": pred_tv,
            "confidence_wasserstein": conf_wass,
            "entropy_wasserstein": ent_wass,
            "accuracy_drop": degradation.get("accuracy_drop"),
            "delta_ece": degradation.get("delta_ece"),
        },
    }

