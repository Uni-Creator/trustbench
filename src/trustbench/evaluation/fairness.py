"""Algorithmic fairness, bias, and demographic disparity evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray | list[Any],
    favorable_label: int = 1,
) -> dict[str, Any]:
    """Compute fairness and disparity metrics across sensitive/protected groups.

    Quantifies:
      - Demographic Parity Difference: max difference in selection rates across groups.
      - Disparate Impact Ratio: min(selection_rate) / max(selection_rate).
      - Equalized Odds Difference: max disparity in True Positive Rate (TPR) and False Positive Rate (FPR).
      - Per-group breakdown of selection rate, accuracy, TPR, and FPR.

    Args:
        y_true: 1D array of binary ground truth labels (0 or 1).
        y_pred: 1D array of binary model predictions (0 or 1).
        sensitive_features: 1D array of sensitive group attributes (e.g. gender, ethnicity, age group).
        favorable_label: Positive/favorable label index (default 1).

    Returns:
        Dictionary containing overall fairness metrics and per-group breakdowns.
    """
    yt = np.asarray(y_true, dtype=int).squeeze()
    yp = np.asarray(y_pred, dtype=int).squeeze()
    groups = np.asarray(sensitive_features).squeeze()

    if len(yt) != len(yp) or len(yt) != len(groups):
        raise ValueError("y_true, y_pred, and sensitive_features must have equal lengths.")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {
            "demographic_parity_difference": 0.0,
            "disparate_impact_ratio": 1.0,
            "equalized_odds_difference": 0.0,
            "groups": {str(g): {} for g in unique_groups},
        }

    group_metrics: dict[str, dict[str, float]] = {}
    selection_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []

    for g in unique_groups:
        mask = groups == g
        yt_g = yt[mask]
        yp_g = yp[mask]
        n_g = int(np.sum(mask))

        if n_g == 0:
            continue

        # Selection Rate: P(Y_pred = favorable)
        sel_rate = float(np.mean(yp_g == favorable_label))
        acc = float(np.mean(yp_g == yt_g))

        # TPR and FPR
        pos_mask = yt_g == favorable_label
        neg_mask = yt_g != favorable_label

        tpr = float(np.mean(yp_g[pos_mask] == favorable_label)) if np.any(pos_mask) else 0.0
        fpr = float(np.mean(yp_g[neg_mask] == favorable_label)) if np.any(neg_mask) else 0.0

        selection_rates.append(sel_rate)
        tprs.append(tpr)
        fprs.append(fpr)

        group_metrics[str(g)] = {
            "n_samples": float(n_g),
            "selection_rate": sel_rate,
            "accuracy": acc,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
        }

    # Disparities
    max_sel = max(selection_rates) if selection_rates else 0.0
    min_sel = min(selection_rates) if selection_rates else 0.0
    demographic_parity_diff = float(max_sel - min_sel)
    disparate_impact = float(min_sel / max_sel) if max_sel > 1e-12 else 1.0

    tpr_diff = float(max(tprs) - min(tprs)) if tprs else 0.0
    fpr_diff = float(max(fprs) - min(fprs)) if fprs else 0.0
    equalized_odds_diff = float(max(tpr_diff, fpr_diff))

    return {
        "demographic_parity_difference": demographic_parity_diff,
        "disparate_impact_ratio": disparate_impact,
        "equalized_odds_difference": equalized_odds_diff,
        "tpr_difference": tpr_diff,
        "fpr_difference": fpr_diff,
        "groups": group_metrics,
    }
