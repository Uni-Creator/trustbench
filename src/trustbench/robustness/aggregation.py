"""Robustness aggregation engine computing corruption-wise, severity-wise, and degradation metrics."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_robustness_aggregation(
    conditions: Sequence[dict[str, Any]],
    clean_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-condition robustness evaluation results.

    Methodology:
      - Corruption-wise Aggregation: For each corruption c, computes the mean accuracy,
        ECE, Brier score, and entropy across tested severities.
      - Severity-wise Aggregation: For each severity s in {1..5}, computes the average
        performance across all tested corruption types.
      - Clean-to-Corrupted Degradation: Absolute drop (Acc_clean - Acc_corr) and
        relative drop ((Acc_clean - Acc_corr) / Acc_clean).
      - Mean Corruption Error (mCE): Normalized corruption error relative to clean error
        or reference baseline.
      - Calibration & Uncertainty Drift: Delta ECE (ECE_corr - ECE_clean) and
        Delta Entropy (Entropy_corr - Entropy_clean).

    Args:
        conditions: Sequence of condition dictionaries containing 'corruption',
            'severity', 'accuracy', 'ece', 'brier_score', 'mean_entropy', etc.
        clean_baseline: Optional dictionary representing clean uncorrupted baseline metrics.

    Returns:
        Structured dictionary containing:
            - 'corruption_wise': dict mapping corruption name to aggregated metrics
            - 'severity_wise': dict mapping severity int to aggregated metrics
            - 'mean_corruption_accuracy': float
            - 'mean_corruption_error': float (1 - mean_corruption_accuracy)
            - 'mean_corruption_ece': float
            - 'mean_accuracy_drop': float | None
            - 'relative_accuracy_drop': float | None
            - 'mean_calibration_degradation': float | None
            - 'worst_case_condition': dict with lowest accuracy corruption and severity
    """
    clean_acc = clean_baseline.get("accuracy") if clean_baseline else None
    clean_ece = clean_baseline.get("ece") if clean_baseline else None
    clean_entropy = clean_baseline.get("mean_entropy") if clean_baseline else None

    # Filter out clean condition from corrupted conditions list if present
    corrupted = [c for c in conditions if c.get("condition") != "Clean" and "corruption" in c]

    if not corrupted:
        return {
            "corruption_wise": {},
            "severity_wise": {},
            "mean_corruption_accuracy": clean_acc or 0.0,
            "mean_corruption_error": (1.0 - clean_acc) if clean_acc is not None else 0.0,
            "mean_corruption_ece": clean_ece or 0.0,
            "mean_accuracy_drop": 0.0,
            "relative_accuracy_drop": 0.0,
            "mean_calibration_degradation": 0.0,
            "worst_case_condition": {},
        }

    # 1. Group by corruption
    by_corr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in corrupted:
        by_corr[str(c["corruption"])].append(c)

    corruption_wise: dict[str, dict[str, Any]] = {}
    corruption_errors: list[float] = []

    for c_name, items in by_corr.items():
        accs = [float(x["accuracy"]) for x in items if "accuracy" in x and x["accuracy"] is not None]
        eces = [float(x["ece"]) for x in items if "ece" in x and x["ece"] is not None]
        ents = [float(x["mean_entropy"]) for x in items if "mean_entropy" in x and x["mean_entropy"] is not None]
        briers = [float(x["brier_score"]) for x in items if "brier_score" in x and x["brier_score"] is not None]

        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_ece = float(np.mean(eces)) if eces else 0.0
        mean_ent = float(np.mean(ents)) if ents else 0.0
        mean_brier = float(np.mean(briers)) if briers else 0.0

        err_rate = float(1.0 - mean_acc)
        corruption_errors.append(err_rate)

        # Absolute and relative degradation from clean
        acc_drop = float(clean_acc - mean_acc) if clean_acc is not None else None
        rel_drop = float(acc_drop / clean_acc) if (acc_drop is not None and clean_acc and clean_acc > 0) else None

        corruption_wise[c_name] = {
            "mean_accuracy": mean_acc,
            "mean_error": err_rate,
            "mean_ece": mean_ece,
            "mean_entropy": mean_ent,
            "mean_brier_score": mean_brier,
            "accuracy_drop": acc_drop,
            "relative_accuracy_drop": rel_drop,
            "delta_ece": float(mean_ece - clean_ece) if clean_ece is not None else None,
            "delta_entropy": float(mean_ent - clean_entropy) if clean_entropy is not None else None,
            "severities_evaluated": [int(x["severity"]) for x in items if "severity" in x],
        }

    # 2. Group by severity
    by_sev: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for c in corrupted:
        by_sev[int(c["severity"])].append(c)

    severity_wise: dict[int, dict[str, Any]] = {}
    for s_lvl in sorted(by_sev.keys()):
        items = by_sev[s_lvl]
        accs = [float(x["accuracy"]) for x in items if "accuracy" in x and x["accuracy"] is not None]
        eces = [float(x["ece"]) for x in items if "ece" in x and x["ece"] is not None]
        ents = [float(x["mean_entropy"]) for x in items if "mean_entropy" in x and x["mean_entropy"] is not None]

        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_ece = float(np.mean(eces)) if eces else 0.0

        severity_wise[s_lvl] = {
            "mean_accuracy": mean_acc,
            "mean_error": float(1.0 - mean_acc),
            "mean_ece": mean_ece,
            "mean_entropy": float(np.mean(ents)) if ents else 0.0,
            "accuracy_drop": float(clean_acc - mean_acc) if clean_acc is not None else None,
            "delta_ece": float(mean_ece - clean_ece) if clean_ece is not None else None,
            "corruptions_count": len(items),
        }

    # 3. Overall summaries
    all_accs = [float(c["accuracy"]) for c in corrupted if "accuracy" in c and c["accuracy"] is not None]
    all_eces = [float(c["ece"]) for c in corrupted if "ece" in c and c["ece"] is not None]

    mean_corr_acc = float(np.mean(all_accs)) if all_accs else 0.0
    mean_corr_ece = float(np.mean(all_eces)) if all_eces else 0.0
    mean_drop = float(clean_acc - mean_corr_acc) if clean_acc is not None else None
    rel_mean_drop = float(mean_drop / clean_acc) if (mean_drop is not None and clean_acc and clean_acc > 0) else None
    delta_ece = float(mean_corr_ece - clean_ece) if clean_ece is not None else None

    # Worst-case condition
    worst = min(corrupted, key=lambda x: float(x.get("accuracy", 1.0)))

    return {
        "corruption_wise": corruption_wise,
        "severity_wise": severity_wise,
        "mean_corruption_accuracy": mean_corr_acc,
        "mean_corruption_error": float(1.0 - mean_corr_acc),
        "mean_corruption_ece": mean_corr_ece,
        "mean_accuracy_drop": mean_drop,
        "relative_accuracy_drop": rel_mean_drop,
        "mean_calibration_degradation": delta_ece,
        "worst_case_condition": {
            "corruption": worst.get("corruption"),
            "severity": worst.get("severity"),
            "condition": worst.get("condition"),
            "accuracy": worst.get("accuracy"),
            "ece": worst.get("ece"),
        },
    }

