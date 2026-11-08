"""Incremental metric aggregation and streaming evaluation for large-scale datasets.

Phase 7: Performance and Production Readiness.

This module provides streaming metric evaluation without requiring all predictions
or probabilities to be retained simultaneously in memory.

Metric Retention Requirements:
------------------------------
- **Streaming / Incremental Metrics** (Memory complexity O(C^2) for C classes):
    - Accuracy: Running count of correct predictions / total samples.
    - Confusion Matrix: Cumulative addition of per-batch contingency tables.
    - Per-Class Metrics: Cumulative True Positives, False Positives, False Negatives.
    - Brier Score: Cumulative sum of squared probability errors / total samples.
    - Negative Log-Likelihood (NLL): Cumulative sum of cross-entropy / total samples.
    - Predictive Entropy: Cumulative sum of Shannon entropy / total samples.
    - Mean Max Probability: Cumulative sum of max confidence / total samples.

- **Full Retention Metrics** (Require retaining all N samples or sorting):
    - AUROC & AUPRC (Error Detection): Global sorting of uncertainty scores across
      all positive (error) and negative (correct) samples.
    - Adaptive ECE: Quantile partitioning requires knowing all confidence quantiles.
    - Risk-Coverage Curve & AURC: Global sorting of all samples by confidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Classification of metric retention requirements for large-scale datasets.
METRIC_RETENTION_REQUIREMENTS: dict[str, str] = {
    "accuracy": "streaming",
    "confusion_matrix": "streaming",
    "brier_score": "streaming",
    "nll": "streaming",
    "mean_entropy": "streaming",
    "mean_max_probability": "streaming",
    "per_class": "streaming",
    "ece": "streaming",  # Equal-width binning can be aggregated via histogram accumulators
    "adaptive_ece": "full_retention",  # Quantile binning requires all samples
    "auroc": "full_retention",  # Rank ordering over entire dataset
    "auprc": "full_retention",  # Precision-recall curve over entire dataset
    "fpr_at_95_tpr": "full_retention",  # Global ROC curve operating point
    "aurc": "full_retention",  # Global risk-coverage curve integration
    "excess_aurc": "full_retention",  # Optimal AURC requires all sample risks
}


class StreamingMetricTracker:
    """Incrementally aggregates classification and calibration metrics batch by batch.

    Peak memory consumption is strictly O(num_classes^2) regardless of whether
    the evaluated dataset contains 1,000 or 10,000,000 samples.

    Args:
        num_classes: Total number of classes.
        class_names: Optional human-readable class names.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | tuple[str, ...] | None = None,
        n_bins: int = 15,
    ) -> None:
        self.num_classes = int(num_classes)
        self.class_names = list(class_names) if class_names else [str(i) for i in range(num_classes)]
        self.n_bins = int(n_bins)
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.total_samples: int = 0
        self.correct_samples: int = 0
        self.confusion_matrix: np.ndarray = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

        # Calibration & uncertainty running sums
        self.sum_brier: float = 0.0
        self.sum_nll: float = 0.0
        self.nll_valid_samples: int = 0
        self.sum_entropy: float = 0.0
        self.sum_max_prob: float = 0.0

        # Equal-width calibration histogram accumulators (bins: [0, 1])
        self.bin_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.bin_correct = np.zeros(self.n_bins, dtype=np.int64)
        self.bin_conf_sums = np.zeros(self.n_bins, dtype=np.float64)

    def update(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> None:
        """Incorporate a new batch into the running metrics.

        Args:
            predictions: 1D array of predicted class integers (B,).
            labels: 1D array of ground truth class integers (B,).
            probabilities: Optional 2D array of class probabilities (B, C).
        """
        preds = np.asarray(predictions, dtype=np.int64)
        lbls = np.asarray(labels, dtype=np.int64)
        batch_n = len(preds)
        if batch_n == 0:
            return

        self.total_samples += batch_n
        correct_mask = preds == lbls
        self.correct_samples += int(np.sum(correct_mask))

        # Confusion matrix accumulation
        valid_mask = (lbls >= 0) & (lbls < self.num_classes) & (preds >= 0) & (preds < self.num_classes)
        if np.any(valid_mask):
            np.add.at(
                self.confusion_matrix,
                (lbls[valid_mask], preds[valid_mask]),
                1,
            )

        if probabilities is not None:
            probs = np.asarray(probabilities, dtype=np.float64)
            # 1. Brier score running sum: sum_i sum_c (p_ic - y_ic)^2
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(batch_n)[valid_mask], lbls[valid_mask]] = 1.0
            self.sum_brier += float(np.sum((probs - one_hot) ** 2))

            # 2. NLL: - sum_i log(p_{i, y_i})
            eps = 1e-15
            clipped_probs = np.clip(probs, eps, 1.0)
            valid_lbls = lbls[valid_mask]
            true_probs = clipped_probs[np.arange(batch_n)[valid_mask], valid_lbls]
            self.sum_nll += float(-np.sum(np.log(true_probs)))
            self.nll_valid_samples += int(np.sum(valid_mask))

            # 3. Max confidence & entropy
            max_conf = np.max(probs, axis=1)
            self.sum_max_prob += float(np.sum(max_conf))
            entropy = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)
            self.sum_entropy += float(np.sum(entropy))

            # 4. Equal-width bin calibration accumulators
            bin_indices = np.clip((max_conf * self.n_bins).astype(int), 0, self.n_bins - 1)
            np.add.at(self.bin_counts, bin_indices, 1)
            np.add.at(self.bin_correct, bin_indices, correct_mask.astype(np.int64))
            np.add.at(self.bin_conf_sums, bin_indices, max_conf)

    def compute(self) -> dict[str, Any]:
        """Compute final aggregated metrics from accumulated statistics.

        Returns:
            Dictionary matching the standard TrustBench metrics schema.
        """
        n = self.total_samples
        if n == 0:
            return {
                "total_samples": 0,
                "accuracy": 0.0,
                "confusion_matrix": self.confusion_matrix.tolist(),
                "brier_score": 0.0,
                "nll": 0.0,
                "ece": 0.0,
                "mean_entropy": 0.0,
                "mean_max_probability": 0.0,
            }

        acc = float(self.correct_samples / n)
        brier = float(self.sum_brier / n)
        nll = float(self.sum_nll / self.nll_valid_samples) if self.nll_valid_samples > 0 else 0.0
        mean_entropy = float(self.sum_entropy / n)
        mean_max_prob = float(self.sum_max_prob / n)

        # Equal-width ECE from bin accumulators
        ece = 0.0
        bin_accs: list[float] = []
        bin_confs: list[float] = []
        for i in range(self.n_bins):
            cnt = self.bin_counts[i]
            if cnt > 0:
                b_acc = float(self.bin_correct[i] / cnt)
                b_conf = float(self.bin_conf_sums[i] / cnt)
                ece += (cnt / n) * abs(b_acc - b_conf)
                bin_accs.append(b_acc)
                bin_confs.append(b_conf)
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)

        # Per-class metrics
        per_class: dict[str, dict[str, Any]] = {}
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []

        cm = self.confusion_matrix
        for c in range(self.num_classes):
            tp = int(cm[c, c])
            fp = int(np.sum(cm[:, c]) - tp)
            fn = int(np.sum(cm[c, :]) - tp)
            support = int(np.sum(cm[c, :]))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            c_name = self.class_names[c] if c < len(self.class_names) else str(c)
            per_class[c_name] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": support,
            }

        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        balanced_acc = float(np.mean(recalls)) if recalls else 0.0

        return {
            "total_samples": n,
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "brier_score": brier,
            "nll": nll,
            "ece": float(ece),
            "mean_entropy": mean_entropy,
            "mean_max_probability": mean_max_prob,
        }
