"""Semantic segmentation metrics: mIoU, Dice, and Pixel Accuracy."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    """Compute semantic segmentation metrics across batches of 2D/3D masks.

    Args:
        y_true: Ground truth class label masks of shape (N, H, W) or (H, W).
        y_pred: Predicted class label masks of shape (N, H, W) or (H, W).
        num_classes: Optional total number of classes.
        class_names: Optional sequence of human-readable class names.
        ignore_index: Optional label index to exclude from evaluation (e.g. 255 for void/boundary).

    Returns:
        Dictionary containing mIoU, Dice, Pixel Accuracy, per-class IoU, and FWIoU.
    """
    yt = np.asarray(y_true, dtype=int).flatten()
    yp = np.asarray(y_pred, dtype=int).flatten()

    if len(yt) != len(yp):
        raise ValueError(
            f"Ground truth and prediction shapes mismatch: {y_true.shape} vs {y_pred.shape}."
        )

    if len(yt) == 0:
        return {
            "miou": 0.0,
            "dice": 0.0,
            "pixel_accuracy": 0.0,
            "fwiou": 0.0,
            "per_class_iou": {},
            "per_class_dice": {},
        }

    # Filter ignore_index
    if ignore_index is not None:
        valid_mask = yt != ignore_index
        yt = yt[valid_mask]
        yp = yp[valid_mask]

    max_c = max(int(np.max(yt)), int(np.max(yp)))
    n_c = num_classes if num_classes is not None else max_c + 1

    # Confusion matrix for all pixels
    # bin index = yt * n_c + yp
    valid = (yt >= 0) & (yt < n_c) & (yp >= 0) & (yp < n_c)
    cm = np.bincount(
        n_c * yt[valid].astype(int) + yp[valid].astype(int),
        minlength=n_c * n_c,
    ).reshape(n_c, n_c)

    # Pixel accuracy
    total_pixels = np.sum(cm)
    correct_pixels = np.trace(cm)
    pixel_acc = float(correct_pixels / total_pixels) if total_pixels > 0 else 0.0

    # Per-class IoU and Dice
    per_class_iou: dict[str, float] = {}
    per_class_dice: dict[str, float] = {}
    ious: list[float] = []
    dices: list[float] = []

    # Frequency-weighted IoU components
    class_frequencies = np.sum(cm, axis=1) / max(1, total_pixels)

    for c in range(n_c):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)

        denominator_iou = tp + fp + fn
        iou = tp / denominator_iou if denominator_iou > 0 else float("nan")

        denominator_dice = 2.0 * tp + fp + fn
        dice = (2.0 * tp) / denominator_dice if denominator_dice > 0 else float("nan")

        name = class_names[c] if (class_names and c < len(class_names)) else str(c)

        if not np.isnan(iou):
            per_class_iou[name] = float(iou)
            ious.append(float(iou))
        if not np.isnan(dice):
            per_class_dice[name] = float(dice)
            dices.append(float(dice))

    miou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0

    # Frequency-Weighted IoU = sum_c (freq_c * IoU_c)
    fwiou = 0.0
    for c in range(n_c):
        name = class_names[c] if (class_names and c < len(class_names)) else str(c)
        if name in per_class_iou:
            fwiou += float(class_frequencies[c] * per_class_iou[name])

    return {
        "miou": miou,
        "dice": mean_dice,
        "pixel_accuracy": pixel_acc,
        "fwiou": float(fwiou),
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
        "total_evaluated_pixels": int(total_pixels),
    }
