"""Task-specific regression metrics and uncertainty estimation."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute comprehensive regression evaluation metrics.

    Args:
        y_true: Array of ground truth continuous targets (N,) or (N, D).
        y_pred: Array of predicted continuous values (N,) or (N, D).
        y_std: Optional array of predicted standard deviations / uncertainties (N,).

    Returns:
        Dictionary containing MAE, MSE, RMSE, R2, MedAE, MAPE, MaxError, and interval metrics.
    """
    yt = np.asarray(y_true, dtype=float).squeeze()
    yp = np.asarray(y_pred, dtype=float).squeeze()

    if len(yt) == 0 or len(yp) == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "medae": 0.0,
            "mape": 0.0,
            "max_error": 0.0,
            "mean_residual": 0.0,
            "n_samples": 0,
        }

    residuals = yt - yp
    abs_residuals = np.abs(residuals)
    sq_residuals = residuals**2

    mae = float(np.mean(abs_residuals))
    mse = float(np.mean(sq_residuals))
    rmse = float(np.sqrt(mse))
    medae = float(np.median(abs_residuals))
    max_error = float(np.max(abs_residuals))
    mean_residual = float(np.mean(residuals))

    # R-squared (coefficient of determination)
    total_ss = np.sum((yt - np.mean(yt)) ** 2)
    residual_ss = np.sum(sq_residuals)
    r2 = float(1.0 - (residual_ss / total_ss)) if total_ss > 1e-12 else 0.0

    # Mean Absolute Percentage Error (MAPE) — safely avoid division by zero
    nonzero_mask = np.abs(yt) > 1e-8
    if np.any(nonzero_mask):
        mape = float(np.mean(abs_residuals[nonzero_mask] / np.abs(yt[nonzero_mask])) * 100.0)
    else:
        mape = 0.0

    metrics: dict[str, Any] = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "medae": medae,
        "mape": mape,
        "max_error": max_error,
        "mean_residual": mean_residual,
        "n_samples": len(yt),
    }

    # Prediction interval and uncertainty metrics if std is available
    if y_std is not None:
        std_arr = np.asarray(y_std, dtype=float).squeeze()
        if len(std_arr) == len(yt):
            # 95% confidence interval corresponds to ~1.96 standard deviations
            z = 1.95996
            lower = yp - z * std_arr
            upper = yp + z * std_arr
            covered = (yt >= lower) & (yt <= upper)

            picp_95 = float(np.mean(covered))  # Prediction Interval Coverage Probability
            mpiw_95 = float(np.mean(upper - lower))  # Mean Prediction Interval Width

            metrics["uncertainty"] = {
                "picp_95": picp_95,
                "mpiw_95": mpiw_95,
                "mean_std": float(np.mean(std_arr)),
                "median_std": float(np.median(std_arr)),
            }

    return metrics
