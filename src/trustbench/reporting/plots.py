"""Pure-data visualisation builders for TrustBench reports.

All public functions return **data-only** dicts (lists of x/y coordinates,
labels, etc.) suitable for rendering with any charting library.

When plotly is installed (optional extra ``trustbench[reporting]``), the
:func:`render_*` family converts a data dict into a self-contained SVG string
ready for embedding inside the HTML report.  When plotly is absent the
functions raise :exc:`ReportingDependencyError` with an actionable install
message; the rest of the reporting pipeline degrades gracefully by omitting
charts.

No metric calculations are performed here.  All chart functions accept
pre-computed arrays or metric dicts sourced from evaluation results.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)



# Dependency error



class ReportingDependencyError(ImportError):
    """Raised when an optional visualisation dependency is missing.

    Install the optional reporting extras to resolve::

        pip install trustbench[reporting]
    """


def _require_plotly():
    """Return the plotly.graph_objects module or raise ReportingDependencyError."""
    try:
        import plotly.graph_objects as go

        return go
    except ImportError as exc:
        raise ReportingDependencyError(
            "plotly is required for chart generation.  "
            "Install it with: pip install trustbench[reporting]"
        ) from exc



# Data builders (no plotly dependency)



def confusion_matrix_data(
    confusion_matrix: list[list[int]] | np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build chart data for a confusion matrix heat-map.

    Args:
        confusion_matrix: Square matrix of shape (C, C).
        class_names: Optional list of class label strings.

    Returns:
        Dict with keys ``z``, ``x``, ``y`` (lists) and ``n_classes``.
    """
    cm = np.asarray(confusion_matrix, dtype=float)
    n = cm.shape[0]
    labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]
    return {
        "z": cm.tolist(),
        "x": labels,
        "y": labels,
        "n_classes": n,
    }


def reliability_diagram_data(
    bin_centers: list[float],
    bin_accuracies: list[float],
    bin_confidences: list[float],
    bin_counts: list[int],
) -> dict[str, Any]:
    """Build chart data for a calibration reliability diagram.

    Args:
        bin_centers: Mid-point of each confidence bin.
        bin_accuracies: Mean accuracy per bin.
        bin_confidences: Mean confidence per bin.
        bin_counts: Number of samples per bin.

    Returns:
        Dict with keys ``bin_centers``, ``bin_accuracies``, ``bin_confidences``,
        ``bin_counts``, and ``calibration_gaps``.
    """
    gaps = [abs(a - c) for a, c in zip(bin_accuracies, bin_confidences, strict=False)]
    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "calibration_gaps": gaps,
    }


def confidence_histogram_data(
    confidences: np.ndarray | list[float],
    n_bins: int = 20,
) -> dict[str, Any]:
    """Build histogram data for the max-softmax-probability distribution.

    Args:
        confidences: 1D array of per-sample confidence scores in [0, 1].
        n_bins: Number of histogram bins.

    Returns:
        Dict with keys ``bin_edges``, ``counts``, and ``n_bins``.
    """
    arr = np.asarray(confidences, dtype=float)
    counts, edges = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "n_bins": n_bins,
    }


def risk_coverage_data(
    curve: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build chart data for a risk-coverage curve.

    Args:
        curve: List of dicts with keys ``coverage`` and ``error_rate`` as
            produced by :func:`~trustbench.evaluation.selective.compute_risk_coverage`.

    Returns:
        Dict with keys ``coverages`` and ``error_rates`` (parallel lists).
    """
    coverages = [float(pt["coverage"]) for pt in curve]
    error_rates = [float(pt["error_rate"]) for pt in curve]
    return {"coverages": coverages, "error_rates": error_rates}


def auroc_curve_data(
    labels: np.ndarray | list[int],
    scores: np.ndarray | list[float],
) -> dict[str, Any]:
    """Compute ROC curve coordinates and AUROC for plotting.

    Args:
        labels: Binary ground-truth labels (1 = positive / error).
        scores: Anomaly/uncertainty scores (higher = more likely positive).

    Returns:
        Dict with keys ``fpr``, ``tpr``, ``thresholds``, and ``auroc``.
        Returns a degenerate diagonal if the inputs are invalid.
    """
    from sklearn.metrics import auc, roc_curve

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if len(y) == 0 or n_pos == 0 or n_neg == 0:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [], "auroc": 0.5}

    fpr, tpr, thresholds = roc_curve(y, s)
    auroc = float(auc(fpr, tpr))
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auroc": auroc,
    }


def robustness_degradation_data(
    corruption_wise: dict[str, dict[str, Any]],
    clean_accuracy: float | None = None,
) -> dict[str, Any]:
    """Build bar-chart data for per-corruption accuracy degradation.

    Args:
        corruption_wise: Mapping of corruption name → aggregated metrics dict
            (output of :func:`~trustbench.robustness.aggregation.compute_robustness_aggregation`).
        clean_accuracy: Clean baseline accuracy, used to draw a reference line.

    Returns:
        Dict with keys ``corruptions``, ``mean_accuracies``, ``accuracy_drops``,
        and ``clean_accuracy``.
    """
    names = list(corruption_wise.keys())
    accs = [float(corruption_wise[n].get("mean_accuracy", 0.0)) for n in names]
    drops = [float(corruption_wise[n].get("accuracy_drop") or 0.0) for n in names]
    return {
        "corruptions": names,
        "mean_accuracies": accs,
        "accuracy_drops": drops,
        "clean_accuracy": clean_accuracy,
    }


def severity_comparison_data(
    severity_wise: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Build line-chart data showing metric degradation across severity levels.

    Args:
        severity_wise: Mapping of severity int → aggregated metrics dict
            (output of :func:`~trustbench.robustness.aggregation.compute_robustness_aggregation`).

    Returns:
        Dict with keys ``severities``, ``mean_accuracies``, ``mean_eces``,
        and ``accuracy_drops``.
    """
    severities = sorted(severity_wise.keys())
    accs = [float(severity_wise[s].get("mean_accuracy", 0.0)) for s in severities]
    eces = [float(severity_wise[s].get("mean_ece", 0.0)) for s in severities]
    drops = [float(severity_wise[s].get("accuracy_drop") or 0.0) for s in severities]
    return {
        "severities": severities,
        "mean_accuracies": accs,
        "mean_eces": eces,
        "accuracy_drops": drops,
    }


def experiment_comparison_data(
    delta_report: dict[str, Any],
) -> dict[str, Any]:
    """Build bar-chart data from an experiment comparison delta report.

    Args:
        delta_report: Output of
            :meth:`~trustbench.experiments.comparison.ExperimentComparator.compare`.

    Returns:
        Dict with keys ``metrics``, ``deltas``, ``improved`` (bool list),
        ``values_a``, and ``values_b``.
    """
    deltas_block = delta_report.get("deltas", {})
    names, deltas, improved, vals_a, vals_b = [], [], [], [], []
    for metric, info in deltas_block.items():
        if not isinstance(info, dict):
            continue
        names.append(metric)
        deltas.append(float(info.get("delta", 0.0)))
        improved.append(bool(info.get("improved", False)))
        vals_a.append(float(info.get("value_a", 0.0)))
        vals_b.append(float(info.get("value_b", 0.0)))
    return {
        "metrics": names,
        "deltas": deltas,
        "improved": improved,
        "values_a": vals_a,
        "values_b": vals_b,
    }



# SVG renderers (require plotly)


_PLOTLY_COMMON = {
    "width": 520,
    "height": 380,
    "margin": {"l": 60, "r": 20, "t": 40, "b": 60},
    "paper_bgcolor": "white",
    "plot_bgcolor": "#f7fafc",
    "font": {"family": "Segoe UI, system-ui, sans-serif", "size": 12},
}


def _fig_to_svg(fig) -> str:
    """Convert a plotly Figure to an inline SVG string or interactive HTML div."""
    try:
        return fig.to_image(format="svg").decode("utf-8")
    except Exception:  # noqa: BLE001
        return fig.to_html(full_html=False, include_plotlyjs=False)


def render_confusion_matrix(data: dict[str, Any]) -> str:
    """Render a confusion matrix heat-map as an inline SVG string.

    Args:
        data: Output of :func:`confusion_matrix_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    z = data["z"]
    labels = data["x"]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            text=[[str(int(v)) for v in row] for row in z],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis_autorange="reversed",
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)


def render_reliability_diagram(data: dict[str, Any]) -> str:
    """Render a reliability (calibration) diagram as an inline SVG string.

    Args:
        data: Output of :func:`reliability_diagram_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    centers = data["bin_centers"]
    accs = data["bin_accuracies"]
    confs = data["bin_confidences"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line={"color": "gray", "dash": "dash"}))
    fig.add_trace(go.Bar(x=centers, y=accs, name="Accuracy", marker_color="#4299e1",
                         width=max(0.04, 1.0 / max(len(centers), 1) * 0.8)))
    fig.add_trace(go.Scatter(x=centers, y=confs, mode="lines+markers", name="Mean confidence",
                             line={"color": "#e53e3e"}))
    fig.update_layout(
        title="Reliability Diagram",
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)


def render_confidence_histogram(data: dict[str, Any]) -> str:
    """Render a confidence score histogram as an inline SVG string.

    Args:
        data: Output of :func:`confidence_histogram_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    edges = data["bin_edges"]
    counts = data["counts"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    width = edges[1] - edges[0] if len(edges) > 1 else 0.05

    fig = go.Figure(go.Bar(x=centers, y=counts, width=width, marker_color="#48bb78"))
    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Max Softmax Probability",
        yaxis_title="Count",
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)


def render_risk_coverage_curve(data: dict[str, Any]) -> str:
    """Render a risk-coverage curve as an inline SVG string.

    Args:
        data: Output of :func:`risk_coverage_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    fig = go.Figure(
        go.Scatter(
            x=data["coverages"],
            y=data["error_rates"],
            mode="lines+markers",
            marker_color="#805ad5",
            fill="tozeroy",
            fillcolor="rgba(128, 90, 213, 0.1)",
        )
    )
    fig.update_layout(
        title="Risk-Coverage Curve",
        xaxis_title="Coverage",
        yaxis_title="Error Rate (Risk)",
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)


def render_auroc_curve(data: dict[str, Any]) -> str:
    """Render a ROC curve as an inline SVG string.

    Args:
        data: Output of :func:`auroc_curve_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line={"color": "gray", "dash": "dash"}))
    fig.add_trace(go.Scatter(
        x=data["fpr"], y=data["tpr"], mode="lines", fill="tozeroy",
        fillcolor="rgba(66, 153, 225, 0.15)",
        name=f"AUROC = {data['auroc']:.4f}", line={"color": "#4299e1"},
    ))
    fig.update_layout(
        title="ROC Curve (Error Detection)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)


def render_robustness_degradation(data: dict[str, Any]) -> str:
    """Render per-corruption accuracy degradation as an inline SVG string.

    Args:
        data: Output of :func:`robustness_degradation_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    fig = go.Figure(go.Bar(
        x=data["corruptions"],
        y=data["mean_accuracies"],
        marker_color="#f6ad55",
    ))
    if data.get("clean_accuracy") is not None:
        fig.add_hline(
            y=data["clean_accuracy"],
            line_dash="dash",
            line_color="gray",
            annotation_text="Clean baseline",
        )
    fig.update_layout(
        title="Per-Corruption Accuracy",
        xaxis_title="Corruption",
        yaxis_title="Mean Accuracy",
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)


def render_severity_comparison(data: dict[str, Any]) -> str:
    """Render severity-vs-accuracy degradation as an inline SVG string.

    Args:
        data: Output of :func:`severity_comparison_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["severities"], y=data["mean_accuracies"],
        mode="lines+markers", name="Accuracy", line={"color": "#4299e1"},
    ))
    fig.add_trace(go.Scatter(
        x=data["severities"], y=data["mean_eces"],
        mode="lines+markers", name="ECE", line={"color": "#e53e3e"},
    ))
    fig.update_layout(
        title="Degradation by Severity",
        xaxis_title="Severity Level",
        yaxis_title="Metric Value",
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)


def render_experiment_comparison(data: dict[str, Any]) -> str:
    """Render an experiment comparison delta chart as an inline SVG string.

    Args:
        data: Output of :func:`experiment_comparison_data`.

    Returns:
        SVG string.

    Raises:
        ReportingDependencyError: If plotly is not installed.
    """
    go = _require_plotly()

    colors = ["#48bb78" if imp else "#fc8181" for imp in data["improved"]]
    fig = go.Figure(go.Bar(
        x=data["metrics"],
        y=data["deltas"],
        marker_color=colors,
        text=[f"{d:+.4f}" for d in data["deltas"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(
        title="Experiment Comparison (Δ metrics, B − A)",
        xaxis_title="Metric",
        yaxis_title="Delta",
        **_PLOTLY_COMMON,
    )
    return _fig_to_svg(fig)
