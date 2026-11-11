"""HTML report generation for TrustBench evaluation results.

Generates a **standalone**, self-contained HTML file that is viewable in any
modern browser without installing TrustBench or any other tool.

Plotly is an optional dependency (``pip install trustbench[reporting]``).
When it is available, charts are embedded as inline SVGs.  When it is absent
all chart slots are replaced with an informational placeholder — the report
remains fully readable as a table-based document.

The function signature is backward-compatible with the original
``generate_html_report(result, output_dir, *, config, experiment_name)``.
New keyword arguments are strictly additive.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, select_autoescape

from trustbench.reporting.plots import (
    ReportingDependencyError,
    auroc_curve_data,
    confidence_histogram_data,
    reliability_diagram_data,
    risk_coverage_data,
    robustness_degradation_data,
    severity_comparison_data,
)

if TYPE_CHECKING:
    from trustbench.config.schema import TrustBenchConfig
    from trustbench.core.results import EvaluationResult

logger = logging.getLogger(__name__)



# Jinja2 template


_TEMPLATE_SRC = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TrustBench Report — {{ experiment_name }}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #1a202c; line-height: 1.6; }
  header { background: linear-gradient(135deg, #2d3748, #4a5568); color: #fff; padding: 2rem 2.5rem; }
  header h1 { font-size: 1.75rem; font-weight: 700; }
  header p  { opacity: 0.75; font-size: 0.9rem; margin-top: 0.25rem; }
  main { max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }
  section { background: #fff; border-radius: 0.75rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 1.5rem; margin-bottom: 1.5rem; }
  h2 { font-size: 1.1rem; font-weight: 600; color: #2d3748; border-bottom: 2px solid #ebf4ff; padding-bottom: 0.5rem; margin-bottom: 1rem; }
  h3 { font-size: 0.95rem; font-weight: 600; color: #4a5568; margin: 1rem 0 0.5rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; }
  .card { background: #ebf8ff; border-radius: 0.5rem; padding: 1rem; text-align: center; }
  .card .label { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: .05em; }
  .card .value { font-size: 1.6rem; font-weight: 700; color: #2b6cb0; margin-top: 0.25rem; }
  .card .value.na { color: #a0aec0; font-size: 1.1rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  th { background: #edf2f7; color: #4a5568; text-align: left; padding: 0.5rem 0.75rem; font-weight: 600; }
  td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #edf2f7; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f7fafc; }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
  .badge.good { background: #c6f6d5; color: #22543d; }
  .badge.warn { background: #fefcbf; color: #744210; }
  .badge.bad  { background: #fed7d7; color: #742a2a; }
  footer { text-align: center; color: #a0aec0; font-size: 0.8rem; padding: 2rem; }
  pre { background: #f7fafc; border-radius: 0.5rem; padding: 1rem; font-size: 0.8rem; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 1rem; margin-top: 1rem; }
  .chart-box { border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 0.5rem; display: flex; align-items: center; justify-content: center; min-height: 400px; width: 100%; overflow: hidden; }
  .chart-box > div { width: 100%; }
  .chart-placeholder { color: #a0aec0; font-size: 0.8rem; text-align: center; padding: 1rem; }
  .js-plotly-plot { width: 100% !important; margin: 0 auto; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>TrustBench Evaluation Report</h1>
  <p>Experiment: <strong>{{ experiment_name }}</strong> &nbsp;·&nbsp; Generated: {{ generated_at }}</p>
</header>
<main>

  <!-- Summary Cards -->
  <section>
    <h2>Summary</h2>
    <div class="cards">
      {% for key, val in summary_cards %}
      <div class="card">
        <div class="label">{{ key }}</div>
        <div class="value {% if val == 'N/A' %}na{% endif %}">{{ val }}</div>
      </div>
      {% endfor %}
    </div>
  </section>

  {% if per_class_rows %}
  <!-- Classification: Per-Class Breakdown -->
  <section>
    <h2>Classification</h2>
    <div class="charts">
      {% if confusion_matrix_svg %}
      <div class="chart-box">{{ confusion_matrix_svg|safe }}</div>
      {% else %}
      <div class="chart-box"><div class="chart-placeholder">Install trustbench[reporting] for charts</div></div>
      {% endif %}
    </div>
    <h3>Per-Class Metrics</h3>
    <table>
      <thead><tr>
        <th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th>
      </tr></thead>
      <tbody>
        {% for row in per_class_rows %}
        <tr>
          <td>{{ row.name }}</td>
          <td>{{ row.precision }}</td>
          <td>{{ row.recall }}</td>
          <td>{{ row.f1 }}</td>
          <td>{{ row.support }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% if confusion_rows %}
    <h3>Top Mis-classifications</h3>
    <table>
      <thead><tr><th>True Class</th><th>Predicted Class</th><th>Count</th></tr></thead>
      <tbody>
        {% for row in confusion_rows %}
        <tr>
          <td>{{ row.true_cls }}</td>
          <td>{{ row.pred_cls }}</td>
          <td><span class="badge bad">{{ row.count }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </section>
  {% endif %}

  {% if calibration_rows %}
  <!-- Calibration -->
  <section>
    <h2>Calibration</h2>
    <div class="charts">
      {% if reliability_diagram_svg %}
      <div class="chart-box">{{ reliability_diagram_svg|safe }}</div>
      {% else %}
      <div class="chart-box"><div class="chart-placeholder">Install trustbench[reporting] for charts</div></div>
      {% endif %}
    </div>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {% for label, val in calibration_rows %}
        <tr><td>{{ label }}</td><td>{{ val }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if uncertainty_rows %}
  <!-- Uncertainty -->
  <section>
    <h2>Uncertainty</h2>
    <div class="charts">
      {% if confidence_histogram_svg %}
      <div class="chart-box">{{ confidence_histogram_svg|safe }}</div>
      {% else %}
      <div class="chart-box"><div class="chart-placeholder">Install trustbench[reporting] for charts</div></div>
      {% endif %}
    </div>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {% for label, val in uncertainty_rows %}
        <tr><td>{{ label }}</td><td>{{ val }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if error_detection_rows %}
  <!-- Error Detection -->
  <section>
    <h2>Error Detection</h2>
    <div class="charts">
      {% if auroc_curve_svg %}
      <div class="chart-box">{{ auroc_curve_svg|safe }}</div>
      {% else %}
      <div class="chart-box"><div class="chart-placeholder">Install trustbench[reporting] for charts</div></div>
      {% endif %}
    </div>
    <table>
      <thead><tr><th>Method</th><th>AUROC</th><th>AUPR</th><th>FPR@95TPR</th><th>Det. Accuracy</th></tr></thead>
      <tbody>
        {% for row in error_detection_rows %}
        <tr>
          <td>{{ row.method }}</td>
          <td>{{ row.auroc }}</td>
          <td>{{ row.auprc }}</td>
          <td>{{ row.fpr95 }}</td>
          <td>{{ row.det_acc }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if selective_rows %}
  <!-- Selective Prediction -->
  <section>
    <h2>Selective Prediction</h2>
    <div class="charts">
      {% if risk_coverage_svg %}
      <div class="chart-box">{{ risk_coverage_svg|safe }}</div>
      {% else %}
      <div class="chart-box"><div class="chart-placeholder">Install trustbench[reporting] for charts</div></div>
      {% endif %}
    </div>
    <table>
      <thead><tr><th>Coverage</th><th>Retained</th><th>Threshold</th><th>Error Rate</th><th>Accuracy</th></tr></thead>
      <tbody>
        {% for row in selective_rows %}
        <tr>
          <td>{{ row.coverage }}</td>
          <td>{{ row.retained }}</td>
          <td>{{ row.threshold }}</td>
          <td>{{ row.error_rate }}</td>
          <td>{{ row.accuracy }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if robustness_rows %}
  <!-- Robustness -->
  <section>
    <h2>Robustness Analysis</h2>
    <div class="charts">
      {% if robustness_degradation_svg %}
      <div class="chart-box">{{ robustness_degradation_svg|safe }}</div>
      {% else %}
      <div class="chart-box"><div class="chart-placeholder">Install trustbench[reporting] for charts</div></div>
      {% endif %}
      {% if severity_comparison_svg %}
      <div class="chart-box">{{ severity_comparison_svg|safe }}</div>
      {% else %}
      <div class="chart-box"><div class="chart-placeholder">Install trustbench[reporting] for charts</div></div>
      {% endif %}
    </div>
    <h3>Per-Corruption Summary</h3>
    <table>
      <thead><tr><th>Corruption</th><th>Mean Accuracy</th><th>Accuracy Drop</th><th>Mean ECE</th></tr></thead>
      <tbody>
        {% for row in robustness_rows %}
        <tr>
          <td>{{ row.corruption }}</td>
          <td>{{ row.mean_accuracy }}</td>
          <td>{{ row.accuracy_drop }}</td>
          <td>{{ row.mean_ece }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if metadata_rows %}
  <!-- Experiment Metadata -->
  <section>
    <h2>Experiment Metadata</h2>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>
        {% for label, val in metadata_rows %}
        <tr><td>{{ label }}</td><td>{{ val }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}

  {% if config_json %}
  <!-- Configuration -->
  <section>
    <h2>Configuration</h2>
    <pre>{{ config_json }}</pre>
  </section>
  {% endif %}

</main>
<footer>
  <p>Generated by <strong>TrustBench</strong> &mdash; {{ generated_at }}</p>
</footer>
</body>
</html>
"""



# Formatting helpers



def _fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _badge_cls(value: float | None, threshold_good: float = 0.8) -> str:
    if value is None:
        return ""
    if value >= threshold_good:
        return "good"
    if value >= 0.6:
        return "warn"
    return "bad"


def _safe_svg(render_fn, *args, **kwargs) -> str | None:
    """Call render_fn; return None (not raise) on any failure."""
    try:
        return render_fn(*args, **kwargs)
    except ReportingDependencyError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Chart render failed: %s", exc)
        return None



# Section builders



def _build_per_class_rows(classification: dict[str, Any], class_names: list[str]) -> list[dict]:
    rows: list[dict] = []
    per_class_report = classification.get("per_class_report")
    if not isinstance(per_class_report, dict):
        return rows
    for cls_key, cls_data in per_class_report.items():
        if not isinstance(cls_data, dict):
            continue
        if cls_key in ("accuracy", "macro avg", "weighted avg"):
            continue
        idx = int(cls_key) if cls_key.isdigit() else None
        name = class_names[idx] if class_names and idx is not None and idx < len(class_names) else cls_key
        rows.append({
            "name": name,
            "precision": _fmt(cls_data.get("precision")),
            "recall": _fmt(cls_data.get("recall")),
            "f1": _fmt(cls_data.get("f1-score")),
            "support": int(cls_data.get("support", 0)),
        })
    return rows


def _build_confusion_rows(classification: dict[str, Any], class_names: list[str]) -> list[dict]:
    import numpy as np

    rows: list[dict] = []
    cm_raw = classification.get("confusion_matrix")
    if cm_raw is None:
        return rows
    try:
        cm = np.asarray(cm_raw)
        n_cls = cm.shape[0]
        entries: list[tuple[int, int, int]] = []
        for i in range(n_cls):
            for j in range(n_cls):
                if i != j and cm[i, j] > 0:
                    entries.append((i, j, int(cm[i, j])))
        entries.sort(key=lambda x: -x[2])
        for ti, pj, count in entries[:10]:
            tname = class_names[ti] if class_names and ti < len(class_names) else str(ti)
            pname = class_names[pj] if class_names and pj < len(class_names) else str(pj)
            rows.append({"true_cls": tname, "pred_cls": pname, "count": count})
    except (TypeError, ValueError, IndexError):
        pass
    return rows


def _build_calibration_rows(calibration: dict[str, Any]) -> list[tuple[str, str]]:
    keys = [
        ("ECE", "ece"),
        ("Adaptive ECE", "adaptive_ece"),
        ("MCE", "mce"),
        ("Brier Score", "brier_score"),
        ("NLL", "nll"),
    ]
    return [(label, _fmt(calibration.get(key))) for label, key in keys if key in calibration]


def _build_uncertainty_rows(uncertainty: dict[str, Any]) -> list[tuple[str, str]]:
    keys = [
        ("Mean Entropy", "mean_entropy"),
        ("Mean Max Probability", "mean_max_probability"),
        ("Mean Energy Score", "mean_energy_score"),
        ("Std Max Probability", "std_max_probability"),
        ("Fraction High Confidence", "fraction_high_confidence"),
        ("Fraction Low Confidence", "fraction_low_confidence"),
    ]
    return [(label, _fmt(calibration.get(key))) for label, key in keys if (calibration := uncertainty) and key in calibration]


def _build_error_detection_rows(error_detection: dict[str, Any]) -> list[dict]:
    rows = []
    method_names = {
        "max_softmax_probability": "Max Softmax Prob.",
        "predictive_entropy": "Predictive Entropy",
        "temperature_scaled_probability": "Temperature Scaled",
        "energy_score": "Energy Score",
    }
    for key, label in method_names.items():
        m = error_detection.get(key)
        if not isinstance(m, dict):
            continue
        rows.append({
            "method": label,
            "auroc": _fmt(m.get("auroc")),
            "auprc": _fmt(m.get("auprc")),
            "fpr95": _fmt(m.get("fpr_at_95_tpr")),
            "det_acc": _fmt(m.get("detection_accuracy")),
        })
    return rows


def _build_selective_rows(selective: dict[str, Any]) -> list[dict]:
    rows = []
    # Prefer MSP-based curve if available
    msp = selective.get("max_softmax_probability", selective)
    curve = msp.get("curve", [])
    for pt in curve:
        rows.append({
            "coverage": f"{float(pt['coverage']) * 100:.0f}%",
            "retained": str(int(pt.get("samples_retained", 0))),
            "threshold": _fmt(pt.get("threshold")),
            "error_rate": _pct(pt.get("error_rate")),
            "accuracy": _pct(pt.get("accuracy")),
        })
    return rows


def _build_robustness_rows(robustness: dict[str, Any]) -> list[dict]:
    rows = []
    aggregation = robustness.get("aggregation", robustness)
    corruption_wise = aggregation.get("corruption_wise", {})
    for cname, cdata in corruption_wise.items():
        rows.append({
            "corruption": cname,
            "mean_accuracy": _pct(cdata.get("mean_accuracy")),
            "accuracy_drop": _fmt(cdata.get("accuracy_drop")),
            "mean_ece": _fmt(cdata.get("mean_ece")),
        })
    return rows


def _build_metadata_rows(result: EvaluationResult, lineage: dict[str, Any] | None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    meta = result.metadata or {}
    for key in ("experiment_name", "model_class", "dataset_name", "num_classes", "num_samples", "device"):
        if key in meta:
            rows.append((key.replace("_", " ").title(), str(meta[key])))
    if lineage:
        for key in ("trustbench_version", "python_version", "execution_start", "checkpoint_sha256"):
            if key in lineage:
                rows.append((key.replace("_", " ").title(), str(lineage[key])))
    return rows



# Public API



def generate_html_report(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    config: TrustBenchConfig | None = None,
    experiment_name: str = "unnamed",
    lineage: dict[str, Any] | None = None,
    plots: list[str] | None = None,
    title: str | None = None,
    report_metadata: dict[str, Any] | None = None,
    filename: str = "report.html",
) -> Path:
    """Generate a self-contained HTML report from an :class:`EvaluationResult`.

    The report is a single ``.html`` file with all CSS inline and optional
    SVG charts embedded directly — no CDN, no JavaScript runtime required,
    viewable offline.

    When plotly is installed (``pip install trustbench[reporting]``) charts
    are rendered as inline SVGs.  Without it, chart slots show a placeholder
    message; all tables and metric values are still present.

    Args:
        result: Completed evaluation result.
        output_dir: Directory where *filename* will be written.
        config: Optional evaluation config to embed as a JSON block.
        experiment_name: Label shown in the report header.
        lineage: Optional provenance dict (from
            :func:`~trustbench.experiments.lineage.build_lineage_record`).
        plots: Explicit list of chart names to include.  When ``None`` all
            relevant charts are included.  Valid names: ``"confusion_matrix"``,
            ``"reliability_diagram"``, ``"confidence_histogram"``,
            ``"risk_coverage"``, ``"auroc_curve"``, ``"robustness_degradation"``,
            ``"severity_comparison"``.
        title: Custom report title (overrides *experiment_name* in the
            ``<title>`` tag only).
        report_metadata: Additional key-value pairs shown in the metadata table.
        filename: Output filename.  Defaults to ``report.html``.

    Returns:
        Path to the written HTML file.
    """
    import numpy as np

    from trustbench.reporting.plots import (
        confusion_matrix_data,
        render_auroc_curve,
        render_confidence_histogram,
        render_confusion_matrix,
        render_reliability_diagram,
        render_risk_coverage_curve,
        render_robustness_degradation,
        render_severity_comparison,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    enabled = set(plots) if plots is not None else None

    def _want(name: str) -> bool:
        return enabled is None or name in enabled

    #  Source data
    summary = result.summary or {}
    classification = result.metrics.get("classification", {})
    calibration = result.metrics.get("calibration", {})
    uncertainty = result.metrics.get("uncertainty", {})
    error_detection = result.metrics.get("error_detection", {})
    selective = result.metrics.get("selective", {})
    robustness = result.metrics.get("robustness", {})
    class_names: list[str] = result.metadata.get("class_names") or []

    #  Summary cards
    summary_cards: list[tuple[str, str]] = [
        ("Accuracy", _fmt(summary.get("accuracy"))),
        ("Macro F1", _fmt(summary.get("macro_f1"))),
        ("ECE", _fmt(summary.get("ece"))),
        ("Brier Score", _fmt(summary.get("brier_score"))),
        ("Mean Entropy", _fmt(summary.get("mean_entropy"))),
        ("Total Samples", str(int(summary.get("total_samples", 0)))),
    ]

    #  Classification
    per_class_rows = _build_per_class_rows(classification, class_names)
    confusion_rows = _build_confusion_rows(classification, class_names)

    #  Charts: confusion matrix
    confusion_matrix_svg: str | None = None
    if _want("confusion_matrix") and classification.get("confusion_matrix") is not None:
        cm_data = confusion_matrix_data(classification["confusion_matrix"], class_names or None)
        confusion_matrix_svg = _safe_svg(render_confusion_matrix, cm_data)

    #  Calibration
    calibration_rows = _build_calibration_rows(calibration)

    # Chart: reliability diagram
    reliability_diagram_svg: str | None = None
    if _want("reliability_diagram") and calibration.get("reliability_diagram"):
        rd = calibration["reliability_diagram"]
        rd_data = reliability_diagram_data(
            bin_centers=rd.get("bin_centers", []),
            bin_accuracies=rd.get("bin_accuracies", []),
            bin_confidences=rd.get("bin_confidences", []),
            bin_counts=rd.get("bin_counts", []),
        )
        reliability_diagram_svg = _safe_svg(render_reliability_diagram, rd_data)

    #  Uncertainty
    uncertainty_rows = _build_uncertainty_rows(uncertainty)

    # Chart: confidence histogram (from probabilities array if available)
    confidence_histogram_svg: str | None = None
    if _want("confidence_histogram") and result.probabilities is not None:
        try:
            confs = np.max(result.probabilities, axis=1)
            ch_data = confidence_histogram_data(confs)
            confidence_histogram_svg = _safe_svg(render_confidence_histogram, ch_data)
        except Exception:
            logger.debug("Failed to render confidence histogram", exc_info=True)

    #  Error detection
    error_detection_rows = _build_error_detection_rows(error_detection)

    # Chart: AUROC curve (MSP-based)
    auroc_curve_svg: str | None = None
    if _want("auroc_curve") and result.probabilities is not None and len(result.labels) > 0:
        try:
            errors = (result.predictions != result.labels).astype(int)
            scores = -np.max(result.probabilities, axis=1)
            auc_data = auroc_curve_data(errors, scores)
            auroc_curve_svg = _safe_svg(render_auroc_curve, auc_data)
        except Exception:
            logger.debug("Failed to render AUROC curve", exc_info=True)

    #  Selective prediction
    selective_rows = _build_selective_rows(selective)

    # Chart: risk-coverage curve
    risk_coverage_svg: str | None = None
    if _want("risk_coverage") and selective:
        msp = selective.get("max_softmax_probability", selective)
        curve = msp.get("curve", [])
        if curve:
            rc_data = risk_coverage_data(curve)
            risk_coverage_svg = _safe_svg(render_risk_coverage_curve, rc_data)

    #  Robustness
    robustness_rows = _build_robustness_rows(robustness)

    # Charts: degradation + severity
    robustness_degradation_svg: str | None = None
    severity_comparison_svg: str | None = None
    aggregation = robustness.get("aggregation", robustness)
    if aggregation:
        corruption_wise = aggregation.get("corruption_wise", {})
        severity_wise = aggregation.get("severity_wise", {})
        clean_acc = aggregation.get("clean_accuracy") or (
            result.metrics.get("classification", {}).get("accuracy")
        )
        if _want("robustness_degradation") and corruption_wise:
            rd_data2 = robustness_degradation_data(corruption_wise, clean_acc)
            robustness_degradation_svg = _safe_svg(render_robustness_degradation, rd_data2)
        if _want("severity_comparison") and severity_wise:
            sv_data = severity_comparison_data(severity_wise)
            severity_comparison_svg = _safe_svg(render_severity_comparison, sv_data)

    #  Metadata table
    metadata_rows = _build_metadata_rows(result, lineage)
    if report_metadata:
        for k, v in report_metadata.items():
            metadata_rows.append((k, str(v)))

    #  Config JSON
    config_json: str | None = None
    if config is not None:
        try:
            config_json = config.model_dump_json(indent=2)
        except (AttributeError, ValueError):
            config_json = None

    #  Render
    env = Environment(autoescape=select_autoescape(["html"]))
    template = env.from_string(_TEMPLATE_SRC)

    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_content = template.render(
        experiment_name=title or experiment_name,
        generated_at=generated_at,
        summary_cards=summary_cards,
        per_class_rows=per_class_rows,
        confusion_rows=confusion_rows,
        confusion_matrix_svg=confusion_matrix_svg,
        calibration_rows=calibration_rows,
        reliability_diagram_svg=reliability_diagram_svg,
        uncertainty_rows=uncertainty_rows,
        confidence_histogram_svg=confidence_histogram_svg,
        error_detection_rows=error_detection_rows,
        auroc_curve_svg=auroc_curve_svg,
        selective_rows=selective_rows,
        risk_coverage_svg=risk_coverage_svg,
        robustness_rows=robustness_rows,
        robustness_degradation_svg=robustness_degradation_svg,
        severity_comparison_svg=severity_comparison_svg,
        metadata_rows=metadata_rows,
        config_json=config_json,
    )

    report_path = out / filename
    report_path.write_text(html_content, encoding="utf-8")
    return report_path
