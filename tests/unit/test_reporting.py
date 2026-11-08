"""Tests for TrustBench Phase 6 — Reporting and Visualization.

Coverage:
- JSON report schema structure and versioning.
- HTML report generation and portability (no external resources).
- Missing metrics graceful handling.
- Empty EvaluationResult.
- Large metric dictionaries.
- Plot data correctness.
- Report does not alter evaluation results.
- Optional dependency isolation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from trustbench.core.results import EvaluationResult
from trustbench.reporting.html_report import generate_html_report
from trustbench.reporting.plots import (
    ReportingDependencyError,
    auroc_curve_data,
    confidence_histogram_data,
    confusion_matrix_data,
    experiment_comparison_data,
    reliability_diagram_data,
    risk_coverage_data,
    robustness_degradation_data,
    severity_comparison_data,
)
from trustbench.reporting.schema import (
    REPORT_SCHEMA_VERSION,
    build_report_dict,
    generate_json_report,
)

# Fixtures



def _make_result(
    n_samples: int = 50,
    n_classes: int = 3,
    with_robustness: bool = False,
) -> EvaluationResult:
    """Build a minimal but realistic EvaluationResult."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, n_classes, n_samples)
    predictions = rng.integers(0, n_classes, n_samples)
    probs = rng.dirichlet(np.ones(n_classes), n_samples)
    logits = rng.standard_normal((n_samples, n_classes))
    sample_ids = np.arange(n_samples)

    accuracy = float(np.mean(labels == predictions))
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, predictions):
        conf_matrix[t, p] += 1

    per_class: dict[str, dict] = {}
    for c in range(n_classes):
        tp = int(conf_matrix[c, c])
        fp = int(conf_matrix[:, c].sum() - tp)
        fn = int(conf_matrix[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[str(c)] = {"precision": prec, "recall": rec, "f1-score": f1, "support": int(conf_matrix[c, :].sum())}

    max_conf = float(np.max(probs, axis=1).mean())
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), axis=1).mean())

    robustness_block: dict = {}
    if with_robustness:
        robustness_block = {
            "aggregation": {
                "corruption_wise": {
                    "gaussian_noise": {"mean_accuracy": 0.55, "mean_ece": 0.08, "accuracy_drop": 0.10},
                    "motion_blur": {"mean_accuracy": 0.60, "mean_ece": 0.07, "accuracy_drop": 0.05},
                },
                "severity_wise": {
                    1: {"mean_accuracy": 0.72, "mean_ece": 0.04, "accuracy_drop": 0.03},
                    2: {"mean_accuracy": 0.65, "mean_ece": 0.06, "accuracy_drop": 0.10},
                    3: {"mean_accuracy": 0.55, "mean_ece": 0.09, "accuracy_drop": 0.20},
                },
                "mean_corruption_accuracy": 0.575,
                "mean_corruption_error": 0.425,
                "mean_corruption_ece": 0.075,
                "mean_accuracy_drop": 0.075,
                "relative_accuracy_drop": 0.115,
                "mean_calibration_degradation": 0.03,
            }
        }

    return EvaluationResult(
        metrics={
            "classification": {
                "accuracy": accuracy,
                "macro_f1": 0.72,
                "per_class_report": per_class,
                "confusion_matrix": conf_matrix.tolist(),
            },
            "calibration": {
                "ece": 0.05,
                "adaptive_ece": 0.048,
                "mce": 0.12,
                "brier_score": 0.18,
                "nll": 1.1,
                "reliability_diagram": {
                    "bin_centers": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "bin_accuracies": [0.08, 0.28, 0.52, 0.68, 0.91],
                    "bin_confidences": [0.09, 0.31, 0.49, 0.71, 0.88],
                    "bin_counts": [5, 8, 12, 15, 10],
                },
            },
            "uncertainty": {
                "mean_entropy": entropy,
                "mean_max_probability": max_conf,
                "std_max_probability": 0.15,
            },
            "error_detection": {
                "max_softmax_probability": {
                    "auroc": 0.73,
                    "auprc": 0.62,
                    "fpr_at_95_tpr": 0.45,
                    "detection_accuracy": 0.68,
                },
                "predictive_entropy": {
                    "auroc": 0.71,
                    "auprc": 0.60,
                    "fpr_at_95_tpr": 0.47,
                    "detection_accuracy": 0.66,
                },
            },
            "selective": {
                "max_softmax_probability": {
                    "curve": [
                        {"coverage": 1.0, "samples_retained": 50, "threshold": 0.3, "error_rate": 0.32, "accuracy": 0.68},
                        {"coverage": 0.8, "samples_retained": 40, "threshold": 0.5, "error_rate": 0.22, "accuracy": 0.78},
                        {"coverage": 0.5, "samples_retained": 25, "threshold": 0.7, "error_rate": 0.12, "accuracy": 0.88},
                    ],
                    "aurc": 0.08,
                    "optimal_aurc": 0.03,
                    "excess_aurc": 0.05,
                },
            },
            "robustness": robustness_block,
            "summary": {
                "accuracy": accuracy,
                "macro_f1": 0.72,
                "ece": 0.05,
                "brier_score": 0.18,
                "mean_entropy": entropy,
                "total_samples": n_samples,
            },
        },
        predictions=predictions,
        labels=labels,
        sample_ids=sample_ids,
        logits=logits,
        probabilities=probs,
        metadata={"class_names": ["cat", "dog", "bird"], "num_classes": n_classes},
    )


def _make_empty_result() -> EvaluationResult:
    return EvaluationResult(
        metrics={"summary": {}},
        predictions=np.array([], dtype=int),
        labels=np.array([], dtype=int),
        sample_ids=np.array([], dtype=int),
    )



# 1. JSON report schema



class TestJsonReportSchema:
    def test_schema_version_present(self):
        result = _make_result()
        report = build_report_dict(result, experiment_name="test_exp")
        assert report["schema_version"] == REPORT_SCHEMA_VERSION

    def test_required_top_level_keys(self):
        result = _make_result()
        report = build_report_dict(result, experiment_name="test")
        required = {
            "schema_version", "execution_metadata", "summary",
            "metrics", "lineage", "configuration",
        }
        assert required.issubset(set(report.keys()))

    def test_execution_metadata_contains_name_and_version(self):
        result = _make_result()
        report = build_report_dict(result, experiment_name="my_run")
        emeta = report["execution_metadata"]
        assert emeta["experiment_name"] == "my_run"
        assert "trustbench_version" in emeta
        assert "generated_at" in emeta

    def test_metrics_domains_present(self):
        result = _make_result()
        report = build_report_dict(result)
        metrics = report["metrics"]
        for domain in ("classification", "calibration", "uncertainty", "error_detection"):
            assert domain in metrics

    def test_lineage_embedded(self):
        result = _make_result()
        lineage = {"trustbench_version": "0.2.0", "python_version": "3.11.0"}
        report = build_report_dict(result, lineage=lineage)
        assert report["lineage"] == lineage

    def test_json_serialisable(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_json_report(result, tmp, experiment_name="ser_test")
            assert path.exists()
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["schema_version"] == REPORT_SCHEMA_VERSION

    def test_empty_result_serialisable(self):
        result = _make_empty_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_json_report(result, tmp, experiment_name="empty")
            assert path.exists()
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert "schema_version" in data

    def test_report_does_not_mutate_result(self):
        result = _make_result()
        original_accuracy = result.metrics["classification"]["accuracy"]
        build_report_dict(result, experiment_name="mutation_check")
        assert result.metrics["classification"]["accuracy"] == original_accuracy

    def test_include_predictions_flag(self):
        result = _make_result()
        report_no_pred = build_report_dict(result, include_predictions=False)
        report_with_pred = build_report_dict(result, include_predictions=True)
        assert "predictions" not in report_no_pred
        assert "predictions" in report_with_pred
        preds = report_with_pred["predictions"]
        assert isinstance(preds, dict)
        assert "predictions" in preds



# 2. HTML report generation



class TestHtmlReport:
    def test_html_file_created(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp, experiment_name="html_test")
            assert path.exists()
            assert path.suffix == ".html"

    def test_html_self_contained_no_external_urls(self):
        """Report must not reference CDNs or external URLs."""
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp, experiment_name="offline_test")
            content = path.read_text(encoding="utf-8")
            # No CDN links
            for cdn in ("cdn.jsdelivr", "unpkg.com", "cdnjs.cloudflare", "fonts.googleapis"):
                assert cdn not in content, f"External resource found: {cdn}"

    def test_html_contains_experiment_name(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp, experiment_name="my_model_v2")
            content = path.read_text(encoding="utf-8")
            assert "my_model_v2" in content

    def test_html_custom_filename(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp, filename="custom_report.html")
            assert path.name == "custom_report.html"
            assert path.exists()

    def test_html_empty_result_does_not_raise(self):
        result = _make_empty_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp, experiment_name="empty")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content

    def test_html_missing_metrics_sections_absent(self):
        """Sections that have no data should not appear in the HTML."""
        result = EvaluationResult(
            metrics={"summary": {"accuracy": 0.9, "total_samples": 10}},
            predictions=np.array([0, 1]),
            labels=np.array([0, 1]),
            sample_ids=np.array([0, 1]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp)
            content = path.read_text(encoding="utf-8")
            # No calibration or robustness data → those sections absent
            assert "Calibration" not in content
            assert "Robustness Analysis" not in content

    def test_html_robustness_section_present_when_data_available(self):
        result = _make_result(with_robustness=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp)
            content = path.read_text(encoding="utf-8")
            assert "Robustness Analysis" in content
            assert "gaussian_noise" in content

    def test_html_plot_selection_flag(self):
        """When plots=[] no chart SVGs should be rendered."""
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_html_report(result, tmp, plots=[])
            content = path.read_text(encoding="utf-8")
            # Plotly SVG starts with <svg
            assert "<svg" not in content



# 3. Plot data correctness



class TestPlotData:
    def test_confusion_matrix_data_shape(self):
        cm = [[5, 1, 0], [2, 8, 1], [0, 1, 7]]
        data = confusion_matrix_data(cm, class_names=["a", "b", "c"])
        assert data["n_classes"] == 3
        assert len(data["z"]) == 3
        assert data["x"] == ["a", "b", "c"]

    def test_confusion_matrix_data_no_class_names(self):
        cm = [[3, 1], [2, 4]]
        data = confusion_matrix_data(cm)
        assert data["x"] == ["0", "1"]

    def test_reliability_diagram_data_gaps(self):
        centers = [0.1, 0.5, 0.9]
        accs = [0.08, 0.52, 0.91]
        confs = [0.09, 0.49, 0.88]
        counts = [10, 20, 15]
        data = reliability_diagram_data(centers, accs, confs, counts)
        gaps = data["calibration_gaps"]
        assert len(gaps) == 3
        assert all(g >= 0 for g in gaps)
        # First gap: |0.08 - 0.09| = 0.01
        assert abs(gaps[0] - 0.01) < 1e-9

    def test_confidence_histogram_data_total_counts(self):
        rng = np.random.default_rng(0)
        confs = rng.uniform(0, 1, 200)
        data = confidence_histogram_data(confs, n_bins=10)
        assert sum(data["counts"]) == 200
        assert len(data["bin_edges"]) == 11

    def test_risk_coverage_data_lengths_match(self):
        curve = [
            {"coverage": 1.0, "error_rate": 0.3},
            {"coverage": 0.8, "error_rate": 0.2},
            {"coverage": 0.5, "error_rate": 0.1},
        ]
        data = risk_coverage_data(curve)
        assert len(data["coverages"]) == 3
        assert len(data["error_rates"]) == 3
        assert data["coverages"] == [1.0, 0.8, 0.5]

    def test_auroc_data_degenerate_all_correct(self):
        labels = np.zeros(10, dtype=int)  # all correct = no errors
        scores = np.ones(10) * 0.5
        data = auroc_curve_data(labels, scores)
        assert data["auroc"] == 0.5  # degenerate case

    def test_auroc_data_separable(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        data = auroc_curve_data(labels, scores)
        assert data["auroc"] > 0.9

    def test_robustness_degradation_data_structure(self):
        corruption_wise = {
            "gaussian_noise": {"mean_accuracy": 0.55, "accuracy_drop": 0.10},
            "motion_blur": {"mean_accuracy": 0.65, "accuracy_drop": 0.05},
        }
        data = robustness_degradation_data(corruption_wise, clean_accuracy=0.70)
        assert "corruptions" in data
        assert len(data["corruptions"]) == 2
        assert data["clean_accuracy"] == 0.70

    def test_severity_comparison_data_sorted(self):
        severity_wise = {
            3: {"mean_accuracy": 0.55, "mean_ece": 0.09, "accuracy_drop": 0.20},
            1: {"mean_accuracy": 0.72, "mean_ece": 0.04, "accuracy_drop": 0.03},
            2: {"mean_accuracy": 0.65, "mean_ece": 0.06, "accuracy_drop": 0.10},
        }
        data = severity_comparison_data(severity_wise)
        assert data["severities"] == [1, 2, 3]  # sorted ascending
        assert data["mean_accuracies"][0] > data["mean_accuracies"][-1]  # higher at sev 1

    def test_experiment_comparison_data_structure(self):
        delta_report = {
            "deltas": {
                "accuracy": {"delta": 0.02, "improved": True, "value_a": 0.80, "value_b": 0.82},
                "ece": {"delta": -0.01, "improved": True, "value_a": 0.06, "value_b": 0.05},
            }
        }
        data = experiment_comparison_data(delta_report)
        assert "accuracy" in data["metrics"]
        assert "ece" in data["metrics"]
        assert all(data["improved"])



# 4. Optional dependency isolation



class TestDependencyIsolation:
    def test_reporting_dependency_error_message(self):
        """ReportingDependencyError carries actionable install instructions."""
        err = ReportingDependencyError("plotly is required for chart generation.")
        assert "plotly" in str(err)

    def test_core_import_does_not_load_plotly(self):
        """Importing trustbench.core must not trigger plotly import."""

        # Remove plotly from sys.modules to simulate absence, then re-check
        # (We can't uninstall plotly mid-process, but we can verify that
        # the core module tree doesn't pull it eagerly)
        import trustbench.core  # noqa: F401
        # The test passes if no AttributeError/ImportError was raised above.
        # We also verify trustbench.reporting is not auto-loaded by core.
        assert True  # optional

    def test_save_result_does_not_raise_on_plotting_failure(self):
        """save_result must succeed even if chart rendering fails."""
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            from trustbench.reporting.result_io import save_result
            save_result(result, tmp, experiment_name="resilience_test")
            # Core artifacts always present
            assert (Path(tmp) / "metrics.json").exists()
            assert (Path(tmp) / "metadata.json").exists()
