# Reporting & Visualization

TrustBench generates both machine-readable JSON reports and standalone, interactive HTML reports viewable in any modern web browser without running a server.

---

## 1. Standalone HTML Reports

Generated via `trustbench evaluate` or `generate_html_report()`:
- **Zero-Dependency Viewing**: Fully self-contained HTML with embedded Plotly charts. Does not require TrustBench, Node.js, or an active internet connection.
- **Interactive Visualizations**:
  - **Confusion Matrix**: Heatmap with normalized percentages and class labels.
  - **Reliability Diagram**: Expected vs observed accuracy across confidence bins with calibration gap shading.
  - **Confidence Histogram**: Frequency of predicted confidence scores.
  - **Risk-Coverage Curve**: Error rate vs fraction of retained predictions.
  - **AUROC Curve**: Error detection trade-off.
  - **Robustness Degradation**: Multi-line plot tracking accuracy degradation across severity levels.

---

## 2. Machine-Readable JSON Reports

Reports adhere to a versioned schema (`REPORT_SCHEMA_VERSION = 1.0`):

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-11-09T00:00:00Z",
  "metadata": {
    "experiment_name": "resnet18_cifar10",
    "model": {"name": "ResNet18", "checkpoint_sha256": "..."},
    "dataset": {"name": "cifar10", "num_samples": 10000}
  },
  "metrics": {
    "classification": {"accuracy": 0.9539, "macro_f1": 0.9539},
    "calibration": {"ece": 0.0262, "brier_score": 0.0712},
    "uncertainty": {"mean_entropy": 0.1824}
  }
}
```

---

## 3. Generating Reports Programmatically

```python
from trustbench.reporting.html_report import generate_html_report
from trustbench.reporting.schema import generate_json_report

# Generate JSON
json_str = generate_json_report(result, experiment_name="run_01")

# Generate HTML
html_str = generate_html_report(result, title="ResNet18 Reliability Report")
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_str)
```
