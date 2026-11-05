"""Format and generate text and markdown reports for robustness evaluation."""

from typing import Any


def format_robustness_markdown_table(summary_data: list[dict[str, Any]]) -> str:
    """Generate a clean markdown summary table of clean vs corrupted conditions.

    Args:
        summary_data: List of condition dicts with keys:
            - 'condition': str
            - 'accuracy': float
            - 'macro_f1': float
            - 'brier_score': float
            - 'ece': float
            - 'mean_entropy': float
            - 'mean_max_probability': float

    Returns:
        Formatted markdown table string.
    """
    headers = [
        "Condition",
        "Accuracy",
        "Macro-F1",
        "Brier Score",
        "ECE",
        "Mean Entropy",
        "Mean Confidence",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |",
    ]

    for item in summary_data:
        cond = item.get("condition", "Unknown")
        acc = f"{item.get('accuracy', 0.0) * 100:.2f}%"
        f1 = f"{item.get('macro_f1', 0.0):.4f}"
        brier = f"{item.get('brier_score', 0.0):.4f}"
        ece = f"{item.get('ece', 0.0):.4f}"
        entropy = f"{item.get('mean_entropy', 0.0):.4f}"
        conf = f"{item.get('mean_max_probability', 0.0) * 100:.2f}%"

        lines.append(f"| {cond} | {acc} | {f1} | {brier} | {ece} | {entropy} | {conf} |")

    return "\n".join(lines)


def format_selective_markdown_table(selective_curve: list[dict[str, Any]]) -> str:
    """Generate markdown table for risk-coverage curve."""
    headers = ["Coverage", "Retained Samples", "Threshold", "Error Rate", "Accuracy"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---:"] * len(headers)) + " |",
    ]
    for row in selective_curve:
        cov = f"{int(row['coverage'] * 100)}%"
        samples = str(row["samples_retained"])
        thresh = f"{row['threshold']:.4f}"
        err = f"{row['error_rate'] * 100:.2f}%"
        acc = f"{row['accuracy'] * 100:.2f}%"
        lines.append(f"| {cov} | {samples} | {thresh} | {err} | {acc} |")
    return "\n".join(lines)
