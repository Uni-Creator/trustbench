"""Example 6: Compare Multiple Model Checkpoints / Experiments.

Demonstrates using ExperimentComparator to evaluate statistical differences
across accuracy, calibration error, and inference metrics between two model checkpoints.

NOTE on the comparison dict schema
------------------------------------
``ExperimentComparator.compare()`` accepts a *list* of experiment dicts/records
and returns::

    {
        "experiment_ids": [...],
        "experiment_names": {id: name, ...},
        "checkpoints": {id: sha256, ...},
        "conditions": [...],       # e.g. ["Clean"]
        "metrics": [...],          # sorted metric names
        "comparison_table": {      # condition → metric → {exp_id: value}
            "Clean": {"accuracy": {"exp_a": 0.912, "exp_b": 0.924}, ...}
        },
        "deltas": {                # only when exactly 2 experiments
            "Clean": {
                "accuracy": {
                    "absolute_difference": +0.012,
                    "relative_percent_change": +1.32,
                    "preferred": "exp_b_name",
                },
                ...
            }
        }
    }

The example below shows how to navigate this structure correctly.
"""

from trustbench.experiments.comparison import ExperimentComparator


def main():
    print("--- Example 6: Compare Model Checkpoints ---")

    # Define results for baseline checkpoint A vs candidate checkpoint B.
    # Metrics must live under a top-level "summary" key (or any condition name)
    # for the comparator to pick them up as scalar values.
    exp_a = {
        "id": "exp_resnet18_baseline",
        "name": "ResNet18 Baseline",
        "metrics": {
            "summary": {
                "accuracy": 0.9120,
                "macro_f1": 0.9080,
                "ece": 0.0450,
                "brier_score": 0.1280,
                "mean_entropy": 0.3120,
            }
        },
        "metadata": {"model": {"arch": "ResNet18", "checkpoint": "ckpt_a.pth"}},
    }

    exp_b = {
        "id": "exp_resnet18_temperature_calibrated",
        "name": "ResNet18 Temperature-Calibrated",
        "metrics": {
            "summary": {
                "accuracy": 0.9240,
                "macro_f1": 0.9210,
                "ece": 0.0150,
                "brier_score": 0.0980,
                "mean_entropy": 0.2840,
            }
        },
        "metadata": {"model": {"arch": "ResNet18", "checkpoint": "ckpt_b.pth"}},
    }

    comparator = ExperimentComparator()
    # compare() expects a list of experiments
    result = comparator.compare([exp_a, exp_b])

    print(f"Comparing: {exp_a['name']}  vs  {exp_b['name']}")

    # --- Per-condition metric table ------------------------------------ #
    print("\nMetric Values per Condition:")
    for condition, metrics in result["comparison_table"].items():
        print(f"  Condition: {condition}")
        for metric_name, exp_values in metrics.items():
            values_str = "  ".join(
                f"{result['experiment_names'].get(eid, eid)}: {v:.4f}"
                for eid, v in exp_values.items()
            )
            print(f"    {metric_name:20s}: {values_str}")

    # --- Deltas (Candidate – Baseline) --------------------------------- #
    print("\nMetric Differentials (Candidate − Baseline):")
    for condition, metrics in result["deltas"].items():
        print(f"  Condition: {condition}")
        for metric_name, delta in metrics.items():
            diff = delta["absolute_difference"]
            pct = delta["relative_percent_change"]
            preferred = delta["preferred"]
            print(
                f"    {metric_name:20s}: {diff:+.4f}  ({pct:+.2f}%)  preferred={preferred}"
            )

    # --- Assertions ---------------------------------------------------- #
    assert "comparison_table" in result
    assert "deltas" in result
    # Exactly two experiments → deltas should be populated
    assert len(result["deltas"]) > 0, "Deltas should be computed for 2 experiments."
    # Spot-check: candidate has higher accuracy
    for metrics in result["deltas"].values():
        if "accuracy" in metrics:
            diff = metrics["accuracy"]["absolute_difference"]
            assert diff > 0.0, (
                f"Expected accuracy to increase from baseline to candidate, got diff={diff:.4f}"
            )
        if "ece" in metrics:
            diff = metrics["ece"]["absolute_difference"]
            assert diff < 0.0, (
                f"Expected ECE to decrease from baseline to candidate, got diff={diff:.4f}"
            )

    print("\n✓ Checkpoint comparison completed successfully.")


if __name__ == "__main__":
    main()
