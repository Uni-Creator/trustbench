"""TrustBench CLI — built on Click."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = """\
# TrustBench configuration  (config_version: 1)
config_version: 1
experiment_name: my_experiment
seed: 42

model:
  checkpoint_path: checkpoints/model.pth
  arch_name: ResNet18
  num_classes: 10
  device: null           # auto-detect

dataset:
  data_dir: data/
  dataset_type: cifar10
  batch_size: 100
  num_workers: 0
  split: test

metrics:
  groups:
    - classification
    - calibration
    - uncertainty
  temperature: null

robustness:
  enabled: false
  corruption_types: []
  severity_levels: [1, 2, 3, 4, 5]

reporting:
  output_dir: trustbench_output
  save_html: true
  save_json: true
"""


def _echo_ok(msg: str) -> None:
    click.echo(click.style("✓ ", fg="green") + msg)


def _echo_err(msg: str) -> None:
    click.echo(click.style("✗ ", fg="red") + msg, err=True)


def _echo_info(msg: str) -> None:
    click.echo(click.style("→ ", fg="cyan") + msg)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="trustbench", prog_name="trustbench")
def app() -> None:
    """TrustBench — model reliability evaluation framework."""


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command("version")
def version_cmd() -> None:
    """Print the installed TrustBench version and exit."""
    try:
        from importlib.metadata import version as _version

        ver = _version("trustbench")
    except Exception:  # noqa: BLE001
        ver = "unknown"
    click.echo(f"trustbench {ver}")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command("init")
@click.option(
    "--output",
    "-o",
    default="trustbench.yaml",
    show_default=True,
    help="Name of the config file to create.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init(output: str, force: bool) -> None:
    """Scaffold a trustbench.yaml config file in the current directory."""
    out = Path(output)
    if out.exists() and not force:
        _echo_err(f"{out} already exists. Use --force to overwrite.")
        sys.exit(1)
    out.write_text(_DEFAULT_CONFIG, encoding="utf-8")
    _echo_ok(f"Created {out}")
    _echo_info("Edit the file and run:  trustbench evaluate --config trustbench.yaml")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command("validate")
@click.argument("config", default="trustbench.yaml")
def validate(config: str) -> None:
    """Validate a YAML config file and report any errors.

    CONFIG is the path to the YAML file (default: trustbench.yaml).
    """
    from trustbench.config.loader import validate_config

    errors = validate_config(config)
    if not errors:
        _echo_ok(f"{config} is valid.")
    else:
        for err in errors:
            _echo_err(err)
        sys.exit(1)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command("evaluate")
@click.option("--config", "-c", default="trustbench.yaml", show_default=True, help="Path to config YAML.")
@click.option("--output-dir", "-o", default=None, help="Override output directory.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output.")
@click.option("--verbose", "-v", is_flag=True, help="Print full metrics JSON.")
def evaluate_cmd(config: str, output_dir: str | None, quiet: bool, verbose: bool) -> None:
    """Run evaluation as defined in CONFIG."""
    from trustbench.benchmark import TrustBench
    from trustbench.config.loader import load_config
    from trustbench.reporting.html_report import generate_html_report
    from trustbench.reporting.result_io import save_result

    if not quiet:
        _echo_info(f"Loading config: {config}")

    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        _echo_err(str(exc))
        sys.exit(1)

    if output_dir:
        from pathlib import Path

        out = Path(output_dir)
    else:
        out = Path(cfg.resolve_path(cfg.reporting.output_dir))

    if not quiet:
        _echo_info(f"Output directory: {out}")
        _echo_info("Running evaluation…")

    try:
        tb = TrustBench.from_config(config)
        result = tb.evaluate()
    except Exception as exc:  # noqa: BLE001
        _echo_err(f"Evaluation failed: {exc}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Save artifacts
    save_result(result, out, experiment_name=cfg.experiment_name)

    # HTML report
    if cfg.reporting.save_html:
        report_path = generate_html_report(
            result, out, config=cfg, experiment_name=cfg.experiment_name
        )
        if not quiet:
            _echo_ok(f"HTML report: {report_path}")

    summary = result.summary
    if not quiet:
        click.echo("\n" + click.style("── Summary ──", bold=True))
        for key, val in summary.items():
            if val is not None:
                click.echo(f"  {key:<20} {val}")

    if verbose:
        click.echo("\n" + click.style("── Full Metrics ──", bold=True))
        click.echo(json.dumps(result.metrics, indent=2, default=str))

    _echo_ok("Done.")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command("inspect")
@click.argument("run_dir")
def inspect(run_dir: str) -> None:
    """Print a metrics summary from an artifact directory RUN_DIR."""
    from trustbench.reporting.artifacts import load_experiment_artifacts

    try:
        artifacts = load_experiment_artifacts(run_dir)
    except FileNotFoundError as exc:
        _echo_err(str(exc))
        sys.exit(1)

    metrics = artifacts.get("metrics", {})
    summary = metrics.get("summary", {})
    metadata = artifacts.get("metadata", {})

    click.echo(click.style(f"\nRun: {run_dir}", bold=True))
    if "experiment_name" in metadata:
        click.echo(f"  experiment : {metadata['experiment_name']}")
    if "timestamp" in metadata:
        click.echo(f"  timestamp  : {metadata['timestamp']}")

    click.echo(click.style("\n── Metrics Summary ──", bold=True))
    if summary:
        for key, val in summary.items():
            if val is not None:
                click.echo(f"  {key:<20} {val}")
    else:
        click.echo("  (no summary block found)")

    pred_data = artifacts.get("predictions", {})
    n_samples = 0
    if "predictions" in pred_data:
        import numpy as np

        n_samples = len(np.asarray(pred_data["predictions"]))
    click.echo(f"\n  total samples: {n_samples}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command("compare")
@click.argument("run_a")
@click.argument("run_b")
def compare(run_a: str, run_b: str) -> None:
    """Compare metrics between two artifact directories RUN_A and RUN_B."""
    from trustbench.reporting.artifacts import load_experiment_artifacts

    try:
        a_art = load_experiment_artifacts(run_a)
        b_art = load_experiment_artifacts(run_b)
    except FileNotFoundError as exc:
        _echo_err(str(exc))
        sys.exit(1)

    a_summary: dict = a_art.get("metrics", {}).get("summary", {})
    b_summary: dict = b_art.get("metrics", {}).get("summary", {})

    all_keys = sorted(set(a_summary) | set(b_summary))

    click.echo(click.style(f"\n{'Metric':<20} {'Run A':>12} {'Run B':>12} {'Δ (B-A)':>12}", bold=True))
    click.echo("─" * 60)
    for key in all_keys:
        a_val = a_summary.get(key)
        b_val = b_summary.get(key)
        try:
            delta = float(b_val) - float(a_val)  # type: ignore[arg-type]
            delta_str = f"{delta:+.4f}"
            color = "green" if delta > 0 else ("red" if delta < 0 else "white")
            delta_str = click.style(delta_str, fg=color)
        except (TypeError, ValueError):
            delta_str = "N/A"

        a_str = f"{a_val:.4f}" if isinstance(a_val, float) else str(a_val)
        b_str = f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)
        click.echo(f"{key:<20} {a_str:>12} {b_str:>12} {delta_str:>12}")
