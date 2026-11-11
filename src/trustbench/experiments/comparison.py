"""Experiment comparison engine validating compatibility and comparing multi-condition metrics."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from trustbench.storage.base import ExperimentRecord, IncompatibleExperimentsError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from trustbench.storage.base import ExperimentStore

# Optimization direction: True = higher is better, False = lower is better
METRIC_DIRECTIONS: dict[str, bool] = {
    # Classification (higher is better)
    "accuracy": True,
    "balanced_accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "macro_f1": True,
    "micro_f1": True,
    "weighted_f1": True,
    # Calibration (lower is better)
    "ece": False,
    "mce": False,
    "adaptive_ece": False,
    "brier_score": False,
    "nll": False,
    "negative_log_likelihood": False,
    # Error Detection (higher is better except FPR)
    "auroc": True,
    "aupr": True,
    "detection_accuracy": True,
    "fpr_at_95_tpr": False,
    "fpr95": False,
    # Selective Prediction (lower is better)
    "aurc": False,
    "excess_aurc": False,
    # Robustness (lower is better)
    "mean_corruption_error": False,
    "relative_mce": False,
    # Runtime & latency (lower is better)
    "duration_seconds": False,
    "inference_latency_ms": False,
}


def validate_compatibility(
    exp_a: ExperimentRecord | dict[str, Any],
    exp_b: ExperimentRecord | dict[str, Any],
) -> dict[str, Any]:
    """Validate that two experiments are compatible for direct numerical comparison.

    Checks:
    - Number of classes
    - Task / experiment type
    - Metric overlap

    Raises:
        IncompatibleExperimentsError: If experiments cannot be reliably compared.
    """
    name_a = exp_a.name if isinstance(exp_a, ExperimentRecord) else exp_a.get("name", "exp_a")
    name_b = exp_b.name if isinstance(exp_b, ExperimentRecord) else exp_b.get("name", "exp_b")

    meta_a = exp_a.metadata if isinstance(exp_a, ExperimentRecord) else exp_a.get("metadata", {})
    meta_b = exp_b.metadata if isinstance(exp_b, ExperimentRecord) else exp_b.get("metadata", {})

    model_a = meta_a.get("model", {}) if isinstance(meta_a, dict) else {}
    model_b = meta_b.get("model", {}) if isinstance(meta_b, dict) else {}

    nc_a = model_a.get("num_classes") if isinstance(model_a, dict) else None
    nc_b = model_b.get("num_classes") if isinstance(model_b, dict) else None

    if (
        nc_a is not None
        and nc_b is not None
        and nc_a != "unavailable"
        and nc_b != "unavailable"
        and int(nc_a) != int(nc_b)
    ):
        raise IncompatibleExperimentsError(
            f"Cannot compare experiment {name_a!r} ({nc_a} classes) with {name_b!r} ({nc_b} classes)."
        )

    metrics_a = exp_a.metrics if isinstance(exp_a, ExperimentRecord) else exp_a.get("metrics", {})
    metrics_b = exp_b.metrics if isinstance(exp_b, ExperimentRecord) else exp_b.get("metrics", {})

    keys_a = set(metrics_a.get("summary", {}).keys()) or set(metrics_a.keys())
    keys_b = set(metrics_b.get("summary", {}).keys()) or set(metrics_b.keys())

    overlap = keys_a.intersection(keys_b)
    # Filter out nested containers
    scalar_overlap = [k for k in overlap if k not in ("summary", "extra")]

    if keys_a and keys_b and not scalar_overlap:
        raise IncompatibleExperimentsError(
            f"Experiments {name_a!r} and {name_b!r} do not have overlapping evaluation metrics."
        )

    return {
        "compatible": True,
        "shared_metrics": sorted(scalar_overlap),
        "num_classes": nc_a,
    }


class ExperimentComparator:
    """Compares multiple evaluation experiments across shared metrics and conditions."""

    def __init__(
        self,
        session: Session | None = None,
        store: ExperimentStore | None = None,
    ):
        self.session_provided = session is not None
        self._session = session
        self.store = store

    @property
    def session(self) -> Session:
        if self._session is None:
            from trustbench.storage.database import get_session_factory

            factory = get_session_factory()
            self._session = factory()
        return self._session

    def close(self) -> None:
        if not self.session_provided and self._session is not None:
            self._session.close()
            self._session = None

    def compare(
        self,
        experiments: list[str] | list[ExperimentRecord] | list[dict[str, Any]],
        conditions: list[str] | None = None,
        metric_names: list[str] | None = None,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Compare multiple experiments.

        Accepts experiment IDs (resolved via store or database session),
        or ExperimentRecord / dict objects directly.
        """
        records: list[Any] = []

        for item in experiments:
            if isinstance(item, (ExperimentRecord, dict)):
                records.append(item)
            elif isinstance(item, str):
                rec: Any = None
                if self.store is not None:
                    rec = self.store.get(item)
                if rec is None and (self._session is not None or self.session_provided):
                    try:
                        from trustbench.storage.repositories import ExperimentRepository

                        repo = ExperimentRepository(self.session)
                        exp_model = repo.get_by_id(item) or repo.get_by_name(item)
                        if exp_model:
                            metrics_models = repo.get_metrics(exp_model.id)
                            rec = {
                                "id": exp_model.id,
                                "name": exp_model.name,
                                "checkpoint_sha256": exp_model.checkpoint_sha256,
                                "metadata": exp_model.config or {},
                                "metrics": {
                                    m.condition: {m.metric_name: m.metric_value}
                                    for m in metrics_models
                                },
                            }
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Failed to query SQLite repository: %s", exc)
                if rec:
                    records.append(rec)

        if not records:
            return {
                "experiment_ids": [],
                "experiment_names": {},
                "checkpoints": {},
                "conditions": [],
                "metrics": [],
                "comparison_table": {},
                "deltas": {},
            }

        # 1. Compatibility validation
        if validate and len(records) >= 2:
            base_rec = records[0]
            for other_rec in records[1:]:
                validate_compatibility(base_rec, other_rec)

        names: dict[str, str] = {}
        checkpoints: dict[str, str | None] = {}
        valid_exp_ids: list[str] = []

        all_conditions: set[str] = set()
        all_metrics: set[str] = set()
        # raw_data: condition -> metric -> exp_id -> val
        raw_data: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for rec in records:
            rid = rec.id if isinstance(rec, ExperimentRecord) else rec.get("id", "unknown")
            rname = rec.name if isinstance(rec, ExperimentRecord) else rec.get("name", "unknown")
            rckpt = rec.checkpoint_sha256 if isinstance(rec, ExperimentRecord) else rec.get("checkpoint_sha256")
            valid_exp_ids.append(rid)
            names[rid] = rname
            checkpoints[rid] = rckpt

            rmetrics = rec.metrics if isinstance(rec, ExperimentRecord) else rec.get("metrics", {})

            # Handle summary metrics (Condition: 'Clean')
            summary = rmetrics.get("summary", {}) if isinstance(rmetrics, dict) else {}
            if summary:
                cond = "Clean"
                if not conditions or cond in conditions:
                    all_conditions.add(cond)
                    for k, v in summary.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool) and (not metric_names or k in metric_names):
                            all_metrics.add(k)
                            raw_data[cond][k][rid] = float(v)

            # Handle structured multi-condition metrics (e.g. robustness corruptions)
            for cond_name, cval in rmetrics.items():
                if cond_name in ("summary", "extra") or not isinstance(cval, dict):
                    continue
                if conditions and cond_name not in conditions:
                    continue
                all_conditions.add(cond_name)
                for mk, mv in cval.items():
                    if isinstance(mv, (int, float)) and not isinstance(mv, bool) and (not metric_names or mk in metric_names):
                        all_metrics.add(mk)
                        raw_data[cond_name][mk][rid] = float(mv)

        sorted_conditions = sorted(all_conditions)
        sorted_metrics = sorted(all_metrics)

        comparison_table = {
            cond: {met: dict(raw_data[cond][met]) for met in sorted_metrics if met in raw_data[cond]}
            for cond in sorted_conditions
        }

        # 2. Compute deltas if comparing exactly 2 experiments (exp_2 - exp_1)
        deltas: dict[str, dict[str, dict[str, Any]]] = {}
        if len(valid_exp_ids) == 2:
            id_a, id_b = valid_exp_ids[0], valid_exp_ids[1]
            for cond in sorted_conditions:
                deltas[cond] = {}
                for met in sorted_metrics:
                    val_a = raw_data[cond][met].get(id_a)
                    val_b = raw_data[cond][met].get(id_b)
                    if val_a is not None and val_b is not None:
                        diff = val_b - val_a
                        pct_change = (diff / val_a * 100.0) if val_a != 0 else 0.0
                        higher_is_better = METRIC_DIRECTIONS.get(met, True)
                        if diff == 0:
                            better = "tie"
                        elif (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
                            better = names.get(id_b, id_b)
                        else:
                            better = names.get(id_a, id_a)

                        deltas[cond][met] = {
                            "absolute_difference": diff,
                            "relative_percent_change": pct_change,
                            "preferred": better,
                        }

        return {
            "experiment_ids": valid_exp_ids,
            "experiment_names": names,
            "checkpoints": checkpoints,
            "conditions": sorted_conditions,
            "metrics": sorted_metrics,
            "comparison_table": comparison_table,
            "deltas": deltas,
        }
