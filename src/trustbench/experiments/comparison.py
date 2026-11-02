"""Experiment comparison logic comparing multiple evaluation runs across conditions and metrics."""

from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from trustbench.storage.database import get_session_factory
from trustbench.storage.repositories import ExperimentRepository


class ExperimentComparator:
    """Compares multiple evaluation experiments across shared metrics and conditions."""

    def __init__(self, session: Session | None = None):
        self.session_provided = session is not None
        self._session = session

    @property
    def session(self) -> Session:
        if self._session is None:
            factory = get_session_factory()
            self._session = factory()
        return self._session

    def close(self) -> None:
        if not self.session_provided and self._session is not None:
            self._session.close()
            self._session = None

    def compare(
        self,
        experiment_ids: list[str],
        conditions: list[str] | None = None,
        metric_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple experiments.

        Returns a structured comparison payload:
        {
            "experiment_ids": [...],
            "experiment_names": {id: name},
            "checkpoints": {id: sha256},
            "conditions": [...],
            "metrics": [...],
            "comparison_table": {
                condition: {
                    metric: {
                        exp_id: value
                    }
                }
            }
        }
        """
        repo = ExperimentRepository(self.session)
        names = {}
        checkpoints = {}
        valid_exp_ids = []

        all_conditions = set()
        all_metrics = set()
        # raw_data: condition -> metric -> exp_id -> val
        raw_data: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for exp_id in experiment_ids:
            exp = repo.get_by_id(exp_id)
            if not exp:
                # Try getting by name if user passed experiment name
                exp = repo.get_by_name(exp_id)
            if not exp:
                continue

            valid_exp_ids.append(exp.id)
            names[exp.id] = exp.name
            checkpoints[exp.id] = exp.checkpoint_sha256

            metrics = repo.get_metrics(exp.id)
            for m in metrics:
                if conditions and m.condition not in conditions:
                    continue
                if metric_names and m.metric_name not in metric_names:
                    continue

                all_conditions.add(m.condition)
                all_metrics.add(m.metric_name)
                raw_data[m.condition][m.metric_name][exp.id] = m.metric_value

        sorted_conditions = sorted(all_conditions)
        sorted_metrics = sorted(all_metrics)

        # Convert defaultdict to normal dict
        comparison_table = {
            cond: {met: dict(raw_data[cond][met]) for met in sorted_metrics if met in raw_data[cond]}
            for cond in sorted_conditions
        }

        return {
            "experiment_ids": valid_exp_ids,
            "experiment_names": names,
            "checkpoints": checkpoints,
            "conditions": sorted_conditions,
            "metrics": sorted_metrics,
            "comparison_table": comparison_table,
        }
