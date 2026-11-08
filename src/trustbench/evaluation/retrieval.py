"""Retrieval task evaluator and structured result definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trustbench.metrics.retrieval import compute_retrieval_metrics


@dataclass
class RetrievalResult:
    """Result of an information retrieval evaluation."""

    metrics: dict[str, Any]
    query_ids: list[str | int]
    ground_truth: dict[str | int, list[str | int]]
    ranked_predictions: dict[str | int, list[str | int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        """Convenience accessor for primary ranking metrics."""
        return {
            "mrr": self.metrics.get("mrr"),
            "map": self.metrics.get("map"),
            "recall@1": self.metrics.get("recall@1"),
            "recall@5": self.metrics.get("recall@5"),
            "ndcg@10": self.metrics.get("ndcg@10"),
        }


def evaluate_retrieval(
    ground_truth: dict[str | int, list[str | int]] | list[list[str | int]],
    ranked_predictions: dict[str | int, list[str | int]] | list[list[str | int]],
    *,
    k_list: tuple[int, ...] = (1, 3, 5, 10),
    extra_metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    """Evaluate retrieval search/ranking predictions against ground truth relevance.

    Args:
        ground_truth: Mapping of query ID -> list of relevant document IDs.
        ranked_predictions: Mapping of query ID -> ordered list of retrieved document IDs.
        k_list: Cutoff thresholds for Recall@K, Precision@K, and nDCG@K.
        extra_metadata: Optional dictionary of evaluation metadata.

    Returns:
        Structured `RetrievalResult`.
    """
    metrics = compute_retrieval_metrics(
        ground_truth=ground_truth,
        ranked_predictions=ranked_predictions,
        k_list=k_list,
    )

    if isinstance(ground_truth, dict):
        query_ids = list(ground_truth.keys())
        gt_dict = ground_truth
        pred_dict = (
            ranked_predictions
            if isinstance(ranked_predictions, dict)
            else {qid: ranked_predictions[i] for i, qid in enumerate(query_ids)}
        )
    else:
        query_ids = list(range(len(ground_truth)))
        gt_dict = {i: ground_truth[i] for i in range(len(ground_truth))}
        pred_dict = {i: ranked_predictions[i] for i in range(len(ranked_predictions))}

    return RetrievalResult(
        metrics=metrics,
        query_ids=query_ids,
        ground_truth=gt_dict,
        ranked_predictions=pred_dict,
        metadata=extra_metadata or {},
    )
