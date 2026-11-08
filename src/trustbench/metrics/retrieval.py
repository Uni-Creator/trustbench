"""Information retrieval and search ranking evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_retrieval_metrics(
    ground_truth: list[list[str | int]] | dict[str | int, list[str | int]],
    ranked_predictions: list[list[str | int]] | dict[str | int, list[str | int]],
    k_list: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Compute standard information retrieval ranking metrics across queries.

    Args:
        ground_truth: List of relevant document IDs per query, or mapping of query_id -> [relevant_doc_ids].
        ranked_predictions: List of retrieved document IDs (in rank order) per query,
            or mapping of query_id -> [retrieved_doc_ids in rank order].
        k_list: Sequence of cutoff thresholds K (e.g. 1, 3, 5, 10).

    Returns:
        Dictionary containing Mean Reciprocal Rank (MRR), Mean Average Precision (MAP),
        Recall@K, Precision@K, and nDCG@K for each cutoff K.
    """
    # Align queries into parallel lists of (relevant_set, retrieved_list)
    aligned_queries: list[tuple[set[str | int], list[str | int]]] = []

    if isinstance(ground_truth, dict) and isinstance(ranked_predictions, dict):
        for q_id, rel_docs in ground_truth.items():
            if q_id in ranked_predictions:
                aligned_queries.append((set(rel_docs), list(ranked_predictions[q_id])))
    elif isinstance(ground_truth, list) and isinstance(ranked_predictions, list):
        for rel_docs, ret_docs in zip(ground_truth, ranked_predictions, strict=False):
            aligned_queries.append((set(rel_docs), list(ret_docs)))
    else:
        raise TypeError("ground_truth and ranked_predictions must both be lists or both be dicts.")

    n_queries = len(aligned_queries)
    if n_queries == 0:
        base: dict[str, Any] = {"mrr": 0.0, "map": 0.0, "n_queries": 0}
        for k in k_list:
            base[f"recall@{k}"] = 0.0
            base[f"precision@{k}"] = 0.0
            base[f"ndcg@{k}"] = 0.0
        return base

    rr_scores: list[float] = []
    ap_scores: list[float] = []
    recalls_at_k: dict[int, list[float]] = {k: [] for k in k_list}
    precisions_at_k: dict[int, list[float]] = {k: [] for k in k_list}
    ndcgs_at_k: dict[int, list[float]] = {k: [] for k in k_list}

    for rel_set, ret_list in aligned_queries:
        if not rel_set:
            continue

        # Reciprocal Rank (RR)
        first_rel_rank: int | None = None
        for rank, doc in enumerate(ret_list, start=1):
            if doc in rel_set:
                first_rel_rank = rank
                break
        rr_scores.append(1.0 / first_rel_rank if first_rel_rank is not None else 0.0)

        # Average Precision (AP)
        num_rel_found = 0
        prec_sum = 0.0
        for rank, doc in enumerate(ret_list, start=1):
            if doc in rel_set:
                num_rel_found += 1
                prec_sum += num_rel_found / rank
        ap_scores.append(prec_sum / len(rel_set) if len(rel_set) > 0 else 0.0)

        # Metrics at each cutoff K
        for k in k_list:
            top_k = ret_list[:k]
            hits = [1 if doc in rel_set else 0 for doc in top_k]
            n_hits = sum(hits)

            # Precision@K & Recall@K
            precisions_at_k[k].append(n_hits / k)
            recalls_at_k[k].append(n_hits / len(rel_set) if len(rel_set) > 0 else 0.0)

            # nDCG@K
            # DCG = sum_{i=1}^K hits[i] / log2(i + 1)
            dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
            # Ideal DCG (all relevant items ranked at top)
            ideal_hits = [1] * min(k, len(rel_set))
            idcg = sum(1.0 / np.log2(idx + 2) for idx in range(len(ideal_hits)))
            ndcg = float(dcg / idcg) if idcg > 0 else 0.0
            ndcgs_at_k[k].append(ndcg)

    result: dict[str, Any] = {
        "mrr": float(np.mean(rr_scores)) if rr_scores else 0.0,
        "map": float(np.mean(ap_scores)) if ap_scores else 0.0,
        "n_queries": n_queries,
    }

    for k in k_list:
        result[f"recall@{k}"] = float(np.mean(recalls_at_k[k])) if recalls_at_k[k] else 0.0
        result[f"precision@{k}"] = (
            float(np.mean(precisions_at_k[k])) if precisions_at_k[k] else 0.0
        )
        result[f"ndcg@{k}"] = float(np.mean(ndcgs_at_k[k])) if ndcgs_at_k[k] else 0.0

    return result
