"""Ranking metrics for Top-K recommendation."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be > 0.")


def hit_rate_at_k(
    ranked_item_ids: Sequence[int],
    relevant_item_ids: Iterable[int],
    k: int,
) -> float:
    """
    Hit Rate@K for one user.

    Returns 1.0 if at least one relevant item occurs in the top-K list,
    otherwise 0.0.
    """
    _validate_k(k)
    relevant = set(relevant_item_ids)

    if not relevant:
        return 0.0

    top_k = ranked_item_ids[:k]
    return float(any(item in relevant for item in top_k))


def precision_at_k(
    ranked_item_ids: Sequence[int],
    relevant_item_ids: Iterable[int],
    k: int,
) -> float:
    """Precision@K for one user."""
    _validate_k(k)
    relevant = set(relevant_item_ids)

    if not relevant:
        return 0.0

    top_k = ranked_item_ids[:k]
    hits = sum(item in relevant for item in top_k)
    return hits / k


def recall_at_k(
    ranked_item_ids: Sequence[int],
    relevant_item_ids: Iterable[int],
    k: int,
) -> float:
    """Recall@K for one user."""
    _validate_k(k)
    relevant = set(relevant_item_ids)

    if not relevant:
        return 0.0

    top_k = ranked_item_ids[:k]
    hits = sum(item in relevant for item in top_k)
    return hits / len(relevant)


def ndcg_at_k(
    ranked_item_ids: Sequence[int],
    relevant_item_ids: Iterable[int],
    k: int,
) -> float:
    """
    Binary-relevance NDCG@K for one user.

    This works both for leave-one-out evaluation and for multiple
    relevant items.
    """
    _validate_k(k)
    relevant = set(relevant_item_ids)

    if not relevant:
        return 0.0

    top_k = ranked_item_ids[:k]

    dcg = 0.0
    for rank, item_id in enumerate(top_k, start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranked_list(
    ranked_item_ids: Sequence[int],
    relevant_item_ids: Iterable[int],
    ks: Sequence[int] = (5, 10, 20),
) -> Mapping[str, float]:
    """
    Evaluate one ranked list at several cutoffs.

    Returns keys such as:
    - HR@10
    - NDCG@10
    - Precision@10
    - Recall@10
    """
    relevant = set(relevant_item_ids)
    results = {}

    for k in ks:
        _validate_k(k)
        results[f"HR@{k}"] = hit_rate_at_k(
            ranked_item_ids, relevant, k
        )
        results[f"NDCG@{k}"] = ndcg_at_k(
            ranked_item_ids, relevant, k
        )
        results[f"Precision@{k}"] = precision_at_k(
            ranked_item_ids, relevant, k
        )
        results[f"Recall@{k}"] = recall_at_k(
            ranked_item_ids, relevant, k
        )

    return results
