"""Ranking and rating metrics. These functions do not touch the datasets."""

from __future__ import annotations

import math


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Return the fraction of the top-k list that is relevant.

    Args:
        recommended: Ranked movie ids, best first.
        relevant: Movie ids counted as hits.
        k: Leading positions to score. A non-positive k returns 0.

    Returns:
        Hits in the first k positions divided by k, in 0–1.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for movie_id in recommended[:k] if movie_id in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Return the fraction of relevant movies that appear in the top-k list.

    Args:
        recommended: Ranked movie ids, best first.
        relevant: Movie ids that should be retrieved.
        k: Leading positions to score. A non-positive k or empty set returns 0.

    Returns:
        Hits in the first k positions divided by the size of relevant, in 0–1.
    """
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for movie_id in recommended[:k] if movie_id in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Return the harmonic mean of precision@k and recall@k.

    Args:
        recommended: Ranked movie ids, best first.
        relevant: Movie ids counted as hits.
        k: Leading positions passed to precision and recall.

    Returns:
        0 when precision and recall are both 0, otherwise their harmonic mean.
    """
    precision = precision_at_k(recommended, relevant, k)
    recall = recall_at_k(recommended, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(recommended: list[int], gains: dict[int, float], k: int) -> float:
    """Return normalized discounted cumulative gain at k.

    Missing ids have gain 0. The ideal list uses only positive gains.

    Args:
        recommended: Ranked movie ids, best first.
        gains: Movie id to relevance. Unlisted ids contribute 0.
        k: Leading positions to score. A non-positive k returns 0.

    Returns:
        DCG divided by the ideal DCG, in 0–1. Returns 0 when every gain is 0.
    """
    if k <= 0:
        return 0.0
    dcg = 0.0
    for rank, movie_id in enumerate(recommended[:k]):
        gain = float(gains.get(movie_id, 0.0))
        dcg += gain / math.log2(rank + 2)
    ideal = sorted((float(value) for value in gains.values() if value > 0), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def coverage(recommended_lists: list[list[int]], catalog_size: int) -> float:
    """Return the fraction of the catalog that appears in at least one list.

    Args:
        recommended_lists: One ranked id list per user. Every id in each list counts.
        catalog_size: Number of distinct movies in the catalog. Non-positive returns 0.

    Returns:
        Unique recommended ids divided by catalog_size, in 0–1.
    """
    if catalog_size <= 0:
        return 0.0
    seen: set[int] = set()
    for recommended in recommended_lists:
        seen.update(recommended)
    return len(seen) / catalog_size


def rmse(actual: list[float], predicted: list[float]) -> float:
    """Return the root mean squared error of paired ratings.

    Args:
        actual: Observed ratings, same length as predicted.
        predicted: Predicted ratings, paired by position with actual.

    Returns:
        Square root of the mean squared difference. Returns 0 when actual is empty.
    """
    if not actual:
        return 0.0
    total = sum((left - right) ** 2 for left, right in zip(actual, predicted, strict=True))
    return math.sqrt(total / len(actual))


def mae(actual: list[float], predicted: list[float]) -> float:
    """Return the mean absolute error of paired ratings.

    Args:
        actual: Observed ratings, same length as predicted.
        predicted: Predicted ratings, paired by position with actual.

    Returns:
        Mean of the absolute differences. Returns 0 when actual is empty.
    """
    if not actual:
        return 0.0
    total = sum(abs(left - right) for left, right in zip(actual, predicted, strict=True))
    return total / len(actual)
