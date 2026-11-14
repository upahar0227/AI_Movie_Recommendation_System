"""Ranking and rating metrics. These functions do not touch the datasets."""

from __future__ import annotations

import math


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-k list that is relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for movie_id in recommended[:k] if movie_id in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant movies that appear in the top-k list."""
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for movie_id in recommended[:k] if movie_id in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Harmonic mean of precision@k and recall@k."""
    precision = precision_at_k(recommended, relevant, k)
    recall = recall_at_k(recommended, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(recommended: list[int], gains: dict[int, float], k: int) -> float:
    """Normalized discounted cumulative gain. Missing items have gain 0."""
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
    """Fraction of the catalog that appears in at least one top-k list."""
    if catalog_size <= 0:
        return 0.0
    seen: set[int] = set()
    for recommended in recommended_lists:
        seen.update(recommended)
    return len(seen) / catalog_size


def rmse(actual: list[float], predicted: list[float]) -> float:
    """Root mean squared error."""
    if not actual:
        return 0.0
    total = sum((left - right) ** 2 for left, right in zip(actual, predicted, strict=True))
    return math.sqrt(total / len(actual))


def mae(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute error."""
    if not actual:
        return 0.0
    total = sum(abs(left - right) for left, right in zip(actual, predicted, strict=True))
    return total / len(actual)
