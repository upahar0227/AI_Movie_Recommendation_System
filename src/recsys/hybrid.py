"""Hybrid ranker. Collaborative weight grows as the user rates more movies."""

from __future__ import annotations

import numpy as np

from .config import COLD_START_K, SENTIMENT_BETA


def adaptive_alpha(n_ratings: int, k: float = COLD_START_K) -> float:
    """``n / (n + k)``. New users stay at 0 so the caller can use popularity."""
    if n_ratings <= 0:
        return 0.0
    return float(n_ratings) / float(n_ratings + k)


def minmax(scores: np.ndarray) -> np.ndarray:
    """Scale a score vector to 0–1. A constant vector becomes zeros."""
    low = float(np.min(scores))
    high = float(np.max(scores))
    if high - low < 1e-8:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - low) / (high - low)


def hybrid_scores(
    collaborative: np.ndarray,
    content: np.ndarray,
    popularity: np.ndarray,
    sentiment: np.ndarray,
    n_ratings: int,
    k: float = COLD_START_K,
    beta: float = SENTIMENT_BETA,
    use_sentiment: bool = False,
) -> np.ndarray:
    """Blend SVD and content. With zero ratings, return popularity.

    ``score = alpha * collaborative + (1 - alpha) * content``, then add ``beta * sentiment``
    when ``use_sentiment`` is true. ``alpha = n_ratings / (n_ratings + k)``.
    """
    if n_ratings <= 0:
        return popularity.astype(np.float64)
    alpha = adaptive_alpha(n_ratings, k)
    blended = alpha * minmax(collaborative) + (1.0 - alpha) * minmax(content)
    if use_sentiment:
        blended = blended + beta * minmax(sentiment)
    return blended
