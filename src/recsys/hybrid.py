"""Hybrid ranker. Collaborative weight grows as the user rates more movies."""

from __future__ import annotations

import numpy as np

from .config import COLD_START_K, SENTIMENT_BETA


def adaptive_alpha(n_ratings: int, k: float = COLD_START_K) -> float:
    """Return ``n / (n + k)`` so new users stay on popularity.

    Args:
        n_ratings: Ratings already known for this user. Zero or less returns 0.
        k: Smoothing constant. Larger k keeps collaborative weight low for longer.

    Returns:
        Collaborative blend weight in 0–1. Zero means the caller should use popularity.
    """
    if n_ratings <= 0:
        return 0.0
    return float(n_ratings) / float(n_ratings + k)


def minmax(scores: np.ndarray) -> np.ndarray:
    """Scale a score vector to 0–1. A constant vector becomes zeros.

    Args:
        scores: One score per movie.

    Returns:
        Array of the same shape, with the minimum at 0 and the maximum at 1.
    """
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
    when ``use_sentiment`` is true. ``alpha = n_ratings / (n_ratings + k)``. Inputs are
    min-max scaled before the blend. Popularity is returned unchanged.

    Args:
        collaborative: SVD scores. content: content scores. popularity: zero-rating fallback.
        sentiment: Reception scores, added only when use_sentiment is true.
        n_ratings: Known ratings. k: alpha smoother. beta: sentiment weight.
        use_sentiment: Add beta times scaled sentiment when true.

    Returns:
        Array of shape (n_movies,).
    """
    if n_ratings <= 0:
        return popularity.astype(np.float64)
    alpha = adaptive_alpha(n_ratings, k)
    blended = alpha * minmax(collaborative) + (1.0 - alpha) * minmax(content)
    if use_sentiment:
        blended = blended + beta * minmax(sentiment)
    return blended
