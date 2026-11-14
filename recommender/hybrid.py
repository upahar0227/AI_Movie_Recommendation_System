from __future__ import annotations

import numpy as np

from .config import HYBRID_SENTIMENT_WEIGHTS, HYBRID_WEIGHTS
from .content_based import ContentBasedRecommender
from .data import MovieLensData
from .sentiment import MovieSentimentIndex


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRecommender:
    name = "hybrid"

    def __init__(self, content, svd, sentiment: MovieSentimentIndex | None = None, weights=None):
        self.content: ContentBasedRecommender = content
        self.svd = svd
        self.sentiment = sentiment
        self.weights = weights or HYBRID_WEIGHTS
        self.data: MovieLensData | None = None

    def fit(self, data: MovieLensData, ratings=None):
        self.data = data
        self.content.fit(data, ratings)
        self.svd.fit(data, ratings)
        return self

    def score_user(self, user_id: int) -> np.ndarray:
        content_scores = _minmax(self.content.score_user(user_id))
        svd_scores = _minmax(self.svd.score_user(user_id))
        combined = (
            self.weights.get("content", 0.0) * content_scores
            + self.weights.get("svd", 0.0) * svd_scores
        )
        sent_w = self.weights.get("sentiment", 0.0)
        if sent_w and self.sentiment is not None:
            sent = np.array(
                [self.sentiment.score(int(mid)) for mid in self.data.movie_ids],
                dtype=np.float32,
            )
            combined = combined + sent_w * sent
        return combined.astype(np.float32)


def make_hybrid(content, svd) -> HybridRecommender:
    return HybridRecommender(content, svd, weights=HYBRID_WEIGHTS)


def make_hybrid_sentiment(content, svd, sentiment: MovieSentimentIndex) -> HybridRecommender:
    rec = HybridRecommender(
        content, svd, sentiment=sentiment, weights=HYBRID_SENTIMENT_WEIGHTS
    )
    rec.name = "hybrid_sentiment"
    return rec
