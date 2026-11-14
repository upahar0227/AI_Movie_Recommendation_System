"""TF-IDF content model over stemmed movie tags."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import MIN_RATING_LIKED, SEED, TFIDF_MAX_FEATURES

logger = logging.getLogger(__name__)


class ContentModel:
    """Cosine similarity over a TF-IDF matrix with at most 5000 features."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.matrix = None
        self.movie_ids: np.ndarray | None = None
        self.id_to_idx: dict[int, int] = {}

    def fit(self, movies: pd.DataFrame) -> ContentModel:
        self.movie_ids = movies["movieId"].astype(int).to_numpy()
        self.id_to_idx = {int(movie_id): index for index, movie_id in enumerate(self.movie_ids)}
        self.matrix = self.vectorizer.fit_transform(movies["tags"].fillna(""))
        logger.info(
            "Content model fit: %s movies, %s features",
            self.matrix.shape[0],
            self.matrix.shape[1],
        )
        return self

    def similar_scores(self, movie_id: int) -> np.ndarray:
        """Cosine similarity of one movie against every movie. Shape ``(n_movies,)``."""
        vector = self.matrix[self._index(movie_id)]
        return cosine_similarity(vector, self.matrix).ravel()

    def recommend_similar(self, movie_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return ``n`` movies most similar to ``movie_id``, excluding itself."""
        scores = self.similar_scores(movie_id)
        scores = scores.copy()
        scores[self._index(movie_id)] = -1.0
        return self._top(scores, n, exclude=set())

    def profile_vector(self, ratings: list[tuple[int, float]]) -> np.ndarray | None:
        """Average TF-IDF vector of movies rated at least 4, else of any rated movie."""
        liked = [movie_id for movie_id, rating in ratings if rating >= MIN_RATING_LIKED and movie_id in self.id_to_idx]
        chosen = liked or [movie_id for movie_id, _rating in ratings if movie_id in self.id_to_idx]
        if not chosen:
            return None
        indexes = [self.id_to_idx[movie_id] for movie_id in chosen]
        return np.asarray(self.matrix[indexes].mean(axis=0))

    def recommend_for_user(self, ratings: list[tuple[int, float]], n: int = 10) -> list[tuple[int, float]]:
        """Rank the catalog by cosine similarity to the user's liked-movie profile."""
        profile = self.profile_vector(ratings)
        if profile is None:
            return []
        scores = cosine_similarity(profile.reshape(1, -1), self.matrix).ravel()
        exclude = {movie_id for movie_id, _rating in ratings}
        return self._top(scores, n, exclude)

    def score_profile(
        self,
        ratings: list[tuple[int, float]],
        clicks: list[int] | None = None,
        click_weight: float = 0.0,
        dislike_weight: float = 0.0,
    ) -> np.ndarray:
        """Similarity to liked movies, plus a small click boost and a dislike penalty.

        ``click_weight`` and ``dislike_weight`` default to 0, which is the evaluation path:
        the average of movies rated at least 4. Serving passes the weights from config.
        """
        profile = self.profile_vector(ratings)
        width = 0 if self.movie_ids is None else len(self.movie_ids)
        if click_weight and clicks:
            clicked = [int(movie_id) for movie_id in clicks if int(movie_id) in self.id_to_idx]
            if clicked:
                indexes = [self.id_to_idx[movie_id] for movie_id in clicked]
                click_vector = np.asarray(self.matrix[indexes].mean(axis=0))
                profile = click_weight * click_vector if profile is None else np.asarray(profile) + click_weight * click_vector
        if profile is None:
            scores = np.zeros(width, dtype=np.float64)
        else:
            scores = cosine_similarity(np.asarray(profile).reshape(1, -1), self.matrix).ravel()
        if dislike_weight:
            disliked = [int(movie_id) for movie_id, rating in ratings if rating <= 2 and int(movie_id) in self.id_to_idx]
            if disliked:
                indexes = [self.id_to_idx[movie_id] for movie_id in disliked]
                negative = np.asarray(self.matrix[indexes].mean(axis=0))
                scores = scores - dislike_weight * cosine_similarity(negative.reshape(1, -1), self.matrix).ravel()
        return scores

    def _index(self, movie_id: int) -> int:
        try:
            return self.id_to_idx[int(movie_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown movie id {movie_id}") from exc

    def _top(self, scores: np.ndarray, n: int, exclude: set[int]) -> list[tuple[int, float]]:
        order = np.argsort(-scores)
        picked: list[tuple[int, float]] = []
        for index in order:
            movie_id = int(self.movie_ids[index])
            if movie_id in exclude or scores[index] < 0:
                continue
            picked.append((movie_id, float(scores[index])))
            if len(picked) >= n:
                break
        return picked


def fit_content(movies: pd.DataFrame) -> ContentModel:
    """Fit the content model. ``SEED`` is recorded so every run stays aligned with other models."""
    _ = SEED
    return ContentModel().fit(movies)
