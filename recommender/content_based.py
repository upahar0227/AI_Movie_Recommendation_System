from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import MIN_RATING_RELEVANT
from .data import MovieLensData
from .preprocess import tokenize


class ContentBasedRecommender:
    """Cosine similarity over TF-IDF of title, genres and tags (count/TF-IDF features)."""

    name = "content"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            min_df=1,
            max_features=30_000,
        )
        self.item_vectors = None
        self.data: MovieLensData | None = None
        self.train_ratings = None

    def fit(self, data: MovieLensData, ratings=None):
        self.data = data
        self.train_ratings = data.ratings if ratings is None else ratings
        self.item_vectors = self.vectorizer.fit_transform(data.movies["soup"].fillna(""))
        return self

    def user_profile(self, user_id: int):
        liked = self.train_ratings[
            (self.train_ratings["userId"] == user_id)
            & (self.train_ratings["rating"] >= MIN_RATING_RELEVANT)
        ]
        idxs = [
            self.data.movie_id_to_idx[mid]
            for mid in liked["movieId"]
            if mid in self.data.movie_id_to_idx
        ]
        if not idxs:
            rated = self.train_ratings[self.train_ratings["userId"] == user_id]
            idxs = [
                self.data.movie_id_to_idx[mid]
                for mid in rated["movieId"]
                if mid in self.data.movie_id_to_idx
            ]
        if not idxs:
            return None
        return np.asarray(self.item_vectors[idxs].mean(axis=0))

    def score_user(self, user_id: int) -> np.ndarray:
        profile = self.user_profile(user_id)
        n_items = len(self.data.movie_ids)
        if profile is None:
            return np.zeros(n_items, dtype=np.float32)
        sims = cosine_similarity(profile, self.item_vectors).ravel()
        return sims.astype(np.float32)

    def score_from_movie_ids(self, movie_ids: list[int]) -> np.ndarray:
        idxs = [
            self.data.movie_id_to_idx[mid]
            for mid in movie_ids
            if mid in self.data.movie_id_to_idx
        ]
        if not idxs:
            return np.zeros(len(self.data.movie_ids), dtype=np.float32)
        profile = np.asarray(self.item_vectors[idxs].mean(axis=0))
        return cosine_similarity(profile, self.item_vectors).ravel().astype(np.float32)
