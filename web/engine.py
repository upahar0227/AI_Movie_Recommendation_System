from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender.content_based import ContentBasedRecommender
from recommender.data import load_movielens

MIN_VOTES = 20


class Catalog:
    def __init__(self):
        self.data = load_movielens()
        self.content = ContentBasedRecommender().fit(self.data)
        grouped = self.data.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        self.stats = grouped
        genres = set()
        for genre_list in self.data.movies["genre_list"]:
            genres.update(genre_list)
        self.genres = sorted(genres)
        self._by_id = self.data.movies.set_index("movieId")

    def movie_card(self, movie_id: int, **extra) -> dict | None:
        if movie_id not in self._by_id.index:
            return None
        row = self._by_id.loc[movie_id]
        if isinstance(row, type(self.data.movies)):
            row = row.iloc[0]
        stats = self.stats.loc[movie_id] if movie_id in self.stats.index else None
        year = row["year"]
        card = {
            "movieId": int(movie_id),
            "title": str(row["title_clean"] or row["title"]),
            "year": None if pd.isna(year) else int(year),
            "genres": list(row["genre_list"]),
            "communityRating": None if stats is None else round(float(stats["mean"]), 2),
            "votes": 0 if stats is None else int(stats["count"]),
        }
        card.update(extra)
        return card

    def top_rated_for_genres(self, genres: list[str], exclude: set[int], k: int = 12) -> list[dict]:
        wanted = {genre for genre in genres if genre in self.genres}
        if not wanted:
            return []
        movies = self.data.movies
        mask = movies["genre_list"].map(lambda items: bool(wanted.intersection(items)))
        subset = movies.loc[mask, ["movieId"]].copy()
        subset = subset.merge(self.stats, left_on="movieId", right_index=True, how="inner")
        subset = subset[~subset["movieId"].isin(exclude)]
        qualified = subset[subset["count"] >= MIN_VOTES]
        if qualified.empty:
            qualified = subset
        qualified = qualified.sort_values(["mean", "count"], ascending=False).head(k)
        cards = []
        label = ", ".join(sorted(wanted))
        for movie_id, mean, count in qualified[["movieId", "mean", "count"]].itertuples(index=False):
            card = self.movie_card(
                int(movie_id),
                score=round(float(mean), 2),
                reason=f"Top rated in {label} · {int(count)} ratings",
            )
            if card:
                cards.append(card)
        return cards

    def from_interactions(self, interactions: list[dict], k: int = 12) -> list[dict]:
        pairs = []
        for item in interactions:
            movie_id = int(item["movieId"])
            if movie_id in self.data.movie_id_to_idx:
                pairs.append((movie_id, float(item["rating"])))
        if not pairs:
            return []

        indexes = [self.data.movie_id_to_idx[movie_id] for movie_id, _ in pairs]
        weights = np.array([rating - 3.0 for _, rating in pairs], dtype=np.float32)
        if np.allclose(weights, 0):
            weights = np.ones(len(pairs), dtype=np.float32)
        matrix = self.content.item_vectors[indexes]
        profile = np.asarray(matrix.T.dot(weights)).reshape(1, -1)
        if not np.any(profile):
            return []
        scores = cosine_similarity(profile, self.content.item_vectors).ravel()

        liked = [(movie_id, rating) for movie_id, rating in pairs if rating >= 4]
        liked_vectors = None
        liked_ids: list[int] = []
        if liked:
            liked_ids = [movie_id for movie_id, _ in liked]
            liked_vectors = self.content.item_vectors[
                [self.data.movie_id_to_idx[movie_id] for movie_id in liked_ids]
            ]

        exclude = {movie_id for movie_id, _ in pairs}
        order = np.argsort(-scores)
        cards = []
        titles = {
            int(movie_id): str(self._by_id.loc[movie_id]["title_clean"])
            for movie_id in liked_ids
        }
        for idx in order:
            movie_id = int(self.data.movie_ids[idx])
            if movie_id in exclude:
                continue
            reason = "Based on movies you rated"
            if liked_vectors is not None:
                candidate = np.asarray(self.content.item_vectors[idx].todense()).reshape(1, -1)
                sims = cosine_similarity(candidate, liked_vectors).ravel()
                best = int(np.argmax(sims))
                reason = f"Because you liked {titles[liked_ids[best]]}"
            card = self.movie_card(movie_id, score=round(float(scores[idx]), 3), reason=reason)
            if card:
                cards.append(card)
            if len(cards) >= k:
                break
        return cards


_CATALOG: Catalog | None = None


def get_catalog() -> Catalog:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = Catalog()
    return _CATALOG
