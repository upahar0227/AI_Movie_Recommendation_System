from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .collaborative import ItemKNNRecommender, SVDRecommender
from .config import K
from .content_based import ContentBasedRecommender
from .data import MovieLensData, load_movielens
from .hybrid import make_hybrid, make_hybrid_sentiment
from .ranking import seen_movie_ids, top_k_from_scores
from .sentiment import fit_production_sentiment, movie_sentiment_scores


@dataclass
class RecommenderSuite:
    data: MovieLensData
    models: dict

    def details(self, movie_id: int) -> dict:
        row = self.data.movies.loc[self.data.movies["movieId"] == movie_id].iloc[0]
        return {
            "movieId": int(movie_id),
            "title": str(row["title"]),
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "genres": list(row["genre_list"]),
            "tags": str(row["tag_text"])[:180],
            "sentiment": round(self.models["hybrid_sentiment"].sentiment.score(int(movie_id)), 3),
        }

    def recommend(self, method: str, user_id: int, k: int = K) -> list[dict]:
        model = self.models[method]
        exclude = seen_movie_ids(self.data.ratings, user_id)
        ranked = top_k_from_scores(self.data, model.score_user(int(user_id)), k, exclude)
        out = []
        for movie_id, score in ranked:
            item = self.details(movie_id)
            item["score"] = round(float(score), 4)
            out.append(item)
        return out

    def recommend_from_titles(self, titles: list[str], k: int = K) -> list[dict]:
        movies = self.data.movies
        picked = []
        lowered = [t.strip().lower() for t in titles if t.strip()]
        for query in lowered:
            hits = movies[movies["title"].str.lower().str.contains(query, regex=False)]
            if hits.empty:
                hits = movies[movies["title_clean"].str.lower() == query]
            if not hits.empty:
                picked.append(int(hits.iloc[0]["movieId"]))
        if not picked:
            return []
        scores = self.models["content"].score_from_movie_ids(picked)
        sent = self.models["hybrid_sentiment"].sentiment
        boost = np.array(
            [sent.score(int(mid)) for mid in self.data.movie_ids], dtype=np.float32
        )
        combined = 0.85 * _minmax(scores) + 0.15 * boost
        ranked = top_k_from_scores(self.data, combined, k, exclude=set(picked))
        out = []
        for movie_id, score in ranked:
            item = self.details(movie_id)
            item["score"] = round(float(score), 4)
            out.append(item)
        return out


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-8:
        return scores
    return (scores - lo) / (hi - lo)


def build_suite() -> RecommenderSuite:
    data = load_movielens()
    content = ContentBasedRecommender().fit(data)
    knn = ItemKNNRecommender().fit(data)
    svd = SVDRecommender().fit(data)
    sentiment_index = movie_sentiment_scores(data, fit_production_sentiment())
    hybrid = make_hybrid(content, svd)
    hybrid.data = data
    hybrid_s = make_hybrid_sentiment(content, svd, sentiment_index)
    hybrid_s.data = data
    return RecommenderSuite(
        data=data,
        models={
            "content": content,
            "item_knn": knn,
            "svd": svd,
            "hybrid": hybrid,
            "hybrid_sentiment": hybrid_s,
        },
    )
