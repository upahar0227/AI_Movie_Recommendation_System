from __future__ import annotations

import numpy as np

from .data import MovieLensData


def seen_movie_ids(ratings, user_id: int) -> set[int]:
    return set(ratings.loc[ratings["userId"] == user_id, "movieId"].astype(int))


def top_k_from_scores(
    data: MovieLensData,
    scores: np.ndarray,
    k: int,
    exclude: set[int] | None = None,
) -> list[tuple[int, float]]:
    exclude = exclude or set()
    order = np.argsort(-scores)
    picked: list[tuple[int, float]] = []
    for idx in order:
        movie_id = int(data.movie_ids[idx])
        if movie_id in exclude:
            continue
        picked.append((movie_id, float(scores[idx])))
        if len(picked) >= k:
            break
    return picked
