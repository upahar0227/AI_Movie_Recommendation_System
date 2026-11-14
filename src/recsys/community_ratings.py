"""One community score per movie, computed once and stored on the movie frame.

The site uses a 0–10 scale. TMDB ``vote_average`` is the primary value when that
film has at least ``COMMUNITY_MIN_TMDB_VOTES`` votes. Otherwise the score falls
back to the MovieLens mean multiplied by 2. A film with some votes but not
enough for a stable raw average is shrunk toward the global mean of that source:

``(v / (v + m)) * R + (m / (v + m)) * C``

``m`` is ``COMMUNITY_PRIOR_M`` (50). Films with no TMDB votes and no MovieLens
ratings stay null. A missing score is never stored as 0.
"""

from __future__ import annotations

import pandas as pd

from .config import COMMUNITY_MIN_TMDB_VOTES, COMMUNITY_PRIOR_M, COMMUNITY_SCALE


def community_table(
    tmdb: pd.DataFrame,
    links: pd.DataFrame,
    ratings: pd.DataFrame,
    prior_m: int = COMMUNITY_PRIOR_M,
    min_tmdb_votes: int = COMMUNITY_MIN_TMDB_VOTES,
) -> pd.DataFrame:
    """Return one row per MovieLens ``movieId`` with the display score and both sources."""
    tmdb_scores = _tmdb_scores(tmdb)
    lens_scores = _movielens_scores(ratings)
    joined = links.dropna(subset=["tmdbId"])[["movieId", "tmdbId"]].drop_duplicates("movieId").copy()
    joined["movieId"] = joined["movieId"].astype(int)
    joined["tmdbId"] = joined["tmdbId"].astype(int)
    joined = joined.merge(tmdb_scores, left_on="tmdbId", right_on="id", how="left")
    joined = joined.merge(lens_scores, on="movieId", how="left")
    tmdb_mean = _global_mean(tmdb_scores["tmdb_vote_average"])
    lens_mean = _global_mean(lens_scores["movielens_average"])
    rows = [
        _choose(row, tmdb_mean, lens_mean, prior_m, min_tmdb_votes)
        for row in joined.itertuples(index=False)
    ]
    table = pd.DataFrame(rows).astype(object)

    def _blank(value):
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            return value
        return None if missing else value

    return table.map(_blank)


def _tmdb_scores(tmdb: pd.DataFrame) -> pd.DataFrame:
    frame = tmdb[["id", "vote_average", "vote_count"]].copy()
    frame["id"] = frame["id"].astype(int)
    frame["tmdb_vote_count"] = frame["vote_count"].fillna(0).astype(int)
    frame["tmdb_vote_average"] = frame["vote_average"].where(frame["tmdb_vote_count"] > 0)
    return frame[["id", "tmdb_vote_average", "tmdb_vote_count"]]


def _movielens_scores(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame(columns=["movieId", "movielens_average", "movielens_count"])
    grouped = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    grouped["movieId"] = grouped["movieId"].astype(int)
    grouped["movielens_average"] = (grouped["mean"] * 2).where(grouped["count"] > 0)
    grouped["movielens_count"] = grouped["count"].astype(int)
    return grouped[["movieId", "movielens_average", "movielens_count"]]


def _global_mean(values: pd.Series) -> float | None:
    clean = values.dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _choose(row, tmdb_mean: float | None, lens_mean: float | None, prior_m: int, min_tmdb_votes: int) -> dict:
    tmdb_count = int(row.tmdb_vote_count) if pd.notna(row.tmdb_vote_count) and int(row.tmdb_vote_count) > 0 else None
    tmdb_average = float(row.tmdb_vote_average) if tmdb_count and pd.notna(row.tmdb_vote_average) else None
    lens_count = int(row.movielens_count) if pd.notna(getattr(row, "movielens_count", None)) and int(row.movielens_count) > 0 else None
    lens_average = float(row.movielens_average) if lens_count and pd.notna(row.movielens_average) else None
    source = None
    raw = None
    count = None
    center = None
    if tmdb_count and tmdb_count >= min_tmdb_votes:
        source, raw, count, center = "tmdb", tmdb_average, tmdb_count, tmdb_mean
    elif lens_count:
        source, raw, count, center = "movielens", lens_average, lens_count, lens_mean
    elif tmdb_count:
        source, raw, count, center = "tmdb", tmdb_average, tmdb_count, tmdb_mean
    avg = None
    if source and raw is not None and count and center is not None:
        avg = round((count / (count + prior_m)) * raw + (prior_m / (count + prior_m)) * center, 1)
    return {
        "movieId": int(row.movieId),
        "avg_rating": avg,
        "rating_count": count,
        "rating_source": source,
        "rating_scale": COMMUNITY_SCALE,
        "tmdb_vote_average": None if tmdb_average is None else round(tmdb_average, 1),
        "tmdb_vote_count": tmdb_count,
        "movielens_average": None if lens_average is None else round(lens_average, 1),
        "movielens_count": lens_count,
    }
