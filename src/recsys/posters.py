"""Poster files and the director, lead, and overview shown on a card.

The TMDB 5000 file has no poster path. Posters are fetched by IMDb id from
Metahub and cached under data/processed/posters. That host's certificate is
expired, so the download retries without verification only for this URL.
"""

from __future__ import annotations

import logging
import re
import ssl
import urllib.request

import pandas as pd

from .community_ratings import community_table
from .config import PROCESSED_DIR
from .data_loader import resolve_raw_paths
from .preprocessing import director_name, top_cast

logger = logging.getLogger(__name__)

POSTER_DIR = PROCESSED_DIR / "posters"
_IMDB = re.compile(r"tt\d{7,8}")
_MAX_BYTES = 2_000_000


def imdb_tt(value: object) -> str | None:
    """Turn a MovieLens imdbId into a tt id, or None when it is missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return f"tt{number:07d}"


def poster_path(imdb_id: str) -> str | None:
    """Site path for a poster. The file is downloaded on the first request."""
    if not _IMDB.fullmatch(imdb_id or ""):
        return None
    return f"/posters/{imdb_id}"


def imdb_url(imdb_id: str | None) -> str | None:
    """Public IMDb title page for a tt id."""
    if not _IMDB.fullmatch(imdb_id or ""):
        return None
    return f"https://www.imdb.com/title/{imdb_id}/"


def cached_poster(imdb_id: str):
    """Return a cached poster file, downloading it once when needed."""
    if not _IMDB.fullmatch(imdb_id or ""):
        return None
    path = POSTER_DIR / f"{imdb_id}.jpg"
    if path.is_file() and path.stat().st_size > 500:
        return path
    payload = _download(f"https://images.metahub.space/poster/medium/{imdb_id}/img")
    if payload is None or len(payload) < 500:
        return None
    if not payload.startswith((b"\xff\xd8", b"\x89PNG", b"RIFF")):
        return None
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def enrich_display(movies: pd.DataFrame) -> pd.DataFrame:
    """Add overview, director, lead, and a poster path. Keeps the model frame if sources are missing."""
    frame = movies.copy()
    try:
        details = _read_details()
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Movie details were not attached: %s", exc)
        return _blank_details(frame)
    community_columns = (
        "avg_rating",
        "rating_count",
        "rating_source",
        "rating_scale",
        "tmdb_vote_average",
        "tmdb_vote_count",
        "movielens_average",
        "movielens_count",
    )
    frame = frame.drop(columns=[name for name in ("overview", "director", "lead", "imdb_id", "link", *community_columns) if name in frame.columns])
    merged = frame.merge(details, on="movieId", how="left")
    for column in ("overview", "director", "lead"):
        merged[column] = merged[column].fillna("").astype(str)
    merged["poster_url"] = merged["imdb_id"].map(lambda value: poster_path(value) if isinstance(value, str) else None)
    merged["link"] = merged["imdb_id"].map(lambda value: imdb_url(value) if isinstance(value, str) else None)
    logger.info("Attached details for %s movies", int(merged["director"].ne("").sum()))
    return merged


def _blank_details(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("overview", "director", "lead"):
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _read_details() -> pd.DataFrame:
    paths = resolve_raw_paths()
    if paths.movielens_dir is None or not paths.tmdb_movies.is_file() or not paths.tmdb_credits.is_file():
        raise FileNotFoundError("TMDB or MovieLens links are not in data/raw")
    links = pd.read_csv(paths.movielens_dir / "links.csv", usecols=["movieId", "imdbId", "tmdbId"])
    ratings = pd.read_csv(paths.movielens_dir / "ratings.csv", usecols=["movieId", "rating"])
    tmdb = pd.read_csv(paths.tmdb_movies, usecols=["id", "overview", "vote_average", "vote_count"])
    credits = pd.read_csv(paths.tmdb_credits, usecols=["movie_id", "cast", "crew"])
    credits["director"] = credits["crew"].map(director_name)
    credits["lead"] = credits["cast"].map(_lead)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    links["imdb_id"] = links["imdbId"].map(imdb_tt)
    tmdb["id"] = tmdb["id"].astype(int)
    joined = links.merge(tmdb, left_on="tmdbId", right_on="id", how="left")
    joined = joined.merge(credits[["movie_id", "director", "lead"]], left_on="tmdbId", right_on="movie_id", how="left")
    details = joined[["movieId", "imdb_id", "overview", "director", "lead"]].copy()
    details["movieId"] = details["movieId"].astype(int)
    details = details.drop_duplicates("movieId")
    scores = community_table(tmdb, links, ratings)
    return details.merge(scores, on="movieId", how="left")


def _lead(raw: object) -> str:
    names = top_cast(raw, limit=1)
    return names[0] if names else ""


def _download(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "movie-recommender"})
    contexts = (None, ssl._create_unverified_context())
    for context in contexts:
        try:
            with urllib.request.urlopen(request, timeout=12, context=context) as response:
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    return None
                return response.read(_MAX_BYTES)
        except Exception as exc:
            logger.info("Poster download failed for %s: %s", url, exc)
    return None
