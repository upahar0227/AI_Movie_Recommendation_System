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
    """Turn a MovieLens imdbId into a tt id, or None when it is missing.

    Args:
        value: Numeric IMDb id without the tt prefix. Null, non-numeric, and
            non-positive values return None.

    Returns:
        ``tt`` plus the number padded to at least seven digits.
    """
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
    """Return the site path for a poster. The file is downloaded on the first request.

    Args:
        imdb_id: A tt id. Anything else returns None.

    Returns:
        ``/posters/{imdb_id}``, or None when the id does not match ``tt`` plus 7–8 digits.
    """
    if not _IMDB.fullmatch(imdb_id or ""):
        return None
    return f"/posters/{imdb_id}"


def imdb_url(imdb_id: str | None) -> str | None:
    """Return the public IMDb title page for a tt id.

    Args:
        imdb_id: A tt id, or None.

    Returns:
        ``https://www.imdb.com/title/{imdb_id}/``, or None for a bad or missing id.
    """
    if not _IMDB.fullmatch(imdb_id or ""):
        return None
    return f"https://www.imdb.com/title/{imdb_id}/"


def cached_poster(imdb_id: str):
    """Return a cached poster file, downloading it once when needed.

    Args:
        imdb_id: A tt id. A bad id, a failed download, or a non-image body returns None.
            A cached file must be larger than 500 bytes to be reused.

    Returns:
        Path to a jpeg, png, or webp file under the posters directory, or None.

    Side effects:
        May download from Metahub and write ``{imdb_id}.jpg``.
    """
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
    """Add overview, director, lead, poster path, IMDb link, and community scores.

    Args:
        movies: Model frame keyed by movieId.

    Returns:
        A copy with those display columns. If the raw files cannot be read, the
        copy keeps empty overview, director, and lead text.

    Side effects:
        Reads MovieLens and TMDB csv files and logs how many directors were attached.
    """
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
    """Ensure overview, director, and lead exist as empty text columns.

    Args:
        frame: Movie frame. Missing columns are added in place.

    Returns:
        The same frame.

    Side effects:
        Adds empty string columns on the caller's frame when they are absent.
    """
    for column in ("overview", "director", "lead"):
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _read_details() -> pd.DataFrame:
    """Join overview, director, lead, IMDb id, and community scores by movieId.

    Returns:
        One row per movieId, including the community-table columns.

    Raises:
        FileNotFoundError: MovieLens links or a TMDB file is not under data/raw.

    Side effects:
        Reads links, ratings, TMDB movies, and TMDB credits from disk.
    """
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
    """Return the first billed cast name.

    Args:
        raw: TMDB cast JSON.

    Returns:
        The lead name, or "" when the cast list is empty.
    """
    names = top_cast(raw, limit=1)
    return names[0] if names else ""


def _download(url: str) -> bytes | None:
    """Download an image, retrying once without certificate checks.

    Args:
        url: Poster URL. The response must declare an image content type.

    Returns:
        At most 2_000_000 bytes, or None when both attempts fail or the body is not an image.

    Side effects:
        Opens the URL. The second attempt uses an unverified TLS context for this host only.
    """
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
