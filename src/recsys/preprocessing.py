"""Build a stemmed tag string for each movie from TMDB metadata."""

from __future__ import annotations

import json
import logging
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z]+")
_STEMMER = PorterStemmer()
_STOP: set[str] | None = None


def stopword_set() -> set[str]:
    """Return the English stopword set, loading it from NLTK on the first call.

    Returns:
        The shared stopword set used by stem_text.

    Side effects:
        Reads the NLTK stopword corpus once and caches it on the module.
    """
    global _STOP
    if _STOP is None:
        _STOP = set(stopwords.words("english"))
    return _STOP


def parse_json_list(raw: object) -> list[dict]:
    """Parse a TMDB JSON list. Invalid or empty values become an empty list.

    Args:
        raw: A list of dicts, a JSON string of that list, or a missing value.

    Returns:
        Dict items only. Non-dicts, bad JSON, and non-lists are dropped.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def names_from(raw: object) -> list[str]:
    """Collect the ``name`` field from a TMDB JSON list.

    Args:
        raw: Value accepted by parse_json_list.

    Returns:
        Names in list order. Items without a name are skipped.
    """
    return [str(item["name"]) for item in parse_json_list(raw) if item.get("name")]


def top_cast(raw: object, limit: int = 3) -> list[str]:
    """Return up to ``limit`` cast names, ordered by TMDB billing order.

    Args:
        raw: TMDB cast JSON. Missing ``order`` sorts after billed names.
        limit: Maximum names to keep.

    Returns:
        Cast names, shortest billing order first.
    """
    people = [item for item in parse_json_list(raw) if item.get("name")]
    people.sort(key=lambda item: item.get("order", 10**6))
    return [str(item["name"]) for item in people[:limit]]


def director_name(raw: object) -> str:
    """Return the first credited director, or an empty string.

    Args:
        raw: TMDB crew JSON. The first item whose job is Director is used.

    Returns:
        That person's name, or "" when no director is listed.
    """
    for person in parse_json_list(raw):
        if person.get("job") == "Director" and person.get("name"):
            return str(person["name"])
    return ""


def stem_text(text: str) -> str:
    """Lowercase, tokenise, drop stopwords, and Porter-stem.

    Args:
        text: Free text. Only letter tokens longer than two characters are kept.

    Returns:
        A single space-separated stemmed string.
    """
    stops = stopword_set()
    tokens = [
        _STEMMER.stem(token)
        for token in _TOKEN.findall(text.lower())
        if token not in stops and len(token) > 2
    ]
    return " ".join(tokens)


def _cell(row: pd.Series, *columns: str) -> str:
    """Return the first present, non-null column as text.

    Args:
        row: One movie row.
        columns: Column names to try, in order.

    Returns:
        The first usable cell, or "" when every column is missing or null.
    """
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return ""


def movie_tag_text(row: pd.Series) -> str:
    """Build stemmed tags from genres, keywords, top 3 cast, director, and overview.

    Args:
        row: A joined movie row. Genre columns may be ``genres_tmdb`` or ``genres``.

    Returns:
        One stemmed tag string for the content model.
    """
    genres = names_from(_cell(row, "genres_tmdb", "genres"))
    keywords = names_from(_cell(row, "keywords"))
    cast = top_cast(_cell(row, "cast"))
    director = director_name(_cell(row, "crew"))
    overview = _cell(row, "overview")
    parts = genres + keywords + cast
    if director:
        parts.append(director)
    parts.append(overview)
    return stem_text(" ".join(parts))


def add_tags(movies: pd.DataFrame) -> pd.DataFrame:
    """Return a modeling frame with a stemmed ``tags`` column and display fields.

    Args:
        movies: Joined MovieLens and TMDB rows, including movieId and tmdbId.

    Returns:
        A copy with title, genre_names, tags, and an empty poster_url. Index is reset.
    """
    logger.info("Building stemmed tags for %s movies", len(movies))
    frame = movies.copy()
    frame["title"] = [
        _cell(row, "title_ml", "title_tmdb", "title") for _, row in frame.iterrows()
    ]
    frame["genre_names"] = [
        ", ".join(names_from(_cell(row, "genres_tmdb", "genres"))) for _, row in frame.iterrows()
    ]
    frame["tags"] = [movie_tag_text(row) for _, row in frame.iterrows()]
    keep = [
        column
        for column in ("movieId", "tmdbId", "title", "genre_names", "overview", "tags", "popularity")
        if column in frame.columns or column in ("title", "genre_names", "tags")
    ]
    prepared = frame.loc[:, [column for column in keep if column in frame.columns]].copy()
    prepared["movieId"] = prepared["movieId"].astype(int)
    prepared["tmdbId"] = prepared["tmdbId"].astype(int)
    prepared["poster_url"] = pd.NA
    logger.info("Movies with empty tags: %s", int((prepared["tags"].str.len() == 0).sum()))
    return prepared.reset_index(drop=True)
