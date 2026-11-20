"""Turn a submitted search into a vector in the content-model space."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    SEARCH_CLICK_STRENGTH,
    SEARCH_EVENT_LIMIT,
    SEARCH_HALF_LIFE_DAYS,
    SEARCH_QUERY_STRENGTH,
    SEARCH_TITLE_MATCH,
    SEARCH_WINDOW_DAYS,
)
from .preprocessing import stem_text
from .serving import RecommenderService


@dataclass
class SearchHit:
    """One stored search: the text, when it was logged, and any click or matches."""

    query: str
    created_at: datetime
    clicked_movie_id: int | None = None
    matched_movie_ids: list[int] | None = None


def normalize_query(text: str) -> str:
    """Lowercase a query and collapse non-alphanumeric characters to single spaces.

    Args:
        text: Raw query. Non-strings are converted with str.

    Returns:
        A stripped, space-separated lowercase string.
    """
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in str(text))
    return " ".join(cleaned.split())


def select_searches(events: list[SearchHit], moment: datetime) -> list[SearchHit]:
    """Keep the last distinct queries inside the search window, newest first.

    Args:
        events: Logged searches. Queries that normalize to fewer than 2 characters
            are dropped. The same normalized text keeps the newest timestamp.
        moment: Current time. Events older than SEARCH_WINDOW_DAYS are dropped.

    Returns:
        At most SEARCH_EVENT_LIMIT hits, newest first.
    """
    moment = _aware(moment)
    cutoff = moment - timedelta(days=SEARCH_WINDOW_DAYS)
    latest: dict[str, SearchHit] = {}
    for event in events:
        if _aware(event.created_at) < cutoff:
            continue
        key = normalize_query(event.query)
        if len(key) < 2:
            continue
        current = latest.get(key)
        if current is None or event.created_at >= current.created_at:
            latest[key] = event
    chosen = sorted(latest.values(), key=lambda item: item.created_at, reverse=True)
    return chosen[:SEARCH_EVENT_LIMIT]


def query_vector(
    service: RecommenderService,
    query: str,
    clicked_movie_id: int | None = None,
    matched_movie_ids: list[int] | None = None,
):
    """Build one profile row. The query weight is split across matches; a click is stronger.

    Args:
        service: Fitted content model used as the vector space.
        query: Search text, used only when matched_movie_ids has no catalog rows.
        clicked_movie_id: Movie added at SEARCH_CLICK_STRENGTH. None adds nothing.
        matched_movie_ids: Movies that share SEARCH_QUERY_STRENGTH equally.

    Returns:
        A sparse content row, or None when neither the query nor the click maps to a vector.
    """
    vector = _matched_rows(service, matched_movie_ids or [])
    if vector is None:
        text = normalize_query(query)
        matched = _match_vector(service, text) if text else None
        if matched is not None:
            vector = matched.multiply(SEARCH_QUERY_STRENGTH)
    if clicked_movie_id is not None:
        clicked = _movie_row(service, int(clicked_movie_id))
        if clicked is not None:
            clicked = clicked.multiply(SEARCH_CLICK_STRENGTH)
            vector = clicked if vector is None else vector + clicked
    return vector


def _matched_rows(service: RecommenderService, movie_ids: list[int]):
    """Average the named movies' content rows at the query strength.

    Args:
        service: Content model. Unknown ids are skipped.
        movie_ids: Match ids. An empty usable list returns None.

    Returns:
        Their sum, each scaled by SEARCH_QUERY_STRENGTH divided by the usable count.
    """
    rows = []
    for movie_id in movie_ids:
        row = _movie_row(service, int(movie_id))
        if row is not None:
            rows.append(row)
    if not rows:
        return None
    share = SEARCH_QUERY_STRENGTH / len(rows)
    combined = None
    for row in rows:
        weighted = row.multiply(share)
        combined = weighted if combined is None else combined + weighted
    return combined


def search_scores(service: RecommenderService, events: list[SearchHit], moment: datetime) -> np.ndarray:
    """Return cosine similarity of the decayed search profile against every movie.

    Args:
        service: Content and collaborative models. Scores are aligned to the latter.
        events: Searches passed through select_searches before they are summed.
        moment: Time used for the window and the half-life decay.

    Returns:
        Array of shape (n_movies,) in collaborative-model order. Zeros when no
        selected search produces a vector.
    """
    width = len(service.collaborative.movie_ids)
    chosen = select_searches(events, moment)
    from .personalised import decay_factor

    combined = None
    for event in chosen:
        vector = query_vector(service, event.query, event.clicked_movie_id, event.matched_movie_ids)
        if vector is None or vector.nnz == 0:
            continue
        age_days = max((_aware(moment) - _aware(event.created_at)).total_seconds(), 0.0) / 86400
        faded = decay_factor(age_days, SEARCH_HALF_LIFE_DAYS)
        if faded == 0:
            continue
        row = vector.multiply(faded)
        combined = row if combined is None else combined + row
    aligned = np.zeros(width, dtype=np.float64)
    if combined is None or combined.nnz == 0:
        return aligned
    scores = cosine_similarity(combined, service.content.matrix).ravel()
    return service._align_content(scores)


def tie_break_search_results(service: RecommenderService, query: str, cards: list[dict], ratings: list[tuple[int, float]]) -> list[dict]:
    """Keep every match. The profile only orders titles that match equally well.

    Args:
        service: Content model used for the liked-movie profile.
        query: Text compared with each card title after normalization.
        cards: Search dicts with title and movie_id. None are removed.
        ratings: (movie id, star rating) pairs. Only ratings of 4 or more form the profile.

    Returns:
        The same cards, exact title first, then prefix, then the rest. Ties break
        toward the liked-movie profile and then the smaller movie id.
    """
    text = normalize_query(query)
    profile = _profile_row(service, ratings)

    def bucket(card: dict) -> int:
        """Rank an exact title above a prefix and both above any other match.

        Args:
            card: Search dict with a title.

        Returns:
            0 for an exact title, 1 for a prefix, 2 otherwise.
        """
        title = normalize_query(card.get("title", ""))
        if title == text:
            return 0
        if text and title.startswith(text):
            return 1
        return 2

    def profile_boost(card: dict) -> float:
        """Return cosine similarity between the liked profile and this card.

        Args:
            card: Search dict with movie_id.

        Returns:
            Similarity in about -1–1, or 0 when the profile or the movie is missing.
        """
        if profile is None:
            return 0.0
        index = service.content.id_to_idx.get(int(card["movie_id"]))
        if index is None:
            return 0.0
        return float(cosine_similarity(profile, service.content.matrix[index])[0, 0])

    return sorted(cards, key=lambda card: (bucket(card), -profile_boost(card), int(card["movie_id"])))


def _match_vector(service: RecommenderService, text: str):
    """Map query text to one content row: exact title, metadata, fuzzy title, then TF-IDF.

    Args:
        service: Catalog and fitted vectorizer.
        text: Already normalized query.

    Returns:
        A sparse content row. Fuzzy title matching runs only when the text has at
        least four characters and the best ratio reaches SEARCH_TITLE_MATCH.
        None when the text vectorizer also finds no tokens.
    """
    titles = _titles(service)
    for movie_id, title in titles:
        if title == text:
            return _movie_row(service, movie_id)
    metadata = _metadata_row(service, text)
    if metadata is not None:
        return metadata
    if len(text) >= 4:
        best_id = None
        best_ratio = 0.0
        for movie_id, title in titles:
            ratio = _ratio(text, title)
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = movie_id
        if best_id is not None and best_ratio >= SEARCH_TITLE_MATCH:
            return _movie_row(service, best_id)
    return _text_row(service, text)


def _titles(service: RecommenderService) -> list[tuple[int, str]]:
    """List normalized titles in catalog order.

    Args:
        service: Movie frame. A frame without a title column returns an empty list.

    Returns:
        (movie id, normalized title) pairs.
    """
    pairs = []
    if "title" not in service.movies.columns:
        return pairs
    for movie_id, title in service.movies[["movieId", "title"]].itertuples(index=False):
        pairs.append((int(movie_id), normalize_query(title)))
    return pairs


def _metadata_row(service: RecommenderService, text: str):
    """Average content rows whose genre, lead, director, or keywords equal the text.

    Args:
        service: Movie frame and content matrix.
        text: Normalized name. Movies not in the content index are skipped.

    Returns:
        The mean sparse row, or None when no name matches.
    """
    movies = service.movies
    indexes: list[int] = []
    for position, row in movies.reset_index(drop=True).iterrows():
        movie_id = int(row["movieId"])
        content_index = service.content.id_to_idx.get(movie_id)
        if content_index is None:
            continue
        names = _names(row)
        if text in names:
            indexes.append(content_index)
    if not indexes:
        return None
    mean = service.content.matrix[indexes].mean(axis=0)
    return sparse.csr_matrix(mean)


def _names(row) -> set[str]:
    """Collect normalized genre, lead, director, and keyword names from one row.

    Args:
        row: A movie row. Missing columns are skipped.

    Returns:
        Normalized name strings. Empty pieces are dropped.
    """
    found = set()
    if "genre_names" in row.index:
        for part in str(row["genre_names"]).split(","):
            name = normalize_query(part)
            if name:
                found.add(name)
    for column in ("lead", "director", "keywords"):
        if column not in row.index or not str(row[column]).strip():
            continue
        for part in str(row[column]).split(","):
            name = normalize_query(part)
            if name:
                found.add(name)
    return found


def _text_row(service: RecommenderService, text: str):
    """Transform stemmed text, then the raw text, with the fitted vectorizer.

    Args:
        service: Content model whose vectorizer is already fit.
        text: Query text.

    Returns:
        The first non-empty sparse row, or None when both transforms are empty.
    """
    stemmed = stem_text(text)
    for candidate in (stemmed, text):
        if not candidate.strip():
            continue
        vector = service.content.vectorizer.transform([candidate])
        if vector.nnz:
            return vector
    return None


def _movie_row(service: RecommenderService, movie_id: int):
    """Return one movie's TF-IDF row.

    Args:
        service: Fitted content model.
        movie_id: MovieLens id.

    Returns:
        The sparse matrix row, or None when the id is not in the content index.
    """
    index = service.content.id_to_idx.get(int(movie_id))
    if index is None:
        return None
    return service.content.matrix[index]


def _profile_row(service: RecommenderService, ratings: list[tuple[int, float]]):
    """Average the content rows of movies rated 4 or higher.

    Args:
        service: Fitted content model.
        ratings: (movie id, star rating) pairs. Unknown ids are ignored.

    Returns:
        The mean sparse row, or None when no liked movie is in the catalog.
    """
    liked = [movie_id for movie_id, rating in ratings if rating >= 4 and int(movie_id) in service.content.id_to_idx]
    if not liked:
        return None
    indexes = [service.content.id_to_idx[int(movie_id)] for movie_id in liked]
    return sparse.csr_matrix(service.content.matrix[indexes].mean(axis=0))


def _aware(value: datetime) -> datetime:
    """Return a UTC datetime. A naive value is labeled UTC without shifting the clock.

    Args:
        value: Timestamp from a search event or the caller.

    Returns:
        An aware UTC datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ratio(left: str, right: str) -> float:
    """Return the SequenceMatcher ratio of two strings.

    Args:
        left: Normalized query. right: Normalized title. Either empty returns 0.

    Returns:
        A similarity in 0–1.
    """
    if not left or not right:
        return 0.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()
