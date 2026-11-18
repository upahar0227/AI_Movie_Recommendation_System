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
    query: str
    created_at: datetime
    clicked_movie_id: int | None = None
    matched_movie_ids: list[int] | None = None


def normalize_query(text: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in str(text))
    return " ".join(cleaned.split())


def select_searches(events: list[SearchHit], moment: datetime) -> list[SearchHit]:
    """Last distinct queries inside the search window, newest first."""
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
    """One profile row. The query weight is split across the top matches; a click is stronger."""
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
    """Cosine similarity of the decayed search profile against every movie."""
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
    """Keep every match. The profile only orders titles that match equally well."""
    text = normalize_query(query)
    profile = _profile_row(service, ratings)

    def bucket(card: dict) -> int:
        title = normalize_query(card.get("title", ""))
        if title == text:
            return 0
        if text and title.startswith(text):
            return 1
        return 2

    def profile_boost(card: dict) -> float:
        if profile is None:
            return 0.0
        index = service.content.id_to_idx.get(int(card["movie_id"]))
        if index is None:
            return 0.0
        return float(cosine_similarity(profile, service.content.matrix[index])[0, 0])

    return sorted(cards, key=lambda card: (bucket(card), -profile_boost(card), int(card["movie_id"])))


def _match_vector(service: RecommenderService, text: str):
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
    pairs = []
    if "title" not in service.movies.columns:
        return pairs
    for movie_id, title in service.movies[["movieId", "title"]].itertuples(index=False):
        pairs.append((int(movie_id), normalize_query(title)))
    return pairs


def _metadata_row(service: RecommenderService, text: str):
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
    stemmed = stem_text(text)
    for candidate in (stemmed, text):
        if not candidate.strip():
            continue
        vector = service.content.vectorizer.transform([candidate])
        if vector.nnz:
            return vector
    return None


def _movie_row(service: RecommenderService, movie_id: int):
    index = service.content.id_to_idx.get(int(movie_id))
    if index is None:
        return None
    return service.content.matrix[index]


def _profile_row(service: RecommenderService, ratings: list[tuple[int, float]]):
    liked = [movie_id for movie_id, rating in ratings if rating >= 4 and int(movie_id) in service.content.id_to_idx]
    if not liked:
        return None
    indexes = [service.content.id_to_idx[int(movie_id)] for movie_id in liked]
    return sparse.csr_matrix(service.content.matrix[indexes].mean(axis=0))


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()
