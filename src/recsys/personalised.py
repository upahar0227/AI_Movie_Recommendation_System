"""Blend a stable taste profile with a capped recent-activity profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    BASE_GENRE_WEIGHT,
    BASE_HALF_LIFE_DAYS,
    CHURN_MAX_FRACTION,
    MMR_LAMBDA,
    RECENT_EVENT_LIMIT,
    RECENT_HALF_LIFE_DAYS,
    RECENT_WINDOW_DAYS,
    SEED,
    SIGNAL_CLICK,
    SIGNAL_RATING_HIGH,
    SIGNAL_RATING_LOW,
    SIGNAL_RATING_MID,
    SIGNAL_SEARCH,
    SIGNAL_WATCHLIST,
    SNAPSHOT_BONUS,
    W_RECENT_K,
    W_RECENT_MAX,
)
from .hybrid import hybrid_scores, minmax
from .serving import RecommenderService

NEUTRAL_RATING = 3.0


@dataclass
class RatedMovie:
    movie_id: int
    rating: float
    updated_at: datetime


@dataclass
class ActivityEvent:
    kind: str
    movie_id: int
    created_at: datetime
    detail: str = ""


@dataclass
class BlendedItem:
    movie_id: int
    score: float
    source: str
    reason: str
    base_score: float
    recent_score: float
    w_recent: float
    snapshot_bonus: float


def recent_weight(n_events: int, w_max: float = W_RECENT_MAX, k: float = W_RECENT_K) -> float:
    """Recent activity grows with evidence and stops at ``w_max``."""
    if n_events <= 0:
        return 0.0
    return min(float(w_max), float(n_events) / float(n_events + k))


def decay_factor(age_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return float(0.5 ** (max(age_days, 0.0) / half_life_days))


def rating_signal(rating: float) -> float:
    """Map a star rating to a profile weight. Implicit signals never use this."""
    if rating >= 4:
        return SIGNAL_RATING_HIGH
    if rating <= 2:
        return SIGNAL_RATING_LOW
    return SIGNAL_RATING_MID


def limit_churn(ranked: list[int], previous: list[int], n: int, fraction: float = CHURN_MAX_FRACTION) -> list[int]:
    """Keep most of the previous list. New items can fill at most ``fraction`` of ``n``."""
    if n <= 0:
        return []
    if not previous:
        return ranked[:n]
    eligible = set(ranked)
    previous_ok = [movie_id for movie_id in previous if movie_id in eligible]
    order = {movie_id: index for index, movie_id in enumerate(ranked)}
    previous_ok.sort(key=lambda movie_id: (order[movie_id], movie_id))
    max_new = int(math.floor(fraction * n + 1e-9))
    keep_target = min(len(previous_ok), max(n - max_new, 0))
    chosen = previous_ok[:keep_target]
    chosen_set = set(chosen)
    previous_set = set(previous_ok)
    new_count = 0
    for movie_id in ranked:
        if len(chosen) >= n:
            break
        if movie_id in chosen_set:
            continue
        if movie_id not in previous_set:
            if new_count >= max_new:
                continue
            new_count += 1
        chosen.append(movie_id)
        chosen_set.add(movie_id)
    return chosen[:n]


def mmr_order(
    movie_ids: list[int],
    relevance: dict[int, float],
    similarity,
    lam: float = MMR_LAMBDA,
) -> list[int]:
    """Re-rank a fixed set. Higher relevance wins, and near-duplicates are pushed down."""
    np.random.seed(SEED)
    remaining = sorted(movie_ids)
    selected: list[int] = []
    while remaining:
        best_id = remaining[0]
        best_key = None
        for movie_id in remaining:
            penalty = 0.0 if not selected else max(float(similarity(movie_id, other)) for other in selected)
            value = lam * float(relevance.get(movie_id, 0.0)) - (1.0 - lam) * penalty
            key = (value, -movie_id)
            if best_key is None or key > best_key:
                best_key = key
                best_id = movie_id
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


def blended_recommendations(
    service: RecommenderService,
    *,
    genres: list[str],
    ratings: list[RatedMovie],
    events: list[ActivityEvent],
    watchlist_ids: set[int],
    previous_ids: list[int],
    n: int,
    now: datetime | None = None,
    w_recent_override: float | None = None,
) -> list[BlendedItem]:
    """Score the catalog from the long-term profile and a capped recent profile."""
    moment = now or datetime.now(timezone.utc)
    genre_names = _genre_names(service)
    selected = {genre.casefold() for genre in genres}
    exclude = {int(item.movie_id) for item in ratings} | {int(movie_id) for movie_id in watchlist_ids}
    recent_events = _recent_events(ratings, events, moment)
    weight = recent_weight(len(recent_events))
    if w_recent_override is not None:
        weight = min(W_RECENT_MAX, max(0.0, float(w_recent_override)))

    base_raw = _base_scores(service, ratings, selected, genre_names, moment)
    recent_raw = _recent_scores(service, recent_events, genre_names, moment)
    candidates = [
        index
        for index, movie_id in enumerate(service.collaborative.movie_ids)
        if int(movie_id) not in exclude and int(movie_id) in genre_names
    ]
    if not ratings and selected:
        candidates = [index for index in candidates if selected.intersection(_folded(genre_names[int(service.collaborative.movie_ids[index])]))]
    base_norm = _normalise_selected(base_raw, candidates)
    recent_norm = _normalise_selected(recent_raw, candidates)
    previous_set = {int(movie_id) for movie_id in previous_ids}
    adjusted: dict[int, float] = {}
    parts: dict[int, tuple[float, float, float]] = {}
    for index in candidates:
        movie_id = int(service.collaborative.movie_ids[index])
        base_score = float(base_norm[index])
        recent_score = float(recent_norm[index])
        final = (1.0 - weight) * base_score + weight * recent_score
        bonus = SNAPSHOT_BONUS if movie_id in previous_set else 0.0
        adjusted[movie_id] = final + bonus
        parts[movie_id] = (base_score, recent_score, bonus)
    ranked = sorted(adjusted, key=lambda movie_id: (-adjusted[movie_id], movie_id))
    kept = limit_churn(ranked, [int(movie_id) for movie_id in previous_ids], n)
    ordered = mmr_order(kept, adjusted, lambda left, right: _similarity(service, left, right))
    items: list[BlendedItem] = []
    for movie_id in ordered:
        base_score, recent_score, bonus = parts[movie_id]
        source, reason = _reason(
            service,
            movie_id,
            genres=selected,
            genre_names=genre_names,
            ratings=ratings,
            events=recent_events,
            base_score=base_score,
            recent_score=recent_score,
            weight=weight,
            moment=moment,
        )
        items.append(
            BlendedItem(
                movie_id=movie_id,
                score=round(adjusted[movie_id], 4),
                source=source,
                reason=reason,
                base_score=round(base_score, 4),
                recent_score=round(recent_score, 4),
                w_recent=round(weight, 4),
                snapshot_bonus=round(bonus, 4),
            )
        )
    return items


def _recent_events(ratings: list[RatedMovie], events: list[ActivityEvent], moment: datetime) -> list[tuple[str, datetime, int, float, str]]:
    """Ratings in the window plus clicks, watchlist adds, and searches. Newest 30 only."""
    cutoff_seconds = RECENT_WINDOW_DAYS * 86400
    collected: list[tuple[datetime, str, int, float, str]] = []
    for item in ratings:
        age = (_utc(moment) - _utc(item.updated_at)).total_seconds()
        if age <= cutoff_seconds:
            collected.append((_utc(item.updated_at), "rating", int(item.movie_id), float(item.rating), ""))
    for event in events:
        if event.kind not in {"click", "watchlist_add", "search"}:
            continue
        age = (_utc(moment) - _utc(event.created_at)).total_seconds()
        if age <= cutoff_seconds:
            collected.append((_utc(event.created_at), event.kind, int(event.movie_id), 0.0, event.detail or ""))
    collected.sort(key=lambda item: item[0], reverse=True)
    chosen = collected[:RECENT_EVENT_LIMIT]
    return [(kind, when, movie_id, rating, detail) for when, kind, movie_id, rating, detail in chosen]


def _base_scores(
    service: RecommenderService,
    ratings: list[RatedMovie],
    selected: set[str],
    genre_names: dict[int, set[str]],
    moment: datetime,
) -> np.ndarray:
    genre_scores = _genre_mean_scores(service, selected, genre_names)
    if not ratings:
        return genre_scores
    weights, explicit = _rating_vectors(ratings, moment, BASE_HALF_LIFE_DAYS)
    hybrid = _hybrid_from_weights(service, weights, explicit)
    mixed = (1.0 - BASE_GENRE_WEIGHT) * minmax(hybrid) + BASE_GENRE_WEIGHT * minmax(genre_scores)
    return mixed


def _recent_scores(
    service: RecommenderService,
    events: list[tuple[str, datetime, int, float, str]],
    genre_names: dict[int, set[str]],
    moment: datetime,
) -> np.ndarray:
    width = len(service.collaborative.movie_ids)
    if not events:
        return np.zeros(width, dtype=np.float64)
    weights: list[tuple[int, float]] = []
    explicit: list[tuple[int, float]] = []
    queries: list[tuple[str, float]] = []
    for kind, when, movie_id, rating, detail in events:
        age_days = (_utc(moment) - _utc(when)).total_seconds() / 86400
        faded = decay_factor(age_days, RECENT_HALF_LIFE_DAYS)
        if kind == "rating":
            weights.append((movie_id, rating_signal(rating) * faded))
            explicit.append((movie_id, NEUTRAL_RATING + (rating - NEUTRAL_RATING) * faded))
        elif kind == "watchlist_add":
            weights.append((movie_id, SIGNAL_WATCHLIST * faded))
        elif kind == "click":
            weights.append((movie_id, SIGNAL_CLICK * faded))
        elif kind == "search" and detail.strip():
            queries.append((detail.strip(), SIGNAL_SEARCH * faded))
            _add_search_genre_weights(weights, detail, genre_names, SIGNAL_SEARCH * faded)
    if explicit:
        return _hybrid_from_weights(service, weights, explicit, queries)
    return _content_scores(service, weights, queries)


def _rating_vectors(ratings: list[RatedMovie], moment: datetime, half_life: float) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    weights: list[tuple[int, float]] = []
    explicit: list[tuple[int, float]] = []
    for item in ratings:
        age_days = (_utc(moment) - _utc(item.updated_at)).total_seconds() / 86400
        faded = decay_factor(age_days, half_life)
        weights.append((int(item.movie_id), rating_signal(item.rating) * faded))
        explicit.append((int(item.movie_id), NEUTRAL_RATING + (float(item.rating) - NEUTRAL_RATING) * faded))
    return weights, explicit


def _hybrid_from_weights(
    service: RecommenderService,
    weights: list[tuple[int, float]],
    explicit: list[tuple[int, float]],
    queries: list[tuple[str, float]] | None = None,
) -> np.ndarray:
    content = _content_scores(service, weights, queries or [])
    popularity = service.collaborative.popularity_scores()
    svd = service.collaborative.svd_scores(explicit)
    sentiment = np.asarray(service.sentiment, dtype=np.float64)
    if len(sentiment) != len(service.collaborative.movie_ids):
        sentiment = np.zeros(len(service.collaborative.movie_ids), dtype=np.float64)
    return hybrid_scores(
        collaborative=svd,
        content=content,
        popularity=popularity,
        sentiment=sentiment,
        n_ratings=len(explicit),
        k=service.hybrid_k,
        beta=service.hybrid_beta,
        use_sentiment=True,
    )


def _content_scores(
    service: RecommenderService,
    weights: list[tuple[int, float]],
    queries: list[tuple[str, float]],
) -> np.ndarray:
    matrix = service.content.matrix
    combined = None
    for movie_id, weight in weights:
        if weight == 0:
            continue
        index = service.content.id_to_idx.get(int(movie_id))
        if index is None:
            continue
        row = matrix[index].multiply(float(weight))
        combined = row if combined is None else combined + row
    for text, weight in queries:
        if weight == 0 or not text.strip():
            continue
        vector = service.content.vectorizer.transform([text])
        if vector.nnz == 0:
            continue
        row = vector.multiply(float(weight))
        combined = row if combined is None else combined + row
    aligned = np.zeros(len(service.collaborative.movie_ids), dtype=np.float64)
    if combined is None:
        return aligned
    scores = cosine_similarity(combined, matrix).ravel()
    return service._align_content(scores)


def _genre_mean_scores(service: RecommenderService, selected: set[str], genre_names: dict[int, set[str]]) -> np.ndarray:
    scores = np.zeros(len(service.collaborative.movie_ids), dtype=np.float64)
    if not selected:
        return scores
    for index, movie_id in enumerate(service.collaborative.movie_ids):
        names = _folded(genre_names.get(int(movie_id), set()))
        if selected.intersection(names):
            scores[index] = float(service.collaborative.item_mean[index])
    return scores


def _add_search_genre_weights(
    weights: list[tuple[int, float]],
    query: str,
    genre_names: dict[int, set[str]],
    weight: float,
) -> None:
    tokens = {token.casefold() for token in query.split() if token.strip()}
    if not tokens:
        return
    for movie_id, names in genre_names.items():
        if tokens.intersection(_folded(names)):
            weights.append((int(movie_id), weight))


def _reason(
    service: RecommenderService,
    movie_id: int,
    *,
    genres: set[str],
    genre_names: dict[int, set[str]],
    ratings: list[RatedMovie],
    events: list[tuple[str, datetime, int, float, str]],
    base_score: float,
    recent_score: float,
    weight: float,
    moment: datetime,
) -> tuple[str, str]:
    base_source, base_reason = _profile_reason(service, movie_id, ratings, [], genres, genre_names, moment, BASE_HALF_LIFE_DAYS)
    recent_source, recent_reason = _profile_reason(
        service, movie_id, [], events, genres, genre_names, moment, RECENT_HALF_LIFE_DAYS
    )
    base_pull = (1.0 - weight) * base_score
    recent_pull = weight * recent_score
    if weight > 0 and recent_pull > base_pull and recent_reason:
        return recent_source, recent_reason
    return base_source, base_reason


def _profile_reason(
    service: RecommenderService,
    movie_id: int,
    ratings: list[RatedMovie],
    events: list[tuple[str, datetime, int, float, str]],
    genres: set[str],
    genre_names: dict[int, set[str]],
    moment: datetime,
    half_life: float,
) -> tuple[str, str]:
    best_score = 0.0
    best = ("genres", "From your genres")
    target = service.content.id_to_idx.get(int(movie_id))
    for item in ratings:
        signal = rating_signal(item.rating)
        if signal <= 0:
            continue
        age_days = (_utc(moment) - _utc(item.updated_at)).total_seconds() / 86400
        score = signal * decay_factor(age_days, half_life) * _similarity(service, movie_id, item.movie_id)
        if score > best_score:
            best_score = score
            best = ("rating", f"Because you rated {service._title(item.movie_id)}")
    for kind, when, source_id, rating, detail in events:
        age_days = (_utc(moment) - _utc(when)).total_seconds() / 86400
        faded = decay_factor(age_days, half_life)
        if kind == "rating" and rating_signal(rating) > 0:
            score = rating_signal(rating) * faded * _similarity(service, movie_id, source_id)
            if score > best_score:
                best_score = score
                best = ("rating", f"Because you rated {service._title(source_id)}")
        elif kind == "watchlist_add":
            score = SIGNAL_WATCHLIST * faded * _similarity(service, movie_id, source_id)
            if score > best_score:
                best_score = score
                best = ("watchlist", f"Because you added {service._title(source_id)} to your watchlist")
        elif kind == "search" and detail.strip() and target is not None:
            vector = service.content.vectorizer.transform([detail.strip()])
            if vector.nnz:
                score = SIGNAL_SEARCH * faded * float(cosine_similarity(vector, service.content.matrix[target])[0, 0])
                if score > best_score:
                    best_score = score
                    best = ("search", "Matches your recent search")
    names = _folded(genre_names.get(int(movie_id), set()))
    if best_score <= 0 and genres.intersection(names):
        return "genres", "From your genres"
    return best


def _similarity(service: RecommenderService, left: int, right: int) -> float:
    if int(left) == int(right):
        return 1.0
    left_index = service.content.id_to_idx.get(int(left))
    right_index = service.content.id_to_idx.get(int(right))
    if left_index is None or right_index is None:
        return 0.0
    return float(cosine_similarity(service.content.matrix[left_index], service.content.matrix[right_index])[0, 0])


def _genre_names(service: RecommenderService) -> dict[int, set[str]]:
    mapping: dict[int, set[str]] = {}
    frame = service.movies
    for _, row in frame.iterrows():
        raw = "" if row.get("genre_names") is None else str(row.get("genre_names"))
        if raw.lower() == "nan":
            raw = ""
        mapping[int(row["movieId"])] = {part.strip() for part in raw.split(",") if part.strip()}
    return mapping


def _folded(names: set[str]) -> set[str]:
    return {name.casefold() for name in names}


def _normalise_selected(scores: np.ndarray, indexes: list[int]) -> np.ndarray:
    normalised = np.zeros_like(scores, dtype=np.float64)
    if not indexes:
        return normalised
    chosen = scores[np.asarray(indexes, dtype=int)]
    scaled = minmax(np.asarray(chosen, dtype=np.float64))
    for offset, index in enumerate(indexes):
        normalised[index] = scaled[offset]
    return normalised


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
