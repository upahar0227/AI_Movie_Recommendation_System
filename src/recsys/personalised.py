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
    K_S,
    MMR_LAMBDA,
    RECENT_EVENT_LIMIT,
    RECENT_HALF_LIFE_DAYS,
    RECENT_WINDOW_DAYS,
    S_MAX,
    SEARCH_MIN_SLOTS,
    SEARCH_SLOT_HOURS,
    SEED,
    SIGNAL_CLICK,
    SIGNAL_RATING_HIGH,
    SIGNAL_RATING_LOW,
    SIGNAL_RATING_MID,
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
    """One account rating and the time it was last updated."""

    movie_id: int
    rating: float
    updated_at: datetime


@dataclass
class ActivityEvent:
    """A click or watchlist add. Other kinds are ignored by the recent window."""

    kind: str
    movie_id: int
    created_at: datetime
    detail: str = ""


@dataclass
class BlendedItem:
    """One blended recommendation and the scores that produced it."""

    movie_id: int
    score: float
    source: str
    reason: str
    base_score: float
    recent_score: float
    search_score: float
    w_base: float
    w_recent: float
    w_search: float
    snapshot_bonus: float


@dataclass
class BlendResult:
    """The ordered list plus the search-driven movie ids considered for it."""

    items: list[BlendedItem]
    search_picks: list[int]


def search_weight(n_events: int, w_max: float = S_MAX, k: float = K_S) -> float:
    """Return search influence. It grows with distinct queries and stops at ``w_max``.

    Args:
        n_events: Distinct searches. Zero or less returns 0.
        w_max: Cap, in 0–1. k: Smoother in ``n / (n + k)``.

    Returns:
        A weight in 0–w_max.
    """
    if n_events <= 0:
        return 0.0
    return min(float(w_max), float(n_events) / float(n_events + k))


def blend_weights(n_recent: int, n_search: int, *, recent_active: bool, search_active: bool) -> tuple[float, float, float]:
    """Return three weights that sum to 1. A missing signal is dropped and the rest are rescaled.

    Args:
        n_recent: Recent events passed to recent_weight.
        n_search: Distinct searches passed to search_weight.
        recent_active: Use the recent weight only when this is true and n_recent is positive.
        search_active: Use the search weight only when this is true and n_search is positive.

    Returns:
        (base, recent, search), each rounded to 10 decimals. All-zero evidence returns (1, 0, 0).
    """
    w_recent = recent_weight(n_recent) if recent_active and n_recent > 0 else 0.0
    w_search = search_weight(n_search) if search_active and n_search > 0 else 0.0
    w_base = max(0.0, 1.0 - w_recent - w_search)
    total = w_base + w_recent + w_search
    if total <= 0:
        return 1.0, 0.0, 0.0
    return round(w_base / total, 10), round(w_recent / total, 10), round(w_search / total, 10)


def recent_weight(n_events: int, w_max: float = W_RECENT_MAX, k: float = W_RECENT_K) -> float:
    """Return recent-activity influence. It grows with evidence and stops at ``w_max``.

    Args:
        n_events: Recent events. Zero or less returns 0.
        w_max: Cap, in 0–1. k: Smoother in ``n / (n + k)``.

    Returns:
        A weight in 0–w_max.
    """
    if n_events <= 0:
        return 0.0
    return min(float(w_max), float(n_events) / float(n_events + k))


def decay_factor(age_days: float, half_life_days: float) -> float:
    """Scale a signal by age. Each half-life cuts the remaining weight in half.

    Args:
        age_days: Age in days. Negative ages are treated as zero.
        half_life_days: Days per halving. Zero or less returns 1.

    Returns:
        A multiplier in 0–1.
    """
    if half_life_days <= 0:
        return 1.0
    return float(0.5 ** (max(age_days, 0.0) / half_life_days))


def rating_signal(rating: float) -> float:
    """Map a star rating to a profile weight. Implicit signals never use this.

    Args:
        rating: Stars. 4 or more maps to the high signal, 2 or less to the low signal.

    Returns:
        The high, low, or middle signal constant from config.
    """
    if rating >= 4:
        return SIGNAL_RATING_HIGH
    if rating <= 2:
        return SIGNAL_RATING_LOW
    return SIGNAL_RATING_MID


def limit_churn(ranked: list[int], previous: list[int], n: int, fraction: float = CHURN_MAX_FRACTION) -> list[int]:
    """Keep most of the previous list. New items can fill at most ``fraction`` of ``n``.

    Args:
        ranked: New order. Only these ids may appear in the result.
        previous: Last snapshot. Ids no longer in ranked are dropped.
        n: List length. Zero or less returns an empty list.
        fraction: Maximum share of n that may be new. The rest stay from previous.

    Returns:
        Up to n ids. With no previous list, this is the head of ranked.
    """
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
    """Re-rank a fixed set. Higher relevance wins, and near-duplicates are pushed down.

    Args:
        movie_ids: Candidates. The set is not grown or filtered.
        relevance: Movie id to the score maximised by the first term.
        similarity: Callable of two movie ids returning a similarity in about 0–1.
        lam: Weight on relevance. The rest penalises similarity to already picked ids.

    Returns:
        Every input id, in MMR order. Ties break toward the smaller movie id.
    """
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
    searches: list | None = None,
) -> BlendResult:
    """Score the catalog from the long-term profile, recent activity, and searches.

    Rated and watchlisted movies are excluded. A fresh search can reserve slots
    even when the churn limit would block them. The keyword is not returned.

    Args:
        service: Fitted models. genres: selected names; with no ratings they also filter.
        ratings: Account ratings. events: clicks and watchlist adds.
        watchlist_ids: Ids excluded from the list. previous_ids: last snapshot for churn.
        n: List length. now: clock; None uses the current UTC time.
        w_recent_override: Replace the recent weight, still capped by the config maximum.
        searches: SearchHit values or objects with the same fields. None means no searches.

    Returns:
        Ordered blended items and up to 10 positive search-score ids when search is active.
    """
    moment = now or datetime.now(timezone.utc)
    genre_names = _genre_names(service)
    selected = {genre.casefold() for genre in genres}
    exclude = {int(item.movie_id) for item in ratings} | {int(movie_id) for movie_id in watchlist_ids}
    recent_events = _recent_events(ratings, events, moment)
    from .search_profile import SearchHit, search_scores, select_searches

    search_hits = [
        item if isinstance(item, SearchHit) else SearchHit(
            item.query,
            _utc(item.created_at),
            item.clicked_movie_id,
            list(getattr(item, "matched_movie_ids", None) or []),
        )
        for item in (searches or [])
    ]
    search_raw = search_scores(service, search_hits, moment)
    active_searches = select_searches(search_hits, moment)
    recent_raw = _recent_scores(service, recent_events, genre_names, moment)
    recent_active = bool(recent_events) and float(np.max(np.abs(recent_raw))) > 1e-12
    search_active = bool(active_searches) and float(np.max(np.abs(search_raw))) > 1e-12
    w_base, w_recent, w_search = blend_weights(
        len(recent_events),
        len(active_searches),
        recent_active=recent_active,
        search_active=search_active,
    )
    if w_recent_override is not None:
        w_recent = min(W_RECENT_MAX, max(0.0, float(w_recent_override)))
        w_base = max(0.0, 1.0 - w_recent - w_search)
        total = w_base + w_recent + w_search
        w_base, w_recent, w_search = w_base / total, w_recent / total, w_search / total

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
    search_norm = _normalise_selected(search_raw, candidates)
    previous_set = {int(movie_id) for movie_id in previous_ids}
    adjusted: dict[int, float] = {}
    parts: dict[int, tuple[float, float, float, float]] = {}
    for index in candidates:
        movie_id = int(service.collaborative.movie_ids[index])
        base_score = float(base_norm[index])
        recent_score = float(recent_norm[index])
        search_score = float(search_norm[index])
        final = w_base * base_score + w_recent * recent_score + w_search * search_score
        bonus = SNAPSHOT_BONUS if movie_id in previous_set else 0.0
        adjusted[movie_id] = final + bonus
        parts[movie_id] = (base_score, recent_score, search_score, bonus)
    ranked = sorted(adjusted, key=lambda movie_id: (-adjusted[movie_id], movie_id))
    kept = limit_churn(ranked, [int(movie_id) for movie_id in previous_ids], n)
    search_order = sorted(adjusted, key=lambda movie_id: (-parts[movie_id][2], movie_id))
    positive_search = [movie_id for movie_id in search_order if parts[movie_id][2] > 0]
    kept = _reserve_search_slots(kept, positive_search, active_searches, moment, n)
    forced_search = set(positive_search[:SEARCH_MIN_SLOTS]) if _search_is_fresh(active_searches, moment) else set()
    ordered = mmr_order(kept, adjusted, lambda left, right: _similarity(service, left, right))
    items: list[BlendedItem] = []
    for movie_id in ordered:
        base_score, recent_score, search_score, bonus = parts[movie_id]
        source, reason = _reason(
            service,
            movie_id,
            genres=selected,
            genre_names=genre_names,
            ratings=ratings,
            events=recent_events,
            searches=active_searches,
            base_score=base_score,
            recent_score=recent_score,
            search_score=search_score,
            w_base=w_base,
            w_recent=w_recent,
            w_search=w_search,
            moment=moment,
            forced=movie_id in forced_search,
        )
        items.append(
            BlendedItem(
                movie_id=movie_id,
                score=round(adjusted[movie_id], 4),
                source=source,
                reason=reason,
                base_score=round(base_score, 4),
                recent_score=round(recent_score, 4),
                search_score=round(search_score, 4),
                w_base=round(w_base, 4),
                w_recent=round(w_recent, 4),
                w_search=round(w_search, 4),
                snapshot_bonus=round(bonus, 4),
            )
        )
    search_picks = [movie_id for movie_id in search_order if parts[movie_id][2] > 0][:10] if search_active else []
    return BlendResult(items, search_picks)


def _search_is_fresh(searches: list, moment: datetime) -> bool:
    """Report whether the newest search is inside the reserved-slot window.

    Args:
        searches: Hits with created_at. An empty list is not fresh.
        moment: Current time.

    Returns:
        True when the newest search is at most SEARCH_SLOT_HOURS old.
    """
    if not searches:
        return False
    latest = max(_utc(item.created_at) for item in searches)
    return (_utc(moment) - latest).total_seconds() <= SEARCH_SLOT_HOURS * 3600


def _reserve_search_slots(
    chosen: list[int],
    search_order: list[int],
    searches: list,
    moment: datetime,
    n: int,
) -> list[int]:
    """Keep a few search-driven movies in the list even when the churn limit would block them.

    Args:
        chosen: Ids already accepted by the churn limit.
        search_order: Candidates with a positive search score, strongest first.
        searches: Active searches. Empty, or older than the slot window, leaves chosen unchanged.
        moment: Current time. n: Maximum length.

    Returns:
        Up to n ids, with the first SEARCH_MIN_SLOTS search movies inserted if needed.
    """
    if n <= 0 or not searches or not search_order:
        return chosen
    latest = max(_utc(item.created_at) for item in searches)
    if (_utc(moment) - latest).total_seconds() > SEARCH_SLOT_HOURS * 3600:
        return chosen
    forced = search_order[:SEARCH_MIN_SLOTS]
    reserved = list(chosen)
    for movie_id in forced:
        if movie_id in reserved:
            continue
        if len(reserved) >= n:
            for index in range(len(reserved) - 1, -1, -1):
                if reserved[index] not in forced:
                    reserved.pop(index)
                    break
        reserved.append(movie_id)
    return reserved[:n]


def _recent_events(ratings: list[RatedMovie], events: list[ActivityEvent], moment: datetime) -> list[tuple[str, datetime, int, float, str]]:
    """Collect ratings, clicks, and watchlist adds inside the recent window.

    Search events are not included. Only the newest RECENT_EVENT_LIMIT rows are kept.

    Args:
        ratings: Account ratings. A rating counts when its update is inside the window.
        events: Activity rows. Only click and watchlist_add are kept.
        moment: Current time.

    Returns:
        (kind, time, movie id, rating, detail) tuples, newest first. Rating is 0
        for clicks and watchlist adds.
    """
    cutoff_seconds = RECENT_WINDOW_DAYS * 86400
    collected: list[tuple[datetime, str, int, float, str]] = []
    for item in ratings:
        age = (_utc(moment) - _utc(item.updated_at)).total_seconds()
        if age <= cutoff_seconds:
            collected.append((_utc(item.updated_at), "rating", int(item.movie_id), float(item.rating), ""))
    for event in events:
        if event.kind not in {"click", "watchlist_add"}:
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
    """Score the long-term profile: genre means, or genre means mixed with decayed ratings.

    Args:
        service: Fitted models.
        ratings: All account ratings, decayed with the long half-life.
        selected: Lowercased genre names. Empty yields zeros until ratings exist.
        genre_names: Movie id to its genre-name set.
        moment: Current time, used to age each rating.

    Returns:
        Array of shape (n_movies,). With no ratings this is the genre-mean vector.
    """
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
    """Score recent ratings, watchlist adds, and clicks with the short half-life.

    Args:
        service: Fitted models.
        events: Tuples from _recent_events. An empty list returns zeros.
        genre_names: Unused. Kept so callers can pass the catalog genre map.
        moment: Current time.

    Returns:
        Array of shape (n_movies,). Hybrid scores when a recent rating exists,
        otherwise content similarity only.
    """
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
    if explicit:
        return _hybrid_from_weights(service, weights, explicit, queries)
    return _content_scores(service, weights, queries)


def _rating_vectors(ratings: list[RatedMovie], moment: datetime, half_life: float) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Decay each rating into a content weight and an SVD rating.

    Args:
        ratings: Account ratings.
        moment: Current time.
        half_life: Days per halving of the signal and of the distance from 3 stars.

    Returns:
        Content (movie id, signed weight) pairs and SVD (movie id, decayed star) pairs.
    """
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
    """Blend content, SVD, popularity, and reception from weighted ratings.

    Args:
        service: Fitted models, including hybrid_k, hybrid_beta, and sentiment.
        weights: Signed content weights.
        explicit: (movie id, star) pairs. Their count sets the hybrid alpha.
        queries: Optional text weights. None adds no text.

    Returns:
        Array of shape (n_movies,) from hybrid_scores. A sentiment vector of the
        wrong length is replaced with zeros.
    """
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
    """Score the catalog by cosine similarity to a weighted sum of movie and text vectors.

    Args:
        service: Content model. Unknown movie ids and zero weights are skipped.
        weights: (movie id, signed weight) pairs.
        queries: (text, weight) pairs transformed by the fitted vectorizer.

    Returns:
        Array of shape (n_movies,) in collaborative order. Zeros when nothing contributes.
    """
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
    """Place each matching movie's item mean on the collaborative index.

    Args:
        service: Source of movie order and item means.
        selected: Lowercased genre names. Empty returns zeros.
        genre_names: Movie id to genre names. Matching is case-insensitive.

    Returns:
        Array of shape (n_movies,). Non-matching movies stay 0.
    """
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
    """Append a content weight for every movie whose genre matches a query token.

    Args:
        weights: List mutated in place.
        query: Text split on whitespace. An empty token set returns immediately.
        genre_names: Movie id to genre names.
        weight: Value appended with each matching movie id.

    Side effects:
        Appends (movie id, weight) pairs to weights. Does not remove existing pairs.
    """
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
    searches: list | None = None,
    base_score: float,
    recent_score: float,
    search_score: float = 0.0,
    w_base: float = 1.0,
    w_recent: float = 0.0,
    w_search: float = 0.0,
    moment: datetime,
    weight: float | None = None,
    forced: bool = False,
) -> tuple[str, str]:
    """Choose the reason whose weighted pull is largest.

    A forced search, or a search pull that beats the other pulls, wins. Otherwise
    the recent reason wins only when its pull beats the base pull.

    Args:
        service: Titles similar movies. movie_id: the card. genres: selected names.
        genre_names: Movie id to genre names. ratings: long-term ratings. events: recent tuples.
        searches: Active searches, or None to skip that reason.
        base_score, recent_score, search_score: normalised pulls before weights.
        w_base, w_recent, w_search: blend weights. moment: current time.
        weight: legacy recent weight, used only when w_recent is 0.
        forced: Use the search reason even when its numeric pull is smaller.

    Returns:
        A source label and a sentence that does not include the query text.
    """
    if weight is not None and w_recent == 0:
        w_recent = weight
        w_base = max(0.0, 1.0 - weight)
    base_source, base_reason = _profile_reason(service, movie_id, ratings, [], genres, genre_names, moment, BASE_HALF_LIFE_DAYS)
    recent_source, recent_reason = _profile_reason(
        service, movie_id, [], events, genres, genre_names, moment, RECENT_HALF_LIFE_DAYS
    )
    base_pull = w_base * base_score
    recent_pull = w_recent * recent_score
    search_pull = w_search * search_score
    if searches and (forced or (search_pull >= recent_pull and search_pull > base_pull)):
        return _search_reason(service, movie_id, searches)
    if w_recent > 0 and recent_pull > base_pull and recent_reason:
        return recent_source, recent_reason
    return base_source, base_reason


def _search_reason(service: RecommenderService, movie_id: int, searches: list) -> tuple[str, str]:
    """Explain a search hit without naming the query.

    Args:
        service: Used to title the clicked movie.
        movie_id: Recommended movie.
        searches: Active hits. Only rows with a clicked movie id are compared.

    Returns:
        ("search", "Similar to {title}") for the most similar clicked movie, otherwise
        ("search", "Based on your recent activity").
    """
    best_title = ""
    best_score = 0.0
    for event in searches:
        if not event.clicked_movie_id:
            continue
        score = _similarity(service, movie_id, int(event.clicked_movie_id))
        if score > best_score:
            best_score = score
            best_title = service._title(int(event.clicked_movie_id))
    if best_title:
        return "search", f"Similar to {best_title}"
    return "search", "Based on your recent activity"


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
    """Pick the strongest rating, watchlist, or old search-detail reason.

    Args:
        service: Content model and titles. movie_id: card being explained.
        ratings: Ratings scored with rating_signal. Non-positive signals are skipped.
        events: Recent tuples. Search rows still in this list can match on detail text.
        genres: Selected names. genre_names: movie id to genre names.
        moment: Current time. half_life: days used to decay each signal.

    Returns:
        (source, sentence). With no positive signal, a genre overlap returns
        ("genres", "From your genres"); otherwise the same default.
    """
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
    """Return content cosine similarity of two movies.

    Args:
        service: Fitted content model.
        left: First movie id. right: second movie id. The same id returns 1.

    Returns:
        Cosine similarity, or 0 when either id is missing from the content index.
    """
    if int(left) == int(right):
        return 1.0
    left_index = service.content.id_to_idx.get(int(left))
    right_index = service.content.id_to_idx.get(int(right))
    if left_index is None or right_index is None:
        return 0.0
    return float(cosine_similarity(service.content.matrix[left_index], service.content.matrix[right_index])[0, 0])


def _genre_names(service: RecommenderService) -> dict[int, set[str]]:
    """Map each movie id to its genre-name set.

    Args:
        service: Movie frame with genre_names. Missing or "nan" text becomes an empty set.

    Returns:
        movie id to stripped genre names. The names are not case-folded.
    """
    mapping: dict[int, set[str]] = {}
    frame = service.movies
    for _, row in frame.iterrows():
        raw = "" if row.get("genre_names") is None else str(row.get("genre_names"))
        if raw.lower() == "nan":
            raw = ""
        mapping[int(row["movieId"])] = {part.strip() for part in raw.split(",") if part.strip()}
    return mapping


def _folded(names: set[str]) -> set[str]:
    """Return the names in case-folded form.

    Args:
        names: Genre strings.

    Returns:
        The same names compared without case.
    """
    return {name.casefold() for name in names}


def _normalise_selected(scores: np.ndarray, indexes: list[int]) -> np.ndarray:
    """Min-max the scores at the candidate indexes and leave every other position at 0.

    Args:
        scores: One score per collaborative movie.
        indexes: Positions allowed into the blend. Empty returns zeros.

    Returns:
        Array of the same shape. Only indexes are in 0–1.
    """
    normalised = np.zeros_like(scores, dtype=np.float64)
    if not indexes:
        return normalised
    chosen = scores[np.asarray(indexes, dtype=int)]
    scaled = minmax(np.asarray(chosen, dtype=np.float64))
    for offset, index in enumerate(indexes):
        normalised[index] = scaled[offset]
    return normalised


def _utc(value: datetime) -> datetime:
    """Return a UTC datetime. A naive value is labeled UTC without shifting the clock.

    Args:
        value: Rating, event, or caller timestamp.

    Returns:
        An aware UTC datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
