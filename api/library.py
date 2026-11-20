"""Ratings, watchlist, and recommendations for the signed-in app user."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db
from src.app.models import (
    EVENT_TYPES,
    RATING_VALUES,
    Interaction,
    Rating,
    RecommendationSnapshot,
    SearchEvent,
    User,
    UserGenre,
    WatchlistItem,
)
from src.app.security import PERSONALISED_AFTER, recommendation_mode
from src.app.settings import recommendations_debug_enabled
from src.recsys.config import GENRE_SELECTION_MAX, SEARCH_DEDUP_MINUTES, SEARCH_MIN_CHARS
from src.recsys.personalised import ActivityEvent, RatedMovie, blended_recommendations
from src.recsys.search_profile import SearchHit, normalize_query
from src.recsys.serving import RecommenderService

router = APIRouter(tags=["library"])


class RatingBody(BaseModel):
    """A MovieLens movie id and a half-star rating."""

    movie_id: int
    rating: float

    @field_validator("rating")
    @classmethod
    def half_star(cls, value: float) -> float:
        """Accept only the ten half-star values from 0.5 to 5.0.

        Args:
            value: Submitted rating.

        Returns:
            The same rating.

        Raises:
            ValueError: The value is not one of those steps. FastAPI reports this as 422.
        """
        if value not in RATING_VALUES:
            raise ValueError("Rating must be from 0.5 to 5.0 in steps of 0.5.")
        return value


class GenreBody(BaseModel):
    """At least one genre name chosen during onboarding."""

    genres: list[str] = Field(min_length=1)


class GenreUpdateBody(BaseModel):
    """Replacement genre list, limited to the configured maximum."""

    genres: list[str] = Field(min_length=1, max_length=GENRE_SELECTION_MAX)


class OnboardingBody(BaseModel):
    """At least five half-star ratings collected before the main site opens."""

    ratings: list[RatingBody] = Field(min_length=5)


class InteractionBody(BaseModel):
    """A movie id and one of the stored interaction event types."""

    movie_id: int
    event_type: str

    @field_validator("event_type")
    @classmethod
    def known_event(cls, value: str) -> str:
        """Accept only impression, click, rating, watchlist_add, or search.

        Args:
            value: Submitted event type.

        Returns:
            The same event type.

        Raises:
            ValueError: The type is not in that set. FastAPI reports this as 422.
        """
        if value not in EVENT_TYPES:
            raise ValueError("Unknown event type.")
        return value


def _service(request: Request) -> RecommenderService:
    """Return the models loaded on the app.

    Args:
        request: Current request.

    Returns:
        The recommender service.

    Raises:
        HTTPException: 503 when the models are not loaded.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Train them first.")
    return service


def _require_movie(service: RecommenderService, movie_id: int) -> None:
    """Reject a movie id that is not in the catalog.

    Args:
        service: Loaded movies.
        movie_id: MovieLens id.

    Raises:
        HTTPException: 404 when the id is absent.
    """
    catalog = {int(value) for value in service.movies["movieId"]}
    if int(movie_id) not in catalog:
        raise HTTPException(status_code=404, detail=f"Unknown movie id {movie_id}")


def _save_rating(session: Session, user: User, movie_id: int, rating: float) -> None:
    """Insert or update one rating and record a rating interaction.

    Args:
        session: Account session. The caller commits.
        user: Signed-in account.
        movie_id: MovieLens id.
        rating: Half-star value already accepted by the body.

    Side effects:
        Writes the rating row and an interaction. A repeat updates the score and timestamp.
    """
    now = datetime.now(timezone.utc)
    existing = session.scalar(select(Rating).where(Rating.user_id == user.id, Rating.movie_id == movie_id))
    if existing is None:
        session.add(Rating(user_id=user.id, movie_id=movie_id, rating=rating, created_at=now, updated_at=now))
    else:
        existing.rating = rating
        existing.updated_at = now
    session.add(Interaction(user_id=user.id, movie_id=movie_id, event_type="rating", created_at=now))


def rating_map(session: Session, user_id: int) -> dict[int, float]:
    """Load every rating the account has saved.

    Args:
        session: Account session.
        user_id: Account id.

    Returns:
        MovieLens id to star rating.
    """
    rows = session.scalars(select(Rating).where(Rating.user_id == user_id)).all()
    return {int(row.movie_id): float(row.rating) for row in rows}


def apply_user_ratings(cards: list[dict], scores: dict[int, float]) -> list[dict]:
    """Copy the account's rating onto each card.

    Args:
        cards: Dicts with ``movie_id``. They are changed in place.
        scores: Movie id to star rating. A missing id stores None.

    Returns:
        The same list.
    """
    for card in cards:
        card["user_rating"] = scores.get(int(card["movie_id"]))
    return cards


def recent_clicks(session: Session, user_id: int, limit: int = 20) -> list[int]:
    """Return recent clicked movie ids, newest first and without repeats.

    Args:
        session: Account session.
        user_id: Account id.
        limit: Maximum click rows to read before de-duplicating.

    Returns:
        Distinct MovieLens ids. A click with no movie id is skipped.
    """
    rows = session.scalars(
        select(Interaction)
        .where(Interaction.user_id == user_id, Interaction.event_type == "click")
        .order_by(Interaction.created_at.desc())
        .limit(limit)
    ).all()
    seen: list[int] = []
    for row in rows:
        if row.movie_id and row.movie_id not in seen:
            seen.append(int(row.movie_id))
    return seen


class SearchEventBody(BaseModel):
    """A submitted search, its matched movie ids, and an optional clicked movie."""

    query: str = ""
    matched_movie_ids: list[int] = Field(default_factory=list)
    clicked_movie_id: int | None = None


def _drop_recommendation_cache(session: Session, user_id: int) -> None:
    """Delete the stored recommendation snapshot so the next list can change.

    Args:
        session: Account session. The caller commits.
        user_id: Account id.

    Side effects:
        Deletes that account's recommendation-snapshot rows.
    """
    session.execute(delete(RecommendationSnapshot).where(RecommendationSnapshot.user_id == user_id))


@router.post("/ratings")
def rate(body: RatingBody, request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """POST /ratings. Sign-in is required.

    Body: movie id and a half-star rating from 0.5 to 5.0. A repeat replaces the old score.

    Returns:
        The saved movie id and rating.

    Raises:
        HTTPException: 401 when signed out, 404 when the movie is unknown, 422 when the
        rating is not a half star, 503 when the models or database are missing.

    Side effects:
        Upserts the rating and adds a rating interaction. ``get_db`` commits.
    """
    service = _service(request)
    _require_movie(service, body.movie_id)
    _save_rating(session, user, body.movie_id, body.rating)
    return {"movie_id": body.movie_id, "rating": body.rating}


@router.get("/genres")
def list_genres(request: Request, user: User = Depends(get_current_user)):
    """GET /genres. Sign-in is required.

    Returns:
        ``genres``: the sorted names present in the catalog.

    Raises:
        HTTPException: 401 when signed out, 503 when the models or database are missing.
    """
    del user
    return {"genres": _service(request).catalog_genres()}


@router.post("/genres")
def save_genres(body: GenreBody, request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """POST /genres. Sign-in is required.

    Body: at least one genre name. Blank and duplicate names are dropped. This also
    marks onboarding complete.

    Returns:
        ``genres``: the names that were stored.

    Raises:
        HTTPException: 401 when signed out, 422 when a name is unknown or none remain,
        503 when the models or database are missing.

    Side effects:
        Replaces the account's genres and clears its recommendation snapshot.
    """
    catalog = set(_service(request).catalog_genres())
    chosen = []
    for genre in body.genres:
        name = genre.strip()
        if name not in catalog:
            raise HTTPException(status_code=422, detail=f"Unknown genre {name}.")
        if name not in chosen:
            chosen.append(name)
    if not chosen:
        raise HTTPException(status_code=422, detail="Pick at least one genre.")
    _replace_genres(session, user, chosen)
    user.onboarding_completed = True
    session.add(user)
    return {"genres": chosen}


@router.get("/users/me/genres")
def my_genres(request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """GET /users/me/genres. Sign-in is required.

    Returns:
        The saved genres, every catalog genre, and the maximum number that can be selected.

    Raises:
        HTTPException: 401 when signed out, 503 when the models or database are missing.
    """
    return {
        "genres": _saved_genres(session, user),
        "allowed": _service(request).catalog_genres(),
        "max": GENRE_SELECTION_MAX,
    }


@router.put("/users/me/genres")
def update_genres(
    body: GenreUpdateBody,
    request: Request,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
):
    """PUT /users/me/genres. Sign-in is required.

    Body: 1 to the configured maximum genre names. Duplicates are dropped.

    Returns:
        ``genres``: the names that were stored.

    Raises:
        HTTPException: 401 when signed out, 422 when a name is unknown, none remain,
        or the list is too long, 503 when the models or database are missing.

    Side effects:
        Replaces the account's genres and clears its recommendation snapshot.
    """
    catalog = set(_service(request).catalog_genres())
    chosen = []
    for genre in body.genres:
        name = genre.strip()
        if name not in catalog:
            raise HTTPException(status_code=422, detail=f"Unknown genre {name}.")
        if name not in chosen:
            chosen.append(name)
    if not chosen:
        raise HTTPException(status_code=422, detail="Pick at least one genre.")
    if len(chosen) > GENRE_SELECTION_MAX:
        raise HTTPException(status_code=422, detail=f"Choose at most {GENRE_SELECTION_MAX} genres.")
    _replace_genres(session, user, chosen)
    session.add(user)
    return {"genres": chosen}


@router.get("/ratings/me")
def my_ratings(request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """GET /ratings/me. Sign-in is required.

    Returns:
        ``ratings``: movie cards, newest update first, each with ``rating`` and ``user_rating``.

    Raises:
        HTTPException: 401 when signed out, 503 when the models or database are missing.
    """
    service = _service(request)
    rows = session.scalars(select(Rating).where(Rating.user_id == user.id).order_by(Rating.updated_at.desc())).all()
    payload = []
    for row in rows:
        card = service.describe(row.movie_id)
        card["rating"] = row.rating
        card["user_rating"] = float(row.rating)
        payload.append(card)
    return {"ratings": payload}


@router.delete("/ratings/{movie_id}")
def delete_rating(movie_id: int, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """DELETE /ratings/{movie_id}. Sign-in is required.

    Args:
        movie_id: MovieLens id of the rating to remove.

    Returns:
        ``{"ok": true}``.

    Raises:
        HTTPException: 401 when signed out, 404 when this account has no such rating,
        503 when the database is missing.

    Side effects:
        Deletes the rating and stores a rating interaction whose detail is ``delete``.
    """
    row = session.scalar(select(Rating).where(Rating.user_id == user.id, Rating.movie_id == movie_id))
    if row is None:
        raise HTTPException(status_code=404, detail="No rating for that movie.")
    session.delete(row)
    session.add(
        Interaction(
            user_id=user.id,
            movie_id=movie_id,
            event_type="rating",
            detail="delete",
            created_at=datetime.now(timezone.utc),
        )
    )
    return {"ok": True}


@router.post("/watchlist/{movie_id}")
def add_watch(movie_id: int, request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """POST /watchlist/{movie_id}. Sign-in is required.

    Args:
        movie_id: MovieLens id to save.

    Returns:
        ``{"movie_id": movie_id}``.

    Raises:
        HTTPException: 401 when signed out, 404 when the movie is unknown, 409 when it
        is already saved, 503 when the models or database are missing.

    Side effects:
        Inserts the watchlist row and a watchlist_add interaction.
    """
    service = _service(request)
    _require_movie(service, movie_id)
    existing = session.scalar(
        select(WatchlistItem).where(WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="That movie is already on your watchlist.")
    now = datetime.now(timezone.utc)
    session.add(WatchlistItem(user_id=user.id, movie_id=movie_id, created_at=now))
    session.add(Interaction(user_id=user.id, movie_id=movie_id, event_type="watchlist_add", created_at=now))
    return {"movie_id": movie_id}


@router.delete("/watchlist/{movie_id}")
def remove_watch(movie_id: int, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """DELETE /watchlist/{movie_id}. Sign-in is required.

    Args:
        movie_id: MovieLens id to remove.

    Returns:
        ``{"ok": true}``.

    Raises:
        HTTPException: 401 when signed out, 404 when that movie is not on the watchlist,
        503 when the database is missing.

    Side effects:
        Deletes the watchlist row. No interaction is written.
    """
    row = session.scalar(select(WatchlistItem).where(WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id))
    if row is None:
        raise HTTPException(status_code=404, detail="That movie is not on your watchlist.")
    session.delete(row)
    return {"ok": True}


@router.get("/watchlist/me")
def my_watchlist(request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """GET /watchlist/me. Sign-in is required.

    Returns:
        ``watchlist``: movie cards, newest save first, each with ``user_rating``.

    Raises:
        HTTPException: 401 when signed out, 503 when the models or database are missing.
    """
    service = _service(request)
    rows = session.scalars(
        select(WatchlistItem).where(WatchlistItem.user_id == user.id).order_by(WatchlistItem.created_at.desc())
    ).all()
    cards = [service.describe(row.movie_id) for row in rows]
    return {"watchlist": apply_user_ratings(cards, rating_map(session, user.id))}


@router.post("/onboarding")
def onboard(body: OnboardingBody, request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """POST /onboarding. Sign-in is required.

    Body: at least five half-star ratings. Each movie may appear only once.

    Returns:
        ``saved``: how many ratings were written.

    Raises:
        HTTPException: 401 when signed out, 404 when a movie is unknown, 422 when a
        movie is repeated or a rating is invalid, 503 when the models or database are missing.

    Side effects:
        Upserts each rating, writes rating interactions, and marks onboarding complete.
    """
    service = _service(request)
    seen: set[int] = set()
    for item in body.ratings:
        if item.movie_id in seen:
            raise HTTPException(status_code=422, detail="Rate each movie only once.")
        seen.add(item.movie_id)
        _require_movie(service, item.movie_id)
        _save_rating(session, user, item.movie_id, item.rating)
    user.onboarding_completed = True
    session.add(user)
    return {"saved": len(body.ratings)}


@router.post("/interactions")
def log_interaction(
    body: InteractionBody,
    request: Request,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
):
    """POST /interactions. Sign-in is required.

    Body: movie id and an event type of impression, click, rating, watchlist_add, or search.

    Returns:
        ``{"ok": true}``.

    Raises:
        HTTPException: 401 when signed out, 404 when the movie is unknown, 422 when the
        event type is unknown, 503 when the models or database are missing.

    Side effects:
        Inserts one interaction row.
    """
    service = _service(request)
    _require_movie(service, body.movie_id)
    session.add(
        Interaction(
            user_id=user.id,
            movie_id=body.movie_id,
            event_type=body.event_type,
            created_at=datetime.now(timezone.utc),
        )
    )
    return {"ok": True}


@router.post("/search/events")
def record_search(
    body: SearchEventBody,
    request: Request,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
):
    """POST /search/events. Sign-in is required.

    Body: query, the first 10 matched movie ids, and an optional clicked movie.
    A normalised query under 2 characters is not stored. The same query within 10
    minutes updates that row; a click updates the latest row for that query.

    Returns:
        ``logged`` false when too short; otherwise the row id and ``deduplicated``.

    Raises:
        HTTPException: 401 when signed out, 404 for an unknown clicked movie, 503 when unconfigured.

    Side effects:
        Inserts or updates a search row and deletes the recommendation snapshot.
    """
    text = body.query.strip()[:200]
    normalised = normalize_query(text)
    if len(normalised) < SEARCH_MIN_CHARS:
        return {"logged": False}
    if body.clicked_movie_id is not None:
        _require_movie(_service(request), body.clicked_movie_id)
    now = datetime.now(timezone.utc)
    existing = session.scalar(
        select(SearchEvent)
        .where(SearchEvent.user_id == user.id, SearchEvent.normalized_query == normalised)
        .order_by(SearchEvent.created_at.desc())
    )
    within_window = existing is not None and _aware(existing.created_at) >= now - timedelta(minutes=SEARCH_DEDUP_MINUTES)
    if existing is not None and (within_window or body.clicked_movie_id is not None):
        if within_window:
            existing.created_at = now
            existing.query = text or existing.query
        if body.matched_movie_ids:
            existing.matched_movie_ids = json.dumps([int(movie_id) for movie_id in body.matched_movie_ids[:10]])
        if body.clicked_movie_id is not None:
            existing.clicked_movie_id = int(body.clicked_movie_id)
        _drop_recommendation_cache(session, user.id)
        session.flush()
        return {"logged": True, "id": existing.id, "deduplicated": True}
    row = SearchEvent(
        user_id=user.id,
        query=text,
        normalized_query=normalised[:200],
        matched_movie_ids=json.dumps([int(movie_id) for movie_id in body.matched_movie_ids[:10]]),
        clicked_movie_id=body.clicked_movie_id,
        created_at=now,
    )
    session.add(row)
    _drop_recommendation_cache(session, user.id)
    session.flush()
    return {"logged": True, "id": row.id, "deduplicated": False}


@router.get("/users/me/searches")
def list_searches(user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """GET /users/me/searches. Sign-in is required.

    Returns:
        ``searches``: id, query, normalised query, matched ids, clicked movie, and
        created time, newest first. The recommendation route does not return the query.

    Raises:
        HTTPException: 401 when signed out, 503 when the database is missing.
    """
    rows = session.scalars(
        select(SearchEvent).where(SearchEvent.user_id == user.id).order_by(SearchEvent.created_at.desc())
    ).all()
    return {
        "searches": [
            {
                "id": row.id,
                "query": row.query,
                "normalized_query": row.normalized_query,
                "matched_movie_ids": _matched_ids(row.matched_movie_ids),
                "clicked_movie_id": row.clicked_movie_id,
                "created_at": _aware(row.created_at).isoformat(),
            }
            for row in rows
        ]
    }


@router.delete("/users/me/searches/{search_id}")
def delete_search(search_id: int, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """DELETE /users/me/searches/{search_id}. Sign-in is required.

    Args:
        search_id: Search row owned by the signed-in account.

    Returns:
        ``deleted``: that id.

    Raises:
        HTTPException: 401 when signed out, 404 when the row is missing or belongs
        to someone else, 503 when the database is missing.

    Side effects:
        Deletes the search and the recommendation snapshot.
    """
    row = session.scalar(select(SearchEvent).where(SearchEvent.id == search_id, SearchEvent.user_id == user.id))
    if row is None:
        raise HTTPException(status_code=404, detail="Search not found.")
    session.delete(row)
    _drop_recommendation_cache(session, user.id)
    return {"deleted": search_id}


@router.delete("/users/me/searches")
def clear_searches(user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """DELETE /users/me/searches. Sign-in is required.

    Returns:
        ``{"cleared": true}``.

    Raises:
        HTTPException: 401 when signed out, 503 when the database is missing.

    Side effects:
        Deletes every search for the account and its recommendation snapshot.
    """
    session.execute(delete(SearchEvent).where(SearchEvent.user_id == user.id))
    _drop_recommendation_cache(session, user.id)
    return {"cleared": True}


def _matched_ids(raw: str) -> list[int]:
    """Parse a JSON list of movie ids stored on a search row.

    Args:
        raw: JSON text. Empty, invalid, or non-list text yields no ids.

    Returns:
        Integer movie ids in the stored order.
    """
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [int(movie_id) for movie_id in parsed]


def _aware(value: datetime) -> datetime:
    """Return a UTC datetime. A naive value is labeled UTC without shifting the clock.

    Args:
        value: A timestamp read from the database.

    Returns:
        An aware UTC datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@router.get("/recommendations/me")
def my_recommendations(
    request: Request,
    n: int = Query(default=10, ge=1, le=50),
    w_recent_override: float | None = Query(default=None),
    include_debug: bool = Query(default=False),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
):
    """GET /recommendations/me. Sign-in is required.

    Query: ``n`` from 1 to 50, default 10. Weight and debug fields apply only when
    ``RECS_DEBUG`` is on. No ratings and no genres returns an empty ask-genres list.
    The response never includes the search text.

    Returns:
        ``mode``, ``rating_count``, ``ratings_until_personalised``, and ``results``.

    Raises:
        HTTPException: 401 when signed out, 422 when ``n`` is out of range, 503 when unconfigured.

    Side effects:
        Replaces the snapshot and writes one impression per movie, except for ask-genres.
    """
    service = _service(request)
    rows = session.scalars(select(Rating).where(Rating.user_id == user.id)).all()
    ratings = [RatedMovie(int(row.movie_id), float(row.rating), row.updated_at) for row in rows]
    chosen = _saved_genres(session, user)
    if not ratings and not chosen:
        return {
            "mode": "ask_genres",
            "rating_count": 0,
            "ratings_until_personalised": PERSONALISED_AFTER,
            "results": [],
        }
    watchlist_ids = set(
        session.scalars(select(WatchlistItem.movie_id).where(WatchlistItem.user_id == user.id)).all()
    )
    activity = session.scalars(
        select(Interaction).where(
            Interaction.user_id == user.id,
            Interaction.event_type.in_(("click", "watchlist_add", "search")),
        )
    ).all()
    events = [
        ActivityEvent(row.event_type, int(row.movie_id), row.created_at, row.detail or "")
        for row in activity
    ]
    previous = session.scalars(
        select(RecommendationSnapshot)
        .where(RecommendationSnapshot.user_id == user.id)
        .order_by(RecommendationSnapshot.rank)
    ).all()
    debug = recommendations_debug_enabled()
    stored_searches = session.scalars(select(SearchEvent).where(SearchEvent.user_id == user.id)).all()
    searches = [
        SearchHit(
            row.query,
            _aware(row.created_at),
            int(row.clicked_movie_id) if row.clicked_movie_id is not None else None,
            _matched_ids(row.matched_movie_ids),
        )
        for row in stored_searches
    ]
    blended = blended_recommendations(
        service,
        genres=chosen,
        ratings=ratings,
        events=events,
        watchlist_ids={int(movie_id) for movie_id in watchlist_ids},
        previous_ids=[int(row.movie_id) for row in previous],
        n=n,
        w_recent_override=w_recent_override if debug else None,
        searches=searches,
    )
    mode = "genre" if not ratings else recommendation_mode(len(ratings))
    now = datetime.now(timezone.utc)
    session.execute(delete(RecommendationSnapshot).where(RecommendationSnapshot.user_id == user.id))
    session.flush()
    payload = []
    for rank, item in enumerate(blended.items, start=1):
        session.add(
            RecommendationSnapshot(user_id=user.id, movie_id=item.movie_id, rank=rank, created_at=now)
        )
        session.add(Interaction(user_id=user.id, movie_id=item.movie_id, event_type="impression", created_at=now))
        card = service._recommendation(item.movie_id, item.score, item.reason, item.source, item.reason).to_dict()
        card["user_rating"] = None
        if debug and include_debug:
            card["base_score"] = item.base_score
            card["recent_score"] = item.recent_score
            card["search_score"] = item.search_score
            card["w_base"] = item.w_base
            card["w_recent"] = item.w_recent
            card["w_search"] = item.w_search
            card["snapshot_bonus"] = item.snapshot_bonus
        payload.append(card)
    count = len(ratings)
    return {
        "mode": mode,
        "rating_count": count,
        "ratings_until_personalised": max(PERSONALISED_AFTER - count, 0),
        "results": payload,
    }


def _replace_genres(session: Session, user: User, chosen: list[str]) -> None:
    """Replace the account's genres and drop the recommendation snapshot.

    Args:
        session: Account session. The caller commits.
        user: Account whose ``favorite_genres`` JSON is also rewritten.
        chosen: Genre names already checked against the catalog.

    Side effects:
        Deletes the old genre and snapshot rows, then inserts the new genres.
    """
    session.execute(delete(UserGenre).where(UserGenre.user_id == user.id))
    session.execute(delete(RecommendationSnapshot).where(RecommendationSnapshot.user_id == user.id))
    session.flush()
    now = datetime.now(timezone.utc)
    for name in chosen:
        session.add(UserGenre(user_id=user.id, genre=name, created_at=now))
    user.favorite_genres = json.dumps(chosen)


def _saved_genres(session: Session, user: User) -> list[str]:
    """Return the account's genres, using the genre table before the old JSON column.

    Args:
        session: Account session.
        user: Account. A broken or non-list JSON value yields no names.

    Returns:
        Genre names. Table rows are ordered by time, then name.
    """
    rows = session.scalars(select(UserGenre).where(UserGenre.user_id == user.id).order_by(UserGenre.created_at, UserGenre.genre)).all()
    if rows:
        return [row.genre for row in rows]
    try:
        parsed = json.loads(user.favorite_genres or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(genre) for genre in parsed if str(genre).strip()]
