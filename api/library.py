"""Ratings, watchlist, and recommendations for the signed-in app user."""

from __future__ import annotations

import json
from datetime import datetime, timezone

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
    User,
    UserGenre,
    WatchlistItem,
)
from src.app.security import PERSONALISED_AFTER, recommendation_mode
from src.app.settings import recommendations_debug_enabled
from src.recsys.config import GENRE_SELECTION_MAX
from src.recsys.personalised import ActivityEvent, RatedMovie, blended_recommendations
from src.recsys.serving import RecommenderService

router = APIRouter(tags=["library"])


class RatingBody(BaseModel):
    movie_id: int
    rating: float

    @field_validator("rating")
    @classmethod
    def half_star(cls, value: float) -> float:
        if value not in RATING_VALUES:
            raise ValueError("Rating must be from 0.5 to 5.0 in steps of 0.5.")
        return value


class GenreBody(BaseModel):
    genres: list[str] = Field(min_length=1)


class GenreUpdateBody(BaseModel):
    genres: list[str] = Field(min_length=1, max_length=GENRE_SELECTION_MAX)


class OnboardingBody(BaseModel):
    ratings: list[RatingBody] = Field(min_length=5)


class InteractionBody(BaseModel):
    movie_id: int
    event_type: str

    @field_validator("event_type")
    @classmethod
    def known_event(cls, value: str) -> str:
        if value not in EVENT_TYPES:
            raise ValueError("Unknown event type.")
        return value


def _service(request: Request) -> RecommenderService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Train them first.")
    return service


def _require_movie(service: RecommenderService, movie_id: int) -> None:
    catalog = {int(value) for value in service.movies["movieId"]}
    if int(movie_id) not in catalog:
        raise HTTPException(status_code=404, detail=f"Unknown movie id {movie_id}")


def _save_rating(session: Session, user: User, movie_id: int, rating: float) -> None:
    now = datetime.now(timezone.utc)
    existing = session.scalar(select(Rating).where(Rating.user_id == user.id, Rating.movie_id == movie_id))
    if existing is None:
        session.add(Rating(user_id=user.id, movie_id=movie_id, rating=rating, created_at=now, updated_at=now))
    else:
        existing.rating = rating
        existing.updated_at = now
    session.add(Interaction(user_id=user.id, movie_id=movie_id, event_type="rating", created_at=now))


def rating_map(session: Session, user_id: int) -> dict[int, float]:
    rows = session.scalars(select(Rating).where(Rating.user_id == user_id)).all()
    return {int(row.movie_id): float(row.rating) for row in rows}


def apply_user_ratings(cards: list[dict], scores: dict[int, float]) -> list[dict]:
    for card in cards:
        card["user_rating"] = scores.get(int(card["movie_id"]))
    return cards


def recent_clicks(session: Session, user_id: int, limit: int = 20) -> list[int]:
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


def log_search(session: Session, user: User, query: str) -> None:
    text = query.strip()[:200]
    if not text:
        return
    session.add(
        Interaction(
            user_id=user.id,
            movie_id=0,
            event_type="search",
            detail=text,
            created_at=datetime.now(timezone.utc),
        )
    )


@router.post("/ratings")
def rate(body: RatingBody, request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    service = _service(request)
    _require_movie(service, body.movie_id)
    _save_rating(session, user, body.movie_id, body.rating)
    return {"movie_id": body.movie_id, "rating": body.rating}


@router.get("/genres")
def list_genres(request: Request, user: User = Depends(get_current_user)):
    del user
    return {"genres": _service(request).catalog_genres()}


@router.post("/genres")
def save_genres(body: GenreBody, request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
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
    row = session.scalar(select(WatchlistItem).where(WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id))
    if row is None:
        raise HTTPException(status_code=404, detail="That movie is not on your watchlist.")
    session.delete(row)
    return {"ok": True}


@router.get("/watchlist/me")
def my_watchlist(request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    service = _service(request)
    rows = session.scalars(
        select(WatchlistItem).where(WatchlistItem.user_id == user.id).order_by(WatchlistItem.created_at.desc())
    ).all()
    cards = [service.describe(row.movie_id) for row in rows]
    return {"watchlist": apply_user_ratings(cards, rating_map(session, user.id))}


@router.post("/onboarding")
def onboard(body: OnboardingBody, request: Request, user: User = Depends(get_current_user), session: Session = Depends(get_db)):
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


@router.get("/recommendations/me")
def my_recommendations(
    request: Request,
    n: int = Query(default=10, ge=1, le=50),
    w_recent_override: float | None = Query(default=None),
    include_debug: bool = Query(default=False),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
):
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
    blended = blended_recommendations(
        service,
        genres=chosen,
        ratings=ratings,
        events=events,
        watchlist_ids={int(movie_id) for movie_id in watchlist_ids},
        previous_ids=[int(row.movie_id) for row in previous],
        n=n,
        w_recent_override=w_recent_override if debug else None,
    )
    mode = "genre" if not ratings else recommendation_mode(len(ratings))
    now = datetime.now(timezone.utc)
    session.execute(delete(RecommendationSnapshot).where(RecommendationSnapshot.user_id == user.id))
    session.flush()
    payload = []
    for rank, item in enumerate(blended, start=1):
        session.add(
            RecommendationSnapshot(user_id=user.id, movie_id=item.movie_id, rank=rank, created_at=now)
        )
        session.add(Interaction(user_id=user.id, movie_id=item.movie_id, event_type="impression", created_at=now))
        card = service._recommendation(item.movie_id, item.score, item.reason, item.source, item.reason).to_dict()
        card["user_rating"] = None
        if debug and include_debug:
            card["base_score"] = item.base_score
            card["recent_score"] = item.recent_score
            card["w_recent"] = item.w_recent
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
    session.execute(delete(UserGenre).where(UserGenre.user_id == user.id))
    session.execute(delete(RecommendationSnapshot).where(RecommendationSnapshot.user_id == user.id))
    session.flush()
    now = datetime.now(timezone.utc)
    for name in chosen:
        session.add(UserGenre(user_id=user.id, genre=name, created_at=now))
    user.favorite_genres = json.dumps(chosen)


def _saved_genres(session: Session, user: User) -> list[str]:
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
