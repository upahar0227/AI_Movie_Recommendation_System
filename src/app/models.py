"""Account tables. These ids are not MovieLens user ids."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base

RATING_VALUES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
EVENT_TYPES = ("impression", "click", "rating", "watchlist_add", "search")


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        The clock time used as the default for account timestamps.
    """
    return datetime.now(timezone.utc)


class User(Base):
    """A person who registered through the app."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    username: Mapped[str] = mapped_column(String(32), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    favorite_genres: Mapped[str] = mapped_column(Text, default="[]", server_default="[]")
    onboarding_completed: Mapped[bool] = mapped_column(default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    is_active: Mapped[bool] = mapped_column(default=True)


class Rating(Base):
    """One app user's score for a MovieLens movie id."""

    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),
        CheckConstraint(
            "rating IN (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)",
            name="ck_ratings_half_star",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    movie_id: Mapped[int] = mapped_column(Integer)
    rating: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class WatchlistItem(Base):
    """A movie an app user saved to watch later."""

    __tablename__ = "watchlist"
    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_watchlist_user_movie"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    movie_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RevokedToken(Base):
    """A JWT id that must no longer be accepted."""

    __tablename__ = "revoked_tokens"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class UserGenre(Base):
    """A genre the user selected. This is the source of truth after onboarding."""

    __tablename__ = "user_genres"
    __table_args__ = (UniqueConstraint("user_id", "genre", name="uq_user_genres_user_genre"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    genre: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RecommendationSnapshot(Base):
    """The last recommendation list served to a user, used to limit churn."""

    __tablename__ = "recommendation_snapshots"
    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_snapshots_user_movie"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    movie_id: Mapped[int] = mapped_column(Integer)
    rank: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SearchEvent(Base):
    """A submitted search. Keystrokes are not stored."""

    __tablename__ = "search_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str] = mapped_column(String(200))
    matched_movie_ids: Mapped[str] = mapped_column(Text, default="[]")
    clicked_movie_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Interaction(Base):
    """A single impression, click, rating, watchlist, or search event."""

    __tablename__ = "interactions"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('impression', 'click', 'rating', 'watchlist_add', 'search')",
            name="ck_interactions_event_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    movie_id: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
