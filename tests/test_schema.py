"""The account migration creates the expected tables and constraints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from src.app.database import make_engine, make_session_factory
from src.app.models import Interaction, Rating, User, WatchlistItem


def _upgrade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    database = tmp_path / "app.db"
    url = "sqlite:///" + database.resolve().as_posix()
    monkeypatch.setenv("DATABASE_URL", url)
    config = Config("alembic.ini")
    command.upgrade(config, "head")
    return url


def test_migration_creates_account_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = _upgrade(tmp_path, monkeypatch)
    engine = make_engine(url)
    names = set(inspect(engine).get_table_names())
    assert names == {
        "users",
        "ratings",
        "watchlist",
        "interactions",
        "revoked_tokens",
        "user_genres",
        "recommendation_snapshots",
        "search_events",
        "alembic_version",
    }

    user_columns = {column["name"] for column in inspect(engine).get_columns("users")}
    assert user_columns == {
        "id",
        "email",
        "username",
        "password_hash",
        "favorite_genres",
        "onboarding_completed",
        "created_at",
        "is_active",
    }
    rating_uniques = {tuple(item["column_names"]) for item in inspect(engine).get_unique_constraints("ratings")}
    assert ("user_id", "movie_id") in rating_uniques
    watch_uniques = {tuple(item["column_names"]) for item in inspect(engine).get_unique_constraints("watchlist")}
    assert ("user_id", "movie_id") in watch_uniques


def test_constraints_reject_bad_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = _upgrade(tmp_path, monkeypatch)
    session = make_session_factory(make_engine(url))()
    now = datetime.now(timezone.utc)
    user = User(email="reader@example.com", username="reader", password_hash="not-a-plaintext-password", created_at=now)
    session.add(user)
    session.commit()

    session.add(Rating(user_id=user.id, movie_id=1, rating=4.0, created_at=now, updated_at=now))
    session.commit()
    session.add(Rating(user_id=user.id, movie_id=1, rating=5.0, created_at=now, updated_at=now))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(Rating(user_id=user.id, movie_id=2, rating=4.2, created_at=now, updated_at=now))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(WatchlistItem(user_id=user.id, movie_id=3, created_at=now))
    session.commit()
    session.add(WatchlistItem(user_id=user.id, movie_id=3, created_at=now))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(Interaction(user_id=user.id, movie_id=3, event_type="bookmark", created_at=now))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    with session.bind.connect() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))
        connection.execute(text("DELETE FROM users WHERE id = :user_id"), {"user_id": user.id})
        connection.commit()
        remaining = connection.execute(text("SELECT COUNT(*) FROM watchlist")).scalar_one()
    assert remaining == 0
    session.close()
