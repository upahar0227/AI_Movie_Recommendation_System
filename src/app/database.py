"""SQLite engine for app accounts. MovieLens artifacts are not stored here."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = ROOT / "data" / "app.db"


class Base(DeclarativeBase):
    """Shared metadata for Alembic and the ORM."""


def database_url() -> str:
    """Return ``DATABASE_URL`` or a SQLite file under ``data/``."""
    configured = os.environ.get("DATABASE_URL", "").strip()
    if configured:
        return configured
    return "sqlite:///" + DEFAULT_DATABASE_PATH.resolve().as_posix()


def make_engine(url: str | None = None) -> Engine:
    """Create an engine and turn on SQLite foreign keys."""
    engine = create_engine(url or database_url(), future=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine: Engine | None = None) -> sessionmaker:
    return sessionmaker(bind=engine or make_engine(), expire_on_commit=False)


def ping(engine: Engine) -> None:
    """Open a connection so callers can fail early on a bad URL."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
