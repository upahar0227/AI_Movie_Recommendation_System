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
    """Return ``DATABASE_URL`` or a SQLite file under ``data/``.

    Returns:
        The configured URL, or ``sqlite:///`` plus the absolute path of ``data/app.db``.

    Side effects:
        Reads the ``DATABASE_URL`` environment variable.
    """
    configured = os.environ.get("DATABASE_URL", "").strip()
    if configured:
        return configured
    return "sqlite:///" + DEFAULT_DATABASE_PATH.resolve().as_posix()


def make_engine(url: str | None = None) -> Engine:
    """Create an engine and turn on SQLite foreign keys.

    Args:
        url: Database URL. None uses ``database_url``.

    Returns:
        A SQLAlchemy engine. SQLite connections run ``PRAGMA foreign_keys=ON``.

    Side effects:
        Registers that pragma on each new SQLite connection.
    """
    engine = create_engine(url or database_url(), future=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
            """Enable SQLite foreign-key checks for one new connection.

            Args:
                dbapi_connection: The sqlite3 connection SQLAlchemy just opened.
                _connection_record: Unused SQLAlchemy connection record.

            Side effects:
                Executes ``PRAGMA foreign_keys=ON`` on that connection.
            """
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine: Engine | None = None) -> sessionmaker:
    """Build a session factory that keeps objects usable after commit.

    Args:
        engine: Engine to bind. None creates one from the default URL.

    Returns:
        A ``sessionmaker`` with ``expire_on_commit`` turned off.
    """
    return sessionmaker(bind=engine or make_engine(), expire_on_commit=False)


def ping(engine: Engine) -> None:
    """Open a connection so callers can fail early on a bad URL.

    Args:
        engine: Engine to test.

    Side effects:
        Connects and runs ``SELECT 1``. A bad URL raises the database driver's error.
    """
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
