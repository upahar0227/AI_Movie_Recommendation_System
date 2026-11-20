"""Alembic environment. The database URL comes from DATABASE_URL when set."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from src.app import models  # noqa: F401
from src.app.database import Base, make_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit migration SQL without opening a database connection.

    The URL comes from ``make_engine``, including a password when one is set.
    Parameter values are rendered literally.

    Side effects:
        Writes the migration SQL through Alembic's offline context.
    """
    context.configure(url=make_engine().url.render_as_string(hide_password=False), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations through a live connection from ``make_engine``.

    Side effects:
        Opens the account database and runs the pending Alembic revisions inside a transaction.
    """
    connectable = make_engine()
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
