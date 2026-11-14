"""Genre rows and the last served recommendation list.

Revision ID: 20261113_0006
Revises: 20261111_0005
Create Date: 2026-11-13
"""

import json
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261113_0006"
down_revision: Union[str, Sequence[str], None] = "20261111_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_genres",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("genre", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "genre", name="uq_user_genres_user_genre"),
    )
    op.create_index("ix_user_genres_user_id", "user_genres", ["user_id"])
    op.create_table(
        "recommendation_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "movie_id", name="uq_snapshots_user_movie"),
    )
    op.create_index("ix_recommendation_snapshots_user_id", "recommendation_snapshots", ["user_id"])

    connection = op.get_bind()
    rows = connection.execute(sa.text("SELECT id, favorite_genres FROM users")).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    for user_id, raw in rows:
        try:
            chosen = json.loads(raw or "[]")
        except json.JSONDecodeError:
            continue
        if not isinstance(chosen, list):
            continue
        seen: set[str] = set()
        for genre in chosen:
            name = str(genre).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            connection.execute(
                sa.text(
                    "INSERT INTO user_genres (user_id, genre, created_at) VALUES (:user_id, :genre, :created_at)"
                ),
                {"user_id": user_id, "genre": name, "created_at": now},
            )


def downgrade() -> None:
    op.drop_index("ix_recommendation_snapshots_user_id", table_name="recommendation_snapshots")
    op.drop_table("recommendation_snapshots")
    op.drop_index("ix_user_genres_user_id", table_name="user_genres")
    op.drop_table("user_genres")
