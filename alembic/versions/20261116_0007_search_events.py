"""Submitted searches, separate from keystroke traffic.

Revision ID: 20261116_0007
Revises: 20261113_0006
Create Date: 2026-11-16
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261116_0007"
down_revision: Union[str, Sequence[str], None] = "20261113_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "search_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.String(length=200), nullable=False),
        sa.Column("matched_movie_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("clicked_movie_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_search_events_user_id", "search_events", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_search_events_user_id", table_name="search_events")
    op.drop_table("search_events")
