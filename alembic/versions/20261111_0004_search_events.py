"""Let interaction rows record a search query.

Revision ID: 20261111_0004
Revises: 20261111_0003
Create Date: 2026-11-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261111_0004"
down_revision: Union[str, Sequence[str], None] = "20261111_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EVENT_CHECK = "event_type IN ('impression', 'click', 'rating', 'watchlist_add', 'search')"


def upgrade() -> None:
    with op.batch_alter_table("interactions", recreate="always") as batch:
        batch.add_column(sa.Column("detail", sa.Text(), nullable=True))
        batch.drop_constraint("ck_interactions_event_type", type_="check")
        batch.create_check_constraint("ck_interactions_event_type", EVENT_CHECK)


def downgrade() -> None:
    with op.batch_alter_table("interactions", recreate="always") as batch:
        batch.drop_constraint("ck_interactions_event_type", type_="check")
        batch.create_check_constraint(
            "ck_interactions_event_type",
            "event_type IN ('impression', 'click', 'rating', 'watchlist_add')",
        )
        batch.drop_column("detail")
