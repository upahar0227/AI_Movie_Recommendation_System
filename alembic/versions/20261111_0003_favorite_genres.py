"""Store the genres a new account picks before rating anything.

Revision ID: 20261111_0003
Revises: 20261111_0002
Create Date: 2026-11-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261111_0003"
down_revision: Union[str, Sequence[str], None] = "20261111_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("favorite_genres", sa.Text(), nullable=False, server_default="[]"))


def downgrade() -> None:
    op.drop_column("users", "favorite_genres")
