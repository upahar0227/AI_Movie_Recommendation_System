"""Record when an account finishes genre or rating onboarding.

Revision ID: 20261111_0005
Revises: 20261111_0004
Create Date: 2026-11-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261111_0005"
down_revision: Union[str, Sequence[str], None] = "20261111_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("onboarding_completed", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.execute(
        "UPDATE users SET onboarding_completed = 1 "
        "WHERE favorite_genres IS NOT NULL AND favorite_genres NOT IN ('', '[]')"
    )
    op.execute("UPDATE users SET onboarding_completed = 1 WHERE id IN (SELECT user_id FROM ratings)")


def downgrade() -> None:
    op.drop_column("users", "onboarding_completed")
