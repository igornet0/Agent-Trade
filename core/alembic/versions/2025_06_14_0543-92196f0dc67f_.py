"""empty message

Revision ID: 92196f0dc67f
Revises: c7b217b61961
Create Date: 2025-06-14 05:43:14.072591

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '92196f0dc67f'
down_revision: Union[str, None] = 'c7b217b61961'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column('agent_trains', sa.Column('epoch_now', sa.Integer(), nullable=False))
    op.drop_constraint(op.f('fk_agent_trains_user_id_users'), 'agent_trains', type_='foreignkey')
    op.drop_column('agent_trains', 'user_id')
    op.add_column('agents', sa.Column('timeframe', sa.String(length=50), nullable=True))
    op.execute("UPDATE agents SET timeframe = '5m'")

    op.alter_column('agents', 'status',
               existing_type=sa.VARCHAR(length=20),
               nullable=False)



def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('agents', 'status',
               existing_type=sa.VARCHAR(length=20),
               nullable=True)
    op.drop_column('agents', 'timeframe')
    op.add_column('agent_trains', sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=False))
    op.create_foreign_key(op.f('fk_agent_trains_user_id_users'), 'agent_trains', 'users', ['user_id'], ['id'])
    op.drop_column('agent_trains', 'epoch_now')
    # ### end Alembic commands ###
