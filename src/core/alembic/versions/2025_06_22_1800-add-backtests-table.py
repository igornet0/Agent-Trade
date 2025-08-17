"""
add backtests table

Revision ID: add_backtests_20250622_1800
Revises: 
Create Date: 2025-06-22 18:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_backtests_20250622_1800'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'backtests',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('pipeline_id', sa.Integer(), sa.ForeignKey('pipelines.id'), nullable=True),
        sa.Column('timeframe', sa.String(length=16), nullable=True),
        sa.Column('start', sa.DateTime(), nullable=True),
        sa.Column('end', sa.DateTime(), nullable=True),
        sa.Column('config_json', sa.JSON(), nullable=False),
        sa.Column('metrics_json', sa.JSON(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('artifacts', sa.JSON(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('backtests')


