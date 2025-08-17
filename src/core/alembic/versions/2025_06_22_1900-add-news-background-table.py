"""
add news_background table

Revision ID: add_news_background_20250622_1900
Revises: add_backtests_20250622_1800
Create Date: 2025-06-22 19:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_news_background_20250622_1900'
down_revision = 'add_backtests_20250622_1800'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'news_background',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('coin_id', sa.Integer(), sa.ForeignKey('coins.id'), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('source_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('sources_breakdown', sa.JSON(), nullable=True),
        sa.Column('window_hours', sa.Integer(), nullable=False, server_default='24'),
        sa.Column('decay_factor', sa.Float(), nullable=False, server_default='0.95'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    )
    
    # Create indexes for efficient querying
    op.create_index('ix_news_background_coin_timestamp', 'news_background', ['coin_id', 'timestamp'])
    op.create_index('ix_news_background_timestamp', 'news_background', ['timestamp'])


def downgrade() -> None:
    op.drop_index('ix_news_background_timestamp', 'news_background')
    op.drop_index('ix_news_background_coin_timestamp', 'news_background')
    op.drop_table('news_background')
