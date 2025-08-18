"""
add artifacts table

Revision ID: add_artifacts_20250622_2000
Revises: add_news_background_20250622_1900
Create Date: 2025-06-22 20:00:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_artifacts_20250622_2000'
down_revision = 'add_news_background_20250622_1900'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'artifacts',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('agent_id', sa.Integer(), sa.ForeignKey('agents.id'), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('path', sa.String(length=500), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),  # model, config, scaler, metadata
        sa.Column('size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('checksum', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    )
    
    # Create unique constraint
    op.create_unique_constraint('uq_artifact_agent_version_type', 'artifacts', ['agent_id', 'version', 'type'])
    
    # Create indexes for efficient querying
    op.create_index('ix_artifacts_agent_id', 'artifacts', ['agent_id'])
    op.create_index('ix_artifacts_type', 'artifacts', ['type'])
    op.create_index('ix_artifacts_created_at', 'artifacts', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_artifacts_created_at', 'artifacts')
    op.drop_index('ix_artifacts_type', 'artifacts')
    op.drop_index('ix_artifacts_agent_id', 'artifacts')
    op.drop_constraint('uq_artifact_agent_version_type', 'artifacts', type_='unique')
    op.drop_table('artifacts')
