from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2025_06_22_1200-add-pipeline-table'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'pipelines',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('config_json', sa.JSON(), nullable=False),
        sa.Column('created', sa.DateTime(), nullable=True),
        sa.Column('updated', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('pipelines')


