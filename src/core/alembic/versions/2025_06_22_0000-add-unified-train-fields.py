from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2025_06_22_0000'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # add columns to agent_trains
    with op.batch_alter_table('agent_trains') as batch_op:
        batch_op.add_column(sa.Column('extra_config', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('metrics', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('artifact_path', sa.String(length=255), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('agent_trains') as batch_op:
        batch_op.drop_column('artifact_path')
        batch_op.drop_column('metrics')
        batch_op.drop_column('extra_config')


