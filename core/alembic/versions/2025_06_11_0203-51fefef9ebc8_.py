"""empty message

Revision ID: 51fefef9ebc8
Revises: 4dd619a528fa
Create Date: 2025-06-11 02:03:04.538466

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '51fefef9ebc8'
down_revision: Union[str, None] = '4dd619a528fa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('portfolios', sa.Column('price_avg', sa.Float(), nullable=False))
    op.drop_constraint(op.f('fk_portfolios_user_id_users'), 'portfolios', type_='foreignkey')
    op.drop_constraint(op.f('fk_portfolios_coin_id_coins'), 'portfolios', type_='foreignkey')
    op.create_foreign_key(op.f('fk_portfolios_user_id_users'), 'portfolios', 'users', ['user_id'], ['id'])
    op.create_foreign_key(op.f('fk_portfolios_coin_id_coins'), 'portfolios', 'coins', ['coin_id'], ['id'])
    op.drop_constraint(op.f('fk_statistic_agents_id_agnet_agents'), 'statistic_agents', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_agents_id_agnet_agents'), 'statistic_agents', 'agents', ['id_agnet'], ['id'])
    op.drop_constraint(op.f('fk_statistic_models_id_model_m_l__models'), 'statistic_models', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_models_id_model_m_l__models'), 'statistic_models', 'm_l__models', ['id_model'], ['id'])
    op.drop_constraint(op.f('fk_transactions_coin_id_coins'), 'transactions', type_='foreignkey')
    op.drop_constraint(op.f('fk_transactions_user_id_users'), 'transactions', type_='foreignkey')
    op.create_foreign_key(op.f('fk_transactions_user_id_users'), 'transactions', 'users', ['user_id'], ['id'])
    op.create_foreign_key(op.f('fk_transactions_coin_id_coins'), 'transactions', 'coins', ['coin_id'], ['id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(op.f('fk_transactions_coin_id_coins'), 'transactions', type_='foreignkey')
    op.drop_constraint(op.f('fk_transactions_user_id_users'), 'transactions', type_='foreignkey')
    op.create_foreign_key(op.f('fk_transactions_user_id_users'), 'transactions', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key(op.f('fk_transactions_coin_id_coins'), 'transactions', 'coins', ['coin_id'], ['id'], ondelete='CASCADE')
    op.drop_constraint(op.f('fk_statistic_models_id_model_m_l__models'), 'statistic_models', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_models_id_model_m_l__models'), 'statistic_models', 'm_l__models', ['id_model'], ['id'], ondelete='CASCADE')
    op.drop_constraint(op.f('fk_statistic_agents_id_agnet_agents'), 'statistic_agents', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_agents_id_agnet_agents'), 'statistic_agents', 'agents', ['id_agnet'], ['id'], ondelete='CASCADE')
    op.drop_constraint(op.f('fk_portfolios_coin_id_coins'), 'portfolios', type_='foreignkey')
    op.drop_constraint(op.f('fk_portfolios_user_id_users'), 'portfolios', type_='foreignkey')
    op.create_foreign_key(op.f('fk_portfolios_coin_id_coins'), 'portfolios', 'coins', ['coin_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key(op.f('fk_portfolios_user_id_users'), 'portfolios', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.drop_column('portfolios', 'price_avg')
    # ### end Alembic commands ###
