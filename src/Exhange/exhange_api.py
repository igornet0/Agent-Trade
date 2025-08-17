import asyncio
from typing import List, Literal
from sqlalchemy.exc import SQLAlchemyError
from core.database import (Coin, User, DataTimeseries, db_helper,
                           orm_get_coins, orm_get_coin_by_name,
                           paginate_coin_prices, orm_add_transaction,
                           orm_add_coin_portfolio, orm_remove_user_balance)

import logging

logger = logging.getLogger("Exhange")

class ExhangeApi:

    @classmethod
    async def get_coins_list(cls) -> List[Coin]:
        async with db_helper.get_session() as session:
            coins = await orm_get_coins(session)
            return coins
        
    @classmethod
    async def get_coin_by_name(cls, name: str) -> Coin:
        async with db_helper.get_session() as session:
            coin = await orm_get_coin_by_name(session, name)
            return coin
        
    @classmethod
    async def get_coin_timeseries(cls, coin: Coin, timeframe: str = "5m", last_timestamp: str = None,
                                  limit: int = 100) -> List[DataTimeseries]:
        async with db_helper.get_session() as session:
            data_coin = await paginate_coin_prices(session, coin, timeframe,
                                                   last_timestamp, limit)
            return data_coin
    
    @classmethod
    async def create_order(cls, user_id: int, coin_id: int, price_order: float, 
                           type_order: Literal["buy", "sell"], amount: float):
        async with db_helper.get_session() as session:
            try:
                user = await session.get(User, user_id, with_for_update=True)
                if not user:
                    raise ValueError("User not found")
                
                coin = await session.get(Coin, coin_id)
                if not coin:
                    raise ValueError("Coin not found")
                
                total_cost = amount * price_order
                if user.balance < total_cost:
                    raise ValueError("Not enough balance")
                
                user.balance -= total_cost

                # Do not commit inside helpers; commit once atomically here
                await orm_add_transaction(session, user_id, coin_id, type_order, amount, price_order, do_commit=False)
                await orm_add_coin_portfolio(session, user_id, coin_id, amount, do_commit=False)
                await session.commit()

            except (SQLAlchemyError, ValueError) as e:
                logger.error(f"Error creating order: {str(e)}")
                await session.rollback()

