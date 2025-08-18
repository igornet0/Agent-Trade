from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.models import Transaction, User, Coin


async def orm_get_user_transactions(session: AsyncSession, user_id: int) -> List[Transaction]:
    """Получает все транзакции пользователя"""
    query = select(Transaction).where(Transaction.user_id == user_id)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_user_coin_transactions(session: AsyncSession, user_id: int, coin_id: int) -> List[Transaction]:
    """Получает все транзакции пользователя для конкретной монеты"""
    query = select(Transaction).where(
        Transaction.user_id == user_id,
        Transaction.coin_id == coin_id
    )
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_transactions_by_id(session: AsyncSession, transaction_id: int) -> Optional[Transaction]:
    """Получает транзакцию по ID"""
    query = select(Transaction).where(Transaction.id == transaction_id)
    result = await session.execute(query)
    return result.scalars().first()


async def orm_get_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int) -> Optional[dict]:
    """Получает портфолио пользователя для конкретной монеты"""
    from core.database.models import Portfolio
    
    query = select(Portfolio).where(
        Portfolio.user_id == user_id,
        Portfolio.coin_id == coin_id
    )
    result = await session.execute(query)
    portfolio = result.scalars().first()
    
    if portfolio:
        return {
            "user_id": portfolio.user_id,
            "coin_id": portfolio.coin_id,
            "amount": portfolio.amount,
            "price_avg": portfolio.price_avg
        }
    return None
