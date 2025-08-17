from typing import List, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from core.database.models import Strategy, StrategyCoin, StrategyAgent, Coin, Agent


async def orm_list_strategies_for_user(session: AsyncSession, user_id: int) -> List[Strategy]:
    query = select(Strategy).where(Strategy.user_id == user_id)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_strategy_detail(session: AsyncSession, strategy_id: int, user_id: int) -> Strategy | None:
    query = select(Strategy).where(Strategy.id == strategy_id, Strategy.user_id == user_id)
    result = await session.execute(query)
    return result.scalars().first()


async def orm_delete_strategy(session: AsyncSession, strategy_id: int, user_id: int) -> bool:
    st = await session.get(Strategy, strategy_id)
    if not st or st.user_id != user_id:
        return False
    await session.execute(delete(StrategyCoin).where(StrategyCoin.strategy_id == strategy_id))
    await session.execute(delete(StrategyAgent).where(StrategyAgent.strategy_id == strategy_id))
    await session.delete(st)
    await session.commit()
    return True


# ---- Validators ----

async def _fetch_existing_ids(session: AsyncSession, model, ids: Sequence[int]) -> set[int]:
    if not ids:
        return set()
    query = select(model.id).where(model.id.in_(ids))
    rows = (await session.execute(query)).scalars().all()
    return set(rows)


async def validate_coins_exist(session: AsyncSession, coin_ids: Sequence[int]) -> list[int]:
    existing = await _fetch_existing_ids(session, Coin, coin_ids)
    return [cid for cid in coin_ids if cid not in existing]


async def validate_agents_exist(session: AsyncSession, agent_ids: Sequence[int]) -> list[int]:
    existing = await _fetch_existing_ids(session, Agent, agent_ids)
    return [aid for aid in agent_ids if aid not in existing]


async def is_strategy_name_taken(session: AsyncSession, user_id: int, name: str) -> bool:
    q = select(Strategy.id).where(Strategy.user_id == user_id, Strategy.name == name)
    return (await session.execute(q)).first() is not None


