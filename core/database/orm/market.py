from typing import Dict, Literal
from datetime import datetime
from sqlalchemy import select, update, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from core.database.models import (
    Coin, Timeseries, DataTimeseries,
)


async def orm_add_coin(session: AsyncSession, name: str, price_now: float = 0) -> Coin:
    query = select(Coin).where(Coin.name == name)
    result = await session.execute(query)
    if not result.scalars().first():
        session.add(Coin(name=name, price_now=price_now))
        await session.commit()
    return await orm_get_coin_by_name(session, name)


async def orm_get_coins(session: AsyncSession, parsed: bool = None) -> list[Coin]:
    query = select(Coin)
    if parsed:
        query = query.where(Coin.parsed == parsed)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_coin_by_id(session: AsyncSession, id: int, parsed: bool = None) -> Coin:
    query = select(Coin).where(Coin.id == id)
    if parsed:
        query = query.where(Coin.parsed == parsed)
    query = query.options(joinedload(Coin.timeseries))
    result = await session.execute(query)
    return result.scalar()


async def orm_get_coin_by_name(session: AsyncSession, name: str) -> Coin:
    query = select(Coin).where(Coin.name == name)
    result = await session.execute(query)
    return result.scalar()


async def orm_update_coin_price(session: AsyncSession, name: str, price_now: float):
    query = update(Coin).where(Coin.name == name).values(price_now=price_now)
    await session.execute(query)
    await session.commit()


async def orm_add_timeseries(session: AsyncSession, coin: Coin | str, timestamp: str, path_dataset: str):
    if isinstance(coin, str):
        coin = await orm_get_coin_by_name(session, coin)
    if not coin:
        raise ValueError(f"Coin {coin} not found")
    tm = await orm_get_timeseries_by_coin(session, coin, timestamp)
    if tm:
        return await orm_update_timeseries_path(session, tm.id, path_dataset)
    timeseries = Timeseries(coin_id=coin.id, timestamp=timestamp, path_dataset=path_dataset)
    session.add(timeseries)
    await session.commit()


async def orm_update_timeseries_path(session: AsyncSession, timeseries_id: int, path_dataset: str):
    query = update(Timeseries).where(Timeseries.id == timeseries_id).values(path_dataset=path_dataset)
    await session.execute(query)
    await session.commit()


async def orm_get_timeseries_by_path(session: AsyncSession, path_dataset: str):
    query = select(Timeseries).where(Timeseries.path_dataset == path_dataset)
    result = await session.execute(query)
    return result.scalars().first()


async def orm_get_timeseries_by_id(session: AsyncSession, id: int):
    query = select(Timeseries).where(Timeseries.id == id)
    result = await session.execute(query)
    return result.scalars().first()


async def orm_get_timeseries_by_coin(session: AsyncSession, coin: Coin | str | int, timeframe: str = None) -> list[Timeseries] | Timeseries:
    if isinstance(coin, str):
        coin = await orm_get_coin_by_name(session, coin)
    elif isinstance(coin, int):
        coin = await orm_get_coin_by_id(session, coin)
    if not coin:
        raise ValueError(f"Coin {coin} not found")
    query = select(Timeseries).options(joinedload(Timeseries.coin)).where(Timeseries.coin_id == coin.id)
    if timeframe:
        query = query.where(Timeseries.timestamp == timeframe)
    result = await session.execute(query)
    if timeframe:
        return result.scalars().first()
    return result.scalars().all()


async def orm_get_data_timeseries(session: AsyncSession, timeseries_id: int) -> list[DataTimeseries]:
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries_id)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_data_timeseries_by_datetime(session: AsyncSession, timeseries_id: int, datetime: str) -> DataTimeseries:
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries_id, DataTimeseries.datetime == datetime)
    result = await session.execute(query)
    return result.scalars().first()


async def paginate_coin_prices(
    session: AsyncSession,
    coin_id: int,
    timeframe: str = "5m",
    last_timestamp: datetime = None,
    limit: int = 100,
    sort: bool = False,
) -> list[DataTimeseries]:
    timeseries = await orm_get_timeseries_by_coin(session, coin_id, timeframe=timeframe)
    if not timeseries:
        raise ValueError(f"Timeseries - {timeframe} for coin - {coin_id} not found")
    query = select(DataTimeseries).where(DataTimeseries.timeseries_id == timeseries.id).order_by(desc(DataTimeseries.datetime))
    if last_timestamp is not None:
        query = query.where(DataTimeseries.datetime < last_timestamp)
    result = await session.execute(query.limit(limit))
    records = result.scalars().all()
    if sort:
        return sorted(records, key=lambda x: x.datetime)
    return records


async def orm_add_data_timeseries(session: AsyncSession, timeseries_id: int, data_timeseries: dict):
    dt = await orm_get_data_timeseries_by_datetime(session, timeseries_id, data_timeseries["datetime"])
    if dt:
        return False
    session.add(DataTimeseries(timeseries_id=timeseries_id, **data_timeseries))
    await session.commit()
    return True


