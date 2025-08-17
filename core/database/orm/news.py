from datetime import datetime
from enum import Enum
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from core.database.models import News, NewsUrl, TelegramChannel, NewsCoin


class NewsType(Enum):
    telegram = "telegram"
    url = "url"
    rss = "RSS"
    api = "API"
    twitter = "TWITTER"


class NewsData(BaseModel):
    id_url: int
    title: str
    text: str
    type: NewsType
    date: datetime


async def orm_add_telegram_chanel(session: AsyncSession, name: str, chat_id: str, parsed: bool = True) -> TelegramChannel:
    channel = await session.execute(select(TelegramChannel).where(TelegramChannel.name == name))
    if channel.scalars().first():
        raise ValueError(f"Channel {name} already exists")
    session.add(TelegramChannel(name=name, chat_id=chat_id, parsed=parsed))
    await session.commit()


async def orm_add_news_url(session: AsyncSession, url: str, a_pup: float = 0.9, parsed: bool = True) -> NewsUrl:
    exists = await session.execute(select(NewsUrl).where(NewsUrl.url == url))
    if exists.scalars().first():
        raise ValueError(f"Url {url} already exists")
    news_url = NewsUrl(url=url, a_pup=a_pup, parsed=parsed)
    session.add(news_url)
    await session.commit()
    await session.refresh(news_url)
    return news_url


async def orm_get_news_url(session: AsyncSession, url: str) -> NewsUrl:
    result = await session.execute(select(NewsUrl).where(NewsUrl.url == url))
    return result.scalars().first()


async def orm_get_news_url_by_id(session: AsyncSession, id: int) -> NewsUrl:
    result = await session.execute(select(NewsUrl).where(NewsUrl.id == id))
    return result.scalars().first()


async def orm_get_telegram_channel(session: AsyncSession, name: str) -> TelegramChannel:
    result = await session.execute(select(TelegramChannel).where(TelegramChannel.name == name))
    return result.scalars().first()


async def orm_get_telegram_channel_by_id(session: AsyncSession, id: int) -> TelegramChannel:
    result = await session.execute(select(TelegramChannel).where(TelegramChannel.id == id))
    return result.scalars().first()


async def orm_get_news_urls(session: AsyncSession, parsed: bool = None) -> list[NewsUrl]:
    query = select(NewsUrl)
    if parsed:
        query = query.where(NewsUrl.parsed == parsed)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_telegram_channels(session: AsyncSession, parsed: bool = None) -> list[TelegramChannel]:
    query = select(TelegramChannel)
    if parsed:
        query = query.where(TelegramChannel.parsed == parsed)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_news(session: AsyncSession, id: int = None, type: str = None, title: str = None, date: datetime = None) -> list[News]:
    query = select(News)
    if id:
        query = query.where(News.id == id)
    if type:
        query = query.where(News.type == type)
    if title:
        query = query.where(News.title == title)
    if date:
        query = query.where(News.date == date)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_add_news(session: AsyncSession, data: NewsData) -> News:
    existing = await orm_get_news(session, title=data.title, date=data.date)
    if existing:
        raise ValueError(f"News {data.id_url} already exists")
    news = News(id_url=data.id_url, type=data.type.value, title=data.title, text=data.text, date=data.date)
    session.add(news)
    await session.commit()
    await session.refresh(news)
    return news


async def orm_add_news_coin(session: AsyncSession, news_id: int, coin_id: int, score: float = 0):
    session.add(NewsCoin(news_id=news_id, coin_id=coin_id, score=score))
    await session.commit()


async def orm_get_news_coin(session: AsyncSession, news_id: int, coin_id: int = None) -> list[NewsCoin]:
    query = select(NewsCoin).where(NewsCoin.news_id == news_id)
    if coin_id:
        query = query.where(NewsCoin.coin_id == coin_id)
    result = await session.execute(query)
    return result.scalars().all()


