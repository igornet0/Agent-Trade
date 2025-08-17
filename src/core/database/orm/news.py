"""
ORM functions for news-related operations
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from core.database.models.main_models import NewsBackground, News, NewsHistoryCoin, Coin


async def orm_get_news_background(
    session: AsyncSession,
    coin_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> List[NewsBackground]:
    """Get news background for a specific coin within time range"""
    query = select(NewsBackground).where(NewsBackground.coin_id == coin_id)
    
    if start_time:
        query = query.where(NewsBackground.timestamp >= start_time)
    if end_time:
        query = query.where(NewsBackground.timestamp <= end_time)
    
    query = query.order_by(NewsBackground.timestamp.desc()).limit(limit)
    
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_latest_news_background(
    session: AsyncSession,
    coin_id: int
) -> Optional[NewsBackground]:
    """Get the latest news background for a specific coin"""
    query = (
        select(NewsBackground)
        .where(NewsBackground.coin_id == coin_id)
        .order_by(NewsBackground.timestamp.desc())
        .limit(1)
    )
    
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def orm_create_news_background(
    session: AsyncSession,
    coin_id: int,
    timestamp: datetime,
    score: float,
    source_count: int = 0,
    sources_breakdown: Optional[Dict[str, Any]] = None,
    window_hours: int = 24,
    decay_factor: float = 0.95
) -> NewsBackground:
    """Create a new news background record"""
    news_bg = NewsBackground(
        coin_id=coin_id,
        timestamp=timestamp,
        score=score,
        source_count=source_count,
        sources_breakdown=sources_breakdown,
        window_hours=window_hours,
        decay_factor=decay_factor
    )
    
    session.add(news_bg)
    await session.commit()
    await session.refresh(news_bg)
    
    return news_bg


async def orm_update_news_background(
    session: AsyncSession,
    news_bg_id: int,
    **kwargs
) -> Optional[NewsBackground]:
    """Update an existing news background record"""
    query = select(NewsBackground).where(NewsBackground.id == news_bg_id)
    result = await session.execute(query)
    news_bg = result.scalar_one_or_none()
    
    if news_bg:
        for key, value in kwargs.items():
            if hasattr(news_bg, key):
                setattr(news_bg, key, value)
        
        await session.commit()
        await session.refresh(news_bg)
    
    return news_bg


async def orm_delete_old_news_background(
    session: AsyncSession,
    older_than_days: int = 30
) -> int:
    """Delete old news background records"""
    cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
    
    query = (
        select(func.count(NewsBackground.id))
        .where(NewsBackground.timestamp < cutoff_date)
    )
    
    result = await session.execute(query)
    count = result.scalar()
    
    if count > 0:
        delete_query = (
            select(NewsBackground)
            .where(NewsBackground.timestamp < cutoff_date)
        )
        
        result = await session.execute(delete_query)
        records_to_delete = result.scalars().all()
        
        for record in records_to_delete:
            await session.delete(record)
        
        await session.commit()
    
    return count


async def orm_get_news_for_background_calculation(
    session: AsyncSession,
    coin_id: int,
    start_time: datetime,
    end_time: datetime
) -> List[Dict[str, Any]]:
    """Get news data for background calculation"""
    query = (
        select(
            News.date,
            NewsHistoryCoin.score,
            News.source,
            News.title
        )
        .join(News, News.id == NewsHistoryCoin.id_news)
        .where(
            and_(
                NewsHistoryCoin.coin_id == coin_id,
                News.date >= start_time,
                News.date <= end_time
            )
        )
        .order_by(News.date)
    )
    
    result = await session.execute(query)
    rows = result.all()
    
    return [
        {
            'date': row.date,
            'score': float(row.score),
            'source': row.source,
            'title': row.title
        }
        for row in rows
    ]


async def orm_get_coins_with_news(
    session: AsyncSession
) -> List[Coin]:
    """Get all coins that have news data"""
    query = (
        select(Coin)
        .distinct()
        .join(NewsHistoryCoin, Coin.id == NewsHistoryCoin.coin_id)
        .options(selectinload(Coin.news_background))
    )
    
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_news_background_summary(
    session: AsyncSession,
    coin_id: int,
    hours: int = 24
) -> Dict[str, Any]:
    """Get summary statistics for news background"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    query = (
        select(
            func.avg(NewsBackground.score).label('avg_score'),
            func.min(NewsBackground.score).label('min_score'),
            func.max(NewsBackground.score).label('max_score'),
            func.count(NewsBackground.id).label('count'),
            func.sum(NewsBackground.source_count).label('total_sources')
        )
        .where(
            and_(
                NewsBackground.coin_id == coin_id,
                NewsBackground.timestamp >= cutoff_time
            )
        )
    )
    
    result = await session.execute(query)
    row = result.one()
    
    return {
        'avg_score': float(row.avg_score) if row.avg_score else 0.0,
        'min_score': float(row.min_score) if row.min_score else 0.0,
        'max_score': float(row.max_score) if row.max_score else 0.0,
        'count': int(row.count) if row.count else 0,
        'total_sources': int(row.total_sources) if row.total_sources else 0,
        'time_window_hours': hours
    }


