from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, or_
from sqlalchemy.orm import selectinload
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from core.database.models.main_models import DataTimeseries, Timeseries, Coin
from core.database.models.Strategy_models import StatisticAgent

logger = logging.getLogger(__name__)


async def orm_get_data_stats(
    db: AsyncSession,
    coin_ids: List[int],
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Получение статистики данных"""
    try:
        # Build base query
        base_query = select(DataTimeseries).join(Timeseries)
        
        # Add filters
        conditions = []
        
        if coin_ids:
            conditions.append(Timeseries.coin_id.in_(coin_ids))
        
        if start_date:
            conditions.append(DataTimeseries.datetime >= start_date)
        
        if end_date:
            conditions.append(DataTimeseries.datetime <= end_date)
        
        if conditions:
            base_query = base_query.where(and_(*conditions))
        
        # Get total records
        total_query = select(func.count(DataTimeseries.id)).select_from(base_query.subquery())
        total_records = await db.scalar(total_query)
        
        # Get date range
        date_range_query = select(
            func.min(DataTimeseries.datetime),
            func.max(DataTimeseries.datetime)
        ).select_from(base_query.subquery())
        
        date_range_result = await db.execute(date_range_query)
        first_record, last_record = date_range_result.first()
        
        # Get coins count
        coins_query = select(func.count(func.distinct(Timeseries.coin_id))).select_from(base_query.subquery())
        coins_count = await db.scalar(coins_query)
        
        # Calculate completeness (assuming expected records per day)
        if first_record and last_record:
            days_diff = (last_record - first_record).days
            expected_records_per_day = 288  # 5-minute intervals
            expected_total = days_diff * expected_records_per_day * len(coin_ids) if coin_ids else 0
            completeness = total_records / expected_total if expected_total > 0 else 1.0
        else:
            completeness = 1.0
        
        # Get missing values count (simplified - records with null values)
        missing_query = select(func.count(DataTimeseries.id)).where(
            or_(
                DataTimeseries.open.is_(None),
                DataTimeseries.max.is_(None),
                DataTimeseries.min.is_(None),
                DataTimeseries.close.is_(None),
                DataTimeseries.volume.is_(None)
            )
        )
        missing_values = await db.scalar(missing_query)
        
        # Get duplicates count (simplified)
        duplicates_query = select(func.count(DataTimeseries.id)).select_from(
            select(DataTimeseries.timeseries_id, DataTimeseries.datetime)
            .group_by(DataTimeseries.timeseries_id, DataTimeseries.datetime)
            .having(func.count(DataTimeseries.id) > 1)
            .subquery()
        )
        duplicates = await db.scalar(duplicates_query)
        
        # Get sample data
        sample_query = base_query.limit(10)
        sample_result = await db.execute(sample_query)
        sample_data = []
        
        for row in sample_result.scalars():
            # Get coin name
            coin_query = select(Coin.name).where(Coin.id == row.timeseries.coin_id)
            coin_name = await db.scalar(coin_query)
            
            sample_data.append({
                'datetime': row.datetime.isoformat(),
                'coin': coin_name,
                'open': row.open,
                'high': row.max,
                'low': row.min,
                'close': row.close,
                'volume': row.volume
            })
        
        # Get detailed stats per coin
        coins_details = []
        if coin_ids:
            for coin_id in coin_ids:
                coin_query = select(Coin.name).where(Coin.id == coin_id)
                coin_name = await db.scalar(coin_query)
                
                coin_conditions = [Timeseries.coin_id == coin_id]
                if start_date:
                    coin_conditions.append(DataTimeseries.datetime >= start_date)
                if end_date:
                    coin_conditions.append(DataTimeseries.datetime <= end_date)
                
                coin_base_query = select(DataTimeseries).join(Timeseries).where(and_(*coin_conditions))
                
                # Get coin stats
                coin_total = await db.scalar(select(func.count(DataTimeseries.id)).select_from(coin_base_query.subquery()))
                coin_date_range = await db.execute(
                    select(func.min(DataTimeseries.datetime), func.max(DataTimeseries.datetime))
                    .select_from(coin_base_query.subquery())
                )
                coin_first, coin_last = coin_date_range.first()
                
                # Calculate coin completeness
                if coin_first and coin_last:
                    coin_days = (coin_last - coin_first).days
                    coin_expected = coin_days * 288  # 5-minute intervals
                    coin_completeness = coin_total / coin_expected if coin_expected > 0 else 1.0
                else:
                    coin_completeness = 1.0
                
                # Get missing records for this coin
                coin_missing = await db.scalar(
                    select(func.count(DataTimeseries.id))
                    .select_from(coin_base_query.subquery())
                    .where(
                        or_(
                            DataTimeseries.open.is_(None),
                            DataTimeseries.max.is_(None),
                            DataTimeseries.min.is_(None),
                            DataTimeseries.close.is_(None),
                            DataTimeseries.volume.is_(None)
                        )
                    )
                )
                
                coins_details.append({
                    'name': coin_name,
                    'records': coin_total,
                    'completeness': coin_completeness,
                    'missing': coin_missing,
                    'first_record': coin_first.isoformat() if coin_first else None,
                    'last_record': coin_last.isoformat() if coin_last else None
                })
        
        return {
            'total_records': total_records,
            'coins_count': coins_count,
            'date_range': f"{first_record.strftime('%Y-%m-%d')} to {last_record.strftime('%Y-%m-%d')}" if first_record and last_record else "No data",
            'completeness': completeness,
            'missing_values': missing_values,
            'duplicates': duplicates,
            'timeframe': timeframe,
            'first_record': first_record.isoformat() if first_record else None,
            'last_record': last_record.isoformat() if last_record else None,
            'sample_data': sample_data,
            'coins_details': coins_details
        }
        
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise


async def orm_export_data(
    db: AsyncSession,
    coin_ids: List[int],
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """Экспорт данных"""
    try:
        # Build query
        query = select(DataTimeseries, Coin.name.label('coin_name')).join(
            Timeseries, DataTimeseries.timeseries_id == Timeseries.id
        ).join(
            Coin, Timeseries.coin_id == Coin.id
        ).where(
            and_(
                Timeseries.coin_id.in_(coin_ids),
                DataTimeseries.datetime >= start_date,
                DataTimeseries.datetime <= end_date
            )
        ).order_by(DataTimeseries.datetime)
        
        result = await db.execute(query)
        data = []
        
        for row in result:
            data.append({
                'datetime': row.DataTimeseries.datetime.isoformat(),
                'coin': row.coin_name,
                'open': row.DataTimeseries.open,
                'high': row.DataTimeseries.max,
                'low': row.DataTimeseries.min,
                'close': row.DataTimeseries.close,
                'volume': row.DataTimeseries.volume
            })
        
        return data
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise


async def orm_import_data(
    db: AsyncSession,
    data: List[Dict[str, Any]],
    timeframe: str
) -> Dict[str, Any]:
    """Импорт данных"""
    try:
        imported_records = 0
        skipped_records = 0
        errors = []
        
        for record in data:
            try:
                # Parse datetime
                if isinstance(record['datetime'], str):
                    dt = datetime.fromisoformat(record['datetime'].replace('Z', '+00:00'))
                else:
                    dt = record['datetime']
                
                # Get or create timeseries for coin
                coin_name = record['coin']
                coin_query = select(Coin.id).where(Coin.name == coin_name)
                coin_id = await db.scalar(coin_query)
                
                if not coin_id:
                    errors.append(f"Coin {coin_name} not found")
                    skipped_records += 1
                    continue
                
                # Get or create timeseries
                timeseries_query = select(Timeseries.id).where(
                    and_(
                        Timeseries.coin_id == coin_id,
                        Timeseries.path_dataset.like(f"%{timeframe}%")
                    )
                )
                timeseries_id = await db.scalar(timeseries_query)
                
                if not timeseries_id:
                    # Create new timeseries
                    new_timeseries = Timeseries(
                        coin_id=coin_id,
                        timestamp=timeframe,
                        path_dataset=f"{coin_name}_{timeframe}_{datetime.now().strftime('%Y%m%d')}"
                    )
                    db.add(new_timeseries)
                    await db.flush()
                    timeseries_id = new_timeseries.id
                
                # Check if record already exists
                existing_query = select(DataTimeseries.id).where(
                    and_(
                        DataTimeseries.timeseries_id == timeseries_id,
                        DataTimeseries.datetime == dt
                    )
                )
                existing = await db.scalar(existing_query)
                
                if existing:
                    skipped_records += 1
                    continue
                
                # Create new record
                new_record = DataTimeseries(
                    timeseries_id=timeseries_id,
                    datetime=dt,
                    open=record['open'],
                    max=record['high'],
                    min=record['low'],
                    close=record['close'],
                    volume=record['volume']
                )
                
                db.add(new_record)
                imported_records += 1
                
            except Exception as e:
                errors.append(f"Error processing record {record}: {str(e)}")
                skipped_records += 1
                continue
        
        # Commit all changes
        await db.commit()
        
        return {
            'imported_records': imported_records,
            'skipped_records': skipped_records,
            'errors': errors
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error importing data: {e}")
        raise


async def orm_get_model_metrics(
    db: AsyncSession,
    model_id: int
) -> Dict[str, Any]:
    """Получение метрик модели"""
    try:
        # Get latest statistics for the model
        query = select(StatisticAgent).where(
            StatisticAgent.agent_id == model_id
        ).order_by(desc(StatisticAgent.id)).limit(1)
        
        result = await db.execute(query)
        latest_stats = result.scalar_one_or_none()
        
        if not latest_stats:
            return {}
        
        return {
            'type': latest_stats.type,
            'loss': latest_stats.loss,
            'accuracy': getattr(latest_stats, 'accuracy', None),
            'precision': getattr(latest_stats, 'precision', None),
            'recall': getattr(latest_stats, 'recall', None),
            'f1score': getattr(latest_stats, 'f1score', None)
        }
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise


async def orm_get_models_list(
    db: AsyncSession,
    type: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = 100
) -> List[Dict[str, Any]]:
    """Получение списка моделей"""
    try:
        from core.database.models.ML_models import Agent
        
        # Build query
        query = select(Agent)
        
        # Add filters
        conditions = []
        if type:
            conditions.append(Agent.type == type)
        if status:
            conditions.append(Agent.status == status)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Add limit and order
        query = query.limit(limit).order_by(desc(Agent.id))
        
        result = await db.execute(query)
        agents = result.scalars().all()
        
        models = []
        for agent in agents:
            models.append({
                'id': agent.id,
                'name': agent.name,
                'type': agent.type,
                'timeframe': agent.timeframe,
                'status': agent.status,
                'version': agent.version,
                'created_at': agent.created_at.isoformat() if hasattr(agent, 'created_at') and agent.created_at else None,
                'updated_at': agent.updated_at.isoformat() if hasattr(agent, 'updated_at') and agent.updated_at else None
            })
        
        return models
        
    except Exception as e:
        logger.error(f"Error getting models list: {e}")
        raise
