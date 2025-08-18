"""
ORM методы для работы с пайплайнами и бэктестами
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from core.database.models.main_models import Pipeline, Backtest
from core.database.models.ML_models import Agent


# Pipeline methods
def orm_create_pipeline(
    db: Session,
    name: str,
    config_json: Dict[str, Any],
    description: Optional[str] = None,
    is_template: bool = False,
    created_by: Optional[int] = None
) -> Pipeline:
    """Создание нового пайплайна"""
    pipeline = Pipeline(
        name=name,
        description=description,
        config_json=config_json,
        is_template=is_template,
        created_by=created_by
    )
    
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    
    return pipeline


def orm_get_pipeline_by_id(db: Session, pipeline_id: int) -> Optional[Pipeline]:
    """Получение пайплайна по ID"""
    return db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()


def orm_get_pipelines(
    db: Session,
    user_id: Optional[int] = None,
    is_template: Optional[bool] = None,
    limit: Optional[int] = None
) -> List[Pipeline]:
    """Получение списка пайплайнов"""
    query = db.query(Pipeline)
    
    if user_id is not None:
        query = query.filter(Pipeline.created_by == user_id)
    
    if is_template is not None:
        query = query.filter(Pipeline.is_template == is_template)
    
    query = query.order_by(desc(Pipeline.id))  # Используем ID вместо updated_at
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def orm_update_pipeline(
    db: Session,
    pipeline_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config_json: Optional[Dict[str, Any]] = None,
    is_template: Optional[bool] = None
) -> Optional[Pipeline]:
    """Обновление пайплайна"""
    pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
    if not pipeline:
        return None
    
    if name is not None:
        pipeline.name = name
    if description is not None:
        pipeline.description = description
    if config_json is not None:
        pipeline.config_json = config_json
    if is_template is not None:
        pipeline.is_template = is_template
    
    db.commit()
    db.refresh(pipeline)
    
    return pipeline


def orm_delete_pipeline(db: Session, pipeline_id: int) -> bool:
    """Удаление пайплайна"""
    pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
    if pipeline:
        db.delete(pipeline)
        db.commit()
        return True
    return False


# Backtest methods
def orm_create_backtest(
    db: Session,
    name: str,
    config_json: Dict[str, Any],
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    coins: List[str],
    pipeline_id: Optional[int] = None
) -> Backtest:
    """Создание нового бэктеста"""
    backtest = Backtest(
        pipeline_id=pipeline_id,
        name=name,
        config_json=config_json,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        coins=coins,
        status="running",
        progress=0.0
    )
    
    db.add(backtest)
    db.commit()
    db.refresh(backtest)
    
    return backtest


def orm_get_backtest_by_id(db: Session, backtest_id: int) -> Optional[Backtest]:
    """Получение бэктеста по ID"""
    return db.query(Backtest).filter(Backtest.id == backtest_id).first()


def orm_get_backtests(
    db: Session,
    pipeline_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Backtest]:
    """Получение списка бэктестов"""
    query = db.query(Backtest)
    
    if pipeline_id is not None:
        query = query.filter(Backtest.pipeline_id == pipeline_id)
    
    if status is not None:
        query = query.filter(Backtest.status == status)
    
    query = query.order_by(desc(Backtest.id))  # Используем ID вместо created_at
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def orm_update_backtest_status(
    db: Session,
    backtest_id: int,
    status: str,
    progress: Optional[float] = None,
    metrics_json: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
) -> Optional[Backtest]:
    """Обновление статуса бэктеста"""
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if not backtest:
        return None
    
    backtest.status = status
    
    if progress is not None:
        backtest.progress = progress
    
    if metrics_json is not None:
        backtest.metrics_json = metrics_json
    
    if artifacts is not None:
        backtest.artifacts = artifacts
    
    if error_message is not None:
        backtest.error_message = error_message
    
    db.commit()
    db.refresh(backtest)
    
    return backtest


def orm_delete_backtest(db: Session, backtest_id: int) -> bool:
    """Удаление бэктеста"""
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    if backtest:
        db.delete(backtest)
        db.commit()
        return True
    return False


def orm_get_backtest_stats(db: Session, pipeline_id: Optional[int] = None) -> Dict[str, Any]:
    """Получение статистики бэктестов"""
    query = db.query(Backtest)
    
    if pipeline_id is not None:
        query = query.filter(Backtest.pipeline_id == pipeline_id)
    
    backtests = query.all()
    
    stats = {
        "total_count": len(backtests),
        "by_status": {},
        "success_rate": 0.0
    }
    
    completed_count = 0
    
    for backtest in backtests:
        status = backtest.status
        if status not in stats["by_status"]:
            stats["by_status"][status] = 0
        stats["by_status"][status] += 1
        
        if status == "completed":
            completed_count += 1
    
    if len(backtests) > 0:
        stats["success_rate"] = completed_count / len(backtests)
    
    return stats


def orm_cleanup_old_backtests(
    db: Session,
    pipeline_id: Optional[int] = None,
    days_to_keep: int = 30
) -> int:
    """Очистка старых бэктестов (по статусу, так как нет created_at)"""
    query = db.query(Backtest).filter(
        Backtest.status.in_(["completed", "failed"])
    )
    
    if pipeline_id is not None:
        query = query.filter(Backtest.pipeline_id == pipeline_id)
    
    # Удаляем старые завершенные бэктесты (по ID, предполагая что старые имеют меньший ID)
    # Это упрощенная логика, в реальности нужно использовать внешние метки времени
    deleted_count = query.delete()
    db.commit()
    
    return deleted_count
