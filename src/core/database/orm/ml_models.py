from __future__ import annotations

from typing import Optional, Literal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from core.database.models import (
    ML_Model,
    ModelAction,
    StatisticModel,
)


async def orm_add_ml_model(
    session: AsyncSession,
    *,
    type: str,
    path_model: str,
    version: str = "0.0.1",
) -> ML_Model:
    existing = await session.execute(
        select(ML_Model).where(ML_Model.path_model == path_model)
    )
    model = existing.scalars().first()
    if model:
        return model

    model = ML_Model(type=type, path_model=path_model, version=version)
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


async def orm_get_ml_model_by_id(session: AsyncSession, model_id: int) -> Optional[ML_Model]:
    res = await session.execute(select(ML_Model).where(ML_Model.id == model_id))
    return res.scalars().first()


async def orm_get_ml_models(
    session: AsyncSession,
    *,
    type: Optional[str] = None,
    version: Optional[str] = None,
    path_like: Optional[str] = None,
):
    query = select(ML_Model)
    if type:
        query = query.where(ML_Model.type == type)
    if version:
        query = query.where(ML_Model.version == version)
    if path_like:
        # простой LIKE через contains
        query = query.where(ML_Model.path_model.contains(path_like))
    res = await session.execute(query)
    return res.scalars().all()


async def orm_delete_ml_model(session: AsyncSession, model_id: int) -> bool:
    model = await orm_get_ml_model_by_id(session, model_id)
    if not model:
        return False
    await session.execute(delete(ModelAction).where(ModelAction.model_id == model_id))
    await session.execute(delete(StatisticModel).where(StatisticModel.model_id == model_id))
    await session.execute(delete(ML_Model).where(ML_Model.id == model_id))
    await session.commit()
    return True


async def orm_add_model_action(
    session: AsyncSession,
    *,
    model_id: int,
    action: str,
    loss: float = 0.0,
) -> ModelAction:
    item = ModelAction(model_id=model_id, action=action, loss=loss)
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return item


async def orm_add_model_stat(
    session: AsyncSession,
    *,
    model_id: int,
    type: Literal[
        "train",
        "val",
        "test",
        "backtest",
        "custom",
    ],
    loss: float,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> StatisticModel:
    stat = StatisticModel(
        model_id=model_id,
        type=type,
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1score=f1,
    )
    session.add(stat)
    await session.commit()
    await session.refresh(stat)
    return stat


