from typing import Literal, List
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from core.database.models import (
    Agent, AgentFeature, AgentFeatureValue, Feature,
    AgentTrain, TrainCoin, AgentType
)
from core.database.models.ML_models import StatisticAgent


async def orm_get_feature_by_id(session: AsyncSession, feature_id: int) -> Feature:
    query = select(Feature).where(Feature.id == feature_id)
    result = await session.execute(query)
    return result.scalars().first()


async def orm_get_agent_by_name(session: AsyncSession, agent_name: str) -> Agent:
    query = select(Agent).where(Agent.name == agent_name)
    result = await session.execute(query)
    return result.scalars().first()


async def orm_add_agent(session: AsyncSession, agent_data: Agent):
    if agent_data.path_model:
        path_model = agent_data.path_model
        if not path_model.endswith(".pth"):
            path_model += ".pth"
    else:
        path_model = f"{agent_data.name}_{agent_data.version}.pth"

    agent = Agent(
        name=agent_data.name,
        type=agent_data.type,
        timeframe=agent_data.timeframe,
        path_model=path_model,
        a_conficent=agent_data.a_conficent,
        active=agent_data.active,
        version=agent_data.version,
    )

    session.add(agent)
    await session.flush()

    features = [
        AgentFeature(
            agent_id=agent.id,
            feature_id=feature.id,
            feature_value=[
                AgentFeatureValue(value=value, feature_name=par_name)
                for par_name, value in feature.parameters.items()
            ],
        )
        for feature in agent_data.features
    ]

    session.add_all(features)
    await session.flush()

    train_data = None

    if agent_data.train_data:
        train_data = AgentTrain(
            agent_id=agent.id,
            epochs=agent_data.train_data.epochs,
            batch_size=agent_data.train_data.batch_size,
            learning_rate=agent_data.train_data.learning_rate,
            weight_decay=agent_data.train_data.weight_decay,
        )
        session.add(train_data)
        await session.flush()
        session.add_all([TrainCoin(train_id=train_data.id, coin_id=coin_id) for coin_id in agent_data.coins])

    await session.commit()
    await session.flush()

    features_data = []

    for feature in features:
        parameters = {value.feature_name: value.value for value in feature.feature_value}
        features_data.append({
            "id": feature.id,
            "feature_id": feature.feature_id,
            "parameters": parameters,
        })

    return {
        "id": agent.id,
        "type": agent.type,
        "path_model": agent.path_model,
        "active": agent.active,
        "created": agent.created.isoformat() if agent.created else None,
        "name": agent.name,
        "a_conficent": agent.a_conficent,
        "version": agent.version,
        "updated": agent.updated.isoformat() if agent.updated else None,
        "features": features_data,
    }, train_data


async def orm_get_agent_feature(session: AsyncSession, agent_id: int) -> list[AgentFeature]:
    query = select(AgentFeature).where(AgentFeature.agent_id == agent_id)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_train_agent(session: AsyncSession, agent_id: int) -> list[AgentTrain]:
    query = select(AgentTrain).where(AgentTrain.agent_id == agent_id)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_train_agents(session: AsyncSession) -> list[AgentTrain]:
    query = select(AgentTrain)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_get_agents(session: AsyncSession, type_agent: str = None,
                 id_agent: int = None, version: str = None,
                 active: bool = None, query_return: bool = False) -> list[Agent]:
    query = select(Agent)
    if type_agent:
        query = query.where(Agent.type == type_agent)
    if id_agent:
        query = query.where(Agent.id == id_agent)
    if version:
        query = query.where(Agent.version == version)
    if active is not None:
        query = query.where(Agent.active == active)
    if status:
        query = query.where(Agent.status == status)

    if query_return:
        return query

    stmt = query.options(
        selectinload(Agent.features).selectinload(AgentFeature.feature_value)
    )
    result = await session.execute(stmt)
    agents = result.scalars().all()
    if not agents:
        return None

    agents_features_data: List[dict] = []
    for agent in agents:
        features_data = []
        for feature in agent.features:
            parameters = {value.feature_name: value.value for value in feature.feature_value}
            feature_t = await orm_get_feature_by_id(session, feature.feature_id)
            features_data.append({
                "id": feature_t.id,
                "name": feature_t.name,
                "feature_id": feature.feature_id,
                "parameters": parameters,
            })

        agents_features_data.append({
            "id": agent.id,
            "type": agent.type,
            "status": agent.status,
            "timeframe": agent.timeframe,
            "path_model": agent.path_model,
            "active": agent.active,
            "created": agent.created.isoformat() if agent.created else None,
            "name": agent.name,
            "a_conficent": agent.a_conficent,
            "version": agent.version,
            "updated": agent.updated.isoformat() if agent.updated else None,
            "features": features_data,
        })
    return agents_features_data


async def orm_get_agents_type(session: AsyncSession) -> list[AgentType]:
    query = select(AgentType)
    result = await session.execute(query)
    return result.scalars().all()


async def orm_delete_agent(session: AsyncSession, agent_id: int):
    agent = await session.get(Agent, agent_id)
    if agent:
        features = await orm_get_agent_feature(session, agent_id)
        await session.execute(delete(StatisticAgent).where(StatisticAgent.agent_id == agent_id))
        for feature in features:
            await session.execute(delete(AgentFeatureValue).where(AgentFeatureValue.agent_feature_id == feature.id))
        await session.execute(delete(AgentFeature).where(AgentFeature.agent_id == agent_id))
        trains = await orm_get_train_agent(session, agent_id)
        for train in trains:
            await session.execute(delete(TrainCoin).where(TrainCoin.train_id == train.id))
        await session.execute(delete(AgentTrain).where(AgentTrain.agent_id == agent_id))
        await session.delete(agent)
        await session.commit()


async def orm_get_agent_by_id(session: AsyncSession, id: int) -> Agent:
    query = select(Agent).where(Agent.id == id)
    result = await session.execute(query)
    return result.scalars().first()


async def orm_get_agents_options(session: AsyncSession, type_agent: str = None,
                         id_agent: int = None, version: str = None,
                         active: bool = None, mod: Literal["actions", "strategies", "stata", "all"] = None) -> list[Agent]:
    query = await orm_get_agents(session, type_agent, id_agent, version, active, query=True)
    if mod in ["actions", "all"]:
        query = query.options(joinedload(Agent.actions))
    if mod in ["strategies", "all"]:
        query = query.options(joinedload(Agent.strategies))
    if mod in ["stata", "all"]:
        query = query.options(joinedload(Agent.stata))
    if mod in ["all"]:
        query = query.options(joinedload(Agent.features))
    result = await session.execute(query)
    return result.scalars().all()


async def orm_set_active_version(session: AsyncSession, agent_id: int):
    agent = await orm_get_agent_by_id(session, agent_id)
    if not agent:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Agent not found")
    await session.execute(update(Agent).where(Agent.name == agent.name).values(active=False))
    await session.execute(update(Agent).where(Agent.id == agent_id).values(active=True))
    await session.commit()


async def orm_get_features(session: AsyncSession) -> List[dict]:
    """Получает все доступные фичи (индикаторы)"""
    query = select(Feature)
    result = await session.execute(query)
    features = result.scalars().all()
    
    features_data = []
    for feature in features:
        features_data.append({
            "id": feature.id,
            "name": feature.name,
            "arguments": []  # Пока возвращаем пустой список аргументов
        })
    
    return features_data


