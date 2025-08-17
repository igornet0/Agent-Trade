
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm_query import (User, Agent, Transaction, Portfolio, 
                                     AgentFeature,
                                     orm_get_train_agent,
                                     orm_delete_agent,
                                     orm_get_agent_by_name,
                                     orm_get_train_agents,
                                     orm_add_agent,
                                     orm_get_agents,
                                     orm_get_agent_by_id,
                                     orm_get_agents_type,
                                     orm_get_features,
                                     orm_get_user_transactions,
                                     orm_get_user_coin_transactions,
                                     orm_get_coin_portfolio,
                                     orm_set_active_version)

from backend.app.configuration import (Server, 
                                       rabbit,
                                       AgentType,
                                       TrainData,
                                       FeatureTypeResponse,
                                       AgentTrainResponse,
                                       AgentTypeResponse,
                                       AgentResponse,
                                       AgentCreate,
                                       AgentTrade,
                                       EvaluateRequest,
                                       UserResponse,
                                       OrderUpdateAmount,
                                       OrderResponse,
                                       OrderCreate,
                                       OrderCancel,
                                       OrderType,
                                       verify_authorization,
                                       verify_authorization_admin)

from backend.celery_app.tasks import train_model_task
from backend.celery_app.tasks import evaluate_model_task
from backend.celery_app.create_app import celery_app
from backend.app.configuration.schemas.agent import TrainRequest

http_bearer = HTTPBearer(auto_error=False)

# Инициализация роутера
router = APIRouter(
    prefix="/api_db_agent",
    dependencies=[Depends(http_bearer)],
    tags=["Api db agent"]
)

@router.post("/train_new_agent/", response_model=AgentResponse)
async def create_agent_train(agent_data: AgentTrainResponse, 
                       _: User = Depends(verify_authorization_admin), 
                       db: AsyncSession = Depends(Server.get_db)):
    try:
        agent = await orm_get_agent_by_name(db, agent_data.name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    if agent:
        raise HTTPException(status_code=400, detail="Agent with this name already exists") 
    
    try:
        agent, train_data = await orm_add_agent(db, agent_data)

        # Kick off async training task (align with your Celery backend/monitoring)
        task = train_model_task.delay(agent_id=agent["id"])

        # Persist task id to AgentTrain for monitoring
        if train_data is not None:
            train_data.task_id = task.id
            await db.commit()

        # Align with response_model: return created agent payload
        return agent
    except Exception as e:
        await db.rollback()  # Откатываем при ошибке
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")

@router.post("/delete_agent/{agent_id}")
async def delete_agent(agent_id: int,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agent = await orm_get_agent_by_id(db, agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        await orm_delete_agent(db, agent_id)

        return {"message": "Agent deleted successfully"}
    except Exception as e:
        await db.rollback()  # Откатываем при ошибке
        raise HTTPException(status_code=500, detail=f"Agent delete failed: {str(e)}")

@router.post("/agent/{id}", response_model=AgentResponse)
async def get_agent_by_id(agent_data: AgentResponse, 
                    _: User = Depends(verify_authorization), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agent = await orm_get_agent_by_id(db, agent_data.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent

@router.get("/agents/", response_model=list[AgentResponse])
async def get_agents(status: str | None = None,
                    type: str | None = None,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_agents(db, type_agent=type, status=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents

@router.get("/train_agents/", response_model=list[TrainData])
async def get_agents(status: str | None = None,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_train_agents(db, status=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents

@router.get("/train_agent/{id}", response_model=list[TrainData])
async def get_agents(id: int,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_agent_by_id(db, id=id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents

@router.get("/agents_types/", response_model=list[AgentTypeResponse])
async def get_agents_type(_: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        get_agents_type = await orm_get_agents_type(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents type failed: {str(e)}")
    
    if not get_agents_type:
        raise HTTPException(status_code=404, detail="Agents type not found")
    
    return get_agents_type

@router.get("/available_features/", response_model=list[FeatureTypeResponse])
async def get_agent_features(_: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        feuters = await orm_get_features(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Features failed: {str(e)}")
    
    if not feuters:
        raise HTTPException(status_code=404, detail="Features not found")
    
    return feuters

# @router.post("/trade/", response_model=list[AgentResponse])
# async def get_agent_trade(trade_data: AgentTrade,
#                     _: User = Depends(verify_authorization_admin), 
#                     db: AsyncSession = Depends(Server.get_db)):
#     try:
#         agents = await orm_get_agents(db)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
#     if not agents:
#         raise HTTPException(status_code=404, detail="Agents not found")
    
#     return agents

@router.get("/health/rabbitmq")
async def rabbitmq_health():
    try:
        connection = await rabbit.get_connection()
        if connection.is_closed:
            return {"status": "down", "detail": "Connection closed"}
        return {"status": "up"}
    except Exception as e:
        return {"status": "down", "detail": str(e)}


@router.get("/train_status/{agent_id}")
async def get_train_status(agent_id: int,
                           _: User = Depends(verify_authorization_admin),
                           db: AsyncSession = Depends(Server.get_db)):
    trains = await orm_get_train_agent(db, agent_id)
    if not trains:
        raise HTTPException(status_code=404, detail="Train not found")
    train = trains[0]
    coins = [c.coin_id for c in (await db.execute(TrainCoin.__table__.select().where(TrainCoin.train_id==train.id))).fetchall()]
    return {
        "id": train.id,
        "agent_id": train.agent_id,
        "epochs": train.epochs,
        "epoch_now": train.epoch_now,
        "loss_now": train.loss_now,
        "loss_avg": train.loss_avg,
        "batch_size": train.batch_size,
        "learning_rate": train.learning_rate,
        "weight_decay": train.weight_decay,
        "status": train.status,
        "task_id": train.task_id,
        "coins": coins,
    }


@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str,
                          _: User = Depends(verify_authorization_admin)):
    result = celery_app.AsyncResult(task_id)
    meta = result.info if isinstance(result.info, dict) else {"detail": str(result.info)}
    return {
        "task_id": task_id,
        "state": result.state,
        "meta": meta,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else False,
    }


@router.post("/evaluate")
async def evaluate_agent(payload: EvaluateRequest,
                         _: User = Depends(verify_authorization_admin)):
    task = evaluate_model_task.delay(agent_id=payload.agent_id,
                                     coins=payload.coins,
                                     timeframe=payload.timeframe or "5m",
                                     start=payload.start.isoformat() if payload.start else None,
                                     end=payload.end.isoformat() if payload.end else None)
    return {"task_id": task.id}


# Unified train/evaluate endpoints by agent type (backward-compatible wrappers)
@router.post("/{agent_type}/train")
async def unified_train(agent_type: AgentType,
                        payload: TrainRequest,
                        _: User = Depends(verify_authorization_admin),
                        db: AsyncSession = Depends(Server.get_db)):
    if payload.type != agent_type:
        raise HTTPException(status_code=400, detail="type mismatch between path and payload")

    try:
        existing = await orm_get_agent_by_name(db, payload.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
    if existing:
        raise HTTPException(status_code=400, detail="Agent with this name already exists")

    try:
        # Reuse existing creation logic
        agent_payload = AgentTrainResponse(
            name=payload.name,
            type=payload.type,
            timeframe=payload.timeframe or "5m",
            features=payload.features or [],
            coins=payload.coins or [],
            train_data=payload.train_data,
            RP_I=False,
        )
        agent, train_data = await orm_add_agent(db, agent_payload)
        task = train_model_task.delay(agent_id=agent["id"])  # kick off training
        if train_data is not None:
            train_data.task_id = task.id
            await db.commit()
        return {"agent": agent, "task_id": task.id}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Unified train failed: {str(e)}")


@router.post("/{agent_type}/evaluate")
async def unified_evaluate(agent_type: AgentType,
                           payload: EvaluateRequest,
                           _: User = Depends(verify_authorization_admin)):
    task = evaluate_model_task.delay(agent_id=payload.agent_id,
                                     coins=payload.coins,
                                     timeframe=payload.timeframe or "5m",
                                     start=payload.start.isoformat() if payload.start else None,
                                     end=payload.end.isoformat() if payload.end else None)
    return {"task_id": task.id}

@router.post("/agents/{agent_id}/promote")
async def promote_agent(agent_id: int,
                        _: User = Depends(verify_authorization_admin),
                        db: AsyncSession = Depends(Server.get_db)):
    try:
        await orm_set_active_version(db, agent_id)
        return {"message": "Agent version promoted", "agent_id": agent_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promote failed: {str(e)}")