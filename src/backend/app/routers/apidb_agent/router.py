
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
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
from backend.celery_app.tasks import evaluate_model_task, train_news_task
from backend.celery_app.create_app import celery_app
from backend.app.configuration.schemas.agent import TrainRequest
from typing import Optional, List, Dict

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
    # Simple branch: news type can run dedicated task in future
    if agent_type == AgentType.NEWS and not payload.agent_id:
        task = train_news_task.delay(coins=payload.coins or [], config=None)
    else:
        task = evaluate_model_task.delay(agent_id=payload.agent_id,
                                         coins=payload.coins,
                                         timeframe=payload.timeframe or "5m",
                                         start=payload.start.isoformat() if payload.start else None,
                                         end=payload.end.isoformat() if payload.end else None)
    return {"task_id": task.id}


# -------- News background endpoints (Stage 1 contract) --------
@router.post("/news/recalc_background")
async def recalc_news_background(
    coins: Optional[str] = Query(None, description="Comma-separated list of coin IDs"),
    window_hours: int = Query(24, description="Time window for background calculation"),
    decay_factor: float = Query(0.95, description="Exponential decay factor"),
    force_recalculate: bool = Query(False, description="Force recalculation ignoring cache"),
    _: str = Depends(verify_authorization_admin)
):
    """Recalculate news background for specified coins or all coins"""
    try:
        from core.services.news_background_service import NewsBackgroundService
        from core.database import db_helper
        
        # Parse coin IDs
        coin_ids = []
        if coins:
            coin_ids = [int(cid.strip()) for cid in coins.split(',') if cid.strip().isdigit()]
        
        # Initialize service
        news_service = NewsBackgroundService()
        
        # Prepare configuration
        config = {
            'coin_ids': coin_ids,
            'window_hours': window_hours,
            'decay_factor': decay_factor,
            'force_recalculate': force_recalculate
        }
        
        # Start Celery task
        from backend.celery_app.create_app import celery_app
        task = celery_app.send_task('backend.celery_app.tasks.train_news_task', kwargs=config)
        
        return {
            "status": "started",
            "task_id": task.id,
            "config": config,
            "message": f"News background recalculation started for {len(coin_ids) if coin_ids else 'all'} coins"
        }
        
    except Exception as e:
        logger.exception("Failed to start news background recalculation")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news/background/{coin_id}")
async def get_news_background(
    coin_id: int,
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(1000, description="Maximum number of records to return"),
    _: str = Depends(verify_authorization_admin)
):
    """Get news background for a specific coin"""
    try:
        from core.services.news_background_service import NewsBackgroundService
        from core.database import db_helper
        from datetime import datetime
        
        # Parse time parameters
        start_dt = None
        end_dt = None
        
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_time format")
        
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_time format")
        
        # Get background data
        async with db_helper.get_session() as session:
            news_service = NewsBackgroundService()
            
            # Try cache first
            cached_data = await news_service.get_cached_background(coin_id)
            if cached_data:
                return {
                    "status": "success",
            "data": cached_data,
            "source": "cache"
        }
            
            # Get from database
            backgrounds = await news_service.get_background_history(
                session, coin_id, start_dt, end_dt, limit
            )
            
            if not backgrounds:
                # Calculate fresh background
                background = await news_service.calculate_news_background(
                    session, coin_id
                )
                return {
                    "status": "success",
                    "data": background,
                    "source": "calculated"
                }
            
            # Return latest background
            latest = await news_service.get_latest_background(session, coin_id)
            
            return {
                "status": "success",
                "data": latest,
                "history": backgrounds,
                "source": "database"
            }
        
    except Exception as e:
        logger.exception(f"Failed to get news background for coin {coin_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news/background/{coin_id}/summary")
async def get_news_background_summary(
    coin_id: int,
    hours: int = Query(24, description="Time window for summary"),
    _: str = Depends(verify_authorization_admin)
):
    """Get summary statistics for news background"""
    try:
        from core.database.orm.news import orm_get_news_background_summary
        from core.database import db_helper
        
        async with db_helper.get_session() as session:
            summary = await orm_get_news_background_summary(session, coin_id, hours)
            
            return {
                "status": "success",
                "coin_id": coin_id,
                "summary": summary
            }
        
    except Exception as e:
        logger.exception(f"Failed to get news background summary for coin {coin_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news/coins")
async def get_coins_with_news(
    _: str = Depends(verify_authorization_admin)
):
    """Get all coins that have news data"""
    try:
        from core.services.news_background_service import NewsBackgroundService
        from core.database import db_helper
        
        async with db_helper.get_session() as session:
            news_service = NewsBackgroundService()
            coins = await news_service.get_coins_with_news_data(session)
            
            return {
                "status": "success",
                "coins": coins,
                "count": len(coins)
            }
        
    except Exception as e:
        logger.exception("Failed to get coins with news data")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/news/evaluate")
async def evaluate_news_model(
    request: dict,
    _: str = Depends(verify_authorization_admin)
):
    """Evaluate News model performance"""
    try:
        from core.database import db_helper
        
        # Validate request
        coin_ids = request.get('coin_ids', [])
        evaluation_hours = request.get('evaluation_hours', 168)
        correlation_threshold = request.get('correlation_threshold', 0.1)
        
        if not coin_ids:
            raise HTTPException(status_code=400, detail="coin_ids is required")
        
        # Start Celery task
        from backend.celery_app.create_app import celery_app
        config = {
            'coin_ids': coin_ids,
            'evaluation_hours': evaluation_hours,
            'correlation_threshold': correlation_threshold
        }
        
        task = celery_app.send_task('backend.celery_app.tasks.evaluate_news_task', kwargs=config)
        
        return {
            "status": "started",
            "task_id": task.id,
            "config": config,
            "message": f"News model evaluation started for {len(coin_ids)} coins"
        }
        
    except Exception as e:
        logger.exception("Failed to start news model evaluation")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/pred_time/train")
async def train_pred_time_model(
    request: dict,
    _: str = Depends(verify_authorization_admin)
):
    """Train Pred_time model for price prediction"""
    try:
        from core.database import db_helper
        
        # Validate request
        coin_ids = request.get('coin_ids', [])
        agent_id = request.get('agent_id')
        config = request.get('config', {})
        
        if not coin_ids:
            raise HTTPException(status_code=400, detail="coin_ids is required")
        
        if not agent_id:
            raise HTTPException(status_code=400, detail="agent_id is required")
        
        # Validate configuration
        required_fields = ['seq_len', 'pred_len', 'model_type']
        for field in required_fields:
            if field not in config:
                raise HTTPException(status_code=400, detail=f"config.{field} is required")
        
        # Start Celery task
        from backend.celery_app.create_app import celery_app
        task_config = {
            'coin_ids': coin_ids,
            'agent_id': agent_id,
            **config
        }
        
        task = celery_app.send_task('backend.celery_app.tasks.train_pred_time_task', kwargs=task_config)
        
        return {
            "status": "started",
            "task_id": task.id,
            "config": task_config,
            "message": f"Pred_time model training started for {len(coin_ids)} coins"
        }
        
    except Exception as e:
        logger.exception("Failed to start Pred_time model training")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pred_time/evaluate")
async def evaluate_pred_time_model(
    request: dict,
    _: str = Depends(verify_authorization_admin)
):
    """Evaluate Pred_time model performance"""
    try:
        from core.database import db_helper
        
        # Validate request
        model_path = request.get('model_path')
        coin_ids = request.get('coin_ids', [])
        evaluation_hours = request.get('evaluation_hours', 168)
        
        if not model_path:
            raise HTTPException(status_code=400, detail="model_path is required")
        
        if not coin_ids:
            raise HTTPException(status_code=400, detail="coin_ids is required")
        
        # Start Celery task
        from backend.celery_app.create_app import celery_app
        task_config = {
            'model_path': model_path,
            'coin_ids': coin_ids,
            'evaluation_hours': evaluation_hours
        }
        
        task = celery_app.send_task('backend.celery_app.tasks.evaluate_pred_time_task', kwargs=task_config)
        
        return {
            "status": "started",
            "task_id": task.id,
            "config": task_config,
            "message": f"Pred_time model evaluation started for {len(coin_ids)} coins"
        }
        
    except Exception as e:
        logger.exception("Failed to start Pred_time model evaluation")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pred_time/models/{agent_id}")
async def get_pred_time_models(
    agent_id: int,
    _: str = Depends(verify_authorization_admin)
):
    """Get Pred_time models for a specific agent"""
    try:
        from core.database import db_helper
        import os
        from pathlib import Path
        
        models_dir = Path("models/pred_time")
        agent_models = []
        
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and f"agent_{agent_id}_" in model_dir.name:
                    # Check if model artifacts exist
                    model_file = model_dir / "model.pth"
                    config_file = model_dir / "config.json"
                    metadata_file = model_dir / "metadata.json"
                    
                    if model_file.exists() and config_file.exists() and metadata_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Get model size
                            model_size = model_file.stat().st_size if model_file.exists() else 0
                            
                            agent_models.append({
                                'model_path': str(model_dir),
                                'model_name': model_dir.name,
                                'config': config,
                                'metadata': metadata,
                                'model_size_bytes': model_size,
                                'created_at': metadata.get('created_at'),
                                'model_type': metadata.get('model_type')
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read model info from {model_dir}: {e}")
                            continue
        
        # Sort by creation date (newest first)
        agent_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "models": agent_models,
            "count": len(agent_models)
        }
        
    except Exception as e:
        logger.exception(f"Failed to get Pred_time models for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pred_time/predict")
async def make_pred_time_prediction(
    request: dict,
    _: str = Depends(verify_authorization_admin)
):
    """Make prediction using trained Pred_time model"""
    try:
        from core.services.pred_time_service import PredTimeService
        from core.database import db_helper
        import asyncio
        
        # Validate request
        model_path = request.get('model_path')
        coin_id = request.get('coin_id')
        features = request.get('features')
        
        if not model_path:
            raise HTTPException(status_code=400, detail="model_path is required")
        
        if not coin_id:
            raise HTTPException(status_code=400, detail="coin_id is required")
        
        if not features:
            raise HTTPException(status_code=400, detail="features is required")
        
        async def _run():
            async with db_helper.get_session() as session:
                # Initialize service
                pred_time_service = PredTimeService()
                
                # Load model
                model = await pred_time_service.load_model(model_path)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found or failed to load")
                
                # Convert features to numpy array
                import numpy as np
                features_array = np.array(features)
                
                # Make prediction
                prediction = await pred_time_service.predict(model, features_array, coin_id)
                if not prediction:
                    raise HTTPException(status_code=500, detail="Prediction failed")
                
                return prediction
        
        result = asyncio.run(_run())
        return {
            "status": "success",
            "prediction": result
        }
        
    except Exception as e:
        logger.exception("Failed to make Pred_time prediction")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

# Trade_time endpoints
@router.post("/trade_time/train")
async def train_trade_time(
    coin_id: int,
    start_date: str,
    end_date: str,
    extra_config: Optional[Dict] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Обучение Trade_time модели"""
    try:
        from core.services.trade_time_service import TradeTimeService
        
        service = TradeTimeService()
        result = service.train_model(str(coin_id), start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trade_time train: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trade_time/evaluate")
async def evaluate_trade_time(
    coin_id: int,
    start_date: str,
    end_date: str,
    model_path: str,
    extra_config: Optional[Dict] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Оценка Trade_time модели"""
    try:
        from core.services.trade_time_service import TradeTimeService
        
        service = TradeTimeService()
        result = service.evaluate_model(str(coin_id), start_date, end_date, model_path, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trade_time evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trade_time/models/{agent_id}")
async def get_trade_time_models(
    agent_id: int,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение списка моделей Trade_time для агента"""
    try:
        import os
        from pathlib import Path
        
        models_dir = Path("models/models_pth/AgentTradeTime")
        if not models_dir.exists():
            return {"status": "success", "agent_id": agent_id, "models": [], "count": 0}
        
        agent_models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and str(agent_id) in model_dir.name:
                config_file = model_dir / "config.json"
                metadata_file = model_dir / "metadata.json"
                
                if config_file.exists() and metadata_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        agent_models.append({
                            'model_path': str(model_dir),
                            'model_name': model_dir.name,
                            'config': config,
                            'metadata': metadata,
                            'created_at': metadata.get('created_at'),
                            'model_type': metadata.get('model_type')
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read model info from {model_dir}: {e}")
                        continue
        
        # Sort by creation date (newest first)
        agent_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "models": agent_models,
            "count": len(agent_models)
        }
        
    except Exception as e:
        logger.exception(f"Failed to get Trade_time models for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trade_time/predict")
async def predict_trade_time(
    coin_id: int,
    start_date: str,
    end_date: str,
    model_path: str,
    current_user: User = Depends(verify_authorization_admin)
):
    """Предсказание торговых сигналов с помощью Trade_time модели"""
    try:
        from core.services.trade_time_service import TradeTimeService
        
        service = TradeTimeService()
        result = service.predict(model_path, str(coin_id), start_date, end_date)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trade_time predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk endpoints
@router.post("/risk/train")
async def train_risk(
    coin_id: int,
    start_date: str,
    end_date: str,
    extra_config: Optional[Dict] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Обучение Risk модели"""
    try:
        from core.services.risk_service import RiskService
        
        service = RiskService()
        result = service.train_model(str(coin_id), start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in risk train: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trade Aggregator endpoints
@router.post("/trade_aggregator/train")
async def train_trade_aggregator(
    coin_id: int,
    start_date: str,
    end_date: str,
    extra_config: Optional[Dict] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Обучение Trade Aggregator модели"""
    try:
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        result = service.train_model(str(coin_id), start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trade_aggregator train: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trade_aggregator/evaluate")
async def evaluate_trade_aggregator(
    coin_id: int,
    start_date: str,
    end_date: str,
    extra_config: Optional[Dict] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Оценка Trade Aggregator модели"""
    try:
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        result = service.evaluate_model(str(coin_id), start_date, end_date, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trade_aggregator evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trade_aggregator/models/{agent_id}")
async def get_trade_aggregator_models(
    agent_id: int,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение моделей Trade Aggregator для агента"""
    try:
        # Здесь должна быть логика получения моделей из БД
        # Пока возвращаем заглушку
        return {
            "agent_id": agent_id,
            "models": [],
            "message": "Trade Aggregator models endpoint - implementation needed"
        }
        
    except Exception as e:
        logger.error(f"Error getting trade_aggregator models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trade_aggregator/predict")
async def predict_trade_aggregator(
    coin_id: int,
    pred_time_signals: Optional[Dict] = None,
    trade_time_signals: Optional[Dict] = None,
    risk_signals: Optional[Dict] = None,
    portfolio_state: Optional[Dict] = None,
    extra_config: Optional[Dict] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Предсказание Trade Aggregator"""
    try:
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        result = service.predict(
            str(coin_id), 
            pred_time_signals, 
            trade_time_signals, 
            risk_signals, 
            portfolio_state, 
            extra_config
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trade_aggregator predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/evaluate")
async def evaluate_risk(
    coin_id: int,
    start_date: str,
    end_date: str,
    model_path: str,
    extra_config: Optional[Dict] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Оценка Risk модели"""
    try:
        from core.services.risk_service import RiskService
        
        service = RiskService()
        result = service.evaluate_model(str(coin_id), start_date, end_date, model_path, extra_config)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in risk evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/models/{agent_id}")
async def get_risk_models(
    agent_id: int,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение списка моделей Risk для агента"""
    try:
        import os
        from pathlib import Path
        
        models_dir = Path("models/models_pth/AgentRisk")
        if not models_dir.exists():
            return {"status": "success", "agent_id": agent_id, "models": [], "count": 0}
        
        agent_models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and str(agent_id) in model_dir.name:
                config_file = model_dir / "config.json"
                metadata_file = model_dir / "metadata.json"
                
                if config_file.exists() and metadata_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        agent_models.append({
                            'model_path': str(model_dir),
                            'model_name': model_dir.name,
                            'config': config,
                            'metadata': metadata,
                            'created_at': metadata.get('created_at'),
                            'model_type': metadata.get('model_type')
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read model info from {model_dir}: {e}")
                        continue
        
        # Sort by creation date (newest first)
        agent_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "models": agent_models,
            "count": len(agent_models)
        }
        
    except Exception as e:
        logger.exception(f"Failed to get Risk models for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/predict")
async def predict_risk(
    coin_id: int,
    start_date: str,
    end_date: str,
    model_path: str,
    current_user: User = Depends(verify_authorization_admin)
):
    """Предсказание рисков и объема с помощью Risk модели"""
    try:
        from core.services.risk_service import RiskService
        
        service = RiskService()
        result = service.predict(model_path, str(coin_id), start_date, end_date)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in risk predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))