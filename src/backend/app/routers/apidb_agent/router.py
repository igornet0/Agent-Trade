
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Response, File, Form, UploadFile
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
from typing import Optional, List, Dict, Any
from datetime import datetime

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

# Pipeline endpoints
@router.post("/pipeline/run")
async def run_pipeline(
    pipeline_config: Dict[str, Any],
    timeframe: str,
    start_date: str,
    end_date: str,
    coins: List[str],
    current_user: User = Depends(verify_authorization_admin)
):
    """Запуск пайплайна на выполнение"""
    try:
        from core.database.orm.pipelines import orm_create_backtest
        from core.database.engine import get_db
        from backend.celery_app.tasks import run_pipeline_backtest_task
        
        # Создаем запись бэктеста
        db = next(get_db())
        backtest = orm_create_backtest(
            db=db,
            name=f"Pipeline Backtest {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            config_json=pipeline_config,
            timeframe=timeframe,
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            coins=coins
        )
        
        # Запускаем Celery задачу
        task = run_pipeline_backtest_task.delay(
            pipeline_config=pipeline_config,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            coins=coins,
            backtest_id=backtest.id
        )
        
        return {
            "task_id": task.id,
            "backtest_id": backtest.id,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Error in pipeline run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/tasks/{task_id}")
async def get_pipeline_task_status(
    task_id: str,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение статуса задачи пайплайна"""
    try:
        from backend.celery_app.tasks import run_pipeline_backtest_task
        
        task = run_pipeline_backtest_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'current': 0,
                'total': 100,
                'status': 'Task is pending...'
            }
        elif task.state == 'PROGRESS':
            response = {
                'state': task.state,
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 100),
                'status': task.info.get('status', '')
            }
        elif task.state == 'SUCCESS':
            response = {
                'state': task.state,
                'current': 100,
                'total': 100,
                'status': 'Task completed successfully',
                'results': task.info.get('results', {})
            }
        else:
            response = {
                'state': task.state,
                'current': 0,
                'total': 100,
                'status': task.info.get('status', 'Task failed'),
                'error': task.info.get('error', 'Unknown error')
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/tasks/{task_id}/revoke")
async def revoke_pipeline_task(
    task_id: str,
    current_user: User = Depends(verify_authorization_admin)
):
    """Отмена задачи пайплайна"""
    try:
        from backend.celery_app.tasks import run_pipeline_backtest_task
        
        task = run_pipeline_backtest_task.AsyncResult(task_id)
        task.revoke(terminate=True)
        
        return {"status": "revoked", "task_id": task_id}
        
    except Exception as e:
        logger.error(f"Error revoking task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/backtests")
async def list_backtests(
    pipeline_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: Optional[int] = 50,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение списка бэктестов"""
    try:
        from core.database.orm.pipelines import orm_get_backtests
        from core.database.engine import get_db
        
        db = next(get_db())
        backtests = orm_get_backtests(db, pipeline_id, status, limit)
        
        return [
            {
                "id": bt.id,
                "pipeline_id": bt.pipeline_id,
                "name": bt.name,
                "timeframe": bt.timeframe,
                "start_date": bt.start_date.isoformat() if bt.start_date else None,
                "end_date": bt.end_date.isoformat() if bt.end_date else None,
                "coins": bt.coins,
                "status": bt.status,
                "progress": bt.progress,
                "error_message": bt.error_message
            }
            for bt in backtests
        ]
        
    except Exception as e:
        logger.error(f"Error listing backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/backtests/{backtest_id}")
async def get_backtest(
    backtest_id: int,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение детальной информации о бэктесте"""
    try:
        from core.database.orm.pipelines import orm_get_backtest_by_id
        from core.database.engine import get_db
        
        db = next(get_db())
        backtest = orm_get_backtest_by_id(db, backtest_id)
        
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        return {
            "id": backtest.id,
            "pipeline_id": backtest.pipeline_id,
            "name": backtest.name,
            "timeframe": backtest.timeframe,
            "start_date": backtest.start_date.isoformat() if backtest.start_date else None,
            "end_date": backtest.end_date.isoformat() if backtest.end_date else None,
            "coins": backtest.coins,
            "status": backtest.status,
            "progress": backtest.progress,
            "error_message": backtest.error_message,
            "metrics": backtest.metrics_json,
            "artifacts": backtest.artifacts
        }
        
    except Exception as e:
        logger.error(f"Error getting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/artifacts/{path:path}")
async def download_artifact(
    path: str,
    current_user: User = Depends(verify_authorization_admin)
):
    """Скачивание артефакта пайплайна"""
    try:
        import os
        
        # Безопасный путь к артефактам
        artifacts_dir = "artifacts/pipelines"
        full_path = os.path.join(artifacts_dir, path)
        
        # Проверяем, что путь находится в разрешенной директории
        if not os.path.abspath(full_path).startswith(os.path.abspath(artifacts_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Определяем тип файла
        file_extension = os.path.splitext(full_path)[1].lower()
        content_type = {
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.pdf': 'application/pdf'
        }.get(file_extension, 'application/octet-stream')
        
        # Читаем файл
        with open(full_path, 'rb') as f:
            content = f.read()
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(full_path)}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error downloading artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Versioning endpoints
@router.post("/models/{agent_id}/versions")
async def create_model_version(
    agent_id: int,
    version: str,
    model_path: str,
    config_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Создание новой версии модели"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        result = service.create_version(
            agent_id=agent_id,
            version=version,
            model_path=model_path,
            config_path=config_path,
            scaler_path=scaler_path,
            metadata=metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{agent_id}/versions/{version}/promote")
async def promote_model_version(
    agent_id: int,
    version: str,
    force: bool = False,
    current_user: User = Depends(verify_authorization_admin)
):
    """Продвижение версии модели в продакшн"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        result = service.promote_version(
            agent_id=agent_id,
            version=version,
            force=force
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error promoting model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{agent_id}/versions/{version}/rollback")
async def rollback_model_version(
    agent_id: int,
    version: str,
    current_user: User = Depends(verify_authorization_admin)
):
    """Откат к указанной версии модели"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        result = service.rollback_version(
            agent_id=agent_id,
            target_version=version
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error rolling back model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{agent_id}/versions")
async def list_model_versions(
    agent_id: int,
    limit: Optional[int] = None,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение списка версий модели"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        versions = service.list_versions(
            agent_id=agent_id,
            limit=limit
        )
        
        return versions
        
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{agent_id}/versions/{version}")
async def get_model_version_info(
    agent_id: int,
    version: str,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение детальной информации о версии модели"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        version_info = service.get_version_info(
            agent_id=agent_id,
            version=version
        )
        
        return version_info
        
    except Exception as e:
        logger.error(f"Error getting model version info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{agent_id}/versions/{version}")
async def delete_model_version(
    agent_id: int,
    version: str,
    force: bool = False,
    current_user: User = Depends(verify_authorization_admin)
):
    """Удаление версии модели"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        result = service.delete_version(
            agent_id=agent_id,
            version=version,
            force=force
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error deleting model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{agent_id}/production")
async def get_model_production_status(
    agent_id: int,
    current_user: User = Depends(verify_authorization_admin)
):
    """Получение статуса продакшн версии модели"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        status = service.get_production_status(agent_id)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting model production status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{agent_id}/versions/cleanup")
async def cleanup_model_versions(
    agent_id: int,
    keep_versions: int = 5,
    current_user: User = Depends(verify_authorization_admin)
):
    """Очистка старых версий модели"""
    try:
        from core.services.model_versioning_service import ModelVersioningService
        
        service = ModelVersioningService()
        result = service.cleanup_old_versions(
            agent_id=agent_id,
            keep_versions=keep_versions
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error cleaning up model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test Model endpoints
@router.post("/test_model")
async def test_model(
    payload: dict,
    _: User = Depends(verify_authorization_admin)
):
    """Тестирование модели с заданными параметрами"""
    try:
        from backend.celery_app.tasks import test_model_task
        
        # Validate payload
        model_id = payload.get('model_id')
        coins = payload.get('coins', [])
        timeframe = payload.get('timeframe', '5m')
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        metrics = payload.get('metrics', [])
        
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        
        if not coins:
            raise HTTPException(status_code=400, detail="coins is required")
        
        # Start Celery task
        task = test_model_task.delay(
            model_id=model_id,
            coins=coins,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics
        )
        
        return {
            "status": "started",
            "task_id": task.id,
            "message": f"Model testing started for {len(coins)} coins"
        }
        
    except Exception as e:
        logger.exception("Failed to start model testing")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: int,
    _: User = Depends(verify_authorization_admin),
    db: AsyncSession = Depends(Server.get_db)
):
    """Получение метрик модели"""
    try:
        from core.database.orm.agents import orm_get_model_metrics
        
        metrics = await orm_get_model_metrics(db, model_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Model metrics not found")
        
        return {
            "model_id": model_id,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.exception(f"Failed to get model metrics for model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Management endpoints
@router.get("/data/stats")
async def get_data_stats(
    coins: Optional[str] = Query(None, description="Comma-separated list of coin IDs"),
    timeframe: str = Query("5m", description="Data timeframe"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    _: User = Depends(verify_authorization_admin),
    db: AsyncSession = Depends(Server.get_db)
):
    """Получение статистики данных"""
    try:
        from core.database.orm.data import orm_get_data_stats
        from datetime import datetime
        
        # Parse coin IDs
        coin_ids = []
        if coins:
            coin_ids = [int(cid.strip()) for cid in coins.split(',') if cid.strip().isdigit()]
        
        # Parse dates
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        
        # Get statistics
        stats = await orm_get_data_stats(db, coin_ids, timeframe, start_dt, end_dt)
        
        return stats
        
    except Exception as e:
        logger.exception("Failed to get data statistics")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/export")
async def export_data(
    coins: str = Query(..., description="Comma-separated list of coin IDs"),
    timeframe: str = Query("5m", description="Data timeframe"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    format: str = Query("csv", description="Export format (csv, json)"),
    _: User = Depends(verify_authorization_admin),
    db: AsyncSession = Depends(Server.get_db)
):
    """Экспорт данных"""
    try:
        from core.database.orm.data import orm_export_data
        from datetime import datetime
        import io
        import csv
        import json
        
        # Parse parameters
        coin_ids = [int(cid.strip()) for cid in coins.split(',') if cid.strip().isdigit()]
        
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
        
        # Get data
        data = await orm_export_data(db, coin_ids, timeframe, start_dt, end_dt)
        
        if format.lower() == "csv":
            # Create CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            if data:
                writer.writerow(data[0].keys())
                for row in data:
                    writer.writerow(row.values())
            
            content = output.getvalue()
            media_type = "text/csv"
            filename = f"trading_data_{start_date}_{end_date}.csv"
            
        elif format.lower() == "json":
            # Create JSON
            content = json.dumps(data, indent=2, default=str)
            media_type = "application/json"
            filename = f"trading_data_{start_date}_{end_date}.json"
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.exception("Failed to export data")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/import")
async def import_data(
    file: UploadFile = File(...),
    timeframe: str = Form("5m"),
    _: User = Depends(verify_authorization_admin),
    db: AsyncSession = Depends(Server.get_db)
):
    """Импорт данных"""
    try:
        from core.database.orm.data import orm_import_data
        import pandas as pd
        import io
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        
        # Parse based on file extension
        file_ext = file.filename.lower().split('.')[-1]
        
        if file_ext == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_ext == 'json':
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate required columns
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        # Import data
        result = await orm_import_data(db, data, timeframe)
        
        return {
            "status": "success",
            "imported_records": result.get('imported_records', 0),
            "skipped_records": result.get('skipped_records', 0),
            "errors": result.get('errors', [])
        }
        
    except Exception as e:
        logger.exception("Failed to import data")
        raise HTTPException(status_code=500, detail=str(e))


# Models endpoint (for listing all models)
@router.get("/models")
async def list_models(
    type: Optional[str] = Query(None, description="Filter by model type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: Optional[int] = Query(100, description="Maximum number of models to return"),
    _: User = Depends(verify_authorization_admin),
    db: AsyncSession = Depends(Server.get_db)
):
    """Получение списка всех моделей"""
    try:
        from core.database.orm.agents import orm_get_models_list
        
        models = await orm_get_models_list(db, type=type, status=status, limit=limit)
        
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail=str(e))


# Missing imports
import logging
import json
from fastapi import File, Form, UploadFile
from core.database.models.Strategy_models import TrainCoin

logger = logging.getLogger(__name__)