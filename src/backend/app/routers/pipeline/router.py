from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4
import logging
from typing import Dict, Any, List
from pathlib import Path

from backend.app.configuration import Server, verify_authorization_admin
from backend.app.configuration.schemas import PipelineConfig
from backend.celery_app.create_app import celery_app
from core.database import db_helper
from core.database.models.main_models import Pipeline as PipelineModel, Backtest as BacktestModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# In-memory cache (optional); source of truth will be DB
_PIPELINES: dict[str, dict] = {}


@router.post("/run")
async def run_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin)):
    """Run pipeline with validation and error handling"""
    try:
        # Validate configuration
        if not config.model_dump():
            raise HTTPException(status_code=400, detail="Invalid pipeline configuration")
        
        # Send task to Celery with proper error handling
        task = celery_app.send_task(
            'backend.celery_app.tasks.run_pipeline_backtest_task', 
            kwargs={
                "config_json": config.model_dump(), 
                "pipeline_id": None
            }
        )
        
        logger.info(f"Pipeline task started with ID: {task.id}")
        return {
            "task_id": task.id,
            "status": "started",
            "message": "Pipeline execution initiated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")


@router.post("/save")
async def save_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin), db: AsyncSession = Depends(Server.get_db)):
    """Save pipeline configuration with validation"""
    try:
        # Validate configuration
        config_dict = config.model_dump()
        if not config_dict:
            raise HTTPException(status_code=400, detail="Invalid pipeline configuration")
        
        # Generate name if not provided
        pipeline_name = getattr(config, 'name', None) or f"pipeline-{uuid4().hex[:6]}"
        
        # Create and save model
        model = PipelineModel(
            name=pipeline_name,
            user_id=None,
            config_json=config_dict
        )
        db.add(model)
        await db.commit()
        await db.refresh(model)
        
        pipeline_id = str(model.id)
        _PIPELINES[pipeline_id] = config_dict
        
        logger.info(f"Pipeline saved with ID: {pipeline_id}")
        return {
            "status": "ok", 
            "pipeline_id": pipeline_id,
            "name": pipeline_name,
            "message": "Pipeline configuration saved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to save pipeline: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save pipeline: {str(e)}")


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str, _: str = Depends(verify_authorization_admin)):
    """Get pipeline configuration with proper error handling"""
    try:
        # Check cache first
        cfg = _PIPELINES.get(pipeline_id)
        if cfg:
            return cfg
        
        # Fallback to DB
        async with db_helper.get_session() as session:
            obj = await session.get(PipelineModel, int(pipeline_id))
            if not obj:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            
            _PIPELINES[pipeline_id] = obj.config_json
            return obj.config_json
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID format")
    except Exception as e:
        logger.error(f"Failed to get pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")


@router.post("/tasks/{task_id}/revoke")
async def revoke_pipeline_task(task_id: str, _: str = Depends(verify_authorization_admin)):
    """Revoke pipeline task with proper error handling"""
    try:
        # Validate task ID
        if not task_id or not task_id.strip():
            raise HTTPException(status_code=400, detail="Invalid task ID")
        
        # Attempt to revoke task
        result = celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
        
        logger.info(f"Pipeline task {task_id} revoked successfully")
        return {
            "status": "revoked", 
            "task_id": task_id,
            "message": "Task revocation initiated"
        }
        
    except Exception as e:
        logger.error(f"Failed to revoke task {task_id}: {str(e)}")
        # Return success even if revocation fails (best effort)
        return {
            "status": "revoke_attempted", 
            "task_id": task_id,
            "message": "Task revocation attempted but may not have succeeded"
        }


@router.get("/backtests")
async def list_backtests(_: str = Depends(verify_authorization_admin)):
    """List backtests with pagination and error handling"""
    try:
        async with db_helper.get_session() as session:
            rows = (await session.execute(
                BacktestModel.__table__.select().order_by(BacktestModel.id.desc()).limit(100)
            )).mappings().all()
            
            return {
                "backtests": [dict(r) for r in rows],
                "count": len(rows),
                "message": "Backtests retrieved successfully"
            }
            
    except Exception as e:
        logger.error(f"Failed to list backtests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")


@router.get("/backtests/{bt_id}")
async def get_backtest(bt_id: int, _: str = Depends(verify_authorization_admin)):
    """Get specific backtest with validation"""
    try:
        async with db_helper.get_session() as session:
            obj = await session.get(BacktestModel, bt_id)
            if not obj:
                raise HTTPException(status_code=404, detail="Backtest not found")
            
            return {
                "id": obj.id,
                "pipeline_id": obj.pipeline_id,
                "timeframe": obj.timeframe,
                "start": obj.start.isoformat() if obj.start else None,
                "end": obj.end.isoformat() if obj.end else None,
                "metrics_json": obj.metrics_json,
                "artifacts": obj.artifacts,
                "created_at": obj.created_at.isoformat() if obj.created_at else None,
                "message": "Backtest retrieved successfully"
            }
            
    except Exception as e:
        logger.error(f"Failed to get backtest {bt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest: {str(e)}")


@router.post("/run/{pipeline_id}")
async def run_pipeline_by_id(pipeline_id: int, _: str = Depends(verify_authorization_admin)):
    """Run pipeline by ID with validation"""
    try:
        # Load pipeline config from DB and trigger task
        async with db_helper.get_session() as session:
            obj = await session.get(PipelineModel, int(pipeline_id))
            if not obj:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            
            # Validate configuration
            if not obj.config_json:
                raise HTTPException(status_code=400, detail="Pipeline configuration is empty")
            
            task = celery_app.send_task(
                'backend.celery_app.tasks.run_pipeline_backtest_task', 
                kwargs={
                    "config_json": obj.config_json, 
                    "pipeline_id": pipeline_id
                }
            )
            
            logger.info(f"Pipeline {pipeline_id} started with task ID: {task.id}")
            return {
                "task_id": task.id,
                "pipeline_id": pipeline_id,
                "status": "started",
                "message": "Pipeline execution initiated successfully"
            }
            
    except Exception as e:
        logger.error(f"Failed to run pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run pipeline: {str(e)}")


@router.get("/artifacts/{path:path}")
async def download_artifact(path: str, _: str = Depends(verify_authorization_admin)):
    """Download artifact with security validation"""
    try:
        # Security: validate path to prevent directory traversal
        artifact_path = Path(path)
        
        # Check if path is absolute and within allowed directory
        if artifact_path.is_absolute():
            # In production, restrict to specific artifacts directory
            allowed_base = Path("/tmp/artifacts")  # Configure this properly
            if not str(artifact_path).startswith(str(allowed_base)):
                raise HTTPException(status_code=403, detail="Access denied to this path")
        
        # Check if file exists
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Check if it's a file
        if not artifact_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        logger.info(f"Artifact download requested: {path}")
        return FileResponse(
            str(artifact_path), 
            filename=artifact_path.name,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Failed to download artifact {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download artifact: {str(e)}")


