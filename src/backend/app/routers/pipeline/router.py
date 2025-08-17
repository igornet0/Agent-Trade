from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from backend.app.configuration import Server, verify_authorization_admin
from backend.app.configuration.schemas import PipelineConfig
from backend.celery_app.create_app import celery_app
from core.database import db_helper
from core.database.models.process_models import Pipeline as PipelineModel

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# In-memory cache (optional); source of truth will be DB
_PIPELINES: dict[str, dict] = {}


@router.post("/run")
async def run_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin)):
    # Placeholder: create async result to simulate orchestrator
    task = celery_app.send_task('backend.celery_app.tasks.run_pipeline_backtest_task', kwargs={"config_json": config.model_dump()})
    return {"task_id": task.id}


@router.post("/save")
async def save_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin), db: AsyncSession = Depends(Server.get_db)):
    # Persist to DB and cache id->config
    model = PipelineModel(name=config.name if hasattr(config, 'name') else f"pipeline-{uuid4().hex[:6]}",
                          user_id=None,
                          config_json=config.model_dump())
    db.add(model)
    await db.commit()
    await db.refresh(model)
    pipeline_id = str(model.id)
    _PIPELINES[pipeline_id] = config.model_dump()
    return {"status": "ok", "pipeline_id": pipeline_id}


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str, _: str = Depends(verify_authorization_admin)):
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


@router.post("/tasks/{task_id}/revoke")
async def revoke_pipeline_task(task_id: str, _: str = Depends(verify_authorization_admin)):
    # Best-effort revoke
    try:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    except Exception:
        pass
    return {"status": "revoked", "task_id": task_id}


