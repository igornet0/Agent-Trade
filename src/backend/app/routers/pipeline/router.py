from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from backend.app.configuration import Server, verify_authorization_admin
from backend.app.configuration.schemas import PipelineConfig
from backend.celery_app.create_app import celery_app

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# In-memory placeholder storage for saved pipelines
_PIPELINES: dict[str, dict] = {}


@router.post("/run")
async def run_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin)):
    # Placeholder: create async result to simulate orchestrator
    task = celery_app.send_task('backend.celery_app.tasks.evaluate_trade_aggregator_task', kwargs={"strategy_config": config.model_dump()})
    return {"task_id": task.id}


@router.post("/save")
async def save_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin), db: AsyncSession = Depends(Server.get_db)):
    # Placeholder persistence: keep in memory and return id
    pipeline_id = str(uuid4())
    _PIPELINES[pipeline_id] = config.model_dump()
    return {"status": "ok", "pipeline_id": pipeline_id}


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str, _: str = Depends(verify_authorization_admin)):
    cfg = _PIPELINES.get(pipeline_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return cfg


