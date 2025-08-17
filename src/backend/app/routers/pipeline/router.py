from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.configuration import Server, verify_authorization_admin
from backend.app.configuration.schemas import PipelineConfig
from backend.celery_app.create_app import celery_app

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/run")
async def run_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin)):
    # Placeholder: create async result to simulate orchestrator
    # Reuse generic task_status infra via celery
    task = celery_app.send_task('backend.celery_app.tasks.evaluate_trade_aggregator_task', kwargs={"strategy_config": config.model_dump()})
    return {"task_id": task.id}


@router.post("/save")
async def save_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin), db: AsyncSession = Depends(Server.get_db)):
    # Placeholder persistence hook — to be implemented
    return {"status": "ok", "saved": True}


