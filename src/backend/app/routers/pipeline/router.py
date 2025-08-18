from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from backend.app.configuration import Server, verify_authorization_admin
from backend.app.configuration.schemas import PipelineConfig
from backend.celery_app.create_app import celery_app
from core.database import db_helper
from core.database.models.main_models import Pipeline as PipelineModel, Backtest as BacktestModel

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# In-memory cache (optional); source of truth will be DB
_PIPELINES: dict[str, dict] = {}


@router.post("/run")
async def run_pipeline(config: PipelineConfig, _: str = Depends(verify_authorization_admin)):
    # Placeholder: create async result to simulate orchestrator
    task = celery_app.send_task('backend.celery_app.tasks.run_pipeline_backtest_task', kwargs={"config_json": config.model_dump(), "pipeline_id": None})
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


@router.get("/backtests")
async def list_backtests(_: str = Depends(verify_authorization_admin)):
    async with db_helper.get_session() as session:
        rows = (await session.execute(
            BacktestModel.__table__.select().order_by(BacktestModel.id.desc()).limit(100)
        )).mappings().all()
        return [dict(r) for r in rows]


@router.get("/backtests/{bt_id}")
async def get_backtest(bt_id: int, _: str = Depends(verify_authorization_admin)):
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
        }


@router.post("/run/{pipeline_id}")
async def run_pipeline_by_id(pipeline_id: int, _: str = Depends(verify_authorization_admin)):
    # Load pipeline config from DB and trigger task
    async with db_helper.get_session() as session:
        obj = await session.get(PipelineModel, int(pipeline_id))
        if not obj:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        task = celery_app.send_task('backend.celery_app.tasks.run_pipeline_backtest_task', kwargs={"config_json": obj.config_json, "pipeline_id": pipeline_id})
        return {"task_id": task.id}


@router.get("/artifacts/{path:path}")
async def download_artifact(path: str, _: str = Depends(verify_authorization_admin)):
    # Security: restrict to temp dir or configured artifacts dir
    # Here we trust generated paths; in production validate path prefix
    return FileResponse(path, filename=path.split('/')[-1])


