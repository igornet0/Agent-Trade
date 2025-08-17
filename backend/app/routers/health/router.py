from fastapi import APIRouter
from sqlalchemy import text
from celery.exceptions import TimeoutError

from backend.app.configuration import Server
from core.database import db_helper
from backend.celery_app.create_app import celery_app

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/db")
async def health_db():
    try:
        async with db_helper.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "up"}
    except Exception as e:
        return {"status": "down", "detail": str(e)}


@router.get("/celery")
async def health_celery():
    try:
        result = celery_app.control.ping(timeout=2.0)
        if result:
            return {"status": "up", "workers": result}
        return {"status": "down", "detail": "no workers"}
    except TimeoutError:
        return {"status": "down", "detail": "timeout"}
    except Exception as e:
        return {"status": "down", "detail": str(e)}

