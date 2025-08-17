from fastapi import APIRouter, Response
from backend.app.monitoring.metrics import export_prometheus

router = APIRouter(prefix="/metrics", tags=["Metrics"])

@router.get("")
async def metrics():
    data, content_type = export_prometheus()
    return Response(content=data, media_type=content_type)

