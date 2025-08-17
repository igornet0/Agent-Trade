from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime


class PipelineNode(BaseModel):
    id: str
    type: str  # DataSource | Indicators | News | Pred_time | Trade_time | Risk | Trade | Metrics
    config: Dict[str, Any] = {}


class PipelineEdge(BaseModel):
    id: Optional[str] = None
    source: str
    target: str


class PipelineConfig(BaseModel):
    nodes: List[PipelineNode]
    edges: List[PipelineEdge] = []
    timeframe: Optional[str] = "5m"
    start: Optional[datetime] = None
    end: Optional[datetime] = None


