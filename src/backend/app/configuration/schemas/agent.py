from pydantic import BaseModel
from enum import Enum as PyEnum
import uuid
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, EmailStr
from datetime import datetime
from datetime import datetime
from typing import List

class AgentType(str, PyEnum):

    PREDTIME = 'AgentPredTime'
    TRADETIME = 'AgentTradeTime'
    NEWS = 'AgentNews'
    RISK = 'AgentRisk'
    TRADE_AGGREGATOR = 'AgentTradeAggregator'

class FeatureType(str, PyEnum):
    pass

class DataTrade(str, PyEnum):
    pass

class Argument(BaseModel):
    id: Optional[int] = 0
    name: Optional[str] = "Argument"
    type: Optional[str] = "int"
    volue: str | int | None = None

class FeatureTypeResponse(BaseModel):
    id: int
    name: str

    arguments: list[Argument]

class AgentTypeResponse(BaseModel):
    id: int
    name: str

class AgentTrade(BaseModel):
    id: int
    agent_id: int
    data: list

class AgentAction(BaseModel):
    id: int
    action: str
    loss: float

class AgentStrategy(BaseModel):
    id: int
    name: str

class AgentStata(BaseModel):
    id: int
    type: str
    loss: float

class FeatureParameters(BaseModel):
    # Динамический словарь для параметров индикатора
    # Пример: {"period": 14, "column": "close"}
    parameters: Dict[str, Any]

class Feature(BaseModel):
    id: int
    # feature_id: int
    name: Optional[str] = ""

    parameters: Dict[str, Any]

class AgentCreate(BaseModel):
    id: Optional[int] = None
    name: str
    type: AgentType
    status: Optional[str] = "open"

    path_model: Optional[str] = ""

    a_conficent: Optional[float] = 0.95
    active: Optional[bool] = True
    version: Optional[str] = "0.0.1"
    created: Optional[datetime] = None

    features: list[Feature] = []

class TrainData(BaseModel):
    class Config:
        from_attributes = True

    id: Optional[int] = None
    agent_id: Optional[int] = None
    status: Optional[str] = "start" 

    epochs: int
    epoch_now: Optional[int] = 0

    batch_size: int
    learning_rate: float
    weight_decay: float
    # Optional extra config bag for Stage 1 contract (DB support may be added later)
    extra_config: Dict[str, Any] | None = None

class AgentTrainResponse(BaseModel):
    class Config:
        from_attributes = True

    id: Optional[int] = None
    name: str
    type: AgentType
    status: Optional[str] = "open"
    timeframe: Optional[str] = "5m"

    RP_I: Optional[bool] = False

    path_model: Optional[str] = ""

    a_conficent: Optional[float] = 0.95
    active: Optional[bool] = True
    version: Optional[str] = "0.0.1"
    created: Optional[datetime] = None

    features: list[Feature] = []
    coins: list[int] = []

    train_data: Optional[TrainData] = None

class AgentResponse(BaseModel):
    class Config:
        from_attributes = True

    id: Optional[int] = None
    name: str
    type: AgentType
    status: Optional[str] = "open"
    timeframe: Optional[str] = "5m"
    RP_I: Optional[bool] = False

    path_model: Optional[str] = ""

    a_conficent: Optional[float] = 0.95
    active: Optional[bool] = True
    version: Optional[str] = "0.0.1"
    created: Optional[datetime] = None

    features: list[Feature] = []





class EvaluateRequest(BaseModel):
    class Config:
        from_attributes = True

    agent_id: int
    coins: List[int] = []
    timeframe: Optional[str] = "5m"
    start: Optional[datetime] = None
    end: Optional[datetime] = None

class EvaluateResponse(BaseModel):
    task_id: str
    detail: Optional[str] = None


# ---------- Training configs per module type ----------
class NewsTrainConfig(BaseModel):
    sources: List[str] = []  # e.g., ["twitter", "telegram", "coindesk"]
    nlp_model: str = "finbert"  # e.g., "bert", "finbert"
    features: List[str] = []  # extracted features list
    window_minutes: int = 240  # rolling window to aggregate influence
    influence_horizon_minutes: int = 180  # horizon for impact propagation


class PredTimeTrainConfig(BaseModel):
    model_name: str = "LSTM"  # e.g., LSTM, GRU, Informer, TimesNet
    seq_len: int = 96
    pred_len: int = 12
    indicators: List[str] = ["SMA", "RSI", "MACD"]
    use_news_background: bool = True


class TradeTimeTrainConfig(BaseModel):
    classifier: str = "LightGBM"  # e.g., LightGBM, CatBoost, Transformer
    target_scheme: str = "direction3"  # mapping to {buy/sell/hold}
    use_news_background: bool = False


class RiskTrainConfig(BaseModel):
    model_name: str = "XGBoost"  # heuristic+ML default
    features: List[str] = ["balance", "pnl", "leverage", "signals"]


class TradeAggregatorConfig(BaseModel):
    mode: str = "rules"  # rules | rl
    weights: Dict[str, float] = {"pred_time": 0.4, "trade_time": 0.4, "news": 0.1, "risk": 0.1}
    rl_enabled: bool = False


class TrainRequest(BaseModel):
    name: str
    type: AgentType
    timeframe: Optional[str] = "5m"
    coins: List[int] = []
    features: List[Feature] = []
    train_data: Optional[TrainData] = None
    # Flexible extra configuration; specific typed configs are optional
    extra_config: Dict[str, Any] | None = None
    news_config: Optional[NewsTrainConfig] = None
    pred_time_config: Optional[PredTimeTrainConfig] = None
    trade_time_config: Optional[TradeTimeTrainConfig] = None
    risk_config: Optional[RiskTrainConfig] = None
    trade_aggregator_config: Optional[TradeAggregatorConfig] = None
