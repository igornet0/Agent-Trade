from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
from datetime import datetime
from enum import Enum

class AgentType(str, Enum):
    """Типы агентов"""
    NEWS = "News"
    PRED_TIME = "Pred_time"
    TRADE_TIME = "Trade_time"
    RISK = "Risk"
    TRADE = "Trade"
    TRADE_AGGREGATOR = "Trade_aggregator"

class TrainRequest(BaseModel):
    """Схема для запроса на обучение"""
    name: str = Field(..., description="Название агента")
    type: AgentType = Field(..., description="Тип агента")
    timeframe: str = Field(default="5m", description="Временной интервал")
    features: List[str] = Field(default=[], description="Список признаков")
    coins: List[int] = Field(default=[], description="Список монет")
    train_data: Optional[Dict] = Field(default=None, description="Данные для обучения")

class EvaluateRequest(BaseModel):
    """Схема для запроса на оценку"""
    agent_id: int = Field(..., description="ID агента")
    coins: List[int] = Field(..., description="Список монет для оценки")
    timeframe: Optional[str] = Field(default="5m", description="Временной интервал")
    start: Optional[datetime] = Field(default=None, description="Начальная дата")
    end: Optional[datetime] = Field(default=None, description="Конечная дата")

class NewsTrainConfig(BaseModel):
    """Конфигурация для обучения News модели"""
    nlp_model: str = Field(default="bert-base-uncased", description="NLP модель")
    window_hours: int = Field(default=24, ge=1, le=168, description="Окно анализа в часах")
    decay_factor: float = Field(default=0.95, ge=0.1, le=1.0, description="Фактор затухания")
    correlation_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Порог корреляции")
    sources: List[str] = Field(default=["twitter", "reddit", "news"], description="Источники новостей")
    min_sources: int = Field(default=2, ge=1, description="Минимальное количество источников")
    force_recalculate: bool = Field(default=False, description="Принудительный пересчет")

class PredTimeTrainConfig(BaseModel):
    """Конфигурация для обучения Pred_time модели"""
    seq_len: int = Field(default=60, ge=10, le=500, description="Длина последовательности")
    pred_len: int = Field(default=1, ge=1, le=30, description="Длина предсказания")
    model_type: str = Field(default="lstm", description="Тип модели (lstm/gru/transformer)")
    hidden_size: Optional[int] = Field(default=128, ge=32, le=512, description="Размер скрытого слоя")
    num_layers: Optional[int] = Field(default=2, ge=1, le=6, description="Количество слоев")
    d_model: Optional[int] = Field(default=256, ge=64, le=1024, description="Размер модели (transformer)")
    n_heads: Optional[int] = Field(default=8, ge=1, le=16, description="Количество голов (transformer)")
    n_layers: Optional[int] = Field(default=4, ge=1, le=12, description="Количество слоев (transformer)")
    d_ff: Optional[int] = Field(default=1024, ge=256, le=4096, description="Размер FF слоя (transformer)")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout")
    batch_size: int = Field(default=32, ge=8, le=128, description="Размер батча")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.01, description="Скорость обучения")
    epochs: int = Field(default=100, ge=10, le=500, description="Количество эпох")
    patience: int = Field(default=20, ge=5, le=100, description="Терпение для early stopping")
    technical_indicators: List[str] = Field(default=["sma", "rsi", "macd"], description="Технические индикаторы")
    news_integration: bool = Field(default=True, description="Интеграция новостей")
    feature_scaling: bool = Field(default=True, description="Масштабирование признаков")
    val_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля валидации")
    test_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля тестирования")

class TradeTimeTrainConfig(BaseModel):
    """Конфигурация для обучения Trade_time модели"""
    model_type: str = Field(default="lightgbm", description="Тип модели (lightgbm/catboost/random_forest)")
    n_estimators: int = Field(default=100, ge=50, le=1000, description="Количество деревьев")
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.3, description="Скорость обучения")
    max_depth: int = Field(default=6, ge=3, le=15, description="Максимальная глубина")
    num_leaves: Optional[int] = Field(default=31, ge=10, le=100, description="Количество листьев (lightgbm)")
    depth: Optional[int] = Field(default=6, ge=3, le=15, description="Глубина (catboost)")
    threshold: float = Field(default=0.02, ge=0.01, le=0.1, description="Порог для сигналов")
    technical_indicators: List[str] = Field(default=["sma", "rsi", "macd", "bb"], description="Технические индикаторы")
    news_integration: bool = Field(default=True, description="Интеграция новостей")
    feature_scaling: bool = Field(default=True, description="Масштабирование признаков")
    val_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля валидации")
    test_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля тестирования")

class RiskTrainConfig(BaseModel):
    """Конфигурация для обучения Risk модели"""
    model_type: str = Field(default="xgboost", description="Тип модели (xgboost)")
    n_estimators: int = Field(default=100, ge=50, le=1000, description="Количество деревьев")
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.3, description="Скорость обучения")
    max_depth: int = Field(default=6, ge=3, le=15, description="Максимальная глубина")
    risk_weight: float = Field(default=0.6, ge=0.1, le=1.0, description="Вес риска в общей оценке")
    volume_weight: float = Field(default=0.4, ge=0.1, le=1.0, description="Вес объема в общей оценке")
    technical_indicators: List[str] = Field(default=["sma", "rsi", "macd", "bb", "atr"], description="Технические индикаторы")
    news_integration: bool = Field(default=True, description="Интеграция новостей")
    feature_scaling: bool = Field(default=True, description="Масштабирование признаков")
    val_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля валидации")
    test_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля тестирования")

class PredTimeModel(BaseModel):
    """Схема для модели Pred_time"""
    model_path: str
    model_name: str
    config: Dict
    metadata: Dict
    model_size_bytes: int
    created_at: str
    model_type: str

class PredTimePrediction(BaseModel):
    """Схема для предсказания Pred_time"""
    timestamp: str
    predicted_change: float
    confidence: Optional[float] = None

class TradeTimeModel(BaseModel):
    """Схема для модели Trade_time"""
    model_path: str
    model_name: str
    config: Dict
    metadata: Dict
    created_at: str
    model_type: str

class TradeTimePrediction(BaseModel):
    """Схема для предсказания Trade_time"""
    timestamp: str
    prediction: int  # -1: Sell, 0: Hold, 1: Buy
    signal: str  # SELL, HOLD, BUY
    probability: Optional[float] = None

class RiskModel(BaseModel):
    """Схема для модели Risk"""
    model_path: str
    model_name: str
    config: Dict
    metadata: Dict
    created_at: str
    model_type: str

class RiskPrediction(BaseModel):
    """Схема для предсказания Risk"""
    timestamp: str
    risk_score: float  # 0-1, где 1 - максимальный риск
    volume_score: float  # 0-1, где 1 - максимальный объем
    risk_level: str  # LOW, MEDIUM, HIGH
    volume_level: str  # LOW, MEDIUM, HIGH

class AgentResponse(BaseModel):
    """Схема для ответа агента"""
    id: int
    name: str
    type: AgentType
    status: str
    created_at: datetime
    updated_at: datetime

class TrainResponse(BaseModel):
    """Схема для ответа на обучение"""
    agent: AgentResponse
    task_id: str
    status: str

class EvaluateResponse(BaseModel):
    """Схема для ответа на оценку"""
    task_id: str
    status: str
    message: str

class TradeAggregatorTrainConfig(BaseModel):
    """Конфигурация для обучения Trade Aggregator модели"""
    mode: str = Field(default="rules", description="Режим работы (rules/ml)")
    weights: Dict[str, float] = Field(
        default={"pred_time": 0.4, "trade_time": 0.4, "risk": 0.2},
        description="Веса для агрегации сигналов"
    )
    thresholds: Dict[str, float] = Field(
        default={"buy_threshold": 0.6, "sell_threshold": 0.4, "hold_threshold": 0.3},
        description="Пороги для принятия решений"
    )
    risk_limits: Dict[str, float] = Field(
        default={"max_position_size": 0.1, "max_leverage": 3.0, "stop_loss_pct": 0.05, "take_profit_pct": 0.15},
        description="Лимиты риск-менеджмента"
    )
    portfolio: Dict[str, Any] = Field(
        default={"max_coins": 10, "rebalance_frequency": "1h", "correlation_threshold": 0.7},
        description="Настройки портфеля"
    )
    # ML параметры (если mode == 'ml')
    n_estimators: Optional[int] = Field(default=100, ge=50, le=1000, description="Количество деревьев")
    learning_rate: Optional[float] = Field(default=0.1, ge=0.01, le=0.3, description="Скорость обучения")
    max_depth: Optional[int] = Field(default=6, ge=3, le=15, description="Максимальная глубина")
    technical_indicators: List[str] = Field(default=["sma", "rsi", "macd", "bb"], description="Технические индикаторы")
    news_integration: bool = Field(default=True, description="Интеграция новостей")
    feature_scaling: bool = Field(default=True, description="Масштабирование признаков")
    val_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля валидации")
    test_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Доля тестирования")

class TradeAggregatorModel(BaseModel):
    """Модель Trade Aggregator"""
    id: int = Field(description="ID модели")
    agent_id: int = Field(description="ID агента")
    mode: str = Field(description="Режим работы")
    weights: Dict[str, float] = Field(description="Веса агрегации")
    thresholds: Dict[str, float] = Field(description="Пороги решений")
    risk_limits: Dict[str, float] = Field(description="Лимиты риска")
    portfolio_config: Dict[str, Any] = Field(description="Конфигурация портфеля")
    artifact_path: Optional[str] = Field(default=None, description="Путь к артефактам")
    created_at: datetime = Field(description="Дата создания")
    is_active: bool = Field(default=False, description="Активна ли модель")

class TradeAggregatorPrediction(BaseModel):
    """Предсказание Trade Aggregator"""
    decision: str = Field(description="Решение (buy/sell/hold)")
    confidence: float = Field(description="Уверенность (0-1)")
    position_size: float = Field(description="Размер позиции")
    stop_loss_pct: float = Field(description="Стоп-лосс в %")
    take_profit_pct: float = Field(description="Тейк-профит в %")
    max_leverage: float = Field(description="Максимальное плечо")
    signals: Dict[str, float] = Field(description="Сигналы от модулей")
    portfolio_metrics: Dict[str, Any] = Field(description="Метрики портфеля")
    metadata: Optional[Dict] = Field(default=None, description="Дополнительные метаданные")



