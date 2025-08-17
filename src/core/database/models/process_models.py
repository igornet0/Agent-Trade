# модели для БД
from typing import Literal, List
from sqlalchemy import ForeignKey, Boolean, String, Integer, JSON, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base

class Processe(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(20), unique=True)
    data: Mapped[str] = mapped_column(String(20))
    is_started: Mapped[bool] = mapped_column(Boolean, default=False)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)


class Pipeline(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey('users.id'), nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    
class Backtest(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pipeline_id: Mapped[int | None] = mapped_column(ForeignKey('pipelines.id'), nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(16), nullable=True)
    start: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    end: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default={})
    artifacts: Mapped[dict] = mapped_column(JSON, nullable=False, default={})
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
