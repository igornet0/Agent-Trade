# модели для БД
from typing import Literal, List
from sqlalchemy import ForeignKey, Boolean, String, Integer, JSON
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
    