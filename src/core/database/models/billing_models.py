# модели для БД (финансовые операции)
from sqlalchemy import ForeignKey, String, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base


class BalanceOperation(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))

    # deposit | withdraw
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)

    # Идемпотентность на уровне БД
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    processed: Mapped[bool] = mapped_column(Boolean, default=False)


