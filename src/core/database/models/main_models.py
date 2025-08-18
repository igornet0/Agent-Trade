# модели для БД
from typing import Literal, List
from sqlalchemy import DateTime, ForeignKey, Float, String, BigInteger, func, Integer, Boolean, UniqueConstraint, CheckConstraint, Column, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base
from datetime import datetime


class NewsCoin(Base):
    
    news_id: Mapped[int] = mapped_column(ForeignKey('newss.id'), primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'), primary_key=True)
    score: Mapped[Float] = mapped_column(Float, default=0)


class Coin(Base):
    __tablename__ = "coins"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    price_now: Mapped[Float] = mapped_column(Float, default=0)
    max_price_now: Mapped[Float] = mapped_column(Float, default=0)
    min_price_now: Mapped[Float] = mapped_column(Float, default=0)
    open_price_now: Mapped[Float] = mapped_column(Float, default=0)
    volume_now: Mapped[Float] = mapped_column(Float, default=0)
    price_change_percentage_24h: Mapped[Float] = mapped_column(Float, default=0, nullable=True)

    news_score_global: Mapped[Float] = mapped_column(Float, default=100)

    parsed: Mapped[bool] = mapped_column(Boolean, default=True)

    timeseries: Mapped[List['Timeseries']] = relationship(back_populates='coin')

    portfolio: Mapped[List['Portfolio']] = relationship(back_populates='coin')
    transaction: Mapped[List['Transaction']] = relationship(back_populates='coin')

    strategies: Mapped[List['Strategy']] = relationship(
        secondary="strategy_coins",
        back_populates='coins'
    )

    # Imported via string to avoid circular import; type is provided by Strategy_models.AgentTrain
    trains: Mapped[List["AgentTrain"]] = relationship("AgentTrain", secondary="train_coins", back_populates="coins", viewonly=False)

    news: Mapped[List['News']] = relationship(
        secondary="news_coins",
        back_populates='news_coin'
    )

    news_background: Mapped[List['NewsBackground']] = relationship(back_populates='coin')


class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(50), unique=True, nullable=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    user_telegram_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    balance: Mapped[Float] = mapped_column(Float, default=0)
    role: Mapped[str] = mapped_column(String(50), default="user")

    active: Mapped[bool] = mapped_column(Boolean, default=True)

    portfolio: Mapped[List['Portfolio']] = relationship(back_populates='user')



class Portfolio(Base):
    __tablename__ = "portfolio"
    __table_args__ = (
        UniqueConstraint('user_id', 'coin_id', name='uq_portfolio_user_coin'),
        CheckConstraint('amount >= 0', name='ck_portfolio_amount_non_negative'),
        {'extend_existing': True}
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))
    amount: Mapped[Float] = mapped_column(Float, default=0.0)
    price_avg: Mapped[Float] = mapped_column(Float, default=0.0)
    
    coin: Mapped['Coin'] = relationship(back_populates='portfolio')
    user: Mapped['User'] = relationship(back_populates='portfolio')
    

class Timeseries(Base):
    __tablename__ = "timeseries"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))  
    timestamp: Mapped[str] = mapped_column(String(50)) 
    path_dataset: Mapped[str] = mapped_column(String(100), unique=True)

    coin: Mapped['Coin'] = relationship(back_populates='timeseries')


class DataTimeseries(Base):
    __tablename__ = "data_timeseries"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timeseries_id: Mapped[int] = mapped_column(ForeignKey('timeseries.id'))  
    datetime: Mapped[DateTime] = mapped_column(DateTime, nullable=False) 
    open: Mapped[Float] = mapped_column(Float)
    max: Mapped[Float] = mapped_column(Float)
    min: Mapped[Float] = mapped_column(Float)
    close: Mapped[Float] = mapped_column(Float)
    volume: Mapped[Float] = mapped_column(Float)

class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(30), default="open")
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))

    type: Mapped[str] = mapped_column(String(20), nullable=False)
    amount_orig: Mapped[Float] = mapped_column(Float, nullable=False)
    amount: Mapped[Float] = mapped_column(Float, nullable=False)
    price: Mapped[Float] = mapped_column(Float, nullable=False)

    coin: Mapped['Coin'] = relationship(back_populates='transaction')
    user: Mapped['User'] = relationship(backref='transaction')

    def set_status(self, new_status: Literal["open", "cancel", "approve"]) -> None:

        assert new_status in ["open", "cancel", "approve"], "Invalid status"

        self.status = new_status

class TelegramChannel(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    chat_id: Mapped[str] = mapped_column(String(50), unique=True)
    parsed: Mapped[bool] = mapped_column(Boolean, default=True)


class NewsUrl(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(200), unique=True)
    a_pup: Mapped[Float] = mapped_column(Float, default=0.9)
    parsed: Mapped[bool] = mapped_column(Boolean, default=True)


class News(Base):
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(50), default="news")
    id_url: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    text: Mapped[str] = mapped_column(String(1000), nullable=False)
    date: Mapped[DateTime] = mapped_column(DateTime, default=func.now())

    news_coin: Mapped[List['Coin']] = relationship(
        'Coin', 
        secondary="news_coins",
        back_populates='news'
    )


class NewsHistoryCoin(Base):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id_news: Mapped[int] = mapped_column(ForeignKey('newss.id'))
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))
    score: Mapped[Float] = mapped_column(Float, default=0)
    news_score_global: Mapped[Integer] = mapped_column(Integer, default=100)

    # news: Mapped['News'] = relationship(back_populates='history_coins')
    # coin: Mapped['Coin'] = relationship(back_populates='history_coins')


class NewsBackground(Base):
    __tablename__ = "news_background"
    
    id = Column(Integer, primary_key=True, index=True)
    coin_id = Column(Integer, ForeignKey("coins.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    score = Column(Float, nullable=False)
    source_count = Column(Integer, nullable=False, default=0)
    sources_breakdown = Column(JSON, nullable=True)
    window_hours = Column(Integer, nullable=False, default=24)
    decay_factor = Column(Float, nullable=False, default=0.95)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    coin = relationship("Coin", back_populates="news_background")
    
    class Config:
        from_attributes = True


class Artifact(Base):
    """Таблица для хранения артефактов моделей"""
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    version = Column(String(50), nullable=False)
    path = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)  # model, config, scaler, metadata
    size_bytes = Column(BigInteger, nullable=True)
    checksum = Column(String(64), nullable=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="artifacts")
    
    __table_args__ = (
        UniqueConstraint('agent_id', 'version', 'type', name='uq_artifact_agent_version_type'),
    )
    
    class Config:
        from_attributes = True


class Backtest(Base):
    """Таблица для хранения результатов бэктестов"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=True)
    name = Column(String(200), nullable=False)
    config_json = Column(JSON, nullable=False)
    timeframe = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    coins = Column(JSON, nullable=False)  # список монет для бэктеста
    metrics_json = Column(JSON, nullable=True)
    artifacts = Column(JSON, nullable=True)  # пути к артефактам
    status = Column(String(20), default="running")  # running, completed, failed
    progress = Column(Float, default=0.0)
    error_message = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    pipeline = relationship("Pipeline", back_populates="backtests")
    
    class Config:
        from_attributes = True


class Pipeline(Base):
    """Таблица для хранения конфигураций пайплайнов"""
    __tablename__ = "pipelines"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(String(1000), nullable=True)
    config_json = Column(JSON, nullable=False)  # граф узлов и связей
    is_template = Column(Boolean, default=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    user = relationship("User", backref="pipelines")
    backtests = relationship("Backtest", back_populates="pipeline")
    
    class Config:
        from_attributes = True