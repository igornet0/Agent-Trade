# модели для БД
from typing import Literal, List
from sqlalchemy import DateTime, ForeignKey, Float, String, BigInteger, func, Integer, Boolean, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base


class NewsCoin(Base):
    
    news_id: Mapped[int] = mapped_column(ForeignKey('newss.id'), primary_key=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'), primary_key=True)
    score: Mapped[Float] = mapped_column(Float, default=0)


class Coin(Base):

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


class User(Base):

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

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))
    amount: Mapped[Float] = mapped_column(Float, default=0.0)
    price_avg: Mapped[Float] = mapped_column(Float, default=0.0)
    
    coin: Mapped['Coin'] = relationship(back_populates='portfolio')
    user: Mapped['User'] = relationship(back_populates='portfolio')

    __table_args__ = (
        UniqueConstraint('user_id', 'coin_id', name='uq_portfolio_user_coin'),
        CheckConstraint('amount >= 0', name='ck_portfolio_amount_non_negative'),
    )
    

class Timeseries(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    coin_id: Mapped[int] = mapped_column(ForeignKey('coins.id'))  
    timestamp: Mapped[str] = mapped_column(String(50)) 
    path_dataset: Mapped[str] = mapped_column(String(100), unique=True)

    coin: Mapped['Coin'] = relationship(back_populates='timeseries')


class DataTimeseries(Base):

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timeseries_id: Mapped[int] = mapped_column(ForeignKey('timeseries.id'))  
    datetime: Mapped[DateTime] = mapped_column(DateTime, nullable=False) 
    open: Mapped[Float] = mapped_column(Float)
    max: Mapped[Float] = mapped_column(Float)
    min: Mapped[Float] = mapped_column(Float)
    close: Mapped[Float] = mapped_column(Float)
    volume: Mapped[Float] = mapped_column(Float)

class Transaction(Base):

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