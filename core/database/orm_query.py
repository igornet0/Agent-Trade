# файл для query запросов
from typing import Tuple, Dict, Literal
from fastapi import HTTPException
from datetime import datetime
from sqlalchemy import select, update, delete, desc, asc, Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from core import data_manager
from core.database.models import (User, Coin, Timeseries, 
                                  DataTimeseries, Transaction, Portfolio, 
                                  AgentType, ModelType,
                                  Feature, FeatureArgument,
                                  News, NewsCoin, NewsHistoryCoin,
                                  ML_Model, ModelAction, StatisticModel,
                                  Strategy, StrategyAgent, AgentTrain,
                                  StrategyCoin, TrainCoin, BalanceOperation)
from core.database.orm import *  # re-export domain-specific ORMs

##################### Добавляем юзера в БД #####################################

async def orm_add_user(
        session: AsyncSession,
        login: str,
        hashed_password: str,
        email: str | None = None,
        user_telegram_id: int | None = None
) -> User:
    
    query = select(User).where(User.login == login)
    result = await session.execute(query)

    if result.first() is None:
        session.add(
            User(login=login,
                 password=hashed_password,
                 user_telegram_id=user_telegram_id,
                 email=email)
        )
        await session.commit()

    else:
        return result.scalars().first()
    
async def orm_get_user_by_login(session: AsyncSession, response) -> Tuple[User, Dict[str, str]] | None:
    if not response.login:
        return None
    
    query = select(User).where(User.login == response.login).options(joinedload(User.portfolio))
     
    result = await session.execute(query)

    return result.scalars().first()

async def orm_get_user_by_email(session: AsyncSession, response) -> Tuple[User, Dict[str, str]] | None:
    if not response.email:
        return None
    
    query = select(User).where(User.email == response.email)

    result = await session.execute(query)

    return result.scalars().first()

async def orm_update_user_place(session: AsyncSession, user_id: int, place_id: int):
    query = update(User).where(User.user_id == user_id).values(place=place_id)
    await session.execute(query)
    await session.commit()

async def orm_update_user_phone(session: AsyncSession, user_id: int, phone: str):
    query = update(User).where(User.user_id == user_id).values(phone=phone)
    await session.execute(query)
    await session.commit()

async def orm_get_user_balance(session: AsyncSession, user_id: int) -> float:
    query = select(User).where(User.user_id == user_id)
    result = await session.execute(query)
    return result.scalars().first().balance

async def orm_remove_user_balance(session: AsyncSession, user_id: int, amount: float):
    new_balance = await orm_get_user_balance(session, user_id) - amount

    if new_balance < 0:
        raise ValueError("Balance cannot be negative")
    
    query = update(User).where(User.user_id == user_id).values(balance=new_balance)

    await session.execute(query)
    await session.commit()

async def orm_add_user_balance(session: AsyncSession, user_id: int, amount: float):
    new_balance = await orm_get_user_balance(session, user_id) + amount

    query = update(User).where(User.user_id == user_id).values(balance=new_balance)

    await session.execute(query)
    await session.commit()

##################### Баланс: идемпотентные операции #####################################

async def orm_apply_balance_operation(
    session: AsyncSession,
    user_id: int,
    amount: float,
    op_type: Literal["deposit", "withdraw"],
    idempotency_key: str,
):
    # Проверяем, не обработана ли операция
    exists_q = select(BalanceOperation).where(BalanceOperation.idempotency_key == idempotency_key)
    exists_res = await session.execute(exists_q)
    existing = exists_res.scalars().first()
    if existing:
        return existing

    # Вставляем запись операции (уникальность ключа защитит от дубликатов)
    op = BalanceOperation(user_id=user_id, amount=amount, type=op_type, idempotency_key=idempotency_key)
    session.add(op)
    await session.flush()

    # Применяем изменение баланса
    current_balance = await orm_get_user_balance(session, user_id)
    if op_type == "withdraw":
        if current_balance < amount:
            raise HTTPException(status_code=400, detail="Insufficient funds")
        new_balance = current_balance - amount
    else:
        new_balance = current_balance + amount

    upd = update(User).where(User.user_id == user_id).values(balance=new_balance)
    await session.execute(upd)

    op.processed = True
    await session.commit()
    await session.refresh(op)
    return op

##################### Фичи #####################################

async def orm_add_feature(session: AsyncSession, feature_name: str):
    session.add(Feature(name=feature_name))
    await session.commit()

async def orm_get_feature_by_name(session: AsyncSession, feature_name: str) -> Feature:
    query = select(Feature).where(Feature.name == feature_name)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_feature_by_id(session: AsyncSession, feature_id: int) -> Feature:
    query = select(Feature).where(Feature.id == feature_id)
    result = await session.execute(query)
    return result.scalars().first()

async def orm_get_features(session: AsyncSession) -> list[Feature]:
    query = select(Feature)
    query = query.options(joinedload(Feature.arguments))

    result = await session.execute(query)

    return result.unique().scalars().all()

async def orm_add_features_argument(session: AsyncSession, arguments: list[FeatureArgument]):
    session.add_all(arguments)
    await session.commit()

async def orm_add_feature_argument(session: AsyncSession, feature_id: int, argument_name: str, type_argument: str):
    session.add(FeatureArgument(feature_id=feature_id, 
                                name=argument_name,
                                type=type_argument))
    await session.commit()

async def orm_get_feature_argument(session: AsyncSession, feature_id: int) -> list[FeatureArgument]:
    query = select(FeatureArgument).where(FeatureArgument.feature_id == feature_id)
    result = await session.execute(query)
    return result.scalars().all()

##################### Portfolio #####################################

async def orm_get_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int) -> Portfolio:
    query = select(Portfolio).where(Portfolio.user_id == user_id, Portfolio.coin_id == coin_id).options(selectinload(Portfolio.coin))
    result = await session.execute(query)

    coin_potfolio = result.scalars().first()

    if not coin_potfolio:
        return None

    return coin_potfolio

async def orm_add_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int, amount: float, do_commit: bool = True):
    coin = await orm_get_coin_portfolio(session, user_id, coin_id)

    if coin:
        return await orm_update_amount_coin_portfolio(session, user_id, coin_id, coin[1] + amount)
    
    session.add(Portfolio(user_id=user_id, coin_id=coin_id, amount=amount))
    if do_commit:
        await session.commit()

async def orm_get_coins_portfolio(session: AsyncSession, user_id: int) -> Dict[Coin, float]:
    query = select(Portfolio).where(Portfolio.user_id == user_id).options(selectinload(Portfolio.coin))
    result = await session.execute(query)
    new_coins = {}
    coins_portfolio = result.scalars().all()

    for coin in coins_portfolio:
        new_coins[coin.coin] = coin.amount

    return new_coins

async def orm_delete_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int):
    query = delete(Portfolio).where(Portfolio.user_id == user_id, Portfolio.coin_id == coin_id)
    await session.execute(query)
    await session.commit()

async def orm_update_amount_coin_portfolio(session: AsyncSession, user_id: int, coin_id: int, amount: float):
    if amount == 0:
        return await orm_delete_coin_portfolio(session, user_id, coin_id)
    
    query = update(Portfolio).where(Portfolio.user_id == user_id, Portfolio.coin_id == coin_id).values(amount=amount)
    await session.execute(query)
    await session.commit()

##################### Transactions #####################################

async def orm_add_transaction(session: AsyncSession, user_id: int, coin_id: int, type_order: Literal["buy", "sell"], amount: float, price: float, do_commit: bool = True):
    transaction = Transaction(user_id=user_id, 
                              coin_id=coin_id, 
                              type=type_order,
                              amount=amount, 
                              price=price)
    session.add(transaction)
    if do_commit:
        await session.commit()

async def orm_get_transactions_by_id(session: AsyncSession, transaction_id: int, status: str = None) -> Transaction:
    query = select(Transaction).where(Transaction.id == transaction_id)
    if status:
        if "!" in status:
            status = status.replace("!", "")
            query = query.where(Transaction.status != status)
        else:
            query = query.where(Transaction.status == status)

    query = query.options(selectinload(Transaction.coin))
    query = query.options(selectinload(Transaction.user))
    result = await session.execute(query)

    return result.scalars().first()

async def orm_get_user_transactions(session: AsyncSession, user_id: int, status: str = None, type_order: Literal["buy", "sell"] = None) -> list[Transaction]:
    query = select(Transaction).where(Transaction.user_id == user_id)

    if status:
        query = query.where(Transaction.status == status)

    if type_order:
        if not type_order in ["buy", "sell"]:
            raise ValueError("type_order must be 'buy' or 'sell'")
        
        query = query.where(Transaction.type == type_order)
        
    result = await session.execute(query)

    return result.scalars().all()

async def orm_get_coin_transactions(session: AsyncSession, coin_id: int, status: str = None, type_order: Literal["buy", "sell"] = None) -> list[Transaction]:
    query = select(Transaction).where(Transaction.coin_id == coin_id)

    if status:
        query = query.where(Transaction.status == status)

    if type_order:
        if not type_order in ["buy", "sell"]:
            raise ValueError("type_order must be 'buy' or 'sell'")
        
        query = query.where(Transaction.type == type_order)

    result = await session.execute(query)
    return result.scalars().all()

async def orm_get_user_coin_transactions(session: AsyncSession, user_id: int, coin_id: int, status: str = None, type_order: Literal["buy", "sell"] = None) -> Dict[Coin, Dict[str, float]]:
    query = select(Transaction).where(Transaction.user_id == user_id, Transaction.coin_id == coin_id)
    
    if status:
        query = query.where(Transaction.status == status)
    
    if type_order:
        if not type_order in ["buy", "sell"]:
            raise ValueError("type_order must be 'buy' or 'sell'")
        
        query = query.where(Transaction.type == type_order)

    query = query.options(selectinload(Transaction.coin))

    result = await session.execute(query)

    new_coins = {}
    coins_portfolio = result.scalars().all()

    for coin in coins_portfolio:
        new_coins[coin.coin] = {"id":coin.id, "amount": coin.amount, "price": coin.price}

    return new_coins

async def orm_update_transaction_status(session: AsyncSession, transaction_id: int, status: Literal["open", "approve", "close", "cancel"]):
    query = select(Transaction).where(Transaction.id == transaction_id)
    result = await session.execute(query)

    transaction = result.scalars().first()
    transaction.set_status(status)

    await session.commit()

async def orm_update_transaction_amount(session: AsyncSession, transaction_id: int, amount: float):
    if amount == 0:
        return await orm_update_transaction_status(session, transaction_id, status="approve")
    
    query = update(Transaction).where(Transaction.id == transaction_id).values(amount=amount)
    await session.execute(query)
    await session.commit()

async def orm_delete_transaction(session: AsyncSession, transaction_id: int):
    query = delete(Transaction).where(Transaction.id == transaction_id)
    await session.execute(query)
    await session.commit()