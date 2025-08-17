from typing import Tuple, Dict
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from core.database.models import User


async def orm_add_user(
    session: AsyncSession,
    login: str,
    hashed_password: str,
    email: str | None = None,
    user_telegram_id: int | None = None,
) -> User:
    query = select(User).where(User.login == login)
    result = await session.execute(query)
    if result.first() is None:
        session.add(
            User(
                login=login,
                password=hashed_password,
                user_telegram_id=user_telegram_id,
                email=email,
            )
        )
        await session.commit()
    else:
        return result.scalars().first()


async def orm_get_user_by_login(session: AsyncSession, response) -> Tuple[User, Dict[str, str]] | None:
    if not getattr(response, "login", None):
        return None
    query = select(User).where(User.login == response.login).options(joinedload(User.portfolio))
    result = await session.execute(query)
    return result.scalars().first()


async def orm_get_user_by_email(session: AsyncSession, response) -> Tuple[User, Dict[str, str]] | None:
    if not getattr(response, "email", None):
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


