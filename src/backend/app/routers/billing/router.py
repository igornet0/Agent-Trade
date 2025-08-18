from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.configuration import Server, verify_authorization
# from core.database.orm_query import orm_apply_balance_operation

router = APIRouter(
    prefix="/billing",
    tags=["Billing"],
    dependencies=[Depends(Server.http_bearer), Depends(verify_authorization)],
)


@router.post("/deposit")
async def deposit(
    amount: float,
    idempotency_key: str = Header(default=None, convert_underscores=False),
    user=Depends(verify_authorization),
    db: AsyncSession = Depends(Server.get_db),
):
    if not idempotency_key:
        raise HTTPException(status_code=400, detail="Missing Idempotency-Key header")
    op = await orm_apply_balance_operation(
        db, user_id=user.id, amount=amount, op_type="deposit", idempotency_key=idempotency_key
    )
    return {"operation_id": op.id, "processed": op.processed, "balance": user.balance + amount}


@router.post("/withdraw")
async def withdraw(
    amount: float,
    idempotency_key: str = Header(default=None, convert_underscores=False),
    user=Depends(verify_authorization),
    db: AsyncSession = Depends(Server.get_db),
):
    if not idempotency_key:
        raise HTTPException(status_code=400, detail="Missing Idempotency-Key header")
    op = await orm_apply_balance_operation(
        db, user_id=user.id, amount=amount, op_type="withdraw", idempotency_key=idempotency_key
    )
    return {"operation_id": op.id, "processed": op.processed}


