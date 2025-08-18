from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm.market import orm_get_coins
from core.database.orm.agents import orm_get_agents
from core.database.orm.data import orm_get_models_list as orm_get_models
from core.database.models import Strategy, StrategyAgent, StrategyCoin
from core.database.orm import (
    orm_list_strategies_for_user,
    orm_get_strategy_detail,
    orm_delete_strategy,
    validate_coins_exist,
    validate_agents_exist,
    is_strategy_name_taken,
)

from backend.app.configuration import (
    Server,
    verify_authorization,
)
from backend.app.configuration.schemas.strategy import (
    StrategyResponse,
    StrategyModelsUpdate,
)
from backend.app.configuration.schemas.strategy import StrategyCreate

# Инициализация роутера
router = APIRouter(
    prefix="/strategy",
    dependencies=[Depends(Server.http_bearer), Depends(verify_authorization)],
    tags=["Strategy"]
)
    
# @router.get("/get_agents/", response_model=list[AgentResponse])
# async def get_agents(db: AsyncSession = Depends(Server.get_db)):
#     try:
#         agents = await orm_get_agents(db, active=True)
        
#         if not agents:
#             raise HTTPException(status_code=404, detail="No agents found")
        
#         agents = sorted(agents, key=lambda x: x.version, reverse=True)

#         return agents 
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")
    
# @router.get("/get_models/", response_model=list[ModelResponse])
# async def get_agents(db: AsyncSession = Depends(Server.get_db)):
#     try:
#         models = await orm_get_agents(db, active=True)
        
#         if not models:
#             raise HTTPException(status_code=404, detail="No agents found")
        
#         models = sorted(models, key=lambda x: int(x.version.replace(".", "")), reverse=True)

#         return models 
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")

@router.post("/create", response_model=StrategyResponse)
async def create_strategy(payload: StrategyCreate,
                          user=Depends(verify_authorization),
                          db: AsyncSession = Depends(Server.get_db)):
    try:
        # Validate uniqueness and references
        if await is_strategy_name_taken(db, user.id, payload.name):
            raise HTTPException(status_code=400, detail="Strategy name already exists")
        missing_coins = await validate_coins_exist(db, payload.coins)
        if missing_coins:
            raise HTTPException(status_code=400, detail=f"Unknown coins: {missing_coins}")
        missing_agents = await validate_agents_exist(db, payload.agents)
        if missing_agents:
            raise HTTPException(status_code=400, detail=f"Unknown agents: {missing_agents}")
        st = Strategy(name=payload.name, user_id=user.id, type=payload.type,
                      model_risk_id=payload.model_risk_id,
                      model_order_id=payload.model_order_id,
                      risk=payload.risk, reward=payload.reward)
        db.add(st)
        await db.flush()
        for coin_id in payload.coins:
            db.add(StrategyCoin(strategy_id=st.id, coin_id=coin_id))
        for agent_id in payload.agents:
            db.add(StrategyAgent(strategy_id=st.id, agent_id=agent_id))
        await db.commit()
        return StrategyResponse(id=st.id, name=st.name, type=st.type,
                                risk=st.risk, reward=st.reward,
                                coins=payload.coins, agents=payload.agents,
                                model_risk_id=st.model_risk_id,
                                model_order_id=st.model_order_id)
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Strategy create failed: {str(e)}")


@router.get("/list", response_model=list[StrategyResponse])
async def list_strategies(user=Depends(verify_authorization),
                          db: AsyncSession = Depends(Server.get_db)):
    strategies = await orm_list_strategies_for_user(db, user.id)
    responses: list[StrategyResponse] = []
    for st in strategies:
        coins_rows = (await db.execute(StrategyCoin.__table__.select().where(StrategyCoin.strategy_id==st.id))).fetchall()
        agents_rows = (await db.execute(StrategyAgent.__table__.select().where(StrategyAgent.strategy_id==st.id))).fetchall()
        responses.append(StrategyResponse(
            id=st.id, name=st.name, type=st.type,
            risk=st.risk, reward=st.reward,
            coins=[r.coin_id for r in coins_rows],
            agents=[r.agent_id for r in agents_rows],
            model_risk_id=st.model_risk_id,
            model_order_id=st.model_order_id,
        ))
    return responses


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: int,
                       user=Depends(verify_authorization),
                       db: AsyncSession = Depends(Server.get_db)):
    st = await orm_get_strategy_detail(db, strategy_id, user.id)
    if not st:
        raise HTTPException(status_code=404, detail="Strategy not found")
    coins_rows = (await db.execute(StrategyCoin.__table__.select().where(StrategyCoin.strategy_id==st.id))).fetchall()
    agents_rows = (await db.execute(StrategyAgent.__table__.select().where(StrategyAgent.strategy_id==st.id))).fetchall()
    return StrategyResponse(
        id=st.id, name=st.name, type=st.type,
        risk=st.risk, reward=st.reward,
        coins=[r.coin_id for r in coins_rows],
        agents=[r.agent_id for r in agents_rows],
        model_risk_id=st.model_risk_id,
        model_order_id=st.model_order_id,
    )


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: int,
                          user=Depends(verify_authorization),
                          db: AsyncSession = Depends(Server.get_db)):
    ok = await orm_delete_strategy(db, strategy_id, user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"message": "Strategy deleted"}

@router.post("/{strategy_id}/models", response_model=StrategyResponse)
async def update_strategy_models(strategy_id: int,
                                 payload: StrategyModelsUpdate,
                                 user=Depends(verify_authorization),
                                 db: AsyncSession = Depends(Server.get_db)):
    st = await db.get(Strategy, strategy_id)
    if not st or st.user_id != user.id:
        raise HTTPException(status_code=404, detail="Strategy not found")
    try:
        st.model_risk_id = payload.model_risk_id
        st.model_order_id = payload.model_order_id
        await db.commit()
        coins_rows = (await db.execute(StrategyCoin.__table__.select().where(StrategyCoin.strategy_id==st.id))).fetchall()
        agents_rows = (await db.execute(StrategyAgent.__table__.select().where(StrategyAgent.strategy_id==st.id))).fetchall()
        return StrategyResponse(id=st.id, name=st.name, type=st.type,
                                risk=st.risk, reward=st.reward,
                                coins=[r.coin_id for r in coins_rows],
                                agents=[r.agent_id for r in agents_rows],
                                model_risk_id=st.model_risk_id,
                                model_order_id=st.model_order_id)
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Strategy update failed: {str(e)}")

# @router.post("/update_order_amount/", response_model=OrderResponse)
# async def cancel_order(order_data: OrderUpdateAmount, 
#                        user: User = Depends(verify_authorization), 
#                        db: AsyncSession = Depends(Server.get_db)):
#     try:
#         order: Transaction = await orm_get_transactions_by_id(db, order_data.id,
#                                                               status="open")

#         if not order:
#             raise HTTPException(status_code=404, detail="Order not found")
        
#         if user.role != "admin" and order.user_id != user.id:
#             raise HTTPException(status_code=403, detail="You do not have permission to cancel this order")
        
#         if order.amount < order_data.amount:
#             raise HTTPException(status_code=400, detail="Insufficient amount in order")
        
#         order.amount -= order_data.amount

#         coin_portfolio = await orm_get_coin_portfolio(db, user.id, order.coin_id)

#         if not coin_portfolio:

#             coin_portfolio = Portfolio(
#                 user_id=user.id,
#                 coin_id=order.coin_id,
#                 amount=0,
#                 price_avg=order.price,  # Можно установить цену, если нужно 224
#             )
#             db.add(coin_portfolio)

#         if order.type == OrderType.BUY:
#             coin_portfolio.price_avg = (coin_portfolio.price_avg * coin_portfolio.amount +
#                                     order.price * order_data.amount) / \
#                                     (coin_portfolio.amount + order_data.amount)
#             coin_portfolio.amount += order_data.amount

#         else:  # OrderType.SELL
#             if coin_portfolio.amount + order.amount != 0:
#                 coin_portfolio.price_avg = (coin_portfolio.price_avg * (coin_portfolio.amount + order.amount_orig) -
#                                 order.price * order_data.amount) / \
#                                 (coin_portfolio.amount + order.amount)
            
#             user.balance += order.price * order_data.amount
            
#         if order.amount == 0:
#             order.set_status(new_status="approve")
#             if coin_portfolio.amount == 0:
#                 await db.delete(coin_portfolio)  # Удаляем портфель, если активов нет
#                 coin_portfolio = None

#         await db.commit()  # Фиксируем изменения
#         await db.refresh(order)  # Обновляем объект из БД
#         if coin_portfolio:
#             await db.refresh(coin_portfolio)  # Обновляем объект из БД

#         return order
        
#     except Exception as e:
#         await db.rollback()
#         raise HTTPException(status_code=500, detail=f"Order Update failed: {str(e)}")

# @router.post("/cancel_order/{id}", response_model=OrderResponse)
# async def cancel_order(order: OrderCancel, 
#                        user: User = Depends(verify_authorization), 
#                        db: AsyncSession = Depends(Server.get_db)):
#     try:
#         order: Transaction = await orm_get_transactions_by_id(db, order.id)

#         if not order:
#             raise HTTPException(status_code=404, detail="Order not found")
        
#         if order.status == "cancel":
#             raise HTTPException(status_code=400, detail="Order already cancelled")
#         elif order.status == "approve":
#             raise HTTPException(status_code=400, detail="Order already approved")
        
#         if user.role != "admin" and order.user_id != user.id:
#             raise HTTPException(status_code=403, detail="You do not have permission to cancel this order")
        
#         if order.type == OrderType.BUY:
#             total_cost = order.amount * order.price
#             user.balance += total_cost
#         else:  # OrderType.SELL
#             # Ищем актив в портфеле
#             portfolio_item = await orm_get_coin_portfolio(db, user.id, order.coin_id)

#             if not portfolio_item:
#                 raise HTTPException(status_code=400, detail="Insufficient assets")
            
#             portfolio_item.amount += order.amount

#         order.set_status(new_status="cancel")
#         await db.commit()  # Фиксируем изменения
#         await db.refresh(order)  # Обновляем объект из БД

#         return order
    
#     except Exception as e:
#         await db.rollback()  # Откатываем при ошибке
#         raise HTTPException(status_code=500, detail=f"Order cancellation failed: {str(e)}")
    
# @router.get("/get_orders/", response_model=list[OrderResponse])
# async def get_orders(user: User = Depends(verify_authorization), 
#                      db: AsyncSession = Depends(Server.get_db)):
#     try:
        
#         orders = await orm_get_user_transactions(db, user.id)
        
#         if not orders:
#             raise HTTPException(status_code=404, detail="No orders found for this user")
        
#         orders = sorted(orders, key=lambda x: x.created, reverse=True)

#         return orders 
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve orders: {str(e)}")
    
