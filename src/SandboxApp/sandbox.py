import asyncio

from typing import List, Dict, Generator, Any, Union, Literal

from src.Dataset import DatasetTimeseries
from src.train_models.transform_data import TimeSeriesTransform
from core.database import orm_get_coins, orm_get_timeseries_by_coin, orm_get_data_timeseries
from core.database.engine import db_helper, set_db_helper

from .boxs import Box, Exhange
from MMM.agents.agent_trade_time import AgentTradeTime
from MMM.mmm_ensemble import aggregate_trade_decision, build_final_order

import logging 

logger = logging.getLogger("Sandbox")

class Sandbox:

    type_box = {
        "Box": Box,
        "Exhange": Exhange
    }

    def __init__(self, data: List[DatasetTimeseries] = [], 
                 agents: List[Any] = [], 
                 db_use: bool = False):
        
        if db_use:
            self.data = asyncio.run(self._load_data_fron_db())
            self.agents = asyncio.run(self._load_agents_from_db())
        else:
            self.data = data
            self.agents = agents

        self._box = None

    @classmethod
    def create_box(cls, type_box: Literal["Box", "Exhange"], **kwargs) -> Union[Box, Exhange]:
        if cls.type_box.get(type_box) is None:
            raise ValueError(f"Unknown box type: {type_box}")
        
        box = cls.type_box[type_box](**kwargs)

        return box
    
    async def _load_data_fron_db(self):
        results = {}
        if not db_helper:
            await set_db_helper()
        async with db_helper.get_session() as session:
            coins = await orm_get_coins(session)
            for coin in coins:
                logger.info(f"Loading data for coin: {coin.name}")
                timeseries = await orm_get_timeseries_by_coin(session, coin)
                for ts in timeseries:
                    logger.info(f"Loading data for timeseries: {ts.id}")
                    result = await orm_get_data_timeseries(session, ts.id)
                    dt = DatasetTimeseries(result)
                    results.setdefault(coin.name, {})
                    results[coin.name][ts.timestamp] = dt

        return results

    async def _load_agents_from_db(self):
        # Заготовка: загрузка активных trade_time агентов по путям чекпойнтов
        # Потребуется ORM-функция для получения активных агентов с путями. Используем уже существующую orm_get_agents
        from core.database.orm.agents import orm_get_agents
        if not db_helper:
            await set_db_helper()
        loaded = []
        async with db_helper.get_session() as session:
            agents = await orm_get_agents(session, type_agent="AgentTradeTime")
            for a in agents or []:
                path = a.get("path_model")
                if not path:
                    continue
                try:
                    agent, *_ = AgentTradeTime._load_agent_from_checkpoint(path)
                    agent.set_mode("test")
                    loaded.append(agent)
                except Exception:
                    continue
        return loaded

    def start(self):
        # Простейший проход по данным и формирование решений (демо)
        if not self.data or not self.agents:
            raise RuntimeError("Sandbox: нет данных или агентов")
        # Здесь можно реализовать логику симуляции. Оставляем как заглушку, т.к. нужна стратегия и риск-агент
        return {"status": "not_implemented", "agents": len(self.agents)}

    def add_data(self, item):
        self.data.append(item)

    def get_data(self):
        return self.data

    def clear_data(self):
        self.data = []