import asyncio

from typing import List, Dict, Generator, Any, Union, Literal

from Dataset import DatasetTimeseries
from train_models.transform_data import TimeSeriesTransform
from core.database import orm_get_coins, orm_get_timeseries_by_coin, orm_get_data_timeseries
from core.database.engine import db_helper, set_db_helper

from .boxs import Box, Exhange
from MMM.agents.agent_trade_time import AgentTradeTime
from MMM.agents.agent_pread_time import AgentPredTime
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
        # Загружаем активных агентов предикта и трейда из БД
        from core.database.orm.agents import orm_get_agents
        if not db_helper:
            await set_db_helper()
        loaded = {"pred": [], "trade": []}
        async with db_helper.get_session() as session:
            pred_agents = await orm_get_agents(session, type_agent="AgentPredTime")
            trade_agents = await orm_get_agents(session, type_agent="AgentTradeTime")
            for a in pred_agents or []:
                path = a.get("path_model")
                if not path:
                    continue
                try:
                    agent, *_ = AgentPredTime._load_agent_from_checkpoint(path)
                    agent.set_mode("test")
                    loaded["pred"].append(agent)
                except Exception:
                    continue
            for a in trade_agents or []:
                path = a.get("path_model")
                if not path:
                    continue
                try:
                    agent, *_ = AgentTradeTime._load_agent_from_checkpoint(path)
                    agent.set_mode("test")
                    loaded["trade"].append(agent)
                except Exception:
                    continue
        return loaded

    def start(self, coin: str | None = None, timeframe: str | None = None, max_steps: int = 100):
        # Минимальный демо-прогон: выбираем первую монету и таймфрейм, считаем решения ансамбля trade_time на основе pred_time
        if not self.data or not self.agents:
            raise RuntimeError("Sandbox: нет данных или агентов")

        # Берём первую монету и первый таймфрейм
        coin_map: Dict[str, Dict[str, DatasetTimeseries]] = self.data
        coin_names = list(coin_map.keys())
        if not coin_names:
            raise RuntimeError("Sandbox: нет данных монет")
        coin_name = coin if coin in coin_names else coin_names[0]
        tf_map = coin_map[coin_name]
        if not tf_map:
            raise RuntimeError("Sandbox: нет доступных таймфреймов")
        timeframe = timeframe if timeframe in tf_map else list(tf_map.keys())[0]
        dt: DatasetTimeseries = tf_map[timeframe]

        # Агенты
        pred_agents: List[Any] = self.agents.get("pred", [])
        trade_agents: List[Any] = self.agents.get("trade", [])
        if not pred_agents or not trade_agents:
            raise RuntimeError("Sandbox: нужны минимум 1 pred и 1 trade агент")

        seq_len = trade_agents[0].model_parameters.get("seq_len", 50)
        pred_len = trade_agents[0].model_parameters.get("pred_len", 5)

        loader = dt.get_time_line_loader(time_line_size=seq_len)

        import torch
        import numpy as np
        from MMM.agents.agent import Agent as _BaseAgent

        # Счётчики решений
        counts = {"hold": 0, "buy": 0, "sell": 0}
        steps = 0

        for window_df in loader:
            steps += 1
            # Подготовим время
            # Используем базовую подготовку времени из первого trade-агента
            t_agent: _BaseAgent = trade_agents[0]
            df_with_ind = t_agent.preprocess_data_for_model(window_df.copy())
            # Временные признаки
            column_time = t_agent.get_column_time()
            time_features = df_with_ind[column_time].values  # [seq_len, 5]

            # Основные фичи для trade-агента
            x_cols = t_agent.get_column_output()
            # Защита на случай отсутствия колонок (например, индикаторы не посчитались на коротком окне)
            try:
                x_arr = df_with_ind[x_cols].values
            except Exception:
                continue
            x = torch.as_tensor(x_arr, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, num_features]
            t = torch.as_tensor(time_features, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 5]

            # Предсказание цены с помощью ансамбля pred-агентов: усредняем
            preds = []
            for pa in pred_agents:
                # Готовим входы под pred-агента
                df_pred = pa.preprocess_data_for_model(window_df.copy())
                try:
                    x_pred_cols = pa.get_column_output()
                    x_pred_arr = df_pred[x_pred_cols].values
                except Exception:
                    continue
                x_pre = torch.as_tensor(x_pred_arr, dtype=torch.float32).unsqueeze(0)
                t_pre = torch.as_tensor(df_pred[pa.get_column_time()].values, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    yhat = pa.model(x_pre, t_pre)  # [1, pred_len]
                    if yhat.dim() == 1:
                        yhat = yhat.unsqueeze(0)
                    preds.append(yhat)
            if not preds:
                continue
            x_pred = torch.stack(preds, dim=0).mean(dim=0)  # [1, pred_len]

            # Предсказания trade-агентов (вероятности [1, pred_len, 3])
            trade_probs = []
            for ta in trade_agents:
                with torch.no_grad():
                    probs = ta.model(x, x_pred, t)
                    trade_probs.append(probs)

            # Агрегируем в финальное решение по батчу
            agg = aggregate_trade_decision(trade_probs, reduce="mean")  # [1, 3]
            decision = int(agg.argmax(dim=-1).item())
            if decision == 0:
                counts["hold"] += 1
            elif decision == 1:
                counts["buy"] += 1
            else:
                counts["sell"] += 1

            # ограничим демо-прогон по параметру max_steps
            if steps >= max_steps:
                break

        return {
            "coin": coin_name,
            "timeframe": timeframe,
            "steps": steps,
            "decisions": counts,
        }

    def add_data(self, item):
        self.data.append(item)

    def get_data(self):
        return self.data

    def clear_data(self):
        self.data = []