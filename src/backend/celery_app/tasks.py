from .create_app import celery_app
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
import math
import os
from uuid import uuid4
from sqlalchemy import select

from core.database import db_helper
from core.database.orm.market import (
    orm_get_timeseries_by_coin as market_get_ts,
    orm_get_data_timeseries as market_get_data,
)
from core.utils.metrics import (
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    aggregate_returns_equal_weight,
)
from core.database.models.main_models import NewsHistoryCoin, News
from core.database.models.process_models import Backtest as BacktestModel
from core.database.orm_query import (
    orm_get_agent_by_id,
    orm_get_train_agent,
    orm_get_timeseries_by_coin,
    orm_get_data_timeseries,
)
from backend.Dataset.loader import LoaderTimeLine
from backend.train_models.loader import Loader

logger = logging.getLogger("celery.train")


@celery_app.task(bind=True)
def train_model_task(self, agent_id: int):
    """Kick off training for a specific Agent and persist artifacts/metrics.

    This task:
    - Loads agent config via ORM
    - Builds Loader/AgentManager
    - Trains and updates progress
    - Saves checkpoints/metrics using agent.save_model/save_json
    """

    async def _run():
        async with db_helper.get_session() as session:  # type: AsyncSession
            agent = await orm_get_agent_by_id(session, agent_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return {"status": "error", "detail": "agent not found"}

            # Fetch training config and target coins
            trains = await orm_get_train_agent(session, agent_id)
            train = trains[0] if isinstance(trains, list) and trains else None

            # Build training components (reusing existing training code)
            loader = Loader(agent_type=agent.type, model_type="MMM")
            agent_manager = loader.load_model(count_agents=1)
            if agent_manager is None:
                return {"status": "error", "detail": "agent manager not built"}

            # Prepare historical loaders for specified coins/timeframe
            loaders: list[LoaderTimeLine] = []
            timeframe = getattr(agent, "timeframe", "5m")
            seq_len = getattr(agent_manager.get_agents(), "model_parameters", {}).get("seq_len", 50)
            pred_len = getattr(agent_manager.get_agents(), "model_parameters", {}).get("pred_len", 5)
            window = seq_len + pred_len

            if train and getattr(train, "coins", None):
                coin_ids = [coin.id for coin in train.coins]
            else:
                coin_ids = []

            async def build_dataset_for_coin(coin_id: int):
                ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe=timeframe)
                if not ts:
                    return
                data_rows = await orm_get_data_timeseries(session, ts.id)
                # Sort ascending by datetime
                data_rows = sorted(data_rows, key=lambda r: r.datetime)
                for row in data_rows:
                    yield {
                        "datetime": row.datetime,
                        "open": row.open,
                        "max": row.max,
                        "min": row.min,
                        "close": row.close,
                        "volume": row.volume,
                    }

            for coin_id in coin_ids:
                dataset_async_gen = build_dataset_for_coin(coin_id)

                # Bridge async generator to sync iterable expected by LoaderTimeLine
                items = []
                async for item in dataset_async_gen:
                    items.append(item)
                if items:
                    loaders.append(LoaderTimeLine(dataset=items, time_line_size=window, timetravel=timeframe))

            # Report starting state
            self.update_state(state='PROGRESS', meta={'progress': 0})

            # Train with progress callback
            try:
                def on_epoch(**kw):
                    # Update task state
                    self.update_state(state='PROGRESS', meta={
                        'progress': kw.get('progress', 0),
                        'epoch': kw.get('epoch'),
                        'avg_loss': kw.get('avg_loss'),
                        'best_loss': kw.get('best_loss'),
                        'lr': kw.get('lr'),
                    })
                    # Persist training metrics
                    async def _update_train():
                        async with db_helper.get_session() as s2:
                            trains_cur = await orm_get_train_agent(s2, agent_id)
                            tr = trains_cur[0] if isinstance(trains_cur, list) and trains_cur else None
                            if tr:
                                tr.epoch_now = kw.get('epoch') or tr.epoch_now
                                tr.loss_now = kw.get('avg_loss') or tr.loss_now
                                tr.loss_avg = kw.get('best_loss') or tr.loss_avg
                                tr.status = 'train'
                                await s2.commit()
                    try:
                        asyncio.create_task(_update_train())
                    except RuntimeError:
                        # If not in an event loop, run synchronously
                        asyncio.run(_update_train())

                loader.train_model(loaders=loaders, agent_manager=agent_manager, mixed=True, on_epoch=on_epoch)
                self.update_state(state='PROGRESS', meta={'progress': 100})
                # Finalize status
                if train:
                    train.status = 'stop'
                    await session.commit()
            except Exception as e:
                logger.exception("Training failed")
                return {"status": "error", "detail": str(e)}

            return {"status": "success", "agent_id": agent_id}

    return asyncio.run(_run())


@celery_app.task(bind=True)
def evaluate_model_task(self, agent_id: int, coins: list[int], timeframe: str = "5m",
                        start: str | None = None, end: str | None = None):
    """Evaluate trained agent offline on a selected time range.

    Returns simple metrics dict; progress updates via Celery state.
    """

    async def _run():
        async with db_helper.get_session() as session:  # type: AsyncSession
            agent = await orm_get_agent_by_id(session, agent_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return {"status": "error", "detail": "agent not found"}

            loader = Loader(agent_type=agent.type, model_type="MMM")
            agent_manager = loader.load_model(count_agents=1)
            if agent_manager is None:
                return {"status": "error", "detail": "agent manager not built"}

            # Load datasets
            def parse_dt(dt: str | None):
                if not dt:
                    return None
                try:
                    return datetime.fromisoformat(dt)
                except Exception:
                    return None

            dt_start = parse_dt(start)
            dt_end = parse_dt(end)

            total = max(len(coins), 1)
            processed = 0
            metrics = {
                "coins": [],
                "samples": 0,
                "avg_loss": None,
            }

            for coin_id in coins:
                ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe=timeframe)
                if not ts:
                    continue
                data_rows = await orm_get_data_timeseries(session, ts.id)
                # sort and clip by time range
                rows = sorted(data_rows, key=lambda r: r.datetime)
                if dt_start:
                    rows = [r for r in rows if r.datetime >= dt_start]
                if dt_end:
                    rows = [r for r in rows if r.datetime <= dt_end]
                items = [
                    {
                        "datetime": r.datetime,
                        "open": r.open,
                        "max": r.max,
                        "min": r.min,
                        "close": r.close,
                        "volume": r.volume,
                    }
                    for r in rows
                ]
                if not items:
                    continue

                # Simple evaluation using loader utilities
                try:
                    result = loader.evaluate_model(dataset=items, agent_manager=agent_manager)
                    metrics["coins"].append({"coin_id": coin_id, **(result or {})})
                    if result and "samples" in result:
                        metrics["samples"] += int(result["samples"])
                    if result and "avg_loss" in result:
                        if metrics["avg_loss"] is None:
                            metrics["avg_loss"] = result["avg_loss"]
                        else:
                            metrics["avg_loss"] = (metrics["avg_loss"] + result["avg_loss"]) / 2.0
                except Exception as e:
                    logger.exception("Evaluate failed for coin %s", coin_id)

                processed += 1
                self.update_state(state='PROGRESS', meta={'progress': int(processed/total*100)})

            return {"status": "success", "agent_id": agent_id, "metrics": metrics}

    return asyncio.run(_run())


# --------- Lightweight placeholders for Stage 1 contracts ---------
@celery_app.task(bind=True)
def train_news_task(self, coins: list[int] | None = None, config: dict | None = None):
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10})
        # Placeholder: here we would fetch recent news, compute influence and persist background
        self.update_state(state='PROGRESS', meta={'progress': 100})
        return {"status": "success", "coins": coins or []}
    except Exception as e:
        logger.exception("train_news_task failed")
        return {"status": "error", "detail": str(e)}


@celery_app.task(bind=True)
def evaluate_trade_aggregator_task(self, strategy_config: dict | None = None):
    try:
        # Simulated orchestration progress and aggregate metrics
        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': 'Initializing pipeline...'})
        cfg = strategy_config or {}
        nodes = cfg.get('nodes', [])
        edges = cfg.get('edges', [])
        metrics = {
            'nodes_count': len(nodes),
            'edges_count': len(edges),
            'Sharpe': 0.0,
            'PnL': 0.0,
        }
        self.update_state(state='PROGRESS', meta={'progress': 50, 'message': 'Running pipeline...', 'metrics': metrics})
        self.update_state(state='PROGRESS', meta={'progress': 100, 'message': 'Completed', 'metrics': metrics})
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.exception("evaluate_trade_aggregator_task failed")
        return {"status": "error", "detail": str(e)}


@celery_app.task(bind=True)
def run_pipeline_backtest_task(self, config_json: dict | None = None, timeframe: str | None = None,
                               start: str | None = None, end: str | None = None, pipeline_id: int | None = None):
    async def _run():
        try:
            cfg = config_json or {}
            nodes = cfg.get('nodes', [])
            edges = cfg.get('edges', [])

            steps = [
                ('load_data', 'Загрузка данных'),
                ('features', 'Расчет индикаторов'),
                ('merge_news', 'Объединение с новостным фоном'),
                ('pred_time', 'Прогноз цены'),
                ('trade_time', 'Сигналы buy/sell/hold'),
                ('risk', 'Оценка риска'),
                ('trade', 'Агрегация и трейдинг'),
                ('metrics', 'Подсчет метрик')
            ]

            metrics: dict = {
                'nodes_count': len(nodes),
                'edges_count': len(edges),
            }

            def step(i: int, code: str, title: str):
                progress = int(i / len(steps) * 100)
                self.update_state(state='PROGRESS', meta={'progress': progress, 'step': code, 'message': title, 'metrics': metrics})

            # Resolve timeframe
            tf = timeframe or cfg.get('timeframe') or '5m'

            # 1) Load data (support multi-coin)
            step(1, *steps[0])
            # pick DataSource
            data_source = next((n for n in nodes if n.get('type') == 'DataSource'), None)
            coin_ids: list[int] = []
            if data_source and isinstance(data_source.get('config'), dict):
                coin_ids = data_source['config'].get('coins', []) or []
            per_asset_closes: list[list[float]] = []
            datetimes: list[str] = []
            async with db_helper.get_session() as session:  # type: AsyncSession
                target_coins = coin_ids or []
                # fallback: if not set, try any available timeseries via ORM helper? Here keep empty -> early return
                for idx, coin_id in enumerate(target_coins):
                    ts = await market_get_ts(session, coin_id, timeframe=tf)
                    if not ts:
                        per_asset_closes.append([])
                        continue
                    rows = await market_get_data(session, ts.id)
                    rows_sorted = sorted(rows, key=lambda r: r.datetime)
                    closes_i: list[float] = []
                    if idx == 0:
                        datetimes = [r.datetime.isoformat() for r in rows_sorted]
                    for r in rows_sorted:
                        closes_i.append(float(r.close))
                    per_asset_closes.append(closes_i)

            # Align by min length
            min_len = min((len(c) for c in per_asset_closes), default=0)
            per_asset_closes = [c[-min_len:] for c in per_asset_closes if min_len > 0]
            metrics['bars'] = min_len
            if min_len < 30:
                # not enough data, return early
                self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'Недостаточно данных', 'metrics': metrics})
                return {"status": "success", "metrics": metrics}

            # 2) Features: simple SMA
            step(2, *steps[1])
            ind_node = next((n for n in nodes if n.get('type') == 'Indicators'), None)
            sma_period = 20
            if ind_node and isinstance(ind_node.get('config'), dict):
                # try to parse SMA(N)
                inds = ind_node['config'].get('indicators', []) or []
                for name in inds:
                    if isinstance(name, str) and name.upper().startswith('SMA(') and name.endswith(')'):
                        try:
                            sma_period = int(name[4:-1])
                        except Exception:
                            pass
                        break
            per_asset_sma: list[list[float | None]] = []
            for closes in per_asset_closes:
                sma = []
                s = 0.0
                for i, c in enumerate(closes):
                    s += c
                    if i >= sma_period:
                        s -= closes[i - sma_period]
                    if i >= sma_period - 1:
                        sma.append(s / sma_period)
                    else:
                        sma.append(None)
                per_asset_sma.append(sma)

            # 3) Merge news background
            step(3, *steps[2])
            news_node = next((n for n in nodes if n.get('type') == 'News'), None)
            news_window_bars = 288
            if news_node and isinstance(news_node.get('config'), dict):
                news_window_bars = int(news_node['config'].get('window', news_window_bars))

            # Parse timeframe into timedelta per bar
            def parse_tf(s: str) -> timedelta:
                try:
                    num = int(''.join([ch for ch in s if ch.isdigit()]))
                    unit = ''.join([ch for ch in s if ch.isalpha()])
                    unit = unit.lower()
                    if unit in ('m', 'min', 'mins', 'minute', 'minutes'):
                        return timedelta(minutes=num)
                    if unit in ('h', 'hour', 'hours'):
                        return timedelta(hours=num)
                    if unit in ('d', 'day', 'days'):
                        return timedelta(days=num)
                    if unit in ('s', 'sec', 'second', 'seconds'):
                        return timedelta(seconds=num)
                except Exception:
                    pass
                return timedelta(minutes=5)

            bar_dt = parse_tf(tf)
            start_dt = None
            end_dt = None
            if datetimes:
                try:
                    start_dt = datetime.fromisoformat(datetimes[-min_len]) if min_len > 0 else None
                    end_dt = datetime.fromisoformat(datetimes[-1])
                except Exception:
                    start_dt = None
                    end_dt = None

            # Load news for all requested coins in [start_dt, end_dt]
            per_asset_news_idx: list[list[float]] = []
            if start_dt and end_dt and coin_ids:
                async with db_helper.get_session() as session:
                    result = await session.execute(
                        select(NewsHistoryCoin.coin_id, News.date, NewsHistoryCoin.score)
                        .join(News, News.id == NewsHistoryCoin.id_news)
                        .where(NewsHistoryCoin.coin_id.in_(coin_ids))
                        .where(News.date >= start_dt - news_window_bars * bar_dt)
                        .where(News.date <= end_dt)
                    )
                    rows = result.all()
                # Build per-coin list of (date, score)
                coin_to_events: dict[int, list[tuple[datetime, float]]] = {cid: [] for cid in coin_ids}
                for cid, ndt, sc in rows:
                    coin_to_events.setdefault(cid, []).append((ndt, float(sc)))
                for cid in coin_ids:
                    events = sorted(coin_to_events.get(cid, []), key=lambda x: x[0])
                    # Build rolling sum over bars
                    idx_series: list[float] = []
                    left = 0
                    right = 0
                    cur_sum = 0.0
                    for i in range(min_len):
                        bt = end_dt - (min_len - 1 - i) * bar_dt
                        window_start = bt - news_window_bars * bar_dt
                        # advance right pointer
                        while right < len(events) and events[right][0] <= bt:
                            cur_sum += events[right][1]
                            right += 1
                        # advance left pointer (drop old)
                        while left < right and events[left][0] < window_start:
                            cur_sum -= events[left][1]
                            left += 1
                        idx_series.append(cur_sum)
                    per_asset_news_idx.append(idx_series)
            else:
                per_asset_news_idx = [[0.0] * min_len for _ in per_asset_closes]

            # 4) Pred_time: naive direction vs SMA
            step(4, *steps[3])
            per_asset_pred_dir: list[list[int]] = []
            for asset_i, (closes, sma) in enumerate(zip(per_asset_closes, per_asset_sma)):
                pred_dir = []  # 1 up, -1 down, 0 none
                for i, c in enumerate(closes):
                    if sma[i] is None:
                        pred_dir.append(0)
                    else:
                        base = 1 if c > sma[i] else -1
                        # Blend simple news influence (normalized) with base direction
                        news_val = per_asset_news_idx[asset_i][i] if asset_i < len(per_asset_news_idx) else 0.0
                        if news_val != 0.0:
                            pred = base + 0.2 * (1 if news_val > 0 else -1)
                            pred_dir.append(1 if pred > 0.5 else (-1 if pred < -0.5 else 0))
                        else:
                            pred_dir.append(base)
                per_asset_pred_dir.append(pred_dir)

            # 5) Trade_time: thresholding -> keep same as pred_dir (simplified)
            step(5, *steps[4])
            per_asset_signals = per_asset_pred_dir

            # 6) Risk: cap position based on simple volatility
            step(6, *steps[5])
            per_asset_rets: list[list[float]] = []
            per_asset_risk_cap: list[float] = []
            for closes in per_asset_closes:
                rets = []
                for i in range(1, len(closes)):
                    if closes[i-1] == 0:
                        rets.append(0.0)
                    else:
                        rets.append((closes[i] / closes[i-1]) - 1.0)
                vol = (sum((r - (sum(rets)/len(rets)))**2 for r in rets)/max(len(rets)-1,1))**0.5 if rets else 0.0
                risk_cap = 1.0 if vol == 0 else min(1.0, 0.02 / max(vol, 1e-6))
                per_asset_rets.append(rets)
                per_asset_risk_cap.append(risk_cap)

            # 7) Trade: backtest PnL
            step(7, *steps[6])
            pnl = 0.0
            wins = 0
            trades = 0
            per_asset_trades: list[int] = []
            per_asset_returns_signal: list[list[float]] = []
            per_asset_trade_rows: list[list[tuple[int, str, int, float]]] = []  # (coin_id, ts, signal, ret)
            for rets, signals, risk_cap in zip(per_asset_rets, per_asset_signals, per_asset_risk_cap):
                asset_trades = 0
                asset_sig_returns: list[float] = []
                asset_rows: list[tuple[int, str, int, float]] = []
                for i in range(1, len(signals)):
                    sig = signals[i-1]
                    if sig != 0:
                        ret = rets[i-1] * sig * risk_cap
                        pnl += ret
                        asset_trades += 1
                        trades += 1
                        asset_sig_returns.append(ret)
                        # timestamp aligned to returns index i
                        if i < len(datetimes[-min_len:]):
                            ts = datetimes[-min_len:][i]
                        else:
                            ts = datetimes[-1] if datetimes else ""
                        asset_rows.append((0, ts, sig, ret))  # coin_id fill later
                        if ret > 0:
                            wins += 1
                per_asset_trades.append(asset_trades)
                per_asset_returns_signal.append(asset_sig_returns)
                per_asset_trade_rows.append(asset_rows)
            winrate = (wins / trades) if trades else 0.0
            # Portfolio equal-weight across assets
            # Align asset signal-returns by min len
            port_rets = aggregate_returns_equal_weight(
                [r for r in per_asset_returns_signal if r]
            )
            sharpe = sharpe_ratio(port_rets)

            # 8) Metrics
            step(8, *steps[7])
            eq = equity_curve(port_rets, start_equity=1.0)
            mdd = max_drawdown(eq)
            sortino = sortino_ratio(port_rets)

            # Optional: write equity curve artifact (CSV) to temp file
            artifacts = {}
            per_coin_pnl: dict[str, float] = {}
            try:
                # per-coin pnl (sum of signal returns)
                for idx, asset_returns in enumerate(per_asset_returns_signal):
                    if idx < len(coin_ids):
                        per_coin_pnl[str(coin_ids[idx])] = round(sum(asset_returns), 6)
            except Exception:
                pass
            try:
                if datetimes:
                    # Align datetimes to returns length
                    # datetimes aligned to min_len bars; returns length = min_len-1
                    dt_aligned = datetimes[-min_len:] if min_len > 0 else []
                    # Fallback if no aligned timestamps present
                    if len(dt_aligned) >= len(eq) - 1:
                        # Build CSV rows ts,equity (skip first equity as it is starting equity)
                        rows = ["timestamp,equity"]
                        for i in range(1, len(eq)):
                            ts = dt_aligned[i]
                            rows.append(f"{ts},{eq[i]:.10f}")
                        out_dir = os.environ.get("PIPELINE_ARTIFACTS_DIR", "/tmp")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"equity_{uuid4().hex}.csv")
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(rows))
                        artifacts["equity_csv"] = out_path
            except Exception:
                # do not fail task if artifacts write fails
                pass

            # Optional: write trades CSV per asset
            try:
                if per_asset_trade_rows and coin_ids:
                    rows = ["timestamp,coin_id,signal,return"]
                    for idx, asset_rows in enumerate(per_asset_trade_rows):
                        coin_id = coin_ids[idx] if idx < len(coin_ids) else 0
                        for (_, ts, sig, ret) in asset_rows:
                            rows.append(f"{ts},{coin_id},{sig},{ret:.10f}")
                    out_dir = os.environ.get("PIPELINE_ARTIFACTS_DIR", "/tmp")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"trades_{uuid4().hex}.csv")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(rows))
                    artifacts["trades_csv"] = out_path
            except Exception:
                pass

            metrics.update({
                'bars': min_len,
                'sma_period': sma_period,
                'signals': trades,
                'PnL': round(pnl, 6),
                'WinRate': round(winrate, 4),
                'Sharpe': round(sharpe, 4),
                'Sortino': round(sortino, 4),
                'MaxDrawdown': round(mdd, 6),
                'timeframe': tf,
                'artifacts': artifacts,
                'per_coin_pnl': per_coin_pnl,
            })
            # persist backtest row
            try:
                async with db_helper.get_session() as session:
                    bt = BacktestModel(
                        pipeline_id=pipeline_id,
                        timeframe=tf,
                        start=start_dt,
                        end=end_dt,
                        config_json=cfg,
                        metrics_json=metrics,
                        artifacts=artifacts,
                    )
                    session.add(bt)
                    await session.commit()
            except Exception:
                pass

            self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'Готово', 'metrics': metrics})
            return {"status": "success", "metrics": metrics}
        except Exception as e:
            logger.exception("run_pipeline_backtest_task failed")
            return {"status": "error", "detail": str(e)}

    return asyncio.run(_run())

    # db = SessionLocal()
    # try:
    #     # Обновляем статус задачи в БД
    #     task = db.query(TrainingTask).filter(TrainingTask.task_id == task_id).first()
    #     task.status = "processing"
    #     db.commit()
        
    #     # Здесь должна быть ваша реальная логика обучения
    #     print(f"Start training model with dataset: {dataset_path}")
    #     print(f"Parameters: {json.dumps(parameters)}")
        
    #     # Имитация долгого обучения
    #     for i in range(10):
    #         time.sleep(1)
    #         self.update_state(state='PROGRESS', meta={'progress': i*10})
        
    #     # Обновляем статус после успешного выполнения
    #     task.status = "completed"
    #     db.commit()
        
    #     return {"status": "success", "task_id": task_id}
    
    # except Exception as e:
    #     task.status = "failed"
    #     db.commit()
    #     raise self.retry(exc=e, countdown=60, max_retries=3)
    
    # finally:
    #     db.close()