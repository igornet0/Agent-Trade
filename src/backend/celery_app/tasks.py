from .create_app import celery_app
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json
import logging

from core.database import db_helper
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