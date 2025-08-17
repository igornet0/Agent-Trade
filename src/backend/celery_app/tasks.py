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
                    return []
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