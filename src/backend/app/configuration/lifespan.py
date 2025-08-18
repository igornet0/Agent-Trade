import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .tasks import tasks
from .rabbitmq_server import rabbit
from core import settings
from core.database.engine import set_db_helper

import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Менеджер жизненного цикла приложения"""
    try:
        logger.info("Initializing database...")
        db = await set_db_helper()
        await db.init_db()
        logger.info("Starting application...")
        # Only initialize RabbitMQ wiring if explicitly enabled
        if settings.run.enable_rabbit:
            await rabbit.setup_dlx()
            asyncio.create_task(rabbit.consume_messages("process_queue", tasks.start_process_train))
        logger.info("Application startup complete")
        yield
    finally:
        logger.info("Shutting down application...")
        if db := await set_db_helper():
            await db.dispose()
        if settings.run.enable_rabbit:
            await rabbit.close()
        logger.info("Application shutdown complete")