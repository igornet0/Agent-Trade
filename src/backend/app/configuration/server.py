import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

from backend.app.configuration.routers import Routers
from core import settings
from core.database import db_helper  # only for type; do not rely on this binding at runtime
from backend.app.middleware.observability import ObservabilityMiddleware

class Server:

    __app: FastAPI

    # Legacy Jinja templates removed; frontend is served by Vite React app
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    http_bearer = HTTPBearer(auto_error=False)
    # Must point to the OAuth2 password flow token endpoint, not a secret key
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login_user/")
    frontend_url = settings.run.frontend_url

    def __init__(self, app: FastAPI):

        self.__app = app
        self.__register_routers(app)
        self.__regist_middleware(app)

    @staticmethod
    async def get_db() -> AsyncGenerator[AsyncSession, None]:
        # Lazily ensure db_helper is initialized (read the live binding from engine module)
        from core.database.engine import set_db_helper, db_helper as engine_db_helper
        if engine_db_helper is None:
            await set_db_helper()
        from core.database.engine import db_helper as live_db_helper
        async with live_db_helper.get_session() as session:
            yield session

    def get_app(self) -> FastAPI:
        return self.__app

    @staticmethod
    def __register_routers(app: FastAPI):

        Routers(Routers._discover_routers()).register(app)

    @staticmethod
    def __regist_middleware(app: FastAPI):
        app.add_middleware(ObservabilityMiddleware)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.run.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

