from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.routes import router as api_router
from server.api.ws import router as ws_router
from server.core.config import get_settings
from server.core.paths import ProjectPaths
from server.core.session_store import SessionStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    app.state.store.bind_main_loop(loop)
    yield
    app.state.store.bind_main_loop(None)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="HouseInsight Agent", version="0.1.0", lifespan=lifespan)

    app.state.store = SessionStore()
    app.state.paths = ProjectPaths()

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")
    return app


app = create_app()
