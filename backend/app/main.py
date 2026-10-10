"""Application entrypoint: ``uvicorn app.main:create_app --factory``."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__, hardware
from app.api import conversations, media, models, openai, system
from app.config import Settings, get_settings
from app.db import Database
from app.engines import EngineManager
from app.jobs import JobRegistry
from app.ollama_admin import OllamaAdmin
from app.providers import ProviderRegistry
from app.security import load_or_create_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(timeout=settings.upstream_timeout)
        if settings.ollama_url:
            app.state.ollama_admin = OllamaAdmin(
                settings.ollama_url, app.state.http, app.state.jobs
            )
        yield
        await app.state.engines.stop_all()
        await app.state.http.aclose()

    app = FastAPI(title="Amy AI Platform", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.api_key = load_or_create_api_key(settings)
    app.state.providers = ProviderRegistry.from_settings(settings)
    app.state.ollama_admin = None
    app.state.jobs = JobRegistry()
    app.state.db = Database(settings.data_dir / "amy.db")
    app.state.engines = EngineManager(
        settings.data_dir,
        app.state.jobs,
        gpu=bool(hardware.nvidia_gpus()),
        torch_index_url=settings.torch_index_url,
        idle_seconds=settings.engine_idle_minutes * 60,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(system.public)
    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(conversations.router)
    app.include_router(media.api)
    app.include_router(media.v1)
    app.include_router(openai.router)

    _mount_frontend(app, settings)
    return app


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built React app, falling back to index.html for client-side routes."""
    static = settings.static_dir
    index = static / "index.html"
    if not index.exists():
        return

    if (static / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        candidate = (static / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(static.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)
