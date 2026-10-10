"""Health check and platform management API (``/api``)."""

import asyncio

from fastapi import APIRouter, Depends, Request

from app import __version__
from app.security import require_api_key

public = APIRouter()
router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


@public.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/system")
async def system(request: Request) -> dict:
    providers = request.app.state.providers.all()
    client = request.app.state.http
    models = await asyncio.gather(*(p.list_models(client) for p in providers))
    return {
        "version": __version__,
        "providers": [
            {"name": p.name, "kind": p.kind, "base_url": p.base_url, "online": bool(m)}
            for p, m in zip(providers, models, strict=True)
        ],
    }
