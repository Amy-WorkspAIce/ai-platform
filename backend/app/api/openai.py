"""OpenAI-compatible API (``/v1``).

Any OpenAI client library can point at ``http://<host>:8484/v1`` with an Amy API key.
This is the contract other Amy WorkspAIce apps (and third parties) use for AI features.
"""

import asyncio
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.security import require_api_key

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@router.get("/models")
async def list_models(request: Request) -> dict:
    registry = request.app.state.providers
    client: httpx.AsyncClient = request.app.state.http
    providers = registry.all()
    results = await asyncio.gather(*(p.list_models(client) for p in providers))
    now = int(time.time())
    data = [
        {"id": f"{p.name}/{m}", "object": "model", "created": now, "owned_by": p.name}
        for p, models in zip(providers, results, strict=True)
        for m in models
    ]
    return {"object": "list", "data": data}


@router.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    model_id = payload.get("model")
    if not isinstance(model_id, str):
        raise HTTPException(400, "'model' is required")

    resolved = request.app.state.providers.resolve(model_id)
    if resolved is None:
        raise HTTPException(404, f"Unknown model '{model_id}'. Use '<provider>/<model>'.")
    provider, upstream_model = resolved
    payload["model"] = upstream_model

    client: httpx.AsyncClient = request.app.state.http
    upstream = client.build_request(
        "POST", provider.url("chat/completions"), json=payload, headers=provider.headers()
    )
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Provider '{provider.name}' unreachable: {exc}") from exc

    if not payload.get("stream"):
        await resp.aread()
        await resp.aclose()
        return JSONResponse(resp.json(), status_code=resp.status_code)

    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
        background=BackgroundTask(resp.aclose),
    )
