"""Settings page API (``/api/settings``)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app import __version__, routing
from app.delegation import CAPABILITIES, installed_for
from app.security import require_api_key
from app.tools.web import provider_chain
from app.user_settings import UserSettings, verify_hf_token

router = APIRouter(prefix="/api/settings", dependencies=[Depends(require_api_key)])


class TokenBody(BaseModel):
    token: str = Field(min_length=8, max_length=500)


class ModelDefaultBody(BaseModel):
    capability: str
    model: str | None = None  # None = best installed


class WebBody(BaseModel):
    enabled: bool


class VoiceBody(BaseModel):
    voice: str | None = Field(default=None, max_length=100)


def store(request: Request) -> UserSettings:
    return request.app.state.user_settings


def huggingface_status(us: UserSettings) -> dict:
    hf = us.huggingface()
    return {"token_set": bool(hf.token), "username": hf.username, "source": hf.source}


@router.get("")
async def get_settings(request: Request) -> dict:
    s = request.app.state.settings
    return {
        "version": __version__,
        "huggingface": huggingface_status(store(request)),
        "search": {"providers": provider_chain(s), "bundled_searxng": s.bundled_searxng},
        "web_access": store(request).web_access,
        "voice": store(request).voice,
        "models": await model_choices(request),
    }


async def model_choices(request: Request) -> dict:
    defaults = store(request).model_defaults()
    out = await routing.choices(request.app)
    for cap in CAPABILITIES:
        installed = installed_for(request.app, cap)
        out[cap] = {
            "pinned": defaults.get(cap),
            "best": installed[0]["id"] if installed else None,
            "installed": [{"id": m["id"], "name": m["name"], "fit": m["fit"]} for m in installed],
        }
    return out


@router.put("/models")
async def set_model_default(request: Request, body: ModelDefaultBody) -> dict:
    if body.capability in routing.PURPOSES:
        available = {m["id"] for m in await routing.chat_models(request.app)}
    elif body.capability in CAPABILITIES:
        available = {m["id"] for m in installed_for(request.app, body.capability)}
    else:
        raise HTTPException(400, f"Unknown capability '{body.capability}'")
    if body.model and body.model not in available:
        raise HTTPException(400, f"'{body.model}' isn't an installed {body.capability} model")
    store(request).set_model_default(body.capability, body.model)
    return await model_choices(request)


@router.put("/web")
async def set_web_access(request: Request, body: WebBody) -> dict:
    store(request).set_web_access(body.enabled)
    return {"web_access": body.enabled}


@router.put("/voice")
async def set_voice(request: Request, body: VoiceBody) -> dict:
    store(request).set_voice(body.voice)
    return {"voice": store(request).voice}


@router.put("/huggingface")
async def set_hf_token(request: Request, body: TokenBody) -> dict:
    us = store(request)
    if us.huggingface().source == "env":
        raise HTTPException(409, "The token is set by AMY_HF_TOKEN in the container config.")
    token = body.token.strip()
    try:
        username = await verify_hf_token(token, request.app.state.http)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    us.save_hf_token(token, username)
    return huggingface_status(us)


@router.delete("/huggingface")
async def clear_hf_token(request: Request) -> dict:
    us = store(request)
    if us.huggingface().source == "env":
        raise HTTPException(409, "The token is set by AMY_HF_TOKEN in the container config.")
    us.clear_hf_token()
    return huggingface_status(us)
