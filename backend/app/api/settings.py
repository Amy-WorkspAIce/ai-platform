"""Settings page API (``/api/settings``)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app import __version__
from app.security import require_api_key
from app.tools.web import provider_chain
from app.user_settings import UserSettings, verify_hf_token

router = APIRouter(prefix="/api/settings", dependencies=[Depends(require_api_key)])


class TokenBody(BaseModel):
    token: str = Field(min_length=8, max_length=500)


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
    }


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
