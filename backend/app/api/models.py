"""Model discovery and management (``/api``): hardware, catalog, install, delete."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app import catalog, hardware
from app.ollama_admin import OllamaAdmin, valid_model_name
from app.security import require_api_key

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


class PullRequest(BaseModel):
    model: str


def get_admin(request: Request) -> OllamaAdmin:
    admin: OllamaAdmin | None = request.app.state.ollama_admin
    if admin is None:
        raise HTTPException(409, "Model management needs an Ollama provider (AMY_OLLAMA_URL).")
    return admin


def detect_hardware(request: Request) -> hardware.Hardware:
    s = request.app.state.settings
    return hardware.detect(s.data_dir, s.gpu_vram_gb)


@router.get("/hardware")
async def get_hardware(request: Request) -> dict:
    return detect_hardware(request).to_dict()


@router.get("/catalog")
async def get_catalog(request: Request) -> dict:
    admin: OllamaAdmin | None = request.app.state.ollama_admin
    installed: set[str] = set()
    ollama_online = False
    if admin:
        try:
            installed = {m["name"] for m in await admin.installed()}
            ollama_online = True
        except httpx.HTTPError:
            pass
    return catalog.build(detect_hardware(request), installed) | {"ollama_online": ollama_online}


@router.get("/models/installed")
async def installed_models(admin: OllamaAdmin = Depends(get_admin)) -> list[dict]:
    try:
        models = await admin.installed()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama unreachable: {exc}") from exc
    return [
        {
            "name": m["name"],
            "size_gb": round(m.get("size", 0) / 1e9, 2),
            "family": m.get("details", {}).get("family"),
            "parameters": m.get("details", {}).get("parameter_size"),
            "quantization": m.get("details", {}).get("quantization_level"),
            "modified_at": m.get("modified_at"),
        }
        for m in models
    ]


@router.post("/models/pull", status_code=202)
async def pull_model(body: PullRequest, admin: OllamaAdmin = Depends(get_admin)) -> dict:
    if not valid_model_name(body.model):
        raise HTTPException(400, "Invalid model name")
    return admin.start_pull(body.model).public()


@router.get("/models/pulls")
async def list_pulls(admin: OllamaAdmin = Depends(get_admin)) -> list[dict]:
    return admin.snapshot()


@router.delete("/models/{model:path}", status_code=204)
async def delete_model(model: str, admin: OllamaAdmin = Depends(get_admin)) -> None:
    if not valid_model_name(model):
        raise HTTPException(400, "Invalid model name")
    try:
        await admin.delete(model)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama unreachable: {exc}") from exc
