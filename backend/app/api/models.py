"""Model discovery and management (``/api``): hardware, catalog, install, delete.

Catalog models that need an engine (image, audio) are installed by the EngineManager,
which installs the engine itself on first use. Everything else goes to Ollama.
"""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app import catalog, hardware
from app.engines import ENGINES, EngineManager
from app.jobs import JobRegistry
from app.ollama_admin import OllamaAdmin, valid_model_name
from app.security import require_api_key

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


class PullRequest(BaseModel):
    model: str


def get_admin(request: Request) -> OllamaAdmin | None:
    return request.app.state.ollama_admin


def require_admin(request: Request) -> OllamaAdmin:
    admin = get_admin(request)
    if admin is None:
        raise HTTPException(409, "This needs an Ollama provider (AMY_OLLAMA_URL).")
    return admin


def get_engines(request: Request) -> EngineManager:
    return request.app.state.engines


def detect_hardware(request: Request) -> hardware.Hardware:
    s = request.app.state.settings
    return hardware.detect(s.data_dir, s.gpu_vram_gb)


def engine_variant(tag: str) -> tuple[str, dict] | None:
    found = catalog.find_variant(tag)
    if found and found[0]["engine"] in ENGINES:
        return found[0]["engine"], found[1]
    return None


@router.get("/hardware")
async def get_hardware(request: Request) -> dict:
    return detect_hardware(request).to_dict()


@router.get("/catalog")
async def get_catalog(request: Request, engines: EngineManager = Depends(get_engines)) -> dict:
    admin = get_admin(request)
    installed: set[str] = set()
    ollama_online = False
    if admin:
        try:
            installed = {m["name"] for m in await admin.installed()}
            ollama_online = True
        except httpx.HTTPError:
            pass
    installed |= engines.installed_tags(catalog.load()["families"])
    ready = {e for e in ENGINES if engines.engine_installed(e)}
    return catalog.build(detect_hardware(request), installed, ready) | {
        "ollama_online": ollama_online
    }


@router.get("/models/installed")
async def installed_models(admin: OllamaAdmin = Depends(require_admin)) -> list[dict]:
    """Models installed in Ollama (engine models are listed in the catalog)."""
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
async def pull_model(request: Request, body: PullRequest) -> dict:
    if not valid_model_name(body.model):
        raise HTTPException(400, "Invalid model name")
    found = engine_variant(body.model)
    if found:
        engine, variant = found
        return get_engines(request).start_install(engine, variant).public()
    return require_admin(request).start_pull(body.model).public()


@router.get("/models/pulls")
async def list_pulls(request: Request) -> list[dict]:
    jobs: JobRegistry = request.app.state.jobs
    return jobs.snapshot()


@router.delete("/models/{model:path}", status_code=204)
async def delete_model(request: Request, model: str) -> None:
    if not valid_model_name(model):
        raise HTTPException(400, "Invalid model name")
    found = engine_variant(model)
    if found:
        await get_engines(request).delete_model(found[0], model)
        return
    try:
        await require_admin(request).delete(model)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama unreachable: {exc}") from exc
