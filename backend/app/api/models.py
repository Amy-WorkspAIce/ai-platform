"""Model discovery and management (``/api``): hardware, catalog, browse, install, delete.

``/api/catalog`` is the curated catalog (with recommendations). ``/api/catalog/browse`` is
everything: curated models plus the whole Ollama library and Hugging Face models Amy's
engines can run, filterable, searchable and sortable.

Models that need an engine (image, video, audio) are installed by the EngineManager, which
installs the engine itself on first use. Everything else goes to Ollama.
"""

import asyncio
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app import catalog, hardware
from app.catalog import live
from app.engines import ENGINES, EngineManager
from app.jobs import JobRegistry
from app.ollama_admin import OllamaAdmin, valid_model_name
from app.security import require_api_key

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

HF_CATEGORIES = ("image", "video", "transcription", "speech")
SORTS = Literal["perfect", "fit", "size_asc", "size_desc", "name", "popular"]


class PullRequest(BaseModel):
    model: str
    # For models outside the curated catalog: which engine runs them.
    engine: Literal["ollama", "image", "audio"] | None = None


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


def hf_token(request: Request) -> str | None:
    return request.app.state.user_settings.hf_token


def engine_variant(request: Request, tag: str) -> tuple[str, dict] | None:
    """A curated engine model, or any installed engine model."""
    found = catalog.find_variant(tag)
    if found and found[0]["engine"] in ENGINES:
        return found[0]["engine"], found[1]
    engines = get_engines(request)
    for engine in ENGINES:
        variant = engines.installed_variant(engine, tag)
        if variant:
            return engine, variant
    return None


async def installed_and_status(request: Request) -> tuple[set[str], bool]:
    admin = get_admin(request)
    installed: set[str] = set()
    online = False
    if admin:
        try:
            installed = {m["name"] for m in await admin.installed()}
            online = True
        except httpx.HTTPError:
            pass
    return installed | get_engines(request).installed_tags(), online


@router.get("/hardware")
async def get_hardware(request: Request) -> dict:
    return detect_hardware(request).to_dict()


@router.get("/catalog")
async def get_catalog(request: Request) -> dict:
    installed, online = await installed_and_status(request)
    engines = get_engines(request)
    ready = {e for e in ENGINES if engines.engine_installed(e)}
    return catalog.build(detect_hardware(request), installed, ready) | {
        "ollama_online": online,
        "hf_token_set": bool(hf_token(request)),
    }


@router.get("/catalog/browse")
async def browse_catalog(
    request: Request,
    category: str = "all",
    q: str | None = Query(default=None, max_length=100),
    sort: SORTS = "perfect",
    runnable: bool = False,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=40, ge=1, le=200),
) -> dict:
    lc: live.LiveCatalog = request.app.state.live_catalog
    hw = detect_hardware(request)
    installed, _ = await installed_and_status(request)
    engines = get_engines(request)
    ready = {e for e in ENGINES if engines.engine_installed(e)}

    jobs = []
    if category in ("all", "chat", "reasoning", "code", "agent", "vision", "embedding"):
        jobs.append(_ollama(lc))
    for cat in HF_CATEGORIES:
        if category in ("all", cat):
            jobs.append(lc.hf_families(cat, q))
    if q and category in ("all", "chat"):
        jobs.append(lc.gguf_search(q))
    live_families = [f for group in await asyncio.gather(*jobs) for f in group]

    merged = catalog.merge(catalog.load()["families"], live_families)
    annotated = [catalog.annotate(f, hw, installed, ready) for f in merged]
    result = catalog.browse(
        annotated,
        category=category,
        query=q,
        sort=sort,
        runnable_only=runnable,
        offset=offset,
        limit=limit,
    )
    return result | {"hardware": hw.to_dict(), "hf_token_set": bool(hf_token(request))}


async def _ollama(lc: live.LiveCatalog) -> list[dict]:
    return lc.ollama_families(await lc.ollama_models())


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
    engines = get_engines(request)
    token = hf_token(request)

    found = catalog.find_variant(body.model)
    if found and found[0]["engine"] in ENGINES:
        return engines.start_install(found[0]["engine"], found[1], token).public()

    if body.engine == "image":
        repo = body.model
        client = request.app.state.http

        async def resolve() -> dict:
            try:
                return await live.diffusers_spec(repo, client, token)
            except PermissionError:
                raise ValueError(
                    f"{repo} requires a Hugging Face token with access. Add your token in "
                    f"Settings and accept the license at https://huggingface.co/{repo}."
                ) from None

        return engines.start_install("image", {"tag": repo}, token, resolve).public()

    if body.engine == "audio":
        return engines.start_install("audio", live.whisper_spec(body.model, None), token).public()

    return require_admin(request).start_pull(body.model).public()


@router.get("/models/pulls")
async def list_pulls(request: Request) -> list[dict]:
    jobs: JobRegistry = request.app.state.jobs
    return jobs.snapshot()


@router.delete("/models/{model:path}", status_code=204)
async def delete_model(request: Request, model: str) -> None:
    if not valid_model_name(model):
        raise HTTPException(400, "Invalid model name")
    found = engine_variant(request, model)
    if found:
        await get_engines(request).delete_model(found[0], model)
        return
    try:
        await require_admin(request).delete(model)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama unreachable: {exc}") from exc
