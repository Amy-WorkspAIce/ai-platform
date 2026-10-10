"""Image generation and audio, OpenAI-compatible where possible.

- ``POST /v1/images/generations``    text-to-image (saved to the gallery)
- ``POST /v1/audio/transcriptions``  speech-to-text (multipart upload)
- ``POST /v1/audio/speech``          text-to-speech (returns audio bytes)
- ``/api/images``                    gallery: list, fetch, delete
- ``GET /api/audio/voices``          voices for a text-to-speech model
"""

import base64
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app import catalog
from app.db import Database
from app.engines import EngineError, EngineManager
from app.security import require_api_key

v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 500 * 1024 * 1024


def resolve(
    request: Request, model: str, engine: str, task: str | None = None
) -> tuple[Path, dict]:
    """Map ``image/<tag>`` (or a bare tag) to an installed engine model."""
    tag = model.removeprefix(f"{engine}/")
    found = catalog.find_variant(tag)
    if not found or found[0]["engine"] != engine:
        raise HTTPException(404, f"Unknown {engine} model '{model}'")
    _, variant = found
    if task and variant["engine_config"].get("task") != task:
        raise HTTPException(400, f"'{model}' does not support {task}")
    engines: EngineManager = request.app.state.engines
    if not engines.model_installed(engine, tag):
        raise HTTPException(409, f"'{model}' is not installed. Install it from the Models page.")
    return engines.model_dir(engine, tag), variant


def engine_error(exc: EngineError) -> HTTPException:
    return HTTPException(502, str(exc))


def images_dir(request: Request) -> Path:
    d = request.app.state.settings.data_dir / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- images ----


class ImageRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    response_format: str = "b64_json"
    # Extensions beyond the OpenAI API:
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0, le=30)
    seed: int | None = None


def parse_size(size: str | None, default: int) -> tuple[int, int]:
    if not size:
        return default, default
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except ValueError as exc:
        raise HTTPException(400, "size must look like 1024x1024") from exc
    # Diffusion models want multiples of 64; keep within sane bounds.
    clamp = lambda x: max(256, min(2048, round(x / 64) * 64))  # noqa: E731
    return clamp(w), clamp(h)


@v1.post("/images/generations")
async def generate_image(request: Request, body: ImageRequest) -> dict:
    model_dir, variant = resolve(request, body.model, "image")
    cfg = variant["engine_config"]
    width, height = parse_size(body.size, cfg["size"])

    # Language models may be holding GPU memory; release it for the image pipeline.
    if request.app.state.ollama_admin:
        await request.app.state.ollama_admin.unload_all()

    payload = {
        "model_dir": str(model_dir),
        "config": cfg,
        "prompt": body.prompt,
        "negative_prompt": body.negative_prompt if cfg.get("negative_prompt") else None,
        "width": width,
        "height": height,
        "n": body.n,
        "steps": body.steps,
        "guidance": body.guidance,
        "seed": body.seed,
    }
    try:
        result = (await request.app.state.engines.call("image", "/generate", payload)).json()
    except EngineError as exc:
        raise engine_error(exc) from exc

    db: Database = request.app.state.db
    params = {
        "width": width,
        "height": height,
        "steps": body.steps or cfg["steps"],
        "guidance": body.guidance if body.guidance is not None else cfg["guidance"],
        "seed": result["seed"],
        "negative_prompt": payload["negative_prompt"],
        "seconds": result.get("generate_seconds"),
    }
    data = []
    for b64 in result["images"]:
        image_id = uuid.uuid4().hex
        (images_dir(request) / f"{image_id}.png").write_bytes(base64.b64decode(b64))
        db.add_image(image_id, f"image/{variant['tag']}", body.prompt, params)
        data.append({"b64_json": b64, "id": image_id})
    return {"created": int(time.time()), "data": data, "seed": result["seed"], "timings": result}


@api.get("/images")
def list_images(request: Request) -> list[dict]:
    return request.app.state.db.list_images()


@api.get("/images/{image_id}")
def get_image(request: Request, image_id: str) -> FileResponse:
    path = images_dir(request) / f"{image_id}.png"
    if not image_id.isalnum() or not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


@api.delete("/images/{image_id}", status_code=204)
def delete_image(request: Request, image_id: str) -> None:
    if not image_id.isalnum() or not request.app.state.db.delete_image(image_id):
        raise HTTPException(404, "Image not found")
    (images_dir(request) / f"{image_id}.png").unlink(missing_ok=True)


# ---- audio ----


@v1.post("/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    model_dir, _ = resolve(request, model, "audio", "transcription")
    tmp = request.app.state.settings.data_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"{uuid.uuid4().hex}{Path(file.filename or '').suffix[:10]}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Audio file too large (max 500 MB)")
                out.write(chunk)
        result = (
            await request.app.state.engines.call(
                "audio",
                "/transcribe",
                {"model_dir": str(model_dir), "path": str(path), "language": language},
            )
        ).json()
    except EngineError as exc:
        raise engine_error(exc) from exc
    finally:
        path.unlink(missing_ok=True)
    if response_format == "text":
        return Response(result["text"], media_type="text/plain")
    if response_format == "verbose_json":
        return result
    return {"text": result["text"]}


class SpeechRequest(BaseModel):
    model: str
    input: str = Field(min_length=1, max_length=10000)
    voice: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: str = "wav"


@v1.post("/audio/speech")
async def speech(request: Request, body: SpeechRequest) -> Response:
    if body.response_format not in ("wav", "flac", "ogg"):
        raise HTTPException(400, "response_format must be wav, flac or ogg")
    model_dir, _ = resolve(request, body.model, "audio", "speech")
    try:
        resp = await request.app.state.engines.call(
            "audio",
            "/speak",
            {
                "model_dir": str(model_dir),
                "text": body.input,
                "voice": body.voice,
                "speed": body.speed,
                "format": body.response_format,
            },
        )
    except EngineError as exc:
        raise engine_error(exc) from exc
    return Response(resp.content, media_type=resp.headers.get("content-type", "audio/wav"))


@api.get("/audio/voices")
async def voices(request: Request, model: str) -> dict:
    model_dir, _ = resolve(request, model, "audio", "speech")
    try:
        resp = await request.app.state.engines.call(
            "audio", "/voices", {"model_dir": str(model_dir)}
        )
    except EngineError as exc:
        raise engine_error(exc) from exc
    return resp.json()
