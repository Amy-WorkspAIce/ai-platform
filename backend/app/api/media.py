"""Image generation and audio, OpenAI-compatible where possible.

- ``POST /v1/images/generations``    text-to-image (saved to the gallery)
- ``POST /v1/videos/generations``    text-to-video, as a background job (minutes)
- ``POST /v1/audio/transcriptions``  speech-to-text (multipart upload)
- ``POST /v1/audio/speech``          text-to-speech (returns audio bytes)
- ``/api/images``                    gallery: list, fetch, delete
- ``GET /api/audio/voices``          voices for a text-to-speech model
"""

import asyncio
import base64
import logging
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
log = logging.getLogger(__name__)


def resolve(
    request: Request, model: str, engine: str, task: str | None = None
) -> tuple[Path, dict]:
    """Map ``image/<tag>`` (or a bare tag) to an installed engine model."""
    tag = model.removeprefix(f"{engine}/")
    engines: EngineManager = request.app.state.engines
    variant = engines.installed_variant(engine, tag)
    if variant is None:
        found = catalog.find_variant(tag)
        if found and found[0]["engine"] == engine:
            raise HTTPException(
                409, f"'{model}' is not installed. Install it from the Models page."
            )
        raise HTTPException(404, f"Unknown {engine} model '{model}'")
    actual = variant["engine_config"].get("task") or ("image" if engine == "image" else None)
    if task and actual != task:
        raise HTTPException(400, f"'{model}' does not support {task}")
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
    model_dir, variant = resolve(request, body.model, "image", "image")
    cfg = variant["engine_config"]
    width, height = parse_size(body.size, cfg.get("size") or 1024)

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
        "steps": body.steps or cfg.get("steps"),
        "guidance": body.guidance if body.guidance is not None else cfg.get("guidance"),
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


# ---- video ----


class VideoRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = None
    width: int | None = Field(default=None, ge=256, le=1920)
    height: int | None = Field(default=None, ge=256, le=1920)
    frames: int | None = Field(default=None, ge=9, le=241)
    fps: int | None = Field(default=None, ge=4, le=60)
    steps: int | None = Field(default=None, ge=1, le=100)
    seed: int | None = None


def videos_dir(request: Request) -> Path:
    d = request.app.state.settings.data_dir / "videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _round(x: int, to: int) -> int:
    return max(to, round(x / to) * to)


@v1.post("/videos/generations", status_code=202)
async def generate_video(request: Request, body: VideoRequest) -> dict:
    """Start a text-to-video job. Poll ``GET /api/videos/jobs`` for progress."""
    model_dir, variant = resolve(request, body.model, "image", "video")
    cfg = variant["engine_config"]
    video_id = uuid.uuid4().hex
    width = _round(body.width or cfg.get("width", 832), 32)
    height = _round(body.height or cfg.get("height", 480), 32)
    # Video models want 4k+1 (Wan) or 8k+1 (LTX) frames; 8k+1 satisfies both.
    frames = (max(9, body.frames or cfg.get("frames", 33)) - 1) // 8 * 8 + 1
    fps = body.fps or cfg.get("fps", 16)
    job = {
        "id": video_id,
        "model": f"image/{variant['tag']}",
        "prompt": body.prompt,
        "status": "queued",
        "phase": "queued",
        "step": 0,
        "steps": 0,
        "error": None,
        "started_at": time.time(),
        "params": {"width": width, "height": height, "frames": frames, "fps": fps},
    }
    jobs: dict = request.app.state.video_jobs
    jobs[video_id] = job
    payload = {
        "model_dir": str(model_dir),
        "config": cfg,
        "prompt": body.prompt,
        "negative_prompt": body.negative_prompt if cfg.get("negative_prompt") else None,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "steps": body.steps,
        "seed": body.seed,
        "output": str(videos_dir(request) / f"{video_id}.mp4"),
    }
    task = asyncio.create_task(_run_video(request.app, job, payload))
    request.app.state.background_tasks.add(task)
    task.add_done_callback(request.app.state.background_tasks.discard)
    return {"id": video_id, "status": "queued"}


async def _run_video(app, job: dict, payload: dict) -> None:
    try:
        job["status"] = "running"
        if app.state.ollama_admin:
            await app.state.ollama_admin.unload_all()
        result = (
            await app.state.engines.call("image", "/generate_video", payload, timeout=4 * 3600)
        ).json()
        params = job["params"] | {
            "seed": result["seed"],
            "seconds": result.get("generate_seconds"),
            "negative_prompt": payload["negative_prompt"],
            "steps": payload["steps"] or payload["config"].get("steps"),
        }
        app.state.db.add_video(job["id"], job["model"], job["prompt"], params)
        job.update(status="done", phase="done", params=params)
    except EngineError as exc:
        job.update(status="error", error=str(exc))
    except Exception as exc:  # surface anything unexpected in the UI
        log.exception("Video generation failed")
        job.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        job["finished_at"] = time.time()


@api.get("/videos/jobs")
async def video_jobs(request: Request) -> list[dict]:
    jobs: dict = request.app.state.video_jobs
    cutoff = time.time() - 3600
    for k in [k for k, j in jobs.items() if j.get("finished_at", time.time()) < cutoff]:
        jobs.pop(k)
    running = [j for j in jobs.values() if j["status"] == "running"]
    if running:
        progress = await request.app.state.engines.peek("image", "/progress")
        if progress:
            running[0].update(
                phase=progress["phase"], step=progress["step"], steps=progress["steps"]
            )
    return sorted(jobs.values(), key=lambda j: j["started_at"], reverse=True)


@api.get("/videos")
def list_videos(request: Request) -> list[dict]:
    return request.app.state.db.list_videos()


@api.get("/videos/{video_id}")
def get_video(request: Request, video_id: str) -> FileResponse:
    path = videos_dir(request) / f"{video_id}.mp4"
    if not video_id.isalnum() or not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


@api.delete("/videos/{video_id}", status_code=204)
def delete_video(request: Request, video_id: str) -> None:
    if not video_id.isalnum() or not request.app.state.db.delete_video(video_id):
        raise HTTPException(404, "Video not found")
    (videos_dir(request) / f"{video_id}.mp4").unlink(missing_ok=True)
