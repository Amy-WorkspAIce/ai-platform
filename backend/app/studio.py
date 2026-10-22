"""Creative operations shared by the HTTP API and the chat's tools.

The chat model delegates to these ("generate me a photo", "make it nighttime", "turn it into
a video", "read that aloud"); the ``/v1`` endpoints expose the same operations to other apps.
Everything produced is saved to the library (images, videos, audio).
"""

import asyncio
import base64
import logging
import mimetypes
import time
import uuid
from pathlib import Path

from app import catalog
from app.engines import EngineError

log = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


class StudioError(Exception):
    """A problem worth showing to the user (or the chat model) as-is."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def data_dir(app, kind: str) -> Path:
    d = app.state.settings.data_dir / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve(app, model: str, engine: str, task: str | None = None) -> tuple[Path, dict]:
    """Map ``image/<tag>`` (or a bare tag) to an installed engine model."""
    tag = model.removeprefix(f"{engine}/")
    engines = app.state.engines
    variant = engines.installed_variant(engine, tag)
    if variant is None:
        found = catalog.find_variant(tag)
        if found and found[0]["engine"] == engine:
            raise StudioError(f"'{model}' is not installed. Install it from the Models page.", 409)
        raise StudioError(f"Unknown {engine} model '{model}'", 404)
    actual = variant["engine_config"].get("task") or ("image" if engine == "image" else None)
    if task and actual != task:
        raise StudioError(f"'{model}' does not support {task}")
    return engines.model_dir(engine, tag), variant


def image_path(app, image_id: str) -> Path | None:
    if not image_id.isalnum():
        return None
    base = data_dir(app, "images")
    for ext in IMAGE_EXTS:
        p = base / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


async def _free_gpu_for_media(app) -> None:
    # Language models may be holding GPU memory; release it for the media pipeline.
    if app.state.ollama_admin:
        await app.state.ollama_admin.unload_all()


async def park_media(app) -> None:
    """Give the GPU back to the chat model after a media job (no-op if nothing to move)."""
    try:
        if "image" in app.state.engines.workers:
            await app.state.engines.call("image", "/park", {}, timeout=60)
    except EngineError as exc:
        log.info("Couldn't park image pipeline: %s", exc)


# ---------------------------------------------------------------- images


async def supports_editing(app, model: str) -> bool:
    model_dir, _ = resolve(app, model, "image", "image")
    try:
        resp = await app.state.engines.call(
            "image", "/supports_image", {"model_dir": str(model_dir)}, timeout=120
        )
        return bool(resp.json().get("supports_image"))
    except EngineError:
        return False


async def create_image(
    app,
    model: str,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    n: int = 1,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    init_image_id: str | None = None,
    source: str = "api",
) -> dict:
    """Generate (or, with ``init_image_id`` on a model that supports it, edit) images."""
    model_dir, variant = resolve(app, model, "image", "image")
    cfg = variant["engine_config"]
    default = cfg.get("size") or 1024
    width, height = width or default, height or default
    init_path = image_path(app, init_image_id) if init_image_id else None
    if init_image_id and init_path is None:
        raise StudioError(f"Image {init_image_id} not found", 404)

    await _free_gpu_for_media(app)
    payload = {
        "model_dir": str(model_dir),
        "config": cfg,
        "prompt": prompt,
        "negative_prompt": negative_prompt if cfg.get("negative_prompt") else None,
        "width": width,
        "height": height,
        "n": n,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "init_image": str(init_path) if init_path else None,
    }
    try:
        result = (await app.state.engines.call("image", "/generate", payload)).json()
    except EngineError as exc:
        raise StudioError(str(exc), 502) from exc

    params = {
        "width": width,
        "height": height,
        "steps": steps or cfg.get("steps"),
        "guidance": guidance if guidance is not None else cfg.get("guidance"),
        "seed": result["seed"],
        "negative_prompt": payload["negative_prompt"],
        "seconds": result.get("generate_seconds"),
        "source": source,
        "edited_from": init_image_id if result.get("edited") else None,
    }
    ids = []
    for b64 in result["images"]:
        image_id = uuid.uuid4().hex
        (data_dir(app, "images") / f"{image_id}.png").write_bytes(base64.b64decode(b64))
        app.state.db.add_image(image_id, f"image/{variant['tag']}", prompt, params)
        ids.append(image_id)
    return {
        "ids": ids,
        "images": result["images"],
        "seed": result["seed"],
        "edited": bool(result.get("edited")),
        "model": f"image/{variant['tag']}",
        "params": params,
        "timings": {k: result.get(k) for k in ("placement", "load_seconds", "generate_seconds")},
    }


def save_upload(app, data: bytes, filename: str, content_type: str | None) -> str:
    """Store a user-provided image in the library (so the chat can edit it)."""
    ext = Path(filename or "").suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = mimetypes.guess_extension(content_type or "") or ".png"
        ext = ".jpg" if ext == ".jpe" else ext
    if ext not in IMAGE_EXTS:
        raise StudioError("Unsupported image type (use PNG, JPEG or WebP)")
    image_id = uuid.uuid4().hex
    (data_dir(app, "images") / f"{image_id}{ext}").write_bytes(data)
    app.state.db.add_image(image_id, "upload", filename or "Uploaded image", {"source": "upload"})
    return image_id


def image_record(app, image_id: str) -> dict | None:
    return next((i for i in app.state.db.list_images(10_000) if i["id"] == image_id), None)


# ---------------------------------------------------------------- video


def _round(x: int, to: int) -> int:
    return max(to, round(x / to) * to)


def start_video(
    app,
    model: str,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
) -> dict:
    """Start a text-to-video job in the background and return it."""
    model_dir, variant = resolve(app, model, "image", "video")
    cfg = variant["engine_config"]
    video_id = uuid.uuid4().hex
    width = _round(width or cfg.get("width", 832), 32)
    height = _round(height or cfg.get("height", 480), 32)
    # Video models want 4k+1 (Wan) or 8k+1 (LTX) frames; 8k+1 satisfies both.
    frames = (max(9, frames or cfg.get("frames", 33)) - 1) // 8 * 8 + 1
    fps = fps or cfg.get("fps", 16)
    job = {
        "id": video_id,
        "model": f"image/{variant['tag']}",
        "prompt": prompt,
        "status": "queued",
        "phase": "queued",
        "step": 0,
        "steps": 0,
        "error": None,
        "started_at": time.time(),
        "params": {"width": width, "height": height, "frames": frames, "fps": fps},
    }
    app.state.video_jobs[video_id] = job
    payload = {
        "model_dir": str(model_dir),
        "config": cfg,
        "prompt": prompt,
        "negative_prompt": negative_prompt if cfg.get("negative_prompt") else None,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "steps": steps,
        "seed": seed,
        "output": str(data_dir(app, "videos") / f"{video_id}.mp4"),
    }
    task = asyncio.create_task(_run_video(app, job, payload))
    app.state.background_tasks.add(task)
    task.add_done_callback(app.state.background_tasks.discard)
    return job


async def _run_video(app, job: dict, payload: dict) -> None:
    try:
        job["status"] = "running"
        await _free_gpu_for_media(app)
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


# ---------------------------------------------------------------- audio


def default_voice(app) -> str | None:
    us = getattr(app.state, "user_settings", None)
    return us.voice if us else None


async def speak(
    app, model: str, text: str, voice: str | None = None, speed: float = 1.0, fmt: str = "wav"
) -> bytes:
    model_dir, _ = resolve(app, model, "audio", "speech")
    voice = voice or default_voice(app)
    try:
        resp = await app.state.engines.call(
            "audio",
            "/speak",
            {
                "model_dir": str(model_dir),
                "text": text,
                "voice": voice,
                "speed": speed,
                "format": fmt,
            },
        )
    except EngineError as exc:
        raise StudioError(str(exc), 502) from exc
    return resp.content


async def speak_to_library(
    app, model: str, text: str, voice: str | None = None, speed: float = 1.0
) -> dict:
    voice = voice or default_voice(app)
    audio = await speak(app, model, text, voice, speed)
    audio_id = uuid.uuid4().hex
    (data_dir(app, "audio") / f"{audio_id}.wav").write_bytes(audio)
    tag = model.removeprefix("audio/")
    return app.state.db.add_audio(
        audio_id, f"audio/{tag}", text, {"voice": voice or "af_heart", "speed": speed}
    )


async def transcribe_file(app, model: str, path: Path, language: str | None = None) -> dict:
    model_dir, _ = resolve(app, model, "audio", "transcription")
    try:
        resp = await app.state.engines.call(
            "audio",
            "/transcribe",
            {"model_dir": str(model_dir), "path": str(path), "language": language},
        )
    except EngineError as exc:
        raise StudioError(str(exc), 502) from exc
    return resp.json()
