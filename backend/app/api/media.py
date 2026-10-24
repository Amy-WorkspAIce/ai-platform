"""Media API, OpenAI-compatible where possible. Thin HTTP layer over ``app.studio``.

- ``POST /v1/images/generations``    text-to-image (saved to the library)
- ``POST /v1/videos/generations``    text-to-video, as a background job (minutes)
- ``POST /v1/audio/transcriptions``  speech-to-text (multipart upload)
- ``POST /v1/audio/speech``          text-to-speech (returns audio bytes)
- ``/api/images``, ``/api/videos``, ``/api/audio``: the library (list, fetch, delete)
- ``POST /api/images/upload``        add your own image (e.g. for the chat to edit)
- ``/api/voices``                    custom voices: recorded samples and described voices
"""

import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app import studio, voice_meta
from app import voice_library as library
from app import voice_runners as runners
from app import voices as custom
from app.api.models import _annotated, _standard_live
from app.catalog import _row_sort_key, flatten
from app.delegation import choose
from app.engines.speech_base import STARTERS
from app.security import require_api_key
from app.studio import StudioError

v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def http_error(exc: StudioError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


async def read_limited(file: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data += chunk
        if len(data) > limit:
            raise HTTPException(413, f"File too large (max {limit // (1024 * 1024)} MB)")
    return bytes(data)


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
    image_id: str | None = None  # edit this library image (models that support editing)


def parse_size(size: str | None) -> tuple[int | None, int | None]:
    if not size:
        return None, None
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except ValueError as exc:
        raise HTTPException(400, "size must look like 1024x1024") from exc
    # Diffusion models want multiples of 64; keep within sane bounds.
    clamp = lambda x: max(256, min(2048, round(x / 64) * 64))  # noqa: E731
    return clamp(w), clamp(h)


@v1.post("/images/generations")
async def generate_image(request: Request, body: ImageRequest) -> dict:
    width, height = parse_size(body.size)
    try:
        result = await studio.create_image(
            request.app,
            body.model,
            body.prompt,
            negative_prompt=body.negative_prompt,
            width=width,
            height=height,
            n=body.n,
            steps=body.steps,
            guidance=body.guidance,
            seed=body.seed,
            init_image_id=body.image_id,
        )
    except StudioError as exc:
        raise http_error(exc) from exc
    data = [
        {"b64_json": b64, "id": i} for b64, i in zip(result["images"], result["ids"], strict=True)
    ]
    return {
        "created": int(time.time()),
        "data": data,
        "seed": result["seed"],
        "edited": result["edited"],
        "timings": result["timings"],
    }


@api.post("/images/upload", status_code=201)
async def upload_image(request: Request, file: UploadFile = File(...)) -> dict:
    data = await read_limited(file, MAX_IMAGE_BYTES)
    try:
        image_id = studio.save_upload(request.app, data, file.filename or "", file.content_type)
    except StudioError as exc:
        raise http_error(exc) from exc
    return {"id": image_id}


@api.get("/images")
def list_images(request: Request) -> list[dict]:
    return request.app.state.db.list_images()


@api.get("/images/{image_id}")
def get_image(request: Request, image_id: str) -> FileResponse:
    path = studio.image_path(request.app, image_id)
    if path is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@api.delete("/images/{image_id}", status_code=204)
def delete_image(request: Request, image_id: str) -> None:
    path = studio.image_path(request.app, image_id)
    if path is None or not request.app.state.db.delete_image(image_id):
        raise HTTPException(404, "Image not found")
    path.unlink(missing_ok=True)


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


@v1.post("/videos/generations", status_code=202)
async def generate_video(request: Request, body: VideoRequest) -> dict:
    """Start a text-to-video job. Poll ``GET /api/videos/jobs`` for progress."""
    try:
        job = studio.start_video(request.app, **body.model_dump())
    except StudioError as exc:
        raise http_error(exc) from exc
    return {"id": job["id"], "status": job["status"]}


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
    path = studio.data_dir(request.app, "videos") / f"{video_id}.mp4"
    if not video_id.isalnum() or not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


@api.delete("/videos/{video_id}", status_code=204)
def delete_video(request: Request, video_id: str) -> None:
    if not video_id.isalnum() or not request.app.state.db.delete_video(video_id):
        raise HTTPException(404, "Video not found")
    (studio.data_dir(request.app, "videos") / f"{video_id}.mp4").unlink(missing_ok=True)


# ---- audio ----


@v1.post("/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    path = (
        studio.data_dir(request.app, "tmp")
        / f"{uuid.uuid4().hex}{Path(file.filename or '').suffix[:10]}"
    )
    try:
        path.write_bytes(await read_limited(file, MAX_UPLOAD_BYTES))
        result = await studio.transcribe_file(request.app, model, path, language)
    except StudioError as exc:
        raise http_error(exc) from exc
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
    save: bool = False  # also keep it in the library


@v1.post("/audio/speech")
async def speech(request: Request, body: SpeechRequest) -> Response:
    if body.response_format not in ("wav", "flac", "ogg"):
        raise HTTPException(400, "response_format must be wav, flac or ogg")
    try:
        if body.save and body.response_format == "wav":
            rec = await studio.speak_to_library(
                request.app, body.model, body.input, body.voice, body.speed
            )
            path = studio.data_dir(request.app, "audio") / f"{rec['id']}.wav"
            return Response(
                path.read_bytes(), media_type="audio/wav", headers={"X-Audio-Id": rec["id"]}
            )
        audio = await studio.speak(
            request.app, body.model, body.input, body.voice, body.speed, body.response_format
        )
    except StudioError as exc:
        raise http_error(exc) from exc
    mime = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}[body.response_format]
    return Response(audio, media_type=mime)


@api.get("/audio/voices")
async def voices(request: Request, model: str) -> dict:
    """A model's built-in voices, plus the custom voices it can speak in (``custom``) and
    which kinds of custom voice it accepts."""
    try:
        model_dir, variant = studio.resolve(request.app, model, "audio", "speech")
        engines = request.app.state.engines
        resp = await engines.call(
            engines.runtime("audio", variant),
            "/voices",
            {"model_dir": str(model_dir), "config": variant["engine_config"]},
        )
    except StudioError as exc:
        raise http_error(exc) from exc
    except studio.EngineError as exc:
        raise HTTPException(502, str(exc)) from exc
    info = resp.json()
    return {
        "voices": info["voices"],
        # Names, languages, genders and other labels, for voices whose ids don't say.
        "details": voice_meta.describe(
            info["voices"], info.get("details") or {}, {n: p for n, (p, _, _) in STARTERS.items()}
        ),
        "accepts": info.get("accepts", []),
        "needs_text": info.get("needs_text", False),
        "custom": custom.usable(request.app, info),
    }


# ---- custom voices ----


def _voice_or_404(request: Request, voice_id: str) -> dict:
    stored = request.app.state.db.get_voice(voice_id.removeprefix(custom.PREFIX))
    if stored is None:
        raise HTTPException(404, "Voice not found")
    return stored


async def _transcribe_clip(request: Request, path: Path) -> str | None:
    """What a sample says, if a speech-to-text model is installed (else None)."""
    model = choose(request.app, "transcription")
    if model is None:
        return None
    try:
        result = await studio.transcribe_file(request.app, model["id"], path)
    except StudioError:
        return None
    return (result.get("text") or "").strip() or None


@api.get("/voices")
def list_voices(request: Request) -> list[dict]:
    return [custom.public(v) for v in request.app.state.db.list_voices()]


@api.post("/voices", status_code=201)
async def create_voice(
    request: Request,
    name: str = Form(..., min_length=1, max_length=60),
    kind: str = Form(...),
    description: str | None = Form(None, max_length=500),
    transcript: str | None = Form(None, max_length=2000),
    file: UploadFile | None = File(None),
) -> dict:
    """Add a voice: ``kind=design`` with a ``description``, or ``kind=clone`` with a WAV
    ``file`` (3 to 60 seconds) and ideally its ``transcript``. Without a transcript, the
    installed speech-to-text model (if any) writes one."""
    db = request.app.state.db
    if kind == "design":
        if not (description or "").strip():
            raise HTTPException(400, "Describe the voice")
        voice = db.add_voice(name.strip(), "design", description=description.strip())
        return custom.public(voice)
    if kind != "clone":
        raise HTTPException(400, "kind must be clone or design")
    if file is None:
        raise HTTPException(400, "Add a voice sample (a WAV file)")
    data = await read_limited(file, custom.MAX_BYTES)
    try:
        seconds = custom.wav_seconds(data)
    except custom.VoiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    voice_id = uuid.uuid4().hex
    path = custom.clip_path(request.app, voice_id)
    path.write_bytes(data)
    given = (transcript or "").strip() or None
    text = given or await _transcribe_clip(request, path)
    voice = db.add_voice(name.strip(), "clone", transcript=text, seconds=seconds, voice_id=voice_id)
    return custom.public(voice) | {"transcribed": bool(text and not given)}


SAMPLE_TEXT = (
    "Hello there. This is the voice I would like you to use. It is warm and clear, "
    "and it speaks at an easy, natural pace."
)


class VoiceFromSpeech(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    model: str
    voice: str | None = None
    text: str = Field(default=SAMPLE_TEXT, min_length=10, max_length=500)


@api.post("/voices/from-speech", status_code=201)
async def voice_from_speech(request: Request, body: VoiceFromSpeech) -> dict:
    """Make a voice sample by speaking a sentence with one of an installed model's voices
    (useful for models that only speak in a sample's voice, such as Qwen3-TTS Base)."""
    if body.voice and body.voice.startswith(custom.PREFIX):
        raise HTTPException(400, "Choose a built-in voice to make a sample from")
    try:
        audio = await studio.speak(request.app, body.model, body.text, body.voice or None)
        seconds = custom.wav_seconds(audio)
    except StudioError as exc:
        raise http_error(exc) from exc
    except custom.VoiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    voice_id = uuid.uuid4().hex
    custom.clip_path(request.app, voice_id).write_bytes(audio)
    voice = request.app.state.db.add_voice(
        body.name.strip(), "clone", transcript=body.text, seconds=seconds, voice_id=voice_id
    )
    return custom.public(voice)


@api.get("/voices/library")
async def voice_library(request: Request) -> dict:
    """Ready-made voice samples to install as your own voices (see ``app.voice_library``)."""
    try:
        found = await library.listing(request.app.state.http, _hf_token(request))
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Couldn't reach Hugging Face: {exc}") from exc
    stored = {v["source"]: v for v in request.app.state.db.list_voices() if v.get("source")}
    models = runners.installed(request.app)
    can_transcribe = bool(choose(request.app, "transcription"))
    voices = []
    for e in found:
        voice = stored.get(f"{library.REPO}:{e['id']}")
        transcript = bool(voice["transcript"]) if voice else can_transcribe
        voices.append(
            e
            | {
                "installed": custom.PREFIX + voice["id"] if voice else None,
                "runs_on": runners.runs_on(e["id"], models, transcript),
            }
        )
    # Something to install for the voices no installed model can speak yet.
    suggested = await _suggested_runners(request) if any(not v["runs_on"] for v in voices) else []
    return {"voices": voices, "models": models, "suggested": suggested}


async def _suggested_runners(request: Request) -> list[dict]:
    try:
        families = await _annotated(request, await _standard_live(request.app.state.live_catalog))
    except httpx.HTTPError:
        return []
    runtime = {f["id"]: f.get("runtime") for f in families}
    rows = [r | {"runtime": runtime.get(r["family_id"])} for r in flatten(families)]
    rows = [r for r in rows if "speech" in r["use_cases"]]
    return runners.suggested(sorted(rows, key=_row_sort_key("perfect")))


class LibraryVoice(BaseModel):
    id: str = Field(min_length=1, max_length=300)
    name: str | None = Field(default=None, min_length=1, max_length=60)


@api.post("/voices/library", status_code=201)
async def install_library_voice(request: Request, body: LibraryVoice) -> dict:
    """Install a voice library sample as a custom voice (transcribed if a speech-to-text
    model is installed)."""
    try:
        entry = next(
            e
            for e in await library.listing(request.app.state.http, _hf_token(request))
            if e["id"] == body.id
        )
    except StopIteration:
        raise HTTPException(404, "That voice isn't in the library") from None
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Couldn't reach Hugging Face: {exc}") from exc
    source = f"{library.REPO}:{entry['id']}"
    db = request.app.state.db
    if any(v.get("source") == source for v in db.list_voices()):
        raise HTTPException(409, "That voice is already installed")
    try:
        data = await library.download(request.app.state.http, _hf_token(request), entry["id"])
        seconds = custom.wav_seconds(data)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Couldn't download the voice: {exc}") from exc
    except custom.VoiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    voice_id = uuid.uuid4().hex
    path = custom.clip_path(request.app, voice_id)
    path.write_bytes(data)
    voice = db.add_voice(
        (body.name or entry["name"]).strip(),
        "clone",
        transcript=await _transcribe_clip(request, path),
        seconds=seconds,
        voice_id=voice_id,
        source=source,
        credit=entry["credit"],
    )
    return custom.public(voice)


def _hf_token(request: Request) -> str | None:
    return request.app.state.user_settings.hf_token


class VoiceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=60)
    description: str | None = Field(default=None, min_length=1, max_length=500)
    transcript: str | None = Field(default=None, max_length=2000)  # "" clears it


@api.patch("/voices/{voice_id}")
def update_voice(request: Request, voice_id: str, body: VoiceUpdate) -> dict:
    stored = _voice_or_404(request, voice_id)
    fields = body.model_dump(exclude_unset=True)
    if "transcript" in fields:
        fields["transcript"] = (fields["transcript"] or "").strip() or None
    for key in ("name", "description"):
        if key in fields:
            if not (fields[key] or "").strip():
                raise HTTPException(400, f"{key.capitalize()} can't be empty")
            fields[key] = fields[key].strip()
    return custom.public(request.app.state.db.update_voice(stored["id"], **fields))


@api.delete("/voices/{voice_id}", status_code=204)
def delete_voice(request: Request, voice_id: str) -> None:
    stored = _voice_or_404(request, voice_id)
    request.app.state.db.delete_voice(stored["id"])
    custom.clip_path(request.app, stored["id"]).unlink(missing_ok=True)
    us = request.app.state.user_settings
    if us.voice == custom.PREFIX + stored["id"]:
        us.set_voice(None)


@api.get("/voices/{voice_id}/sample")
def voice_sample(request: Request, voice_id: str) -> FileResponse:
    stored = _voice_or_404(request, voice_id)
    path = custom.clip_path(request.app, stored["id"])
    if stored["kind"] != "clone" or not path.exists():
        raise HTTPException(404, "This voice has no sample")
    return FileResponse(path, media_type="audio/wav")


@api.get("/audio")
def list_audio(request: Request) -> list[dict]:
    return request.app.state.db.list_audio()


@api.get("/audio/{audio_id}")
def get_audio(request: Request, audio_id: str) -> FileResponse:
    path = studio.data_dir(request.app, "audio") / f"{audio_id}.wav"
    if not audio_id.isalnum() or not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/wav")


@api.delete("/audio/{audio_id}", status_code=204)
def delete_audio(request: Request, audio_id: str) -> None:
    if not audio_id.isalnum() or not request.app.state.db.delete_audio(audio_id):
        raise HTTPException(404, "Audio not found")
    (studio.data_dir(request.app, "audio") / f"{audio_id}.wav").unlink(missing_ok=True)
