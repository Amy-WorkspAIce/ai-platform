"""Custom voices: the user's own voices, usable by speech models that accept them.

Two kinds:

- **clone**: a short recording (a WAV clip, 3 to 60 seconds) and, ideally, what it says. Models
  such as XTTS, Chatterbox, F5-TTS, Qwen3-TTS Base, OmniVoice and VoxCPM speak in that voice.
- **design**: a voice described in words ("a calm older man with a deep voice"), for
  Qwen3-TTS VoiceDesign, OmniVoice and VoxCPM.

A custom voice is chosen like any other, by id: ``custom:<id>``. When speaking, the platform
turns it into a ``profile`` for the speech worker (see ``app.engines.speech_worker``). Clips
live in ``<data>/voices/<id>.wav``; the browser converts recordings to WAV before upload, so
the platform needs no audio libraries.
"""

import io
import wave
from pathlib import Path

from app import voice_meta

PREFIX = "custom:"
MIN_SECONDS = 3
MAX_SECONDS = 60
MAX_BYTES = 20 * 1024 * 1024


class VoiceError(ValueError):
    pass


def voices_dir(app) -> Path:
    path = Path(app.state.settings.data_dir) / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clip_path(app, voice_id: str) -> Path:
    return voices_dir(app) / f"{voice_id}.wav"


def wav_seconds(data: bytes) -> float:
    """The length of a WAV clip, checking it's one the speech engines can read."""
    if len(data) > MAX_BYTES:
        raise VoiceError(f"The sample is too large (max {MAX_BYTES // (1024 * 1024)} MB)")
    try:
        with wave.open(io.BytesIO(data)) as w:
            seconds = w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError) as exc:
        raise VoiceError("The sample must be a WAV file (16-bit PCM)") from exc
    if seconds < MIN_SECONDS:
        raise VoiceError(f"The sample is too short: record at least {MIN_SECONDS} seconds")
    if seconds > MAX_SECONDS:
        raise VoiceError(f"The sample is too long: keep it under {MAX_SECONDS} seconds")
    return round(seconds, 1)


def public(voice: dict) -> dict:
    """A stored voice as the API shows it."""
    out = voice | {"id": f"{PREFIX}{voice['id']}"}
    repo, _, path = (voice.get("source") or "").partition(":")
    if repo == voice_meta.REPO:  # installed from the voice library: keep its labels
        out["meta"] = voice_meta.lookup(path)
    return out


def get(app, voice: str | None) -> dict | None:
    """The stored custom voice a voice id names (None for built-in voices)."""
    if not voice or not voice.startswith(PREFIX):
        return None
    return app.state.db.get_voice(voice.removeprefix(PREFIX))


def profile(app, voice: str | None) -> dict | None:
    """What the speech worker needs to speak in a custom voice."""
    stored = get(app, voice)
    if stored is None:
        return None
    if stored["kind"] == "design":
        return {"kind": "design", "description": stored["description"]}
    return {
        "kind": "clone",
        "audio": str(clip_path(app, stored["id"])),
        "text": stored["transcript"] or None,
    }


def usable(app, info: dict) -> list[dict]:
    """The custom voices a model can speak in, given what its ``/voices`` reported."""
    accepts = set(info.get("accepts") or [])
    out = []
    for v in app.state.db.list_voices():
        if v["kind"] not in accepts:
            continue
        if v["kind"] == "clone" and info.get("needs_text") and not v["transcript"]:
            continue
        out.append(public(v))
    return out
