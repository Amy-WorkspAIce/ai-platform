"""Audio worker: speech-to-text (faster-whisper) and text-to-speech (Kokoro, ONNX).

Runs in /data/engines/audio/venv. Whisper uses the GPU when CUDA libraries are present,
otherwise int8 on the CPU. Kokoro is small enough to run on the CPU in real time.
"""

import io
import threading
from pathlib import Path

import _worker_common as common

_lock = threading.Lock()
_whisper: dict[str, object] = {}
_kokoro: dict[str, object] = {}

# Kokoro voice prefix -> espeak language code.
LANGS = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}
FORMATS = {"wav": ("WAV", "audio/wav"), "flac": ("FLAC", "audio/flac"), "ogg": ("OGG", "audio/ogg")}


def _whisper_model(model_dir: str):
    if model_dir not in _whisper:
        import ctranslate2
        from faster_whisper import WhisperModel

        cuda = ctranslate2.get_cuda_device_count() > 0
        _whisper.clear()
        _whisper[model_dir] = WhisperModel(
            model_dir,
            device="cuda" if cuda else "cpu",
            compute_type="float16" if cuda else "int8",
        )
        print(f"[amy-audio] loaded whisper {model_dir} on {'cuda' if cuda else 'cpu'}", flush=True)
    return _whisper[model_dir]


def _kokoro_model(model_dir: str):
    if model_dir not in _kokoro:
        from kokoro_onnx import Kokoro

        d = Path(model_dir)
        _kokoro[model_dir] = Kokoro(str(d / "kokoro-v1.0.onnx"), str(d / "voices-v1.0.bin"))
    return _kokoro[model_dir]


def transcribe(body: dict) -> common.Response:
    with _lock:
        model = _whisper_model(body["model_dir"])
        segments, info = model.transcribe(
            body["path"],
            language=body.get("language") or None,
            task="translate" if body.get("translate") else "transcribe",
            vad_filter=True,
        )
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    return common.json_response(
        {
            "text": "".join(s["text"] for s in segs).strip(),
            "language": info.language,
            "duration": info.duration,
            "segments": segs,
        }
    )


def speak(body: dict) -> common.Response:
    import soundfile as sf

    fmt, mime = FORMATS.get(body.get("format", "wav"), FORMATS["wav"])
    voice = body.get("voice") or "af_heart"
    with _lock:
        kokoro = _kokoro_model(body["model_dir"])
        if voice not in kokoro.get_voices():
            voice = "af_heart"  # the saved default may be another model's voice
        samples, rate = kokoro.create(
            body["text"],
            voice=voice,
            speed=float(body.get("speed") or 1.0),
            lang=LANGS.get(voice[:1], "en-us"),
        )
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format=fmt)
    return 200, mime, buf.getvalue()


def voices(body: dict) -> common.Response:
    with _lock:
        kokoro = _kokoro_model(body["model_dir"])
        names = sorted(kokoro.get_voices())
    return common.json_response({"voices": names})


if __name__ == "__main__":
    common.main({"/transcribe": transcribe, "/speak": speak, "/voices": voices})
