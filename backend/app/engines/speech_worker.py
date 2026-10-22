"""Speech runtimes: text-to-speech models that need their own library.

One worker script serves every speech runtime (``tts-chatterbox``, ``tts-f5``...); each
runtime has its own virtualenv with only its library installed, and each request says which
backend to use (``config["backend"]``, saved with the model at install). Libraries are
imported lazily, so a runtime's venv only needs its own.

Every backend turns text into float samples at some rate; the worker encodes the audio.
Unknown voices fall back to the backend's default, since the platform's default voice may
belong to another model.
"""

import io
import json
import threading
from pathlib import Path

import _worker_common as common

_lock = threading.Lock()
_loaded: dict[str, "Backend"] = {}
FORMATS = {"wav": ("WAV", "audio/wav"), "flac": ("FLAC", "audio/flac"), "ogg": ("OGG", "audio/ogg")}
DEFAULT = "default"


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _numpy(audio):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).squeeze()


class Backend:
    def __init__(self, model_dir: Path, config: dict):
        self.dir = model_dir
        self.config = config

    def voices(self) -> list[str]:
        return [DEFAULT]

    def speak(self, text: str, voice: str | None, speed: float):
        raise NotImplementedError


class Transformers(Backend):
    """VITS/MMS (one voice, 1,100+ languages) and Bark (voice presets) via transformers."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        model_type = json.loads((model_dir / "config.json").read_text()).get("model_type")
        self.presets: dict[str, dict] = {}
        if model_type == "bark":
            from transformers import AutoProcessor, BarkModel

            self.processor = AutoProcessor.from_pretrained(str(model_dir))
            self.model = BarkModel.from_pretrained(str(model_dir)).to(_device())
            # The preset index points at another repo, so load the files from this one.
            index = model_dir / "speaker_embeddings_path.json"
            if index.exists():
                self.presets = {
                    name: files
                    for name, files in json.loads(index.read_text()).items()
                    if name.startswith("v2/") and isinstance(files, dict)
                }
        else:
            from transformers import pipeline

            self.pipe = pipeline("text-to-speech", model=str(model_dir), device=_device())

    def voices(self):
        return sorted(self.presets) or [DEFAULT]

    def speak(self, text, voice, speed):
        if not self.presets and not hasattr(self, "pipe"):
            raise ValueError("This Bark model has no voice presets")
        if self.presets:
            import numpy as np

            name = voice if voice in self.presets else "v2/en_speaker_6"
            files = self.presets.get(name) or next(iter(self.presets.values()))
            preset = {k: np.load(self.dir / path) for k, path in files.items()}
            inputs = self.processor(text, voice_preset=preset).to(_device())
            audio = self.model.generate(**inputs)
            return _numpy(audio), self.model.generation_config.sample_rate
        out = self.pipe(text)
        return _numpy(out["audio"]), out["sampling_rate"]


class Chatterbox(Backend):
    """Resemble AI's Chatterbox: natural English speech with an expressiveness control."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_local(str(model_dir), device=_device())

    def speak(self, text, voice, speed):
        return _numpy(self.model.generate(text)), self.model.sr


class F5(Backend):
    """F5-TTS: speaks in the voice of a short reference clip (its bundled English sample)."""

    REF_TEXT = "Some call me nature, others call me mother nature."

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from importlib.resources import files

        from f5_tts.api import F5TTS

        ckpts = sorted(
            p
            for p in model_dir.rglob("*")
            if p.suffix in (".safetensors", ".pt") and "vocos" not in p.parts
        )
        vocab = next(iter(sorted(model_dir.rglob("vocab.txt"))), None)
        self.model = F5TTS(
            ckpt_file=str(ckpts[-1]),
            vocab_file=str(vocab) if vocab else "",
            vocoder_local_path=str(model_dir / "vocos"),
            device=_device(),
        )
        self.ref = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))

    def speak(self, text, voice, speed):
        wav, sr, _ = self.model.infer(
            ref_file=self.ref, ref_text=self.REF_TEXT, gen_text=text, speed=speed, nfe_step=24
        )
        return _numpy(wav), sr


class Xtts(Backend):
    """Coqui XTTS v2: 17 languages and 58 built-in speakers."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts as Model

        cfg = XttsConfig()
        cfg.load_json(str(model_dir / "config.json"))
        self.model = Model.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=str(model_dir), use_deepspeed=False)
        self.model.to(_device())
        self.speakers = self.model.speaker_manager.speakers

    def voices(self):
        return sorted(self.speakers)

    def speak(self, text, voice, speed):
        name = voice if voice in self.speakers else sorted(self.speakers)[0]
        s = self.speakers[name]
        out = self.model.inference(
            text, "en", s["gpt_cond_latent"], s["speaker_embedding"], speed=speed
        )
        return _numpy(out["wav"]), 24000


class Qwen(Backend):
    """Qwen3-TTS (CustomVoice models): several named speakers across languages."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map=_device())
        self.speakers = list(self.model.get_supported_speakers() or [])

    def voices(self):
        return self.speakers or [DEFAULT]

    def speak(self, text, voice, speed):
        speaker = voice if voice in self.speakers else (self.speakers or [None])[0]
        wavs, sr = self.model.generate_custom_voice(text=text, speaker=speaker)
        return _numpy(wavs[0]), sr


BACKENDS = {
    "transformers": Transformers,
    "chatterbox": Chatterbox,
    "f5": F5,
    "xtts": Xtts,
    "qwen": Qwen,
}


def _model(body: dict) -> Backend:
    model_dir = body["model_dir"]
    if model_dir not in _loaded:
        _loaded.clear()  # one model at a time: these are big
        config = body.get("config") or {}
        _loaded[model_dir] = BACKENDS[config["backend"]](Path(model_dir), config)
        print(f"[amy-speech] loaded {config['backend']} {model_dir}", flush=True)
    return _loaded[model_dir]


def speak(body: dict) -> common.Response:
    import soundfile as sf

    fmt, mime = FORMATS.get(body.get("format", "wav"), FORMATS["wav"])
    with _lock:
        samples, rate = _model(body).speak(
            body["text"], body.get("voice"), float(body.get("speed") or 1.0)
        )
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format=fmt)
    return 200, mime, buf.getvalue()


def voices(body: dict) -> common.Response:
    with _lock:
        names = _model(body).voices()
    return common.json_response({"voices": names})


def after_download(dest: Path, config: dict) -> None:
    """Fetch the extra repos a runtime needs next to the model (e.g. F5-TTS's vocoder)."""
    for name, spec in (config.get("extras") or {}).items():
        common.download(spec, dest / name)


if __name__ == "__main__":
    common.main({"/speak": speak, "/voices": voices}, after_download)
