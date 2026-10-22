"""Speech runtimes: text-to-speech models that need their own library.

One worker script serves every speech runtime (``tts-chatterbox``, ``tts-f5``...); each
runtime has its own virtualenv with only its library installed, and each request says which
backend to use (``config["backend"]``, saved with the model at install). Libraries are
imported lazily, so a runtime's venv only needs its own.

Every backend turns text into float samples at some rate; the worker encodes the audio.
Unknown voices fall back to the backend's default, since the platform's default voice may
belong to another model.

**Custom voices.** Besides its built-in voices, a backend may accept the user's own voices,
which the platform sends as ``profile``:

- ``{"kind": "clone", "audio": "/data/voices/x.wav", "text": "what the clip says" | None}``:
  speak like a short recording. Some backends need the clip's transcript (``needs_text``).
- ``{"kind": "design", "description": "A calm older man with a deep voice"}``: a voice
  described in words.

``/voices`` reports which kinds a model accepts, so the platform only offers voices it can use.
"""

import io
import json
import re
import threading
from pathlib import Path

import _worker_common as common

_lock = threading.Lock()
_loaded: dict[str, "Backend"] = {}
FORMATS = {"wav": ("WAV", "audio/wav"), "flac": ("FLAC", "audio/flac"), "ogg": ("OGG", "audio/ogg")}
DEFAULT = "default"

# Ready-made described voices, for models that make a voice from a description.
DESIGNS = {
    "Warm narrator": "A middle-aged man with a warm, deep and calm voice, like an audiobook "
    "narrator.",
    "Bright host": "A young woman with a bright, friendly and upbeat voice.",
    "Gentle storyteller": "An elderly woman with a soft, gentle and kind voice.",
    "Newsreader": "A man in his thirties with a clear, confident and neutral newsreader voice.",
    "Soft whisper": "A young woman whispering softly and slowly.",
    "Lively teen": "A teenage boy with an energetic, fast and cheerful voice.",
}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _numpy(audio):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).squeeze()


class Backend:
    # Which custom voice kinds this backend accepts, and whether clones need a transcript.
    accepts: tuple[str, ...] = ()
    needs_text = False

    def __init__(self, model_dir: Path, config: dict):
        self.dir = model_dir
        self.config = config

    def voices(self) -> list[str]:
        return [DEFAULT]

    def speak(self, text: str, voice: str | None, speed: float, profile: dict | None = None):
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "voices": self.voices(),
            "accepts": list(self.accepts),
            "needs_text": self.needs_text,
        }

    def check(self, profile: dict | None) -> dict | None:
        """The profile if this backend can use it (else None: use a built-in voice)."""
        if not profile or profile.get("kind") not in self.accepts:
            return None
        if profile["kind"] == "clone" and self.needs_text and not profile.get("text"):
            raise ValueError("This model needs the words spoken in the voice sample")
        return profile

    @staticmethod
    def described(voice: str | None, profile: dict | None) -> str | None:
        """The description to speak with: the user's designed voice or a ready-made one."""
        if profile and profile["kind"] == "design":
            return profile["description"]
        return DESIGNS.get(voice or "")


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

    def speak(self, text, voice, speed, profile=None):
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
    """Resemble AI's Chatterbox: natural English speech; clones a voice from a short clip."""

    accepts = ("clone",)

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_local(str(model_dir), device=_device())

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        audio = profile["audio"] if profile else None
        return _numpy(self.model.generate(text, audio_prompt_path=audio)), self.model.sr


class F5(Backend):
    """F5-TTS: speaks in the voice of a short reference clip (its bundled English sample, or
    the user's own with its transcript)."""

    accepts = ("clone",)
    needs_text = True
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

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        ref, ref_text = (
            (profile["audio"], profile["text"]) if profile else (self.ref, self.REF_TEXT)
        )
        wav, sr, _ = self.model.infer(
            ref_file=ref, ref_text=ref_text, gen_text=text, speed=speed, nfe_step=24
        )
        return _numpy(wav), sr


class Xtts(Backend):
    """Coqui XTTS v2: 17 languages and 58 built-in speakers; clones a voice from a clip."""

    accepts = ("clone",)

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts as Model

        cfg = XttsConfig()
        cfg.load_json(str(model_dir / "config.json"))
        self.model = Model.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=str(model_dir), use_deepspeed=False)
        self.model.to(_device())
        self.speakers = dict(self.model.speaker_manager.speakers)
        self.clones: dict[str, dict] = {}

    def voices(self):
        return sorted(self.speakers)

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        if profile:
            if profile["audio"] not in self.clones:
                latent, embedding = self.model.get_conditioning_latents(
                    audio_path=[profile["audio"]]
                )
                self.clones[profile["audio"]] = {
                    "gpt_cond_latent": latent,
                    "speaker_embedding": embedding,
                }
            s = self.clones[profile["audio"]]
        else:
            s = self.speakers[voice if voice in self.speakers else self.voices()[0]]
        out = self.model.inference(
            text, "en", s["gpt_cond_latent"], s["speaker_embedding"], speed=speed
        )
        return _numpy(out["wav"]), 24000


class Qwen(Backend):
    """Qwen3-TTS. Three kinds of model: CustomVoice (named speakers), VoiceDesign (a voice
    described in words) and Base (clones a voice from a clip; the transcript is optional but
    makes it closer)."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map=_device())
        self.kind = getattr(self.model.model, "tts_model_type", "custom_voice")
        self.accepts = {"voice_design": ("design",), "base": ("clone",)}.get(self.kind, ())
        self.speakers = list(self.model.get_supported_speakers() or [])
        self.prompts: dict[str, object] = {}

    def voices(self):
        if self.kind == "voice_design":
            return list(DESIGNS)
        if self.kind == "base":
            return []  # only the user's own voices
        return self.speakers or [DEFAULT]

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        if self.kind == "voice_design":
            description = self.described(voice, profile) or next(iter(DESIGNS.values()))
            wavs, sr = self.model.generate_voice_design(text=text, instruct=description)
        elif self.kind == "base":
            if not profile:
                raise ValueError(
                    "This model speaks in a voice you give it: add a voice sample in Settings first"
                )
            ref_text = profile.get("text") or None
            key = f"{profile['audio']}|{ref_text}"
            if key not in self.prompts:
                self.prompts[key] = self.model.create_voice_clone_prompt(
                    ref_audio=profile["audio"], ref_text=ref_text, x_vector_only_mode=not ref_text
                )
            wavs, sr = self.model.generate_voice_clone(
                text=text, voice_clone_prompt=self.prompts[key]
            )
        else:
            speaker = voice if voice in self.speakers else (self.speakers or [None])[0]
            wavs, sr = self.model.generate_custom_voice(text=text, speaker=speaker)
        return _numpy(wavs[0]), sr


# OmniVoice describes voices with a fixed vocabulary; pick out what a free description says.
# One tag per group, first match wins.
OMNI_TAGS = [
    (r"\b(woman|female|girl|lady|she|her)\b", "female"),
    (r"\b(man|male|boy|guy|gentleman|he|his)\b", "male"),
    (r"\b(child|kid|little)\b", "child"),
    (r"\b(teen|teenage|teenager|adolescent)\b", "teenager"),
    (r"\b(elderly|old|older|aged|grandmother|grandfather|senior)\b", "elderly"),
    (r"\b(middle-aged|thirties|forties|fifties|mature)\b", "middle-aged"),
    (r"\b(young|twenties)\b", "young adult"),
    (r"\b(very deep|very low)\b", "very low pitch"),
    (r"\b(deep|low|husky|gravelly)\b", "low pitch"),
    (r"\b(very high|squeaky)\b", "very high pitch"),
    (r"\b(high|bright|light)\b", "high pitch"),
    (r"\b(whisper|whispers|whispering|whispered)\b", "whisper"),
]
OMNI_GROUPS = [
    {"female", "male"},
    {"child", "teenager", "young adult", "middle-aged", "elderly"},
    {"very low pitch", "low pitch", "high pitch", "very high pitch"},
    {"whisper"},
]
OMNI_ACCENTS = (
    "american",
    "british",
    "australian",
    "chinese",
    "canadian",
    "indian",
    "korean",
    "portuguese",
    "russian",
    "japanese",
)


def omni_tags(description: str) -> str:
    """ "An older British man with a deep voice" -> "male, elderly, low pitch, british accent"."""
    text = description.lower()
    picked: list[str] = []
    for pattern, tag in OMNI_TAGS:
        group = next(g for g in OMNI_GROUPS if tag in g)
        if not group & set(picked) and re.search(pattern, text):
            picked.append(tag)
    accent = next((a for a in OMNI_ACCENTS if re.search(rf"\b{a}\b", text)), None)
    if accent:
        picked.append(f"{accent} accent")
    return ", ".join(picked)


class OmniVoice(Backend):
    """k2-fsa OmniVoice: 600+ languages. Picks a voice itself, designs one from a description
    (gender, age, pitch, accent) or clones one from a clip and its transcript."""

    accepts = ("clone", "design")
    needs_text = True  # without one it would download a speech recogniser

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        import torch
        from omnivoice import OmniVoice as Model

        dtype = torch.float16 if _device() == "cuda" else torch.float32
        self.model = Model.from_pretrained(str(model_dir), device_map=_device(), dtype=dtype)
        self.prompts: dict[str, object] = {}

    def voices(self):
        return [DEFAULT, *DESIGNS]

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        kwargs: dict = {"speed": speed if speed != 1.0 else None}
        if profile and profile["kind"] == "clone":
            key = f"{profile['audio']}|{profile['text']}"
            if key not in self.prompts:
                self.prompts[key] = self.model.create_voice_clone_prompt(
                    ref_audio=profile["audio"], ref_text=profile["text"]
                )
            kwargs["voice_clone_prompt"] = self.prompts[key]
        elif description := self.described(voice, profile):
            kwargs["instruct"] = omni_tags(description) or None
        audio = self.model.generate(text=text, **kwargs)
        return _numpy(audio[0]), self.model.sampling_rate


class VoxCpm(Backend):
    """OpenBMB VoxCPM2: 30 languages at 48 kHz. Picks a voice itself, designs one from a
    description, or clones one from a clip (its transcript makes the clone closer)."""

    accepts = ("clone", "design")

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        from voxcpm import VoxCPM

        self.model = VoxCPM(
            voxcpm_model_path=str(model_dir),
            zipenhancer_model_path=None,
            enable_denoiser=False,
            optimize=False,
            device=_device(),
        )

    def voices(self):
        return [DEFAULT, *DESIGNS]

    def speak(self, text, voice, speed, profile=None):
        profile = self.check(profile)
        kwargs: dict = {}
        if profile and profile["kind"] == "clone":
            kwargs["reference_wav_path"] = profile["audio"]
            if profile.get("text"):
                kwargs.update(prompt_wav_path=profile["audio"], prompt_text=profile["text"])
        elif description := self.described(voice, profile):
            text = f"({description.strip('()')}){text}"
        wav = self.model.generate(text=text, cfg_value=2.0, inference_timesteps=10, **kwargs)
        return _numpy(wav), self.model.tts_model.sample_rate


BACKENDS = {
    "transformers": Transformers,
    "chatterbox": Chatterbox,
    "f5": F5,
    "xtts": Xtts,
    "qwen": Qwen,
    "omnivoice": OmniVoice,
    "voxcpm": VoxCpm,
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
            body["text"],
            body.get("voice"),
            float(body.get("speed") or 1.0),
            body.get("profile"),
        )
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format=fmt)
    return 200, mime, buf.getvalue()


def voices(body: dict) -> common.Response:
    with _lock:
        info = _model(body).info()
    return common.json_response(info)


def after_download(dest: Path, config: dict) -> None:
    """Fetch the extra repos a runtime needs next to the model (e.g. F5-TTS's vocoder)."""
    for name, spec in (config.get("extras") or {}).items():
        common.download(spec, dest / name)


if __name__ == "__main__":
    common.main({"/speak": speak, "/voices": voices}, after_download)
