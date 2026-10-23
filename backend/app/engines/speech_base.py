"""What every speech backend shares (see ``speech_worker``): the ``Backend`` base class, the
ready-made described voices and the starter voice samples.

**Custom voices.** Besides its built-in voices, a backend may accept the user's own voices,
which the platform sends as ``profile``:

- ``{"kind": "clone", "audio": "/data/voices/x.wav", "text": "what the clip says" | None}``:
  speak like a short recording. Some backends need the clip's transcript (``needs_text``).
- ``{"kind": "design", "description": "A calm older man with a deep voice"}``: a voice
  described in words.

``/voices`` reports which kinds a model accepts, so the platform only offers voices it can use.

**Starter voices.** Models that only copy voices would have nothing to say anything with until
the user adds a sample, so they offer a few public-domain and CC BY recordings from
``kyutai/tts-voices`` as built-in voices. The catalog downloads them into the shared Hugging
Face cache at install (``app.catalog.speech.extras``).
"""

import json
from pathlib import Path

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

VOICE_REPO = "kyutai/tts-voices"
# name -> (file in VOICE_REPO, its transcript, gender). Voice-Zero (LibriVox) is CC0; Alba
# MacKenna's recordings are CC BY 4.0.
STARTERS = {
    "Alba": (
        "alba-mackenna/a-moment-by.wav",
        "The scent feels familiar to you, strangely nostalgic, as it brings you back to your "
        "favorite childhood meal.",
        "Female",
    ),
    "Caro": (
        "voice-zero/caro_davy.wav",
        "All LibriVox recordings are in the public domain. For more information or to "
        "volunteer, please visit LibriVox.org.",
        "Female",
    ),
    "Peter": (
        "voice-zero/peter_yearsley.wav",
        "The north wind and the sun were disputing which was the stronger, when a traveller "
        "came along wrapped in a warm cloak.",
        "Male",
    ),
    "Stuart": (
        "voice-zero/stuart_bell.wav",
        "The north wind and the sun were disputing which was the stronger, when a traveller "
        "came along wrapped in a warm cloak.",
        "Male",
    ),
}


def hub_file(repo: str, filename: str) -> str:
    """A file the catalog downloaded into the shared Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, local_files_only=True)


def available_starters() -> dict[str, str]:
    """Starter voice -> its downloaded clip (models installed before starter voices existed
    don't have them)."""
    found = {}
    for name, (path, _, _) in STARTERS.items():
        try:
            found[name] = hub_file(VOICE_REPO, path)
        except Exception:  # noqa: BLE001, S112 - not downloaded
            continue
    return found


def starter(voice: str | None) -> dict:
    """A starter voice as a clone profile (the first one for unknown names)."""
    clips = available_starters()
    if not clips:
        raise ValueError(
            "This model speaks in a voice copied from a sample: add one on the Voices page"
        )
    name = voice if voice in clips else next(iter(clips))
    return {"kind": "clone", "audio": clips[name], "text": STARTERS[name][1]}


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def numpy(audio):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).squeeze()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Backend:
    # Which custom voice kinds this backend accepts, and whether clones need a transcript.
    accepts: tuple[str, ...] = ()
    needs_text = False
    # Only copies voices: offer the starter voices as built-in ones.
    starters = False

    def __init__(self, model_dir: Path, config: dict):
        self.dir = model_dir
        self.config = config

    def voices(self) -> list[str]:
        return list(available_starters()) if self.starters else [DEFAULT]

    def details(self) -> dict[str, dict]:
        """Display names, languages and genders for voices whose ids don't say."""
        if self.starters:
            return {
                n: {"name": n, "language": "Starter voices", "gender": STARTERS[n][2]}
                for n in available_starters()
            }
        return {}

    def speak(self, text: str, voice: str | None, speed: float, profile: dict | None = None):
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "voices": self.voices(),
            "details": self.details(),
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

    def clone(self, voice: str | None, profile: dict | None) -> dict:
        """For backends that only copy voices: the user's sample, else a starter voice."""
        return self.check(profile) or starter(voice)

    @staticmethod
    def described(voice: str | None, profile: dict | None) -> str | None:
        """The description to speak with: the user's designed voice or a ready-made one."""
        if profile and profile["kind"] == "design":
            return profile["description"]
        return DESIGNS.get(voice or "")
