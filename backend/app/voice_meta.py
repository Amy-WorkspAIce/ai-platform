"""What is known about the voices in ``kyutai/tts-voices``: gender, age, accent, region,
language and speaking style, so the Voices page can filter them.

The labels live in ``voice_meta.json`` (made by ``scripts/voice_meta.py``) and come from:

- the VCTK corpus's speaker list (gender, age, accent, region) for the VCTK speakers;
- what the recordings are known to be (Alba MacKenna's styles, the LibriVox readers);
- otherwise, gender estimated from the sample's pitch, marked ``"estimated": true``. Pitches
  between typical male and female voices are left without a gender rather than guessed.

Anything not known is simply left out, and shows as "Uncategorized".
"""

import json
import re
from functools import cache
from pathlib import Path

REPO = "kyutai/tts-voices"
DATA = Path(__file__).with_name("voice_meta.json")
FIELDS = ("gender", "estimated", "age", "accent", "region", "language", "style")
# Kyutai's own voices include one VCTK speaker too.
VCTK = re.compile(r"(?:^|/)(?:vctk|unmute-prod-website)/p(\d+)_")


def key(path: str) -> str:
    """One recording's key: the path without extension, the "_enhanced" suffix, or (for VCTK)
    the sentence, so every recording of a speaker shares the speaker's labels."""
    m = VCTK.search(path)
    if m:
        return f"vctk/p{m[1]}"
    stem = re.sub(r"(\.(wav|mp3|flac))+(\..*)?$", "", path)
    return stem.removesuffix("_enhanced")


@cache
def _table() -> dict[str, dict]:
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def lookup(path: str) -> dict:
    """The labels for one recording in the voice repo ({} if nothing is known)."""
    return dict(_table().get(key(path), {}))


# Labels some speech models put in "language" that name a set of voices, not a language.
GROUPS = {
    "Voices",
    "Starter voices",
    "Donated voices",
    "VCTK (English accents)",
    "Alba MacKenna",
    "Unmute",
}


def describe(voices: list[str], details: dict[str, dict], paths: dict[str, str]) -> dict:
    """A model's voice details with the labels added: ``paths`` maps voices named otherwise
    (the starter voices) to their recording in the voice repo. A set name a model gave as the
    language moves to ``group``."""
    out = {}
    for v in voices:
        d = dict(details.get(v) or {})
        if d.get("language") in GROUPS:
            d["group"] = d.pop("language")
        meta = lookup(paths.get(v, v))
        if meta.get("gender") and d.get("gender"):
            meta.pop("estimated", None)  # the model says so: no need to guess
            meta.pop("gender")
        d = meta | {k: val for k, val in d.items() if val}
        if d:
            out[v] = d
    return out
