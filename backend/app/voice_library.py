"""The voice library: ready-made voice samples to install as custom voices.

Samples come from Kyutai's ``kyutai/tts-voices`` repository on Hugging Face, keeping only
collections that may be used commercially (CC0 and CC BY 4.0; the non-commercial Expresso
and EARS recordings are left out). An installed sample is an ordinary custom voice (a clone),
so every model that copies voices can speak in it, and the credit its license asks for is kept
with it. Where a cleaned-up ("enhanced") recording exists, that one is used.
"""

import time

import httpx

from app import voice_meta
from app.catalog.live import HF_API, hf_headers

REPO = voice_meta.REPO
TTL = 24 * 3600

# Folder -> (collection, language, license, credit). Each voice's own labels (gender, accent,
# and so on) come from ``app.voice_meta``.
COLLECTIONS = {
    "alba-mackenna": ("Alba MacKenna", "English", "CC BY 4.0", "Voice acting by Alba MacKenna"),
    "voice-zero": ("LibriVox readers", "English", "CC0", "Voice-Zero, from LibriVox"),
    "vctk": ("VCTK speakers", "English", "CC BY 4.0", "VCTK corpus, University of Edinburgh"),
    "voice-donations": ("Donated voices", "", "CC0", "Unmute voice donation project"),
    "cml-tts/fr": ("French speakers", "French", "CC BY 4.0", "CML-TTS dataset"),
}

_cache: dict[str, tuple[float, list[dict]]] = {}


def file_url(path: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/main/{path}"


def _name(folder: str, stem: str) -> str:
    if folder == "vctk":
        return f"Speaker {stem.split('_')[0].lstrip('p')}"
    if folder == "cml-tts/fr":
        return f"Speaker {stem.split('_')[0]}"
    if folder == "voice-zero":
        return stem.replace("_", " ").title()
    if folder == "alba-mackenna":
        return f"Alba ({stem.replace('-', ' ')})"
    return f"Donor {stem}" if stem.isalnum() and stem == stem.lower() else stem


def entries(files: list[str]) -> list[dict]:
    """Library entries from the repo's file list: one per recording, enhanced if there is one."""
    paths = set(files)
    out = []
    for folder, (collection, language, license_, credit) in COLLECTIONS.items():
        prefix = folder + "/"
        for path in sorted(paths):
            if not path.startswith(prefix) or not path.endswith(".wav"):
                continue
            stem = path[len(prefix) : -len(".wav")]
            if "/" in stem or stem.endswith("_enhanced"):
                continue
            enhanced = f"{prefix}{stem}_enhanced.wav"
            chosen = enhanced if enhanced in paths else path
            meta = voice_meta.lookup(chosen)
            out.append(
                meta
                | {
                    "id": chosen,
                    "name": _name(folder, stem),
                    "collection": collection,
                    "language": meta.get("language", language),
                    "license": license_,
                    "credit": f"{credit} ({license_})",
                    "preview": file_url(chosen),
                }
            )
    return out


async def listing(client: httpx.AsyncClient, token: str | None) -> list[dict]:
    cached = _cache.get(REPO)
    if cached and time.time() - cached[0] < TTL:
        return cached[1]
    resp = await client.get(f"{HF_API}/models/{REPO}", headers=hf_headers(token), timeout=30)
    resp.raise_for_status()
    found = entries([s["rfilename"] for s in resp.json().get("siblings") or []])
    _cache[REPO] = (time.time(), found)
    return found


async def download(client: httpx.AsyncClient, token: str | None, path: str) -> bytes:
    resp = await client.get(
        file_url(path), headers=hf_headers(token), timeout=60, follow_redirects=True
    )
    resp.raise_for_status()
    return resp.content
