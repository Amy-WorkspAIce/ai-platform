"""Which speech models can speak a voice library voice.

Library voices are samples, so any model that copies a voice from a sample can speak them
once they're installed (models that need the sample's words written down only when it has a
transcript). Kyutai TTS 1.6B also has the VCTK and Alba MacKenna voices built in.
"""

from app import delegation
from app.catalog import speech
from app.voice_meta import key

# Runtimes whose voice copying needs the sample's transcript.
NEEDS_TEXT = {"tts-f5", "tts-omnivoice", "tts-neutts", "tts-kitten", "tts-transformers"}
KYUTAI_BUILT_IN = ("vctk/", "alba-mackenna/")
MAX_SUGGESTED = 5


def copies(runtime: str | None, repo: str) -> bool:
    style = speech.style_for(runtime, repo, []) or ""
    return "cop" in style  # "copied from your sample", "can copy a voice", "describe or copy"


def kyutai_voices(runtime: str | None, repo: str) -> bool:
    """Kyutai TTS 1.6B (not the 0.75B, which copies voices) has some library voices built in."""
    return runtime == "tts-kyutai" and not copies(runtime, repo)


def built_into_kyutai(voice_id: str) -> bool:
    return key(voice_id).startswith(KYUTAI_BUILT_IN)


def installed(app) -> list[dict]:
    """Installed speech models that can speak library voices: [{id, name, how}], where ``how``
    is "built_in" (Kyutai 1.6B, some voices), "copies", or "copies_with_text"."""
    variants = {
        f"{engine}/{tag}": variant for engine, tag, variant in app.state.engines.installed_models()
    }
    out = []
    for m in delegation.installed_for(app, "speech"):
        variant = variants.get(m["id"]) or {}
        runtime = (variant.get("engine_config") or {}).get("runtime")
        repo = variant.get("tag", "")
        if kyutai_voices(runtime, repo):
            how = "built_in"
        elif copies(runtime, repo):
            how = "copies_with_text" if runtime in NEEDS_TEXT else "copies"
        else:
            continue
        out.append({"id": m["id"], "name": m["name"], "how": how})
    return out


def runs_on(voice_id: str, models: list[dict], transcript: bool) -> list[str]:
    """The installed models (ids) that can speak one library voice. ``transcript``: whether
    the voice would have its words written down (installed with one, or a speech-to-text
    model is there to transcribe it)."""
    out = []
    for m in models:
        if m["how"] == "built_in":
            ok = built_into_kyutai(voice_id)
        else:
            ok = m["how"] == "copies" or transcript
        if ok:
            out.append(m["id"])
    return out


def suggested(rows: list[dict]) -> list[dict]:
    """Speech models to install so library voices have something to run on, best first:
    catalog rows (already sorted) that copy voices, plus Kyutai TTS 1.6B for its built-in ones."""
    out = []
    for r in rows:
        if r.get("unsupported") or r.get("requires_token") or r.get("installed"):
            continue
        runtime, repo = r.get("runtime"), r.get("tag", "")
        if kyutai_voices(runtime, repo):
            note = "has the VCTK and Alba MacKenna voices built in"
        elif copies(runtime, repo):
            note = "copies any of them"
            if runtime in NEEDS_TEXT:
                note += " (with a speech-to-text model installed)"
        else:
            continue
        out.append(
            {
                "tag": repo,
                "name": r["name"],
                "fit": r.get("fit"),
                "size_gb": r.get("size_gb"),
                "note": note,
            }
        )
        if len(out) >= MAX_SUGGESTED:
            break
    return out
