"""Which installed speech models can speak a voice: one of your voices, or a voice library voice.

- A recorded voice (a clone) works with models that copy voices from a sample; some of them
  also need the sample's words written down (``NEEDS_TEXT``).
- A described voice works with models that make a voice from a description.
- Kyutai TTS 1.6B can't copy voices, but has the library's VCTK and Alba MacKenna voices built
  in, so those (and your voices installed from them) can use its built-in copy.

When nothing installed can, ``suggested`` picks catalog models that could.
"""

from app import delegation
from app.catalog import speech
from app.voice_meta import REPO, key

# Runtimes whose voice copying needs the sample's transcript.
NEEDS_TEXT = {"tts-f5", "tts-omnivoice", "tts-neutts", "tts-kitten", "tts-transformers"}
KYUTAI_BUILT_IN = ("vctk/", "alba-mackenna/")
MAX_SUGGESTED = 5


def kinds(runtime: str | None, repo: str) -> list[str]:
    """The kinds of custom voice a model can use: "clone" and/or "design"."""
    style = speech.style_for(runtime, repo, []) or ""
    out = []
    if "cop" in style:  # "copied from your sample", "can copy a voice", "describe or copy"
        out.append("clone")
    if "describe" in style:
        out.append("design")
    return out


def kyutai_voices(runtime: str | None, repo: str) -> bool:
    """Kyutai TTS 1.6B (not the 0.75B, which copies voices) has some library voices built in."""
    return runtime == "tts-kyutai" and not kinds(runtime, repo)


def kyutai_voice(library_path: str) -> str | None:
    """Kyutai TTS 1.6B's own id for a library voice it has built in (None if it hasn't)."""
    if not key(library_path).startswith(KYUTAI_BUILT_IN):
        return None
    return library_path.replace("_enhanced.wav", ".wav")  # it has the original recordings


def installed(app) -> list[dict]:
    """Installed speech models that can speak voices beyond their own:
    [{id, name, kinds, needs_text, built_in, how}]. ``built_in``: Kyutai 1.6B's library voices;
    ``how`` (for the voice library): "built_in", "copies" or "copies_with_text"."""
    variants = {
        f"{engine}/{tag}": variant for engine, tag, variant in app.state.engines.installed_models()
    }
    out = []
    for m in delegation.installed_for(app, "speech"):
        variant = variants.get(m["id"]) or {}
        runtime = (variant.get("engine_config") or {}).get("runtime")
        repo = variant.get("tag", "")
        can = kinds(runtime, repo)
        built_in = kyutai_voices(runtime, repo)
        if not can and not built_in:
            continue
        needs_text = "clone" in can and runtime in NEEDS_TEXT
        if built_in:
            how = "built_in"
        elif "clone" in can:
            how = "copies_with_text" if needs_text else "copies"
        else:
            how = "describes"
        out.append(
            {
                "id": m["id"],
                "name": m["name"],
                "kinds": can,
                "needs_text": needs_text,
                "built_in": built_in,
                "how": how,
            }
        )
    return out


def runs_on(voice_id: str, models: list[dict], transcript: bool) -> list[str]:
    """The installed models (ids) that can speak one library voice. ``transcript``: whether
    the voice would have its words written down (installed with one, or a speech-to-text
    model is there to transcribe it)."""
    out = []
    for m in models:
        if m["built_in"]:
            ok = kyutai_voice(voice_id) is not None
        else:
            ok = "clone" in m["kinds"] and (transcript or not m["needs_text"])
        if ok:
            out.append(m["id"])
    return out


def usable_with(voice: dict, models: list[dict]) -> list[dict]:
    """Where one of your voices can be used: [{model, voice}], ``voice`` being the id to speak
    with on that model (your voice, or Kyutai 1.6B's built-in copy of a library voice)."""
    out = []
    repo, _, path = (voice.get("source") or "").partition(":")
    for m in models:
        if m["built_in"]:
            builtin = kyutai_voice(path) if repo == REPO and voice["kind"] == "clone" else None
            if builtin:
                out.append({"model": m["id"], "voice": builtin})
        elif voice["kind"] in m["kinds"]:
            if voice["kind"] == "clone" and m["needs_text"] and not voice.get("transcript"):
                continue
            out.append({"model": m["id"], "voice": f"custom:{voice['id']}"})
    return out


def suggested(rows: list[dict], kind: str = "clone", library: bool = False) -> list[dict]:
    """Speech models to install for voices of one kind, best first, from catalog rows (already
    sorted). ``library``: also offer Kyutai TTS 1.6B for its built-in library voices."""
    out = []
    for r in rows:
        if r.get("unsupported") or r.get("requires_token") or r.get("installed"):
            continue
        runtime, repo = r.get("runtime"), r.get("tag", "")
        can = kinds(runtime, repo)
        if library and kyutai_voices(runtime, repo):
            note = "has the VCTK and Alba MacKenna voices built in"
        elif kind in can:
            if kind == "design":
                note = "makes voices from a description"
            else:
                note = "copies voices from a sample"
                if runtime in NEEDS_TEXT:
                    note += " (needs the sample's words written down)"
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
