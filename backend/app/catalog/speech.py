"""Which speech runtime can run a Hugging Face text-to-speech model, and how to download it.

Almost every popular voice model ships with its own library (Chatterbox, F5-TTS, Coqui XTTS,
Qwen3-TTS...). Amy supports a model when it has a **runtime** for that library: an engine
installed on demand into its own virtualenv (see ``app.engines``). The model's Hugging Face
``library_name`` and tags say which one it needs.
"""

KOKORO_REPOS = ("hexgrad/kokoro-82m", "onnx-community/kokoro-82m")

# Download sizes (GB) of the files Amy fetches, measured, for repos whose listing has no
# safetensors size (so the Models page can show a size and a fit before install).
KNOWN_GB = {"coqui/xtts-v2": 2.09, "resembleai/chatterbox": 3.19, "swivid/f5-tts": 1.4}

# Runtime engine id -> what it runs, for messages.
RUNTIMES = {
    "tts-transformers": "Transformers (MMS, VITS, Bark)",
    "tts-chatterbox": "Chatterbox",
    "tts-f5": "F5-TTS",
    "tts-xtts": "Coqui XTTS",
    "tts-qwen": "Qwen3-TTS",
}


def runtime_for(model: dict) -> str | None:
    """The runtime engine for a Hugging Face model listing (needs ``library_name`` and tags)."""
    lib = (model.get("library_name") or "").lower()
    tags = set(model.get("tags") or [])
    repo = model["id"].lower()
    if lib == "chatterbox" or repo.startswith("resembleai/chatterbox"):
        return "tts-chatterbox"
    if repo == "swivid/f5-tts":
        return "tts-f5"  # fine-tunes and E2-TTS use other model shapes
    if lib == "coqui" and "xtts" in repo:
        return "tts-xtts"
    if (lib == "qwen-tts" or repo.startswith("qwen/qwen3-tts")) and "customvoice" in repo:
        return "tts-qwen"  # Base and VoiceDesign models need a voice sample or description
    if lib == "transformers" and tags & {"vits", "bark"}:
        return "tts-transformers"
    return None


def unsupported_reason(model: dict) -> str | None:
    """Why Amy can't run a speech model (None when it can)."""
    if runtime_for(model):
        return None
    repo = model["id"].lower()
    if repo.startswith("qwen/qwen3-tts"):
        return (
            "Needs a voice sample or description, which Amy can't give it yet "
            "(Qwen3-TTS CustomVoice models work)."
        )
    if repo.startswith(KOKORO_REPOS):
        return "This is Kokoro: install Kokoro 82M from the recommended models (same voices)."
    lib = model.get("library_name")
    if lib == "transformers":
        return (
            "Uses custom code that Amy's Transformers speech engine can't run "
            "(it runs MMS, VITS and Bark)."
        )
    if lib == "f5-tts":
        return "An F5-TTS variant Amy hasn't been set up for yet (the original F5-TTS works)."
    if lib:
        return f"Needs the {lib} library, which Amy has no engine for yet."
    return "Needs a speech engine Amy doesn't have yet."


def allow_patterns(runtime: str, repo: str, files: list[str]) -> list[str] | None:
    """Only the files the runtime loads (None: the whole repo)."""
    if runtime == "tts-transformers":
        patterns = ["*.json", "*.txt", "*.model", "speaker_embeddings/**"]
        weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
        return [*patterns, weights]
    if runtime == "tts-chatterbox" and repo.lower() == "resembleai/chatterbox":
        return [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]
    if runtime == "tts-f5":
        return ["F5TTS_v1_Base/*"]
    if runtime == "tts-xtts":
        return ["*.json", "*.pth"]
    return None


def extras(runtime: str) -> dict:
    """Other repos a runtime needs next to the model (downloaded into subfolders)."""
    if runtime == "tts-f5":
        return {"vocos": {"type": "hf", "repo": "charactr/vocos-mel-24khz"}}
    return {}
