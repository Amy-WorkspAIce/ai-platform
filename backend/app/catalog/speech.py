"""Which speech runtime can run a Hugging Face text-to-speech model, and how to download it.

Almost every popular voice model ships with its own library (Chatterbox, F5-TTS, Coqui XTTS,
Qwen3-TTS...). Amy supports a model when it has a **runtime** for that library: an engine
installed on demand into its own virtualenv (see ``app.engines``). The model's Hugging Face
``library_name`` and tags say which one it needs; models whose listing names no library are
matched by repo.
"""

KOKORO_REPOS = ("hexgrad/kokoro-82m", "onnx-community/kokoro-82m")

# Download sizes (GB) of the files Amy fetches, measured, for repos whose listing has no
# safetensors size (so the Models page can show a size and a fit before install).
KNOWN_GB = {
    "coqui/xtts-v2": 2.09,
    "resembleai/chatterbox": 3.19,
    "swivid/f5-tts": 1.4,
    "swivid/e2-tts": 1.38,
    "misha24-10/f5-tts_russian": 1.4,
    "firstpixel/f5-tts-pt-br": 1.4,
    "k2-fsa/omnivoice": 3.27,
    "openbmb/voxcpm2": 4.96,
    "supertone/supertonic-3": 0.41,
    "supertone/supertonic-2": 0.27,
    "kittenml/kitten-tts-2": 1.6,
    "kyutai/tts-1.6b-en_fr": 3.85,
    "kyutai/tts-0.75b-en-public": 1.94,
    "kyutai/pocket-tts": 0.5,
    "kyutai/pocket-tts-without-voice-cloning": 0.5,
    "myshell-ai/melotts-english": 0.62,
    "samuel-vitorino/sopro-v2-turbo": 0.95,
    "zyphra/zonos-v0.1-transformer": 3.25,
    "neuphonic/neutts-air": 3.04,
    "neuphonic/neutts-nano": 0.96,
    "indexteam/indextts-2": 8.1,
    "microsoft/vibevoice-realtime-0.5b": 2.04,
    "pnnbao-ump/vieneu-tts-v3-turbo": 2.2,
}

# Runtime engine id -> what it runs, for messages.
RUNTIMES = {
    "tts-transformers": "Transformers (MMS, VITS, Bark, CSM)",
    "tts-chatterbox": "Chatterbox",
    "tts-f5": "F5-TTS",
    "tts-xtts": "Coqui XTTS",
    "tts-qwen": "Qwen3-TTS",
    "tts-omnivoice": "OmniVoice",
    "tts-voxcpm": "VoxCPM",
    "tts-supertonic": "Supertonic",
    "tts-kitten": "KittenTTS",
    "tts-kyutai": "Kyutai TTS",
    "tts-pocket": "Pocket TTS",
    "tts-parler": "Parler-TTS",
    "tts-melo": "MeloTTS",
    "tts-sopro": "Sopro",
    "tts-soprano": "Soprano",
    "tts-zonos": "Zonos",
    "tts-neutts": "NeuTTS",
    "tts-outetts": "OuteTTS",
    "tts-indextts": "IndexTTS",
    "tts-vibevoice": "VibeVoice",
    "tts-vieneu": "VieNeu-TTS",
}

# Hugging Face library -> runtime.
LIBRARIES = {
    "chatterbox": "tts-chatterbox",
    "omnivoice": "tts-omnivoice",
    "voxcpm": "tts-voxcpm",
    "supertonic": "tts-supertonic",
    "pocket-tts": "tts-pocket",
    "sopro": "tts-sopro",
    "zonos": "tts-zonos",
    "outetts": "tts-outetts",
}

# Repo prefix -> runtime, for repos whose listing names no library (or a generic one).
REPOS = {
    "resembleai/chatterbox": "tts-chatterbox",
    "qwen/qwen3-tts": "tts-qwen",
    "kittenml/kitten-tts": "tts-kitten",
    "kyutai/tts-": "tts-kyutai",
    "ekwek/soprano": "tts-soprano",
    "neuphonic/neutts-air": "tts-neutts",
    "neuphonic/neutts-nano": "tts-neutts",
    "myshell-ai/melotts-": "tts-melo",
    "indexteam/indextts-2": "tts-indextts",
    "microsoft/vibevoice-realtime": "tts-vibevoice",
    "pnnbao-ump/vieneu-tts-v3-turbo": "tts-vieneu",
}

# F5-TTS models: repo -> (model architecture, the checkpoint and vocabulary files to fetch).
# Fine-tunes share F5-TTS's code but differ in architecture and file layout.
F5_MODELS = {
    "swivid/f5-tts": ("F5TTS_v1_Base", ["F5TTS_v1_Base/*"]),
    "swivid/e2-tts": ("E2TTS_Base", ["E2TTS_Base/*.safetensors"]),
    "misha24-10/f5-tts_russian": (
        "F5TTS_v1_Base",
        ["F5TTS_v1_Base_v4_winter/model_212000.safetensors", "F5TTS_v1_Base/vocab.txt"],
    ),
    "firstpixel/f5-tts-pt-br": ("F5TTS_Base", ["pt-br/model_last.safetensors"]),
}

# Why a model in a supported family still can't run, by repo fragment.
NOT_YET = {
    "nari-labs/dia": "Dia needs a GPU: on a CPU it takes most of an hour per sentence.",
    "kyutai/tts-voices": "Kyutai's voice library: its voices come with Kyutai TTS, and you can "
    "add them as your own voices on the Voices page.",
    "-gguf": "This is a GGUF build, which needs llama.cpp. Install the original model instead.",
    "zonos-v0.1-hybrid": "The hybrid Zonos needs a CUDA-only kernel (mamba-ssm). Install the "
    "transformer one instead.",
    "microsoft/vibevoice-1.5b": "Microsoft withdrew the code for this model because of misuse. "
    "VibeVoice Realtime 0.5B works.",
    "vibevoice/": "Microsoft withdrew the code for this model because of misuse. "
    "VibeVoice Realtime 0.5B works.",
}


def runtime_for(model: dict) -> str | None:
    """The runtime engine for a Hugging Face model listing (needs ``library_name`` and tags)."""
    lib = (model.get("library_name") or "").lower()
    tags = set(model.get("tags") or [])
    repo = model["id"].lower()
    # A "gguf" tag alone isn't enough: some repos ship llama.cpp files beside the real model.
    if any(part in repo for part in NOT_YET) or "tokenizer" in repo:
        return None
    if lib == "f5-tts" or repo in F5_MODELS:
        return "tts-f5" if repo in F5_MODELS else None
    if lib == "coqui" and "xtts" in repo:
        return "tts-xtts"
    if lib == "qwen-tts":
        return "tts-qwen"  # Base models clone a voice sample; VoiceDesign ones a description
    if lib in LIBRARIES:
        return LIBRARIES[lib]
    if lib == "moshi" and repo.startswith("kyutai/tts-"):
        return "tts-kyutai"
    for prefix, runtime in REPOS.items():
        if repo.startswith(prefix):
            return runtime
    if "parler_tts" in tags:
        return "tts-parler"
    if lib == "transformers" and tags & {"vits", "bark", "csm"}:
        return "tts-transformers"
    return None


def voice_style(model: dict) -> str | None:
    """What voices a supported model offers beyond its built-in ones, for its description."""
    runtime = runtime_for(model)
    repo = model["id"].lower()
    if runtime == "tts-qwen":
        if "voicedesign" in repo:
            return "speaks in voices you describe in words"
        if "base" in repo:
            return "speaks in a voice copied from your sample"
        return None
    if runtime in ("tts-omnivoice", "tts-voxcpm"):
        return "voices you describe or copy from a sample"
    if runtime == "tts-parler":
        return "speaks in voices you describe in words"
    if runtime in ("tts-sopro", "tts-zonos", "tts-neutts", "tts-indextts"):
        return "speaks in a voice copied from your sample"
    if runtime == "tts-kyutai" and "public" in repo:
        return "speaks in a voice copied from your sample"
    if runtime == "tts-pocket" and "without-voice-cloning" not in repo:
        return "can copy a voice from your sample"
    if runtime in ("tts-xtts", "tts-chatterbox", "tts-f5", "tts-kitten", "tts-vieneu"):
        return "can copy a voice from your sample"
    if runtime == "tts-transformers" and "csm" in (model.get("tags") or []):
        return "can copy a voice from your sample"
    return None


def unsupported_reason(model: dict) -> str | None:
    """Why Amy can't run a speech model (None when it can)."""
    if runtime_for(model):
        return None
    repo = model["id"].lower()
    if repo.startswith(KOKORO_REPOS):
        return "This is Kokoro: install Kokoro 82M from the recommended models (same voices)."
    for part, reason in NOT_YET.items():
        if part in repo:
            return reason
    lib = model.get("library_name")
    if lib == "transformers":
        return (
            "Uses custom code that Amy's Transformers speech engine can't run "
            "(it runs MMS, VITS, Bark and Sesame CSM)."
        )
    if lib == "f5-tts":
        return "An F5-TTS fine-tune Amy hasn't been set up for yet (F5-TTS and E2-TTS work)."
    if lib in ("mlx", "mlx-audio", "coreml", "executorch", "litert", "transformers.js"):
        return f"A {lib} build for Apple, phones or browsers, not servers."
    if lib:
        return f"Needs the {lib} library, which Amy has no engine for yet."
    return "Needs a speech engine Amy doesn't have yet."


def allow_patterns(runtime: str, repo: str, files: list[str]) -> list[str] | None:
    """Only the files the runtime loads (None: the whole repo)."""
    repo = repo.lower()
    if runtime == "tts-transformers":
        patterns = ["*.json", "*.txt", "*.model", "speaker_embeddings/**"]
        weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
        return [*patterns, weights]
    if runtime == "tts-chatterbox" and repo == "resembleai/chatterbox":
        return [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]
    if runtime == "tts-f5":
        return F5_MODELS[repo][1]
    if runtime == "tts-xtts":
        return ["*.json", "*.pth"]
    if runtime == "tts-kitten" and any(f.startswith("lm/") for f in files):
        # KittenTTS 2: the packed weights, not the llama.cpp and other runtimes' copies.
        return [
            "config.json",
            "lm/model-ternary.safetensors",
            "lm/*.json",
            "lm/*.txt",
            "voices/*",
            "speaker/*",
        ]
    if runtime == "tts-pocket":
        return ["languages/english/*"]
    if runtime == "tts-melo":
        return ["config.json", "checkpoint.pth"]
    if runtime == "tts-vieneu":
        return ["*.json", "*.md"]  # the library loads the weights by repo, from the shared cache
    if runtime == "tts-soprano":
        return ["*.json", "*.safetensors", "decoder.pth"]
    if runtime in ("tts-parler", "tts-neutts", "tts-outetts", "tts-sopro"):
        return ["*.json", "*.txt", "*.model", "*.safetensors", "*.yaml", "*.py"]
    return None


def options(runtime: str, repo: str) -> dict:
    """Per-model settings the backend needs (saved in its engine config)."""
    if runtime == "tts-f5":
        return {"model": F5_MODELS[repo.lower()][0]}
    return {}


def extras(runtime: str) -> dict:
    """Other repos a runtime needs next to the model (downloaded into subfolders)."""
    if runtime == "tts-f5":
        return {"vocos": {"type": "hf", "repo": "charactr/vocos-mel-24khz"}}
    if runtime == "tts-vibevoice":
        base = (
            "https://raw.githubusercontent.com/microsoft/VibeVoice/"
            "16fb2cb1217c9934a886e1948ffb06120caa2df5/demo/voices/streaming_model/"
        )
        names = [f"{lang}-{who}" for lang, who in VIBEVOICE_PRESETS]
        return {"voices": {"type": "urls", "files": {f"{n}.pt": f"{base}{n}.pt" for n in names}}}
    return {}


# VibeVoice Realtime's preset voices live in Microsoft's GitHub repo, not on Hugging Face.
VIBEVOICE_PRESETS = [
    ("de", "Spk0_man"), ("de", "Spk1_woman"), ("en", "Carter_man"), ("en", "Davis_man"),
    ("en", "Emma_woman"), ("en", "Frank_man"), ("en", "Grace_woman"), ("en", "Mike_man"),
    ("fr", "Spk0_man"), ("fr", "Spk1_woman"), ("in", "Samuel_man"), ("it", "Spk0_woman"),
    ("it", "Spk1_man"), ("jp", "Spk0_man"), ("jp", "Spk1_woman"), ("kr", "Spk0_woman"),
    ("kr", "Spk1_man"), ("nl", "Spk0_man"), ("nl", "Spk1_woman"), ("pl", "Spk0_man"),
    ("pl", "Spk1_woman"), ("pt", "Spk0_woman"), ("pt", "Spk1_man"), ("sp", "Spk0_woman"),
    ("sp", "Spk1_man"),
]  # fmt: skip
