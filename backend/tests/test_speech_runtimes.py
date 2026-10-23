"""Speech models that need their own library: classification, install specs, routing."""

import asyncio
import json

import httpx

from app.catalog import live, speech
from app.engines import ENGINES, EngineManager
from tests.test_delegation import AUTH, client, install  # noqa: F401 - fixture


def listing(repo, lib=None, tags=()):
    return {"id": repo, "library_name": lib, "tags": list(tags)}


def test_runtime_is_picked_from_the_library():
    assert speech.runtime_for(listing("ResembleAI/chatterbox", "chatterbox")) == "tts-chatterbox"
    assert speech.runtime_for(listing("SWivid/F5-TTS", "f5-tts")) == "tts-f5"
    assert speech.runtime_for(listing("SWivid/E2-TTS", "f5-tts")) == "tts-f5"
    assert speech.runtime_for(listing("someone/F5-TTS-finetune", "f5-tts")) is None
    assert speech.runtime_for(listing("coqui/XTTS-v2", "coqui")) == "tts-xtts"
    assert speech.runtime_for(listing("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")) == "tts-qwen"
    assert speech.runtime_for(listing("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")) == "tts-qwen"
    assert speech.runtime_for(listing("Qwen/Qwen3-TTS-12Hz-0.6B-Base")) == "tts-qwen"
    assert speech.runtime_for(listing("Qwen/Qwen3-TTS-Tokenizer-12Hz")) is None
    assert speech.runtime_for(listing("k2-fsa/OmniVoice", "omnivoice")) == "tts-omnivoice"
    assert speech.runtime_for(listing("openbmb/VoxCPM2", "voxcpm")) == "tts-voxcpm"
    assert speech.runtime_for(listing("Serveurperso/OmniVoice-GGUF", "gguf")) is None
    vits = listing("facebook/mms-tts-eng", "transformers", ["transformers", "vits"])
    assert speech.runtime_for(vits) == "tts-transformers"
    # transformers alone isn't enough: VibeVoice etc. need custom code.
    assert speech.runtime_for(listing("microsoft/VibeVoice-1.5B", "transformers")) is None
    assert all(rt in ENGINES for rt in speech.RUNTIMES)


def test_newer_engines_are_routed():
    rt = speech.runtime_for
    assert rt(listing("Supertone/supertonic-3", "supertonic")) == "tts-supertonic"
    assert rt(listing("KittenML/kitten-tts-2", None, ["gguf"])) == "tts-kitten"
    assert rt(listing("kyutai/tts-1.6b-en_fr", "moshi")) == "tts-kyutai"
    assert rt(listing("kyutai/moshiko-pytorch-bf16", "moshi")) is None  # speech-to-speech
    assert rt(listing("kyutai/tts-voices")) is None  # a voice library, not a model
    assert rt(listing("kyutai/pocket-tts", "pocket-tts")) == "tts-pocket"
    parler = listing("parler-tts/parler-tts-mini-v1", "transformers", ["parler_tts"])
    assert rt(parler) == "tts-parler"
    assert rt(listing("myshell-ai/MeloTTS-English", "transformers")) == "tts-melo"
    assert rt(listing("samuel-vitorino/sopro-v2-turbo", "sopro")) == "tts-sopro"
    assert rt(listing("ekwek/Soprano-1.1-80M", "transformers")) == "tts-soprano"
    assert rt(listing("Zyphra/Zonos-v0.1-transformer", "zonos")) == "tts-zonos"
    assert rt(listing("Zyphra/Zonos-v0.1-hybrid", "zonos")) is None  # CUDA-only kernel
    assert rt(listing("neuphonic/neutts-air")) == "tts-neutts"
    assert rt(listing("neuphonic/neutts-air-q4-gguf", None, ["gguf"])) is None
    assert rt(listing("OuteAI/Llama-OuteTTS-1.0-1B", "outetts")) == "tts-outetts"
    assert rt(listing("OuteAI/Llama-OuteTTS-1.0-1B-GGUF", "outetts")) is None
    assert rt(listing("IndexTeam/IndexTTS-2")) == "tts-indextts"
    assert rt(listing("microsoft/VibeVoice-Realtime-0.5B", "transformers")) == "tts-vibevoice"
    assert rt(listing("pnnbao-ump/VieNeu-TTS-v3-Turbo")) == "tts-vieneu"
    csm = listing("sesame/csm-1b", "transformers", ["transformers", "csm"])
    assert rt(csm) == "tts-transformers"
    assert rt(listing("nari-labs/Dia-1.6B-0626", None, ["dia"])) is None  # GPU only
    assert "withdrew" in speech.unsupported_reason(listing("microsoft/VibeVoice-1.5B"))
    assert "Apple" in speech.unsupported_reason(listing("mlx-community/Kokoro-82M-bf16", "mlx"))


def test_engines_that_need_another_python_or_source_overlay():
    assert ENGINES["tts-melo"].python == "3.11"
    assert ENGINES["tts-indextts"].python == "3.11"
    assert ENGINES["tts-zonos"].overlays[0][1] == "zonos"
    assert all(" @ https://github.com/" in p for p in ENGINES["tts-melo"].packages[:1])


def test_f5_models_pick_their_architecture_and_files():
    assert speech.options("tts-f5", "SWivid/E2-TTS") == {"model": "E2TTS_Base"}
    assert speech.allow_patterns("tts-f5", "SWivid/E2-TTS", []) == ["E2TTS_Base/*.safetensors"]
    assert speech.options("tts-xtts", "coqui/XTTS-v2") == {}


def test_vibevoice_voices_come_from_github():
    files = speech.extras("tts-vibevoice")["voices"]["files"]
    assert len(files) == 25 and files["en-Emma_woman.pt"].startswith("https://raw.github")


def test_unsupported_models_say_why():
    assert "Kokoro 82M" in speech.unsupported_reason(listing("hexgrad/Kokoro-82M"))
    assert "sanotts library" in speech.unsupported_reason(listing("ampixa/sanoTTS", "sanotts"))
    fam = live.hf_family(listing("coqui/XTTS-v2", "coqui"), "speech", False)
    assert fam["unsupported"] is None and fam["runtime"] == "tts-xtts"
    assert "copy a voice from your sample" in fam["description"]


def test_descriptions_say_what_voices_a_model_offers():
    design = live.hf_family(listing("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"), "speech", False)
    assert "voices you describe in words" in design["description"]
    base = live.hf_family(listing("Qwen/Qwen3-TTS-12Hz-0.6B-Base"), "speech", False)
    assert "copied from your sample" in base["description"]
    custom = live.hf_family(listing("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"), "speech", False)
    assert "sample" not in custom["description"]


def test_audio_spec_for_a_speech_model():
    siblings = [
        {"rfilename": "config.json", "size": 1000},
        {"rfilename": "model.pth", "size": 1_800_000_000},
        {"rfilename": "samples/demo.wav", "size": 5_000_000},
    ]

    def handle(req):
        return httpx.Response(
            200, json={"id": "coqui/XTTS-v2", "library_name": "coqui", "siblings": siblings}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as c:
            return await live.audio_spec("coqui/XTTS-v2", c, None)

    spec = asyncio.run(run())
    cfg = spec["engine_config"]
    assert cfg["runtime"] == "tts-xtts" and cfg["backend"] == "xtts" and cfg["task"] == "speech"
    assert cfg["source"]["allow_patterns"] == ["*.json", "*.pth"]
    assert spec["size_gb"] == 1.8  # the demo audio isn't downloaded


def test_runtime_routing():
    xtts = {"engine_config": {"runtime": "tts-xtts"}}
    assert EngineManager.runtime("audio", xtts) == "tts-xtts"
    assert EngineManager.runtime("audio", {"engine_config": {}}) == "audio"
    assert EngineManager.runtime("audio", {"engine_config": {"runtime": "nope"}}) == "audio"


def test_speech_is_served_by_the_models_runtime(client):  # noqa: F811
    variant = {
        "tag": "coqui/XTTS-v2",
        "size_gb": 1.8,
        "engine_config": {
            "task": "speech",
            "runtime": "tts-xtts",
            "backend": "xtts",
            "source": {"type": "hf", "repo": "coqui/XTTS-v2"},
        },
    }
    install(client.app, "audio", "coqui/XTTS-v2", variant)
    r = client.post(
        "/v1/audio/speech", headers=AUTH, json={"model": "audio/coqui/XTTS-v2", "input": "hi"}
    )
    assert r.status_code == 200
    payload = client.fake.calls[-1][1]
    assert payload["config"]["backend"] == "xtts" and payload["text"] == "hi"
    assert client.fake.engines[-1] == "tts-xtts"
    # It's an ordinary installed speech model for everything else.
    models = client.get("/api/settings", headers=AUTH).json()["models"]["speech"]["installed"]
    assert [m["id"] for m in models] == ["audio/coqui/XTTS-v2"]


def _fake_model(m, tag, runtime):
    d = m.model_dir("audio", tag)
    d.mkdir(parents=True)
    config = {"task": "speech", "runtime": runtime, "backend": "f5"}
    (d / ".amy-model.json").write_text(json.dumps({"tag": tag, "engine_config": config}))


def test_a_runtime_goes_with_its_last_model(tmp_path):
    from app.jobs import JobRegistry

    m = EngineManager(tmp_path, JobRegistry(), gpu=False, torch_index_url="")
    for eng in ("audio", "tts-f5"):
        (m.root / eng).mkdir(parents=True)
    _fake_model(m, "SWivid/F5-TTS", "tts-f5")
    _fake_model(m, "SWivid/E2-TTS", "tts-f5")

    asyncio.run(m.delete_model("audio", "SWivid/F5-TTS"))
    assert (m.root / "tts-f5").exists()  # E2-TTS still uses it
    m._installing["tts-f5"] += 1
    asyncio.run(m.delete_model("audio", "SWivid/E2-TTS"))
    assert (m.root / "tts-f5").exists()  # another model is installing onto it
    m._installing["tts-f5"] -= 1
    _fake_model(m, "SWivid/E2-TTS", "tts-f5")
    asyncio.run(m.delete_model("audio", "SWivid/E2-TTS"))
    assert not (m.root / "tts-f5").exists()
    assert (m.root / "audio").exists()  # the base engine stays


def test_installs_that_would_fill_the_disk_are_refused(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    from app.engines import EngineError
    from app.jobs import JobRegistry

    m = EngineManager(tmp_path, JobRegistry(), gpu=True, torch_index_url="")
    usage = namedtuple("usage", "total used free")
    variant = {"tag": "SWivid/F5-TTS", "size_gb": 1.35, "engine_config": {"runtime": "tts-f5"}}
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(0, 0, 10e9))
    try:
        m.check_space("tts-f5", variant)  # 1.35 GB + a GPU runtime + the margin > 10 GB
    except EngineError as exc:
        assert "Not enough disk space" in str(exc)
    else:
        raise AssertionError("expected a refusal")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(0, 0, 20e9))
    m.check_space("tts-f5", variant)
