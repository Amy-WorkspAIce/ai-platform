"""Speech models that need their own library: classification, install specs, routing."""

import asyncio

import httpx

from app.catalog import live, speech
from app.engines import ENGINES, EngineManager
from tests.test_delegation import AUTH, client, install  # noqa: F401 - fixture


def listing(repo, lib=None, tags=()):
    return {"id": repo, "library_name": lib, "tags": list(tags)}


def test_runtime_is_picked_from_the_library():
    assert speech.runtime_for(listing("ResembleAI/chatterbox", "chatterbox")) == "tts-chatterbox"
    assert speech.runtime_for(listing("SWivid/F5-TTS", "f5-tts")) == "tts-f5"
    assert speech.runtime_for(listing("SWivid/E2-TTS", "f5-tts")) is None  # other shape
    assert speech.runtime_for(listing("coqui/XTTS-v2", "coqui")) == "tts-xtts"
    assert speech.runtime_for(listing("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")) == "tts-qwen"
    vits = listing("facebook/mms-tts-eng", "transformers", ["transformers", "vits"])
    assert speech.runtime_for(vits) == "tts-transformers"
    # transformers alone isn't enough: VibeVoice etc. need custom code.
    assert speech.runtime_for(listing("microsoft/VibeVoice-1.5B", "transformers")) is None
    assert all(rt in ENGINES for rt in speech.RUNTIMES)


def test_unsupported_models_say_why():
    assert "Kokoro 82M" in speech.unsupported_reason(listing("hexgrad/Kokoro-82M"))
    assert "omnivoice library" in speech.unsupported_reason(
        listing("k2-fsa/OmniVoice", "omnivoice")
    )
    fam = live.hf_family(listing("coqui/XTTS-v2", "coqui"), "speech", False)
    assert fam["unsupported"] is None and fam["runtime"] == "tts-xtts"


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
