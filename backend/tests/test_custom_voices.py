"""Custom voices: recorded samples and described voices, offered to models that accept them."""

import io
import sys
import wave
from pathlib import Path

import httpx

from tests.test_delegation import AUTH, client, install  # noqa: F401 - fixture

sys.path.insert(0, str(Path(__file__).parents[1] / "app" / "engines"))
import speech_worker  # noqa: E402

QWEN_BASE = {
    "tag": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "size_gb": 2.5,
    "engine_config": {
        "task": "speech",
        "runtime": "tts-qwen",
        "backend": "qwen",
        "source": {"type": "hf", "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"},
    },
}


def wav(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(seconds * rate))
    return buf.getvalue()


def answer_voices(client, info):  # noqa: F811
    """Make the fake engine answer /voices with ``info`` and /speak with a real WAV."""
    fake = client.fake

    async def call(engine, path, payload, timeout=0):
        if path == "/voices":
            fake.calls.append((path, payload))
            fake.engines.append(engine)
            return httpx.Response(200, json=info)
        if path == "/speak":
            fake.calls.append((path, payload))
            fake.engines.append(engine)
            return httpx.Response(200, content=wav(4), headers={"content-type": "audio/wav"})
        return await fake(engine, path, payload, timeout)

    client.app.state.engines.call = call


def add_clone(client, name="Me", transcript=None, seconds=5.0):  # noqa: F811
    data = {"name": name, "kind": "clone"}
    if transcript:
        data["transcript"] = transcript
    return client.post(
        "/api/voices",
        headers=AUTH,
        data=data,
        files={"file": ("me.wav", wav(seconds), "audio/wav")},
    )


def test_add_and_list_voices(client):  # noqa: F811
    r = client.post(
        "/api/voices",
        headers=AUTH,
        data={"name": "Captain", "kind": "design", "description": "A gruff old sea captain"},
    )
    assert r.status_code == 201 and r.json()["id"].startswith("custom:")
    r = add_clone(client, transcript="Hello, this is me.")
    assert r.status_code == 201
    clone = r.json()
    assert clone["seconds"] == 5.0 and clone["transcript"] == "Hello, this is me."
    assert not clone["transcribed"]
    voices = client.get("/api/voices", headers=AUTH).json()
    assert [v["name"] for v in voices] == ["Captain", "Me"]
    assert client.get(f"/api/voices/{clone['id']}/sample", headers=AUTH).content == wav(5)


def test_bad_samples_are_refused(client):  # noqa: F811
    assert "too short" in add_clone(client, seconds=1).json()["detail"]
    assert "too long" in add_clone(client, seconds=61).json()["detail"]
    r = client.post(
        "/api/voices",
        headers=AUTH,
        data={"name": "x", "kind": "clone"},
        files={"file": ("x.webm", b"\x1aE\xdf\xa3webm", "audio/webm")},
    )
    assert r.status_code == 400 and "WAV" in r.json()["detail"]
    r = client.post("/api/voices", headers=AUTH, data={"name": "x", "kind": "design"})
    assert r.status_code == 400


def test_models_are_offered_the_voices_they_accept(client):  # noqa: F811
    install(client.app, "audio", QWEN_BASE["tag"], QWEN_BASE)
    client.post(
        "/api/voices",
        headers=AUTH,
        data={"name": "Captain", "kind": "design", "description": "A gruff sea captain"},
    )
    add_clone(client, name="No words")
    add_clone(client, name="With words", transcript="Hello there.")
    model = "audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    answer_voices(client, {"voices": [], "accepts": ["clone"], "needs_text": False})
    names = [
        v["name"]
        for v in client.get(f"/api/audio/voices?model={model}", headers=AUTH).json()["custom"]
    ]
    assert names == ["No words", "With words"]

    answer_voices(client, {"voices": [], "accepts": ["clone"], "needs_text": True})
    names = [
        v["name"]
        for v in client.get(f"/api/audio/voices?model={model}", headers=AUTH).json()["custom"]
    ]
    assert names == ["With words"]

    answer_voices(client, {"voices": ["Warm narrator"], "accepts": ["design"]})
    r = client.get(f"/api/audio/voices?model={model}", headers=AUTH).json()
    assert [v["name"] for v in r["custom"]] == ["Captain"] and r["voices"] == ["Warm narrator"]


def test_speaking_in_a_custom_voice_sends_its_profile(client):  # noqa: F811
    install(client.app, "audio", QWEN_BASE["tag"], QWEN_BASE)
    answer_voices(client, {"voices": [], "accepts": ["clone"]})
    voice = add_clone(client, transcript="Hello there.").json()
    r = client.post(
        "/v1/audio/speech",
        headers=AUTH,
        json={"model": "audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base", "input": "hi", "voice": voice["id"]},
    )
    assert r.status_code == 200
    profile = client.fake.calls[-1][1]["profile"]
    assert profile["kind"] == "clone" and profile["text"] == "Hello there."
    assert Path(profile["audio"]).read_bytes() == wav(5)

    # A built-in voice sends no profile.
    client.post(
        "/v1/audio/speech",
        headers=AUTH,
        json={"model": "audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base", "input": "hi", "voice": "ryan"},
    )
    assert client.fake.calls[-1][1]["profile"] is None


def test_voice_from_speech_saves_a_sample_with_its_words(client):  # noqa: F811
    install(client.app, "audio", QWEN_BASE["tag"], QWEN_BASE)
    answer_voices(client, {"voices": [], "accepts": ["clone"]})
    r = client.post(
        "/api/voices/from-speech",
        headers=AUTH,
        json={"name": "Narrator", "model": "audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base", "voice": "x"},
    )
    assert r.status_code == 201
    v = r.json()
    assert v["kind"] == "clone" and v["seconds"] == 4.0 and "Hello there" in v["transcript"]


def test_edit_and_delete_a_voice(client):  # noqa: F811
    voice = add_clone(client, transcript="Hello.").json()
    r = client.patch(
        f"/api/voices/{voice['id']}", headers=AUTH, json={"name": "Renamed", "transcript": ""}
    )
    assert r.json()["name"] == "Renamed" and r.json()["transcript"] is None
    client.put("/api/settings/voice", headers=AUTH, json={"voice": voice["id"]})
    assert client.delete(f"/api/voices/{voice['id']}", headers=AUTH).status_code == 204
    assert client.get("/api/settings", headers=AUTH).json()["voice"] is None
    assert client.get("/api/voices", headers=AUTH).json() == []
    assert client.delete(f"/api/voices/{voice['id']}", headers=AUTH).status_code == 404


def test_omnivoice_tags_from_a_description():
    tags = speech_worker.omni_tags
    assert tags("A calm older British man with a deep voice") == (
        "male, elderly, low pitch, british accent"
    )
    assert tags("A young woman whispering softly") == "female, young adult, whisper"
    # "woman" doesn't count as "man", and only one tag per group.
    assert tags("A woman, a man") == "female"
    assert tags("Something indescribable") == ""


def test_backends_check_profiles():
    class Clone(speech_worker.Backend):
        accepts = ("clone",)
        needs_text = True

    b = Clone(Path("."), {})
    assert b.check({"kind": "design", "description": "x"}) is None
    assert b.check(None) is None
    try:
        b.check({"kind": "clone", "audio": "a.wav", "text": None})
    except ValueError as exc:
        assert "words spoken" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
    assert speech_worker.Backend.described("Bright host", None).startswith("A young woman")
