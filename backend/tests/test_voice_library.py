"""The voice library, starter voices, engine overlays and the HTTPS server."""

import io
import sys
import tarfile
from pathlib import Path

import pytest

from app import serve
from app import voice_library as library
from app.engines import EngineError, extract_package
from tests.test_custom_voices import answer_voices, wav
from tests.test_delegation import AUTH, client, install  # noqa: F401 - fixture

sys.path.insert(0, str(Path(__file__).parents[1] / "app" / "engines"))
import speech_base  # noqa: E402

FILES = [
    "README.md",
    "vctk/p225_023.wav",
    "vctk/p225_023_enhanced.wav",
    "vctk/p225_023.wav.1e68beda@240.safetensors",
    "voice-donations/Selfie.wav",
    "voice-donations/0a67.wav",
    "voice-zero/caro_davy.wav",
    "expresso/ex01-ex02_default_001_channel1_168s.wav",  # non-commercial: left out
    "ears/p001/freeform_speech_01.wav",  # non-commercial: left out
]


def test_library_keeps_free_collections_and_prefers_enhanced_clips():
    entries = {e["id"]: e for e in library.entries(FILES)}
    assert set(entries) == {
        "vctk/p225_023_enhanced.wav",
        "voice-donations/Selfie.wav",
        "voice-donations/0a67.wav",
        "voice-zero/caro_davy.wav",
    }
    vctk = entries["vctk/p225_023_enhanced.wav"]
    assert vctk["name"] == "Speaker 225" and vctk["license"] == "CC BY 4.0"
    assert "Edinburgh" in vctk["credit"] and vctk["preview"].startswith("https://huggingface.co/")
    assert entries["voice-donations/0a67.wav"]["name"] == "Donor 0a67"
    assert entries["voice-donations/Selfie.wav"]["name"] == "Selfie"
    assert entries["voice-zero/caro_davy.wav"]["name"] == "Caro Davy"


@pytest.fixture
def fake_library(monkeypatch):
    async def listing(http, token):
        return library.entries(FILES)

    async def download(http, token, path):
        return wav(6)

    async def no_suggestions(request):
        return []

    from app.api import media

    monkeypatch.setattr(library, "listing", listing)
    monkeypatch.setattr(library, "download", download)
    monkeypatch.setattr(media, "_suggested_runners", no_suggestions)  # no catalog fetch


def test_install_a_library_voice(client, fake_library):  # noqa: F811
    voices = client.get("/api/voices/library", headers=AUTH).json()["voices"]
    assert all(v["installed"] is None for v in voices)
    r = client.post("/api/voices/library", headers=AUTH, json={"id": "voice-zero/caro_davy.wav"})
    assert r.status_code == 201, r.text
    voice = r.json()
    assert voice["name"] == "Caro Davy" and voice["kind"] == "clone" and voice["seconds"] == 6
    assert voice["credit"] == "Voice-Zero, from LibriVox (CC0)"
    again = client.post(
        "/api/voices/library", headers=AUTH, json={"id": "voice-zero/caro_davy.wav"}
    )
    assert again.status_code == 409
    missing = client.post("/api/voices/library", headers=AUTH, json={"id": "expresso/x.wav"})
    assert missing.status_code == 404
    listed = {v["id"]: v for v in client.get("/api/voices/library", headers=AUTH).json()["voices"]}
    assert listed["voice-zero/caro_davy.wav"]["installed"] == voice["id"]
    # It's an ordinary custom voice: offered to models that copy voices, with its sample.
    assert client.get(f"/api/voices/{voice['id']}/sample", headers=AUTH).status_code == 200


def test_voice_details_get_labels(client):  # noqa: F811
    variant = {
        "tag": "kyutai/tts-1.6b-en_fr",
        "size_gb": 2,
        "engine_config": {"task": "speech", "runtime": "tts-kyutai", "backend": "kyutai"},
    }
    install(client.app, "audio", variant["tag"], variant)
    details = {
        "F1": {"name": "Female 1", "language": "Voices", "gender": "Female"},
        "vctk/p225_023.wav": {"name": "Speaker 225", "language": "VCTK (English accents)"},
        "Caro": {"name": "Caro", "language": "Starter voices", "gender": "Female"},
    }
    voices = [*details, "plain"]
    answer_voices(client, {"voices": voices, "details": details, "accepts": []})
    r = client.get("/api/audio/voices?model=audio/kyutai/tts-1.6b-en_fr", headers=AUTH)
    got = r.json()["details"]
    # A set's name moves out of "language"; labels the voice repo has are added.
    assert got["F1"] == {"name": "Female 1", "gender": "Female", "group": "Voices"}
    assert got["vctk/p225_023.wav"] == {
        "name": "Speaker 225",
        "group": "VCTK (English accents)",
        "gender": "Female",
        "age": 23,
        "accent": "English",
        "region": "Southern England",
        "language": "English",
    }
    assert got["Caro"]["style"] == "Audiobook" and "estimated" not in got["Caro"]
    assert "plain" not in got


def test_library_voices_carry_labels(client, fake_library):  # noqa: F811
    voices = {v["id"]: v for v in client.get("/api/voices/library", headers=AUTH).json()["voices"]}
    assert voices["vctk/p225_023_enhanced.wav"]["accent"] == "English"
    donor = voices["voice-donations/0a67.wav"]
    assert donor["language"] == "" and donor["estimated"] is True and donor["gender"]
    r = client.post("/api/voices/library", headers=AUTH, json={"id": "voice-zero/caro_davy.wav"})
    assert r.json()["meta"]["gender"] == "Female"  # kept once installed


def test_starter_voices_only_when_downloaded(monkeypatch):
    def missing(repo, filename):
        raise FileNotFoundError(filename)

    monkeypatch.setattr(speech_base, "hub_file", missing)

    class CloneOnly(speech_base.Backend):
        accepts = ("clone",)
        starters = True

    b = CloneOnly(Path("."), {})
    assert b.voices() == [] and b.details() == {}
    with pytest.raises(ValueError, match="Voices page"):
        b.clone(None, None)
    monkeypatch.setattr(speech_base, "hub_file", lambda repo, f: f"/cache/{f}")
    assert b.voices() == list(speech_base.STARTERS)
    clip = b.clone("Peter", None)
    assert clip["audio"] == "/cache/voice-zero/peter_yearsley.wav" and "north wind" in clip["text"]
    mine = {"kind": "clone", "audio": "/data/voices/x.wav", "text": None}
    assert b.clone("Peter", mine) is mine


def tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_overlay_replaces_the_installed_package(tmp_path):
    dest = tmp_path / "site-packages" / "zonos"
    dest.mkdir(parents=True)
    (dest / "__init__.py").write_text("partial")
    archive = tarball({
        "Zonos-abc/zonos/__init__.py": b"full",
        "Zonos-abc/zonos/backbone/__init__.py": b"",
        "Zonos-abc/README.md": b"not copied",
    })  # fmt: skip
    extract_package(archive, "zonos", dest)
    assert (dest / "__init__.py").read_text() == "full"
    assert (dest / "backbone" / "__init__.py").exists() and not (dest / "README.md").exists()
    with pytest.raises(EngineError):
        extract_package(archive, "nothere", tmp_path / "x")


def test_https_server_only_with_a_certificate(tmp_path, monkeypatch):
    monkeypatch.setenv("AMY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AMY_TLS_DIR", str(tmp_path / "tls"))
    monkeypatch.setenv("AMY_CATALOG_WARMUP", "false")
    assert [c.port for c in serve.configs()] == [8484]
    (tmp_path / "tls").mkdir()
    (tmp_path / "tls" / "cert.pem").write_text("cert")
    (tmp_path / "tls" / "key.pem").write_text("key")
    plain, secure = serve.configs()
    assert secure.port == 8443 and secure.ssl_certfile.endswith("cert.pem")
    assert secure.lifespan == "off" and plain.app is secure.app
    monkeypatch.setenv("AMY_HTTPS_PORT", "0")
    assert len(serve.configs()) == 1


def test_library_says_which_models_speak_each_voice(client, fake_library, monkeypatch):  # noqa: F811
    from app import voice_runners
    from app.api import media

    async def suggestions(request):
        return [{"tag": "ResembleAI/chatterbox", "name": "chatterbox"}]

    monkeypatch.setattr(media, "_suggested_runners", suggestions)
    listing = client.get("/api/voices/library", headers=AUTH).json()
    assert listing["models"] == [] and listing["suggested"][0]["name"] == "chatterbox"
    assert all(v["runs_on"] == [] for v in listing["voices"])

    for repo, runtime in (
        ("kyutai/tts-1.6b-en_fr", "tts-kyutai"),  # VCTK and Alba voices built in
        ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "tts-qwen"),  # copies any sample
        ("SWivid/F5-TTS", "tts-f5"),  # copies, but needs the sample's words
        ("Supertone/supertonic-3", "tts-supertonic"),  # its own voices only
    ):
        cfg = {"task": "speech", "runtime": runtime, "backend": runtime.removeprefix("tts-")}
        install(client.app, "audio", repo, {"tag": repo, "size_gb": 1, "engine_config": cfg})
    listing = client.get("/api/voices/library", headers=AUTH).json()
    how = {m["id"]: m["how"] for m in listing["models"]}
    assert how == {
        "audio/kyutai/tts-1.6b-en_fr": "built_in",
        "audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base": "copies",
        "audio/SWivid/F5-TTS": "copies_with_text",
    }
    runs = {v["id"]: set(v["runs_on"]) for v in listing["voices"]}
    # No speech-to-text model to write down the words, so F5 can't take them yet.
    assert runs["vctk/p225_023_enhanced.wav"] == {
        "audio/kyutai/tts-1.6b-en_fr",
        "audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    }
    assert runs["voice-zero/caro_davy.wav"] == {"audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base"}
    assert listing["suggested"] == []  # every voice has a model

    rows = [
        {"tag": "kokoro-82m", "name": "Kokoro", "runtime": None},
        {"tag": "ResembleAI/chatterbox", "name": "chatterbox", "runtime": "tts-chatterbox"},
        {"tag": "x/gated", "name": "g", "runtime": "tts-xtts", "requires_token": True},
        {"tag": "kyutai/tts-1.6b-en_fr", "name": "tts-1.6b", "runtime": "tts-kyutai"},
    ]
    assert [s["name"] for s in voice_runners.suggested(rows, library=True)] == [
        "chatterbox",
        "tts-1.6b",
    ]
    assert [s["name"] for s in voice_runners.suggested(rows)] == ["chatterbox"]


def test_your_voices_say_where_they_can_be_used(client, fake_library, monkeypatch):  # noqa: F811
    from app.api import media

    async def rows(request):
        return [
            {"tag": "ResembleAI/chatterbox", "name": "chatterbox", "runtime": "tts-chatterbox"},
            {"tag": "x/parler", "name": "parler", "runtime": "tts-parler"},
        ]

    monkeypatch.setattr(media, "_speech_rows", rows)
    vctk = client.post(
        "/api/voices/library", headers=AUTH, json={"id": "vctk/p225_023_enhanced.wav"}
    ).json()
    caro = client.post(
        "/api/voices/library", headers=AUTH, json={"id": "voice-zero/caro_davy.wav"}
    ).json()
    form = {"name": "Calm", "kind": "design", "description": "A calm older man"}
    calm = client.post("/api/voices", headers=AUTH, data=form).json()

    got = client.get("/api/voices/runners", headers=AUTH).json()
    assert got["voices"] == {vctk["id"]: [], caro["id"]: [], calm["id"]: []}
    assert [s["name"] for s in got["suggested"]["clone"]] == ["chatterbox"]
    assert [s["name"] for s in got["suggested"]["design"]] == ["parler"]

    for repo, runtime in (
        ("kyutai/tts-1.6b-en_fr", "tts-kyutai"),
        ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "tts-qwen"),
        ("SWivid/F5-TTS", "tts-f5"),  # no transcripts here, so no use for these clones
    ):
        cfg = {"task": "speech", "runtime": runtime, "backend": runtime.removeprefix("tts-")}
        install(client.app, "audio", repo, {"tag": repo, "size_gb": 1, "engine_config": cfg})
    got = client.get("/api/voices/runners", headers=AUTH).json()
    qwen = "audio/Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    kyutai = "audio/kyutai/tts-1.6b-en_fr"
    # Kyutai 1.6B speaks the VCTK speaker with its own built-in copy.
    assert sorted(got["voices"][vctk["id"]], key=lambda w: w["model"]) == [
        {"model": qwen, "voice": vctk["id"]},
        {"model": kyutai, "voice": "vctk/p225_023.wav"},
    ]
    assert got["voices"][caro["id"]] == [{"model": qwen, "voice": caro["id"]}]
    assert got["voices"][calm["id"]] == []
    assert list(got["suggested"]) == ["design"]
