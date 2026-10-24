import asyncio
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import catalog
from app.config import Settings
from app.engines import ENGINES, EngineError, EngineManager, torch_fix
from app.hardware import Gpu, Hardware
from app.jobs import JobRegistry
from app.main import create_app

AUTH = {"Authorization": "Bearer k"}
PNG = base64.b64encode(b"\x89PNG fake").decode()


def variant(tag):
    return catalog.find_variant(tag)[1]


@pytest.mark.parametrize(
    ("tag", "hw", "fit"),
    [
        ("flux2-klein-4b", Hardware(gpus=[Gpu("2080 Ti", 11, 10)], ram_total_gb=32), "gpu"),
        ("z-image-turbo", Hardware(gpus=[Gpu("2080 Ti", 11, 10)], ram_total_gb=64), "partial"),
        ("z-image-turbo", Hardware(gpus=[Gpu("2080 Ti", 11, 10)], ram_total_gb=16), "too_large"),
        ("sd15", Hardware(ram_total_gb=16), "cpu"),
        ("kokoro-82m", Hardware(), "light"),
    ],
)
def test_engine_variant_fit(tag, hw, fit):
    assert catalog.assess_variant(variant(tag), hw) == fit


def test_catalog_engine_entries_are_complete():
    for fam in catalog.load()["families"]:
        if fam["engine"] in ENGINES:
            for v in fam["variants"]:
                assert "source" in v["engine_config"], v["tag"]
                cfg = v["engine_config"]
                if fam["engine"] == "image":
                    shape = (
                        {"width", "height", "frames"} if cfg.get("task") == "video" else {"size"}
                    )
                    assert {"steps", "guidance", "full_gb", "component_gb"} | shape <= cfg.keys()


def mark_installed(engines: EngineManager, engine: str, tag: str):
    engines.root.joinpath(engine).mkdir(parents=True, exist_ok=True)
    engines._marker(engine).write_text(
        json.dumps({"version": ENGINES[engine].version, "gpu": engines.gpu})
    )
    d = engines.model_dir(engine, tag)
    d.mkdir(parents=True)
    (d / ".amy-model.json").write_text(json.dumps(variant(tag)))


class FakeEngines:
    """Stands in for EngineManager.call, recording requests."""

    def __init__(self):
        self.calls = []

    async def __call__(self, engine, path, payload, timeout=0):
        self.calls.append((engine, path, payload))
        if path == "/generate":
            return httpx.Response(200, json={"images": [PNG] * payload["n"], "seed": 42})
        if path == "/transcribe":
            return httpx.Response(200, json={"text": "hello world", "language": "en"})
        if path == "/speak":
            return httpx.Response(200, content=b"RIFFfake", headers={"content-type": "audio/wav"})
        if path == "/voices":
            return httpx.Response(200, json={"voices": ["af_heart"]})
        if path == "/generate_video":
            from pathlib import Path

            Path(payload["output"]).write_bytes(b"fake mp4")
            return httpx.Response(
                200,
                json={
                    "seed": 7,
                    "frames": payload["frames"],
                    "fps": payload["fps"],
                    "generate_seconds": 1.0,
                },
            )
        return httpx.Response(404)


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, api_key="k", ollama_url="", static_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        fake = FakeEngines()
        monkeypatch.setattr(c.app.state.engines, "call", fake)
        c.fake = fake
        yield c


def test_not_installed_is_409(client):
    r = client.post(
        "/v1/images/generations", headers=AUTH, json={"model": "image/sdxl-base-1.0", "prompt": "x"}
    )
    assert r.status_code == 409


def test_generate_saves_to_gallery(client):
    mark_installed(client.app.state.engines, "image", "sdxl-base-1.0")
    r = client.post(
        "/v1/images/generations",
        headers=AUTH,
        json={"model": "image/sdxl-base-1.0", "prompt": "a fox", "size": "1000x700", "n": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 2 and body["seed"] == 42
    sent = client.fake.calls[0][2]
    assert (sent["width"], sent["height"]) == (1024, 704)

    gallery = client.get("/api/images", headers=AUTH).json()
    assert len(gallery) == 2 and gallery[0]["prompt"] == "a fox"
    img = client.get(f"/api/images/{gallery[0]['id']}", headers=AUTH)
    assert img.status_code == 200 and img.content.startswith(b"\x89PNG")
    assert client.delete(f"/api/images/{gallery[0]['id']}", headers=AUTH).status_code == 204
    assert len(client.get("/api/images", headers=AUTH).json()) == 1


def test_negative_prompt_only_when_supported(client):
    mark_installed(client.app.state.engines, "image", "flux2-klein-4b")
    client.post(
        "/v1/images/generations",
        headers=AUTH,
        json={"model": "flux2-klein-4b", "prompt": "x", "negative_prompt": "blurry"},
    )
    assert client.fake.calls[0][2]["negative_prompt"] is None


def test_transcription(client):
    mark_installed(client.app.state.engines, "audio", "whisper-small")
    r = client.post(
        "/v1/audio/transcriptions",
        headers=AUTH,
        data={"model": "audio/whisper-small"},
        files={"file": ("clip.webm", b"audio-bytes", "audio/webm")},
    )
    assert r.json() == {"text": "hello world"}
    assert client.fake.calls[0][2]["path"].endswith(".webm")
    assert not list((client.app.state.settings.data_dir / "tmp").iterdir())  # temp file cleaned


def test_speech_and_task_check(client):
    mark_installed(client.app.state.engines, "audio", "kokoro-82m")
    mark_installed(client.app.state.engines, "audio", "whisper-small")
    r = client.post(
        "/v1/audio/speech", headers=AUTH, json={"model": "audio/kokoro-82m", "input": "hi"}
    )
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    r = client.post(
        "/v1/audio/speech", headers=AUTH, json={"model": "audio/whisper-small", "input": "hi"}
    )
    assert r.status_code == 400


def test_models_list_includes_installed_engine_models(client):
    mark_installed(client.app.state.engines, "audio", "kokoro-82m")
    models = client.get("/v1/models", headers=AUTH).json()["data"]
    kokoro = next(m for m in models if m["id"] == "audio/kokoro-82m")
    assert kokoro["capability"] == "speech"


def test_pull_routes_engine_models(client, monkeypatch):
    started = []
    monkeypatch.setattr(
        client.app.state.engines,
        "start_install",
        lambda engine, v, *rest: (
            started.append((engine, v["tag"])) or client.app.state.jobs.start(v["tag"], _noop)
        ),
    )
    r = client.post("/api/models/pull", headers=AUTH, json={"model": "kokoro-82m"})
    assert r.status_code == 202 and started == [("audio", "kokoro-82m")]


async def _noop(state):
    pass


def test_catalog_marks_engine_install_state(client):
    mark_installed(client.app.state.engines, "audio", "kokoro-82m")
    cat = client.get("/api/catalog", headers=AUTH).json()
    kokoro = next(f for f in cat["families"] if f["id"] == "kokoro")
    flux = next(f for f in cat["families"] if f["id"] == "flux2-klein")
    assert kokoro["engine_installed"] and kokoro["variants"][0]["installed"]
    assert not flux["engine_installed"] and "7 GB" in flux["engine_note"]
    assert "engine_config" not in flux["variants"][0]


def test_install_flow(tmp_path, monkeypatch):
    asyncio.run(_install_flow(tmp_path, monkeypatch))


async def _install_flow(tmp_path, monkeypatch):
    engines = EngineManager(tmp_path, JobRegistry(), gpu=False, torch_index_url="x")
    steps = []

    async def fake_engine(spec, state):
        steps.append("engine")
        engines.root.joinpath(spec.id).mkdir(parents=True, exist_ok=True)
        engines._marker(spec.id).write_text(json.dumps({"version": spec.version, "gpu": False}))

    async def fake_download(engine, v, state, hf_token=None):
        steps.append("download")
        d = engines.model_dir(engine, v["tag"])
        d.mkdir(parents=True)
        (d / ".amy-model.json").write_text("{}")

    monkeypatch.setattr(engines, "_install_engine", fake_engine)
    monkeypatch.setattr(engines, "_download_model", fake_download)
    state = engines.start_install("audio", variant("kokoro-82m"))
    await engines.jobs._tasks.copy().pop()
    assert state.done and state.error is None and steps == ["engine", "download"]
    assert engines.model_installed("audio", "kokoro-82m")

    # Second model reuses the engine.
    engines.start_install("audio", variant("whisper-small"))
    await engines.jobs._tasks.copy().pop()
    assert steps == ["engine", "download", "download"]


def test_worker_env_adds_nvidia_libs(tmp_path):
    engines = EngineManager(tmp_path, JobRegistry(), gpu=True, torch_index_url="x")
    lib = engines.venv("audio") / "lib/python3.12/site-packages/nvidia/cublas/lib"
    lib.mkdir(parents=True)
    env = engines._env("audio")
    assert str(lib) in env["LD_LIBRARY_PATH"] and env["HF_HUB_OFFLINE"] == "1"


def test_video_job_flow(client):
    import time

    mark_installed(client.app.state.engines, "image", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    r = client.post(
        "/v1/videos/generations",
        headers=AUTH,
        json={
            "model": "image/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt": "a fox",
            "frames": 40,
            "width": 830,
        },
    )
    assert r.status_code == 202
    vid = r.json()["id"]
    for _ in range(50):
        job = next(j for j in client.get("/api/videos/jobs", headers=AUTH).json() if j["id"] == vid)
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    sent = client.fake.calls[-1][2]
    assert sent["frames"] == 33 and sent["width"] == 832  # 8k+1 frames, multiples of 32
    gallery = client.get("/api/videos", headers=AUTH).json()
    assert gallery[0]["id"] == vid and gallery[0]["params"]["seed"] == 7
    assert client.get(f"/api/videos/{vid}", headers=AUTH).content == b"fake mp4"
    assert client.delete(f"/api/videos/{vid}", headers=AUTH).status_code == 204


def test_image_endpoint_rejects_video_models(client):
    mark_installed(client.app.state.engines, "image", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    r = client.post(
        "/v1/images/generations",
        headers=AUTH,
        json={"model": "image/Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "prompt": "x"},
    )
    assert r.status_code == 400


@pytest.mark.parametrize(
    ("versions", "flavour", "fix"),
    [
        # A CUDA 12.8 torchaudio next to the newest torch from PyPI (its library won't load).
        (
            {"torch": "2.14.1", "torchaudio": "2.11.0+cu128", "cuda": "13"},
            "cu128",
            ["torch==2.11.0+cu128"],
        ),
        ({"torch": "2.11.0+cu128", "torchaudio": "2.11.0+cu128"}, "cu128", []),
        # A library pins torch and PyPI's build matches the CUDA index: left alone.
        ({"torch": "2.9.1", "cuda": "12"}, "cu128", []),
        ({"torch": "2.12.0", "cuda": "13"}, "cu128", ["torch==2.12.0+cu128"]),
        ({"torch": "2.14.1", "torchaudio": "2.11.0"}, "cpu", ["torch==2.11.0+cpu"]),
        ({"torch": "2.8.0"}, "cpu", []),
        ({}, "cu128", []),
    ],
)
def test_torch_matches_torchaudio_and_machine(versions, flavour, fix):
    assert torch_fix(versions, flavour) == fix


def test_out_of_gpu_memory_frees_the_gpu_and_retries(tmp_path, monkeypatch):
    engines = EngineManager(tmp_path, JobRegistry(), gpu=True, torch_index_url="x")
    calls, freed = [], []

    async def fake_call(engine, path, payload, timeout):
        calls.append(engine)
        if len(calls) == 1:
            raise EngineError("OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB")
        return httpx.Response(200, json={"ok": True})

    async def release():
        freed.append("ollama")

    monkeypatch.setattr(engines, "_call", fake_call)
    engines.release_gpu = release
    resp = asyncio.run(engines.call("tts-qwen", "/speak", {}))
    assert resp.json() == {"ok": True} and calls == ["tts-qwen", "tts-qwen"] and freed == ["ollama"]

    # Other errors aren't retried.
    async def broken(engine, path, payload, timeout):
        raise EngineError("Could not load this library")

    monkeypatch.setattr(engines, "_call", broken)
    with pytest.raises(EngineError, match="Could not load"):
        asyncio.run(engines.call("tts-qwen", "/speak", {}))


def test_repair_matches_torch_once(tmp_path, monkeypatch):
    engines = EngineManager(tmp_path, JobRegistry(), gpu=True, torch_index_url="x")
    spec = ENGINES["tts-qwen"]
    engines.root.joinpath(spec.id).mkdir(parents=True)
    engines._marker(spec.id).write_text(json.dumps({"version": spec.version, "gpu": True}))
    aligned = []

    async def fake_align(engine, state):
        aligned.append(engine)

    monkeypatch.setattr(engines, "_align_torch", fake_align)
    asyncio.run(engines.repair())
    asyncio.run(engines.repair())
    assert aligned == ["tts-qwen"]
    assert json.loads(engines._marker(spec.id).read_text())["torch_aligned"] is True
